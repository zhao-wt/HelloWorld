"""
forecast/models.py — walk-forward, expanding-window member models.

Each MEMBER is a model family that maps the predictor panel to a forecast of
the forward h-month total return. Members (Welch-Goyal benchmark + four ML
families):

    mean   prevailing historical mean (the benchmark every member must beat)
    enet   ElasticNetCV — regularized linear regression
    knn    k-nearest-neighbor regression
    rf     random forest regression (shallow, regularized)
    mlp    neural network (single hidden layer, strongly regularized)

No-look-ahead discipline
------------------------
A forecast made at month t for the h-month forward return can only be trained on
months t' whose h-month window has ALREADY realized, i.e. t' <= t - h. (The
predictors are already publication-lagged in forecast/features.py.) The training
window is standardized on itself; any not-yet-published predictor is mean-filled
(0 after standardization), matching bear/inference.py.

Forecasts are clipped to the training-window 1st-99th percentile of realized
returns (a mild Campbell-Thompson style sanity bound) so extrapolating learners
(enet/mlp) cannot emit absurd values.

Outputs (read by forecast/ensemble.py and the app — no ML on the app path):
    data/forecast_members_{1,3,6,12}m.csv   walk-forward OOS preds (cols=members)
    data/forecast_current.json              current forecast per (member, horizon)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from forecast.features import PREDICTORS

_FORECAST_DIR = Path(__file__).resolve().parent
_DATA_DIR = _FORECAST_DIR.parent / "data"

HORIZONS = [1, 3, 6, 12]

TRAIN_START = pd.Timestamp("1930-01-31")
OOS_START = pd.Timestamp("1960-01-31")
MIN_TRAIN = 120          # need >= 10 years of observed targets before predicting

# Member registry. refit_every = months between full refits in the walk-forward
# (cheap learners refit monthly; rf/mlp refit twice a year to keep runtime sane —
# still honest OOS, just a slightly staler-but-past-only model between refits).
MEMBERS: dict[str, dict] = {
    "mean": {"label": "Prevailing mean",  "type": "mean", "refit_every": 1},
    "enet": {"label": "Elastic net",      "type": "enet", "refit_every": 3},
    "knn":  {"label": "k-Nearest neighbor", "type": "knn", "refit_every": 1},
    "rf":   {"label": "Random forest",    "type": "rf",   "refit_every": 6},
    "mlp":  {"label": "Neural net (MLP)", "type": "mlp",  "refit_every": 6},
}

# Members that combine into the ensemble (the benchmark `mean` is shown for
# reference / R²_OS but is NOT part of the predictive combination).
ENSEMBLE_MEMBERS = ["enet", "knn", "rf", "mlp"]


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------

def _make_estimator(mtype: str):
    """Return a fresh sklearn estimator for the model family (or None=mean)."""
    if mtype == "mean":
        return None
    if mtype == "enet":
        from sklearn.linear_model import ElasticNetCV
        return ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], n_alphas=30, cv=3,
                            max_iter=8000, random_state=42)
    if mtype == "knn":
        from sklearn.neighbors import KNeighborsRegressor
        return KNeighborsRegressor(n_neighbors=40, weights="uniform")
    if mtype == "rf":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=300, max_depth=4,
                                     min_samples_leaf=20, max_features="sqrt",
                                     random_state=42, n_jobs=-1)
    if mtype == "mlp":
        from sklearn.neural_network import MLPRegressor
        # Strongly regularized, small net with early stopping: returns are mostly
        # noise, so an unregularized MLP fits noise and explodes OOS.
        return MLPRegressor(hidden_layer_sizes=(8,), activation="relu",
                            alpha=3.0, max_iter=2000, random_state=42,
                            early_stopping=True, n_iter_no_change=20,
                            validation_fraction=0.15)
    raise ValueError(f"unknown member type {mtype}")


def _standardize(train: pd.DataFrame, test: pd.DataFrame):
    """Standardize on the training window; mean-fill missing (0) afterwards."""
    mu = train.mean()
    sig = train.std(ddof=1).replace(0.0, np.nan)
    tr = ((train - mu) / sig).fillna(0.0)
    te = ((test - mu) / sig).fillna(0.0)
    return tr.values, te.values


def _fit_predict(mtype: str, X_tr, y_tr, X_te) -> np.ndarray:
    """Fit a member on (X_tr, y_tr), return predictions for X_te rows."""
    if mtype == "mean":
        return np.full(len(X_te), float(np.mean(y_tr)))
    est = _make_estimator(mtype)
    est.fit(X_tr, y_tr)
    return est.predict(X_te)


# ---------------------------------------------------------------------------
# Walk-forward OOS for one (member, horizon)
# ---------------------------------------------------------------------------

def walk_forward(
    features: pd.DataFrame,
    y: pd.Series,
    member: str,
    horizon: int,
) -> pd.Series:
    """
    Expanding-window OOS forecasts for a member at a horizon.

    At each test month t >= OOS_START, train only on months t' <= t - horizon
    whose target has realized (no look-ahead), refit at the member's cadence,
    predict t, and clip to the training-window 1-99 pct of realized returns.
    """
    mtype = MEMBERS[member]["type"]
    refit_every = MEMBERS[member]["refit_every"]
    X = features[PREDICTORS]
    idx = features.index
    out = pd.Series(np.nan, index=idx, name=member)

    cached = None            # (model_state) — we just re-call _fit_predict
    last_fit_pos = -10**9
    test_positions = [i for i, t in enumerate(idx) if t >= OOS_START]

    for i in test_positions:
        t = idx[i]
        cutoff = t - pd.DateOffset(months=horizon)     # label must have realized
        tr_mask = (idx <= cutoff) & (idx >= TRAIN_START) & y.notna()
        if tr_mask.sum() < MIN_TRAIN:
            continue

        # Refit only every `refit_every` months (expanding window of past data).
        need_refit = (i - last_fit_pos) >= refit_every or cached is None
        X_tr = X.loc[tr_mask]
        y_tr = y.loc[tr_mask].values.astype(float)
        lo, hi = np.percentile(y_tr, [1, 99])

        if mtype == "mean":
            pred = float(np.mean(y_tr))
        else:
            if need_refit:
                X_tr_sc, _ = _standardize(X_tr, X.loc[[t]])
                est = _make_estimator(mtype)
                est.fit(X_tr_sc, y_tr)
                cached = (est, X_tr.mean(), X_tr.std(ddof=1).replace(0.0, np.nan))
                last_fit_pos = i
            est, mu, sig = cached
            x_te = ((X.loc[[t]] - mu) / sig).fillna(0.0).values
            pred = float(est.predict(x_te)[0])

        out.loc[t] = float(np.clip(pred, lo, hi))

    return out.dropna()


def current_forecast(features: pd.DataFrame, y: pd.Series,
                     member: str, horizon: int) -> tuple[float, str]:
    """Fit on all observed history, predict the latest complete feature row."""
    mtype = MEMBERS[member]["type"]
    X = features[PREDICTORS]
    as_of = X.dropna(how="all").index[-1]

    tr_mask = (X.index >= TRAIN_START) & y.notna()
    X_tr = X.loc[tr_mask]
    y_tr = y.loc[tr_mask].values.astype(float)
    lo, hi = np.percentile(y_tr, [1, 99])

    if mtype == "mean":
        pred = float(np.mean(y_tr))
    else:
        X_tr_sc, X_te_sc = _standardize(X_tr, X.loc[[as_of]])
        pred = float(_fit_predict(mtype, X_tr_sc, y_tr, X_te_sc)[0])

    return float(np.clip(pred, lo, hi)), str(as_of.date())


# ---------------------------------------------------------------------------
# Build everything
# ---------------------------------------------------------------------------

def build_all() -> None:
    feats = pd.read_csv(_DATA_DIR / "forecast_features.csv",
                        index_col=0, parse_dates=True)
    targets = pd.read_csv(_DATA_DIR / "forecast_targets.csv",
                          index_col=0, parse_dates=True)

    current: dict[str, dict] = {}
    for h in HORIZONS:
        y = targets[f"ret_{h}m"]
        print(f"\n{'='*64}\n  Horizon {h}m — walk-forward OOS members\n{'='*64}")
        cols = {}
        for member in MEMBERS:
            t0 = time.time()
            oos = walk_forward(feats, y, member, h)
            cur, as_of = current_forecast(feats, y, member, h)
            cols[member] = oos
            current.setdefault(str(h), {})[member] = cur
            current.setdefault("as_of", as_of)
            print(f"  {member:<6} OOS n={len(oos):>4}  "
                  f"current={cur:+.2%}  ({time.time()-t0:.1f}s)")
        df = pd.DataFrame(cols).sort_index()
        df["realized"] = y.reindex(df.index)
        out = _DATA_DIR / f"forecast_members_{h}m.csv"
        df.to_csv(out, date_format="%Y-%m-%d", float_format="%.6f")
        print(f"  wrote {out.name}  ({len(df)} rows)")

    with open(_DATA_DIR / "forecast_current.json", "w") as fh:
        json.dump(current, fh, indent=2)
    print(f"\nWrote data/forecast_current.json  (as_of {current.get('as_of')})")


if __name__ == "__main__":
    build_all()
