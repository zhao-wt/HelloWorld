"""
forecast/univariate.py — single-predictor leaderboard for the Market Forecast.

For each predictor x horizon, fits a one-variable linear (OLS) walk-forward
forecast of the forward h-month return and reports its out-of-sample skill. This
mirrors bear/univariate.py but for the continuous-return target and the
Campbell-Thompson R²_OS metric.

Columns (per predictor, for each horizon):
    R2_OS   out-of-sample R^2 vs prevailing mean
    hit     directional hit-rate
    HAC p   Newey-West p-value of the in-sample slope (lag = horizon), shown for
            information only — return predictability is weak and we do NOT gate
            on significance (members earn inclusion by OOS skill, not p-values).
    coef    in-sample standardized slope (sign = direction of the relationship)
    current the predictor's latest forecast at this horizon

Output: data/univariate_forecast.csv  (one row per predictor; multi-horizon cols)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from forecast.features import PREDICTORS, PREDICTOR_INFO
from forecast.models import TRAIN_START, OOS_START, MIN_TRAIN
from forecast.ensemble import r2_oos

_FORECAST_DIR = Path(__file__).resolve().parent
_DATA_DIR = _FORECAST_DIR.parent / "data"
HORIZONS = [1, 3, 6, 12]


def _hac_pvalue_slope(x: np.ndarray, y: np.ndarray, maxlags: int) -> float:
    """Newey-West HAC p-value for the slope of a simple OLS y ~ a + b x."""
    from math import erf, sqrt
    n = len(x)
    X = np.column_stack([np.ones(n), x])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta
    u = resid[:, None] * X
    S = u.T @ u
    for lag in range(1, min(maxlags, n - 1) + 1):
        w = 1.0 - lag / (maxlags + 1.0)
        G = u[lag:].T @ u[:-lag]
        S += w * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    se = np.sqrt(max(V[1, 1], 1e-18))
    z = beta[1] / se
    return 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(z) / sqrt(2.0))))


def _walk_forward_ols(x: pd.Series, y: pd.Series, horizon: int) -> pd.Series:
    """Expanding-window single-predictor OLS forecast (no look-ahead)."""
    idx = x.index
    out = pd.Series(np.nan, index=idx)
    for t in idx[idx >= OOS_START]:
        cutoff = t - pd.DateOffset(months=horizon)
        m = (idx <= cutoff) & (idx >= TRAIN_START) & x.notna() & y.notna()
        if m.sum() < MIN_TRAIN:
            continue
        xtr = x.loc[m].values
        ytr = y.loc[m].values
        mu, sig = xtr.mean(), xtr.std(ddof=1)
        if sig == 0 or np.isnan(x.loc[t]):
            continue
        xs = (xtr - mu) / sig
        b1, b0 = np.polyfit(xs, ytr, 1)
        out.loc[t] = b0 + b1 * (x.loc[t] - mu) / sig
    return out.dropna()


def build_and_save() -> pd.DataFrame:
    feats = pd.read_csv(_DATA_DIR / "forecast_features.csv",
                        index_col=0, parse_dates=True)
    targets = pd.read_csv(_DATA_DIR / "forecast_targets.csv",
                          index_col=0, parse_dates=True)

    rows = []
    for pred in PREDICTORS:
        label, family = PREDICTOR_INFO[pred]
        x = feats[pred]
        row = {"Predictor": pred, "Description": label, "Family": family,
               "Data since": str(x.dropna().index[0].date()) if x.notna().any() else "-"}
        for h in HORIZONS:
            y = targets[f"ret_{h}m"]
            oos = _walk_forward_ols(x, y, h)
            bench = pd.read_csv(_DATA_DIR / f"forecast_members_{h}m.csv",
                                index_col=0, parse_dates=True)["mean"]
            realized = y
            r2 = r2_oos(oos, realized, bench)
            idx = oos.dropna().index.intersection(realized.dropna().index)
            hit = float(np.mean(np.sign(oos.loc[idx]) == np.sign(realized.loc[idx]))) \
                if len(idx) else float("nan")
            # In-sample HAC slope p-value (full observed history)
            m = x.notna() & y.notna() & (x.index >= TRAIN_START)
            xs = ((x.loc[m] - x.loc[m].mean()) / x.loc[m].std(ddof=1)).values
            pv = _hac_pvalue_slope(xs, y.loc[m].values, h) if m.sum() > 30 else float("nan")
            cur = float(oos.iloc[-1]) if len(oos) else float("nan")
            row[f"R2_OS_{h}m"] = r2
            row[f"hit_{h}m"] = hit
            row[f"p_{h}m"] = pv
            row[f"cur_{h}m"] = cur
        rows.append(row)

    df = pd.DataFrame(rows)
    # Sort by the longest-horizon OOS skill (most informative), within family.
    df = df.sort_values(["Family", "R2_OS_12m"], ascending=[True, False]).reset_index(drop=True)
    out = _DATA_DIR / "univariate_forecast.csv"
    df.to_csv(out, index=False, float_format="%.6f")
    return df


if __name__ == "__main__":
    df = build_and_save()
    print(f"\n{'='*84}\n  Univariate predictor leaderboard (R2_OS by horizon)\n{'='*84}")
    show = df[["Predictor", "Family", "R2_OS_1m", "R2_OS_3m", "R2_OS_6m", "R2_OS_12m"]]
    print(show.to_string(index=False))
    print(f"\nWrote data/univariate_forecast.csv  ({len(df)} predictors)")
