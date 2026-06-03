"""
bear/inference.py — final model inference for the dashboard.

Loads the Phase 2 feature CSVs and Phase 3 targets, fits the two FINAL
models (weight-constrained ≤30%, signs fixed to expected), and exposes
a single helper, load_assessment(), returning everything the Streamlit
app needs:

    * current_prob   — latest probability reading
    * as_of          — date of that reading
    * factors        — per-feature table (raw, z-score, coef, weight, contribution)
    * history        — full historical fitted-probability series
    * coef / intercept

The historical curve is the FINAL model's fitted probability applied to
the full feature history (a consistent single-parameter view for plotting).
Out-of-sample AUC from Phase 6 remains the honest performance metric.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

_BEAR_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Final model specifications  (from Phase 4 / Phase 5 winners)
# ---------------------------------------------------------------------------

BEAR_SPEC = {
    "kind":        "bear",
    "title":       "Bear Market",
    "subtitle":    ">20% drawdown over next 12 months",
    "horizon":     12,
    "features_csv": "bear_features.csv",
    "target_col":  "y_bear",
    "features": ["ntfs_3m_chg", "ts_inv_dummy", "ebp_3m_chg",
                 "baa_zscore_60m", "lei_6m_growth", "ffr_6m_chg"],
    "signs": {"ntfs_3m_chg": +1, "ts_inv_dummy": +1, "ebp_3m_chg": +1,
              "baa_zscore_60m": +1, "lei_6m_growth": -1, "ffr_6m_chg": -1},
    "labels": {
        "ntfs_3m_chg":    ("Near-term forward spread, 3m change", "Yield curve"),
        "ts_inv_dummy":   ("Yield curve inversion (10y-3m < 0)",  "Yield curve"),
        "ebp_3m_chg":     ("Excess bond premium, 3m change",      "Credit"),
        "baa_zscore_60m": ("BAA spread, 5yr z-score",             "Credit"),
        "lei_6m_growth":  ("Leading indicator, 6m growth",        "Leading"),
        "ffr_6m_chg":     ("Fed funds rate, 6m change",           "Policy"),
    },
}

CORR_SPEC = {
    "kind":        "correction",
    "title":       "Correction",
    "subtitle":    "10–20% drawdown over next 6 months",
    "horizon":     6,
    "features_csv": "correction_features.csv",
    "target_col":  "y_corr",
    "features": ["vts_slope", "spx_vs_10ma", "m12_1_mom",
                 "anfci_3m_chg", "cape_20yr_pct", "baa_zscore_24m"],
    "signs": {"vts_slope": +1, "spx_vs_10ma": -1, "m12_1_mom": +1,
              "anfci_3m_chg": +1, "cape_20yr_pct": -1, "baa_zscore_24m": +1},
    "labels": {
        "vts_slope":      ("VIX term structure slope (VIX3M-VIX)", "Volatility"),
        "spx_vs_10ma":    ("S&P 500 vs 10-month MA",               "Trend"),
        "m12_1_mom":      ("12-1 month price momentum",            "Trend"),
        "anfci_3m_chg":   ("Adj. financial conditions, 3m change", "Fin. cond."),
        "cape_20yr_pct":  ("Shiller CAPE, 20yr percentile",        "Valuation"),
        "baa_zscore_24m": ("BAA spread, 2yr z-score",              "Credit"),
    },
}

MAX_WEIGHT = 0.30


# ---------------------------------------------------------------------------
# Weight-constrained logistic fit
# ---------------------------------------------------------------------------

def _fit_constrained(
    X_sc:  np.ndarray,
    y_arr: np.ndarray,
    signs: np.ndarray,
    max_w: float = MAX_WEIGHT,
    C:     float = 1.0,
) -> tuple[np.ndarray, float]:
    """
    Logistic regression with |weight| ≤ max_w per feature, signs fixed.

    Fit with NATURAL (unweighted) likelihood so the output probabilities
    are calibrated to the true event base rate — essential for a dashboard
    that displays probabilities. (Phase 4/5 used balanced weights to
    optimise AUC ranking; ranking is preserved here, only the level is
    recalibrated to reality.)
    """
    n  = len(signs)

    # Warm start from unconstrained natural-weight fit
    m0 = LogisticRegression(C=C, solver="lbfgs",
                            max_iter=1000, random_state=42)
    m0.fit(X_sc, y_arr)
    gamma0 = np.abs(m0.coef_[0])

    def obj(p):
        g = p[:n]; b = p[n]
        logits = np.clip(X_sc @ (signs * g) + b, -500, 500)
        nll = -(y_arr * np.log(expit(logits) + 1e-12)
                + (1 - y_arr) * np.log(1 - expit(logits) + 1e-12)).sum()
        return nll + 0.5 / C * float(np.sum(g ** 2))

    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda p, i=i: max_w * p[:n].sum() - p[i]})

    res = minimize(obj, np.append(gamma0, m0.intercept_[0]),
                   method="SLSQP", constraints=cons,
                   bounds=[(1e-8, None)] * n + [(None, None)],
                   options={"maxiter": 3000, "ftol": 1e-11})
    gamma = np.abs(res.x[:n])
    return signs * gamma, float(res.x[n])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_assessment(kind: Literal["bear", "correction"]) -> dict:
    """
    Fit the final model and return a dict with current reading, factor
    table, and full historical fitted-probability series.
    """
    spec = BEAR_SPEC if kind == "bear" else CORR_SPEC
    feats = spec["features"]

    features_df = pd.read_csv(_BEAR_DIR / spec["features_csv"],
                              index_col=0, parse_dates=True)
    targets     = pd.read_csv(_BEAR_DIR / "targets.csv",
                              index_col=0, parse_dates=True)
    y = targets[spec["target_col"]]

    # Complete training rows
    mask  = features_df[feats].notna().all(axis=1) & y.notna()
    X_tr  = features_df.loc[mask, feats]
    y_tr  = y.loc[mask]
    mu    = X_tr.mean()
    sig   = X_tr.std(ddof=1).replace(0.0, np.nan)
    X_sc  = ((X_tr - mu) / sig).fillna(0.0)

    signs = np.array([spec["signs"][f] for f in feats])
    coef, intercept = _fit_constrained(X_sc.values, y_tr.values, signs)

    weights = np.abs(coef) / np.abs(coef).sum() * 100

    # Historical fitted probability over ALL rows with complete features
    valid   = features_df[feats].notna().all(axis=1)
    X_all   = ((features_df[feats] - mu) / sig).fillna(0.0)
    history = pd.Series(np.nan, index=features_df.index, name="prob")
    history.loc[valid] = expit(X_all.loc[valid].values @ coef + intercept)

    # Current reading = last complete row
    avail     = features_df[feats].dropna()
    as_of     = avail.index[-1]
    row_raw   = avail.loc[as_of]
    row_sc    = (row_raw - mu) / sig
    contribs  = row_sc.values * coef
    current_p = float(expit(row_sc.values @ coef + intercept))

    factor_rows = []
    for f, c, w, vr, vz, ctr in zip(feats, coef, weights,
                                     row_raw.values, row_sc.values, contribs):
        label, category = spec["labels"][f]
        factor_rows.append({
            "Feature":      f,
            "Description":  label,
            "Category":     category,
            "Raw value":    round(float(vr), 4),
            "Z-score":      round(float(vz), 3),
            "Coefficient":  round(float(c), 4),
            "Weight %":     round(float(w), 1),
            "Contribution": round(float(ctr), 4),
            "Direction":    "Bearish" if ctr > 0 else "Bullish",
        })
    factors = pd.DataFrame(factor_rows).sort_values(
        "Weight %", ascending=False
    ).reset_index(drop=True)

    return {
        "kind":         kind,
        "title":        spec["title"],
        "subtitle":     spec["subtitle"],
        "horizon":      spec["horizon"],
        "current_prob": current_p,
        "as_of":        as_of,
        "intercept":    intercept,
        "factors":      factors,
        "history":      history.dropna(),
        "base_rate":    float(y_tr.mean()),
    }


if __name__ == "__main__":
    for k in ("bear", "correction"):
        a = load_assessment(k)
        print(f"\n{'='*60}\n  {a['title']} model — {a['subtitle']}\n{'='*60}")
        print(f"  Current probability: {a['current_prob']:.1%}  (as of {a['as_of'].date()})")
        print(f"  Base rate: {a['base_rate']:.1%}")
        print(f"  History: {len(a['history'])} months, "
              f"{a['history'].index[0].date()} to {a['history'].index[-1].date()}")
        print(f"\n{a['factors'].to_string(index=False)}")
