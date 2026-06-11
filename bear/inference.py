"""
bear/inference.py — final model inference for the dashboard.

Two layers, deliberately decoupled so the Streamlit app needs NO ML libraries:

  * fit_and_save(kind)   — OFFLINE. Fits the final weight-constrained model
                           (needs scikit-learn + scipy) and writes the fitted
                           parameters to bear/<kind>_model_params.json.

  * load_assessment(kind) — RUNTIME. Reads the JSON params + feature CSV and
                           produces the current probability, factor readings,
                           and historical curve using only numpy / pandas.
                           No scikit-learn or scipy import on this path.

The app calls load_assessment(). If the params JSON is missing it will try to
fit on the fly (lazy-importing the ML libs); if those libs are unavailable it
raises a clear message telling the user to run the offline fit step.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

_BEAR_DIR = Path(__file__).resolve().parent
_DATA_DIR = _BEAR_DIR.parent / "data"     # all data files live in repo data/

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
                 "baa_zscore_60m", "lei_6m_growth", "ffr_6m_chg", "sahm_level"],
    "signs": {"ntfs_3m_chg": +1, "ts_inv_dummy": +1, "ebp_3m_chg": +1,
              "baa_zscore_60m": +1, "lei_6m_growth": -1, "ffr_6m_chg": -1,
              "sahm_level": +1},
    "labels": {
        "ntfs_3m_chg":    ("Near-term forward spread, 3m change", "Yield curve"),
        "ts_inv_dummy":   ("Yield curve inversion (10y-3m < 0)",  "Yield curve"),
        "ebp_3m_chg":     ("Excess bond premium, 3m change",      "Credit"),
        "baa_zscore_60m": ("BAA spread, 5yr z-score",             "Credit"),
        "lei_6m_growth":  ("Leading indicator, 6m growth",        "Leading"),
        "ffr_6m_chg":     ("Fed funds rate, 6m change",           "Policy"),
        "sahm_level":     ("Sahm rule (unemployment momentum)",   "Labor"),
    },
    "min_w": 0.10,   # every factor must carry >= 10% weight
    "max_w": 0.40,
    "oos_start": "1995-01-31",   # walk-forward OOS begins here
    "model_type": "logistic",
    "value_kind": "probability",
}

# Bear+ : same 7 factors as the Bear model, but the dependent variable is the
# CONTINUOUS rolling 12-month forward drawdown. We model the drawdown SEVERITY
#   s_t = -mdd_12m_t  in  [0, 1]
# with a weight-constrained FRACTIONAL LOGISTIC regression (quasi-binomial):
#   E[s_t] = sigma(z_t)  in (0, 1)
# so the output is interpretable as a probability-like expected-drawdown
# fraction (e.g. 0.13 = expected 13% drawdown). Signs are the effect on
# severity (higher = worse), identical to the logistic Bear model.
# Bear: UNCONSTRAINED binary logistic for P(12-month forward drawdown > 20%).
# Long-history (1960+) model from the feature-engineering research
# (bear/research_features.py): inflation (Chen 2009 — top predictor) + trend +
# valuation. All four factors are real (Shiller-based) back to ~1905, so the
# model uses NO imputed data. OOS AUC ~0.75 (vs ~0.65 for the prior set).
BEARPLUS_SPEC = {
    "kind":        "bearplus",
    "title":       "Bear",
    "subtitle":    ">20% drawdown over next 12 months",
    "horizon":     12,
    "features_csv": "bear_features.csv",
    "target_col":  "mdd_12m",
    "target_transform": "exceeds_20",   # y = 1 if mdd_12m <= -0.20
    "features": ["infl_zscore_120m", "infl_yoy", "spx_vs_10ma", "cape_z_120m"],
    # Economic priors (for reference only — the unconstrained fit sets signs).
    "signs": {"infl_zscore_120m": +1, "infl_yoy": +1,
              "spx_vs_10ma": -1, "cape_z_120m": +1},
    "labels": {
        "infl_zscore_120m": ("CPI inflation, 10yr z-score",  "Inflation"),
        "infl_yoy":         ("CPI inflation, YoY %",         "Inflation"),
        "spx_vs_10ma":      ("S&P 500 vs 10-month MA",       "Trend"),
        "cape_z_120m":      ("Shiller CAPE, 10yr z-score",   "Valuation"),
    },
    "min_w": 0.0,
    "max_w": 1.0,
    "unconstrained": True,           # free signs, free weights (MLE)
    "train_start": "1910-01-31",     # all 4 factors real from ~1905 (no fill)
    "oos_start": "1935-01-31",       # 25y initial training; long OOS window
    "model_type": "logistic",        # binary logistic on the >20% event
    "value_kind": "probability",     # output = P(bear) in [0,1]
}

CORR_SPEC = {
    "kind":        "correction",
    "title":       "Correction",
    "subtitle":    ">10% drawdown over next 6 months",
    "horizon":     6,
    "features_csv": "correction_features.csv",
    "target_col":  "mdd_6m",
    "target_transform": "exceeds_10",   # y = 1 if mdd_6m <= -0.10 (>10% drawdown)
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
    "min_w": 0.0,
    "max_w": 0.30,
    "oos_start": "2010-01-31",
    "model_type": "logistic",
    "value_kind": "probability",
}

# Correction+ : same 6 factors as the Correction model, but the dependent
# variable is broader — a binary 6-month rolling correction defined as ANY
# drawdown deeper than 10% (mdd_6m <= -10%, i.e. corrections AND bears that
# pass through 10%). Weight-constrained binary logistic; output is the
# probability of a >10% drawdown within the next 6 months.
CORRECTIONPLUS_SPEC = {
    "kind":        "correctionplus",
    "title":       "Correction",
    "subtitle":    ">10% drawdown within a 6-month rolling window",
    "horizon":     6,
    "features_csv": "correction_features.csv",
    "target_col":  "mdd_6m",
    "target_transform": "exceeds_10",   # y = 1 if mdd_6m <= -0.10
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
    "min_w": 0.0,
    "max_w": 0.30,
    "train_start": "1960-01-31",     # extend sample to the 1960s; mean-fill gaps
    "oos_start": "1975-01-31",
    "model_type": "logistic",
    "value_kind": "probability",
}

# ---------------------------------------------------------------------------
# Ensemble member — MODEL A (long history, trained from the 1920s)
# ---------------------------------------------------------------------------
# First member of the bear-market ensemble. CONSTRAINT: every feature is built
# only from raw series that START BEFORE 1920 (SPX/CPI/DGS10 1871, CAPE 1900,
# BAA/AAA/INDPRO 1919), so the model trains from 1920 and learns pre-WWII regime
# dynamics (1929 crash, Great Depression). Selected by bear/search_model_a.py:
# a parsimonious Trend + Valuation + Credit set that maximizes walk-forward OOS
# AUC (0.64) and beats richer combinations out-of-sample. Inflation is excluded
# because the deflationary 1930s break the modern inflation-bear link, so the
# long-history signal rests on the BAA-10y credit spread instead.
MODELA_SPEC = {
    "kind":        "modela",
    "title":       "Bear — Model A (1920s)",
    "subtitle":    ">20% drawdown over next 12 months (long-history member)",
    "horizon":     12,
    "features_csv": "bear_features.csv",
    "target_col":  "mdd_12m",
    "target_transform": "exceeds_20",   # y = 1 if mdd_12m <= -0.20
    "features": ["spx_vs_10ma", "cape_20yr_pct", "baa_10y_spread"],
    "signs": {"spx_vs_10ma": -1, "cape_20yr_pct": +1, "baa_10y_spread": +1},
    "labels": {
        "spx_vs_10ma":    ("S&P 500 vs 10-month MA",            "Trend"),
        "cape_20yr_pct":  ("Shiller CAPE, 20yr percentile",     "Valuation"),
        "baa_10y_spread": ("BAA - 10y Treasury credit spread",  "Credit"),
    },
    "min_w": 0.0,
    "max_w": 1.0,
    "unconstrained": True,           # free signs, free weights (MLE)
    "train_start": "1920-01-31",     # all 3 factors real from 1919-1920 (no fill)
    "oos_start": "1950-01-31",       # 30y initial training; long OOS window
    "model_type": "logistic",
    "value_kind": "probability",
}

# ---------------------------------------------------------------------------
# Ensemble member — MODEL B (post-war, trained from 1950)
# ---------------------------------------------------------------------------
# Second member of the bear-market ensemble. Trains from 1950, so beyond Model
# A's pre-1920 series it may use post-war data: a real 10y-3m term spread
# (DGS10-TB3MS, TB3MS from 1920) and unemployment (UNRATE, 1948). Selected by
# bear/search_model_b.py under a HARD HAC-significance constraint (every
# coefficient p<0.05 after Newey-West, lag=12): the OOS-maximizing 6-factor set
# left credit/IP/term insignificant, so the committed model is the largest
# all-significant set. Adds a RATES factor (10y yield change) over bearplus and,
# unlike Model A, finds INFLATION significant — the post-war inflation bears
# (1973-74, 1980-82) restore the Chen-2009 link that the deflationary 1930s broke.
# The unconstrained fit gives the 10y yield change a NEGATIVE sign: falling
# long-term yields over the prior year (the bond market pricing weaker growth /
# Fed easing ahead of downturns) raise bear risk.
MODELB_SPEC = {
    "kind":        "modelb",
    "title":       "Bear — Model B (1950s)",
    "subtitle":    ">20% drawdown over next 12 months (post-war member)",
    "horizon":     12,
    "features_csv": "bear_features.csv",
    "target_col":  "mdd_12m",
    "target_transform": "exceeds_20",   # y = 1 if mdd_12m <= -0.20
    "features": ["spx_vs_10ma", "infl_zscore_120m", "cape_20yr_pct", "dgs10_12m_chg"],
    "signs": {"spx_vs_10ma": -1, "infl_zscore_120m": +1,
              "cape_20yr_pct": +1, "dgs10_12m_chg": -1},
    "labels": {
        "spx_vs_10ma":      ("S&P 500 vs 10-month MA",          "Trend"),
        "infl_zscore_120m": ("CPI inflation, 10yr z-score",     "Inflation"),
        "cape_20yr_pct":    ("Shiller CAPE, 20yr percentile",   "Valuation"),
        "dgs10_12m_chg":    ("10y Treasury yield, 12m change",  "Rates"),
    },
    "min_w": 0.0,
    "max_w": 1.0,
    "unconstrained": True,           # free signs, free weights (MLE)
    "train_start": "1950-01-31",     # post-war start (UNRATE / TB3MS available)
    "oos_start": "1970-01-31",       # 20y initial training; OOS 1970-2026
    "model_type": "logistic",
    "value_kind": "probability",
}

# ---------------------------------------------------------------------------
# Ensemble member — MODEL C (1960s, trained from 1962)
# ---------------------------------------------------------------------------
# Third member of the bear-market ensemble. Trains from 1962, so beyond Models
# A/B it may use the 1960s-era families (NTFS 1961, OECD LEI 1955, real-time
# Sahm 1959, fed funds 1954). Selected by bear/search_model_c.py under the hard
# HAC-significance rule via significance-constrained forward selection (no
# multi-factor set was all-significant in a plain OOS ranking — the shorter
# 1962+ sample inflates Newey-West SEs). The committed 5-factor set is fully
# HAC-significant and posts the ensemble's best OOS AUC (~0.79). Its distinct
# contribution vs A/B is the 10y-3m YIELD-CURVE INVERSION dummy (Estrella-
# Mishkin recession signal). Like Model B, the 10y yield change carries a
# negative sign (falling long rates -> higher bear risk).
MODELC_SPEC = {
    "kind":        "modelc",
    "title":       "Bear — Model C (1960s)",
    "subtitle":    ">20% drawdown over next 12 months (1960s member)",
    "horizon":     12,
    "features_csv": "bear_features.csv",
    "target_col":  "mdd_12m",
    "target_transform": "exceeds_20",   # y = 1 if mdd_12m <= -0.20
    "features": ["spx_vs_10ma", "infl_zscore_120m", "cape_z_120m",
                 "dgs10_12m_chg", "ts_10y3m_inv_dummy"],
    "signs": {"spx_vs_10ma": -1, "infl_zscore_120m": +1, "cape_z_120m": +1,
              "dgs10_12m_chg": -1, "ts_10y3m_inv_dummy": +1},
    "labels": {
        "spx_vs_10ma":        ("S&P 500 vs 10-month MA",            "Trend"),
        "infl_zscore_120m":   ("CPI inflation, 10yr z-score",       "Inflation"),
        "cape_z_120m":        ("Shiller CAPE, 10yr z-score",        "Valuation"),
        "dgs10_12m_chg":      ("10y Treasury yield, 12m change",    "Rates"),
        "ts_10y3m_inv_dummy": ("Yield-curve inversion (10y-3m<0)",  "Term"),
    },
    "min_w": 0.0,
    "max_w": 1.0,
    "unconstrained": True,           # free signs, free weights (MLE)
    "train_start": "1962-01-31",     # 1960s start (NTFS / LEI / Sahm available)
    "oos_start": "1985-01-31",       # ~23y initial training; OOS 1985-2026
    "model_type": "logistic",
    "value_kind": "probability",
}

# ---------------------------------------------------------------------------
# Ensemble member — MODEL D (1980s, modern, trained from 1986)
# ---------------------------------------------------------------------------
# Fourth and final member of the bear-market ensemble. Trains from 1986, so it
# may use the full modern toolkit (NFCI/ANFCI 1971, EBP 1973, native term
# spreads 1976-1982, BAA10Y 1986, VIX 1990). Selected by bear/search_model_d.py
# via significance-constrained forward selection (the short modern sample + the
# hard HAC rule admit no fully-significant set under a plain OOS ranking). The
# committed 3-factor set is fully HAC-significant and posts the ensemble's
# highest OOS AUC (~0.90, on the 2005+ window spanning the GFC, COVID and 2022).
# It is the only member with a POLICY factor: fed funds FALLING over 6 months
# (the Fed easing into deteriorating conditions) raises bear risk. Trend and
# valuation are not individually significant post-1986 — the modern bears were
# policy/credit/inflation events.
MODELD_SPEC = {
    "kind":        "modeld",
    "title":       "Bear — Model D (1980s)",
    "subtitle":    ">20% drawdown over next 12 months (modern member)",
    "horizon":     12,
    "features_csv": "bear_features.csv",
    "target_col":  "mdd_12m",
    "target_transform": "exceeds_20",   # y = 1 if mdd_12m <= -0.20
    "features": ["ffr_6m_chg", "infl_zscore_120m", "baa_aaa_z60"],
    "signs": {"ffr_6m_chg": -1, "infl_zscore_120m": +1, "baa_aaa_z60": +1},
    "labels": {
        "ffr_6m_chg":       ("Fed funds rate, 6m change",          "Policy"),
        "infl_zscore_120m": ("CPI inflation, 10yr z-score",        "Inflation"),
        "baa_aaa_z60":      ("BAA-AAA quality spread, 5yr z-score", "Credit"),
    },
    "min_w": 0.0,
    "max_w": 1.0,
    "unconstrained": True,           # free signs, free weights (MLE)
    "train_start": "1986-01-31",     # modern start (BAA10Y / native spreads)
    "oos_start": "2005-01-31",       # OOS spans GFC, COVID, 2022
    "model_type": "logistic",
    "value_kind": "probability",
}

_SPECS = {
    "bear": BEAR_SPEC,
    "correction": CORR_SPEC,
    "bearplus": BEARPLUS_SPEC,
    "correctionplus": CORRECTIONPLUS_SPEC,
    "modela": MODELA_SPEC,
    "modelb": MODELB_SPEC,
    "modelc": MODELC_SPEC,
    "modeld": MODELD_SPEC,
}


def _oos_path(kind: str) -> Path:
    return _DATA_DIR / f"{kind}_oos.csv"


def _params_path(kind: str) -> Path:
    return _DATA_DIR / f"{kind}_model_params.json"


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def _hac_pvalues(
    X: np.ndarray,
    y: np.ndarray,
    beta: np.ndarray,
    maxlags: int,
) -> np.ndarray:
    """
    Newey-West HAC p-values for a (quasi-)logistic model evaluated at beta.

    Overlapping rolling-window targets (a 12-month forward drawdown shares 11/12
    of its window with the next month) induce strong serial correlation in the
    score, which deflates naive standard errors. The HAC sandwich with a
    Bartlett kernel and lag = horizon corrects for this.

        score_t  = (y_t - p_t) x_t                 (p_t = sigma(x_t'beta))
        bread    = (sum_t p_t(1-p_t) x_t x_t')^{-1}
        meat     = S_0 + sum_{l=1}^{L}(1 - l/(L+1))(S_l + S_l')
        V_HAC    = bread . meat . bread

    Works for binary (correction) and fractional [0,1] (bear severity) y alike.
    X includes the intercept column; returns a p-value per column of X.
    """
    from math import erf, sqrt

    n, k = X.shape
    eta = np.clip(X @ beta, -500, 500)
    p   = 1.0 / (1.0 + np.exp(-eta))
    w   = np.clip(p * (1.0 - p), 1e-8, None)

    bread = np.linalg.inv(X.T @ (w[:, None] * X) + 1e-8 * np.eye(k))
    scores = (y - p)[:, None] * X
    S = scores.T @ scores
    for lag in range(1, min(maxlags, n - 1) + 1):
        wl = 1.0 - lag / (maxlags + 1.0)
        G  = scores[lag:].T @ scores[:-lag]
        S += wl * (G + G.T)

    V  = bread @ S @ bread
    se = np.sqrt(np.clip(np.diag(V), 1e-12, None))
    z  = beta / se
    return np.array([2.0 * (1.0 - 0.5 * (1.0 + erf(abs(zi) / sqrt(2.0)))) for zi in z])


def _apply_target_transform(y: pd.Series, transform: str | None) -> pd.Series:
    """
    Map the raw target to the modeling target.
      "severity"   : y = -mdd clipped to [0, 1)  (drawdown magnitude as a fraction)
      "exceeds_10" : y = 1 if forward drawdown deeper than 10% (mdd <= -0.10)
      None         : identity
    NaN observations (unresolved forward window) are preserved as NaN.
    """
    if transform == "severity":
        return (-y).clip(lower=0.0, upper=0.999)
    if transform == "exceeds_10":
        return (y <= -0.10).astype(float).where(y.notna())
    if transform == "exceeds_20":
        return (y <= -0.20).astype(float).where(y.notna())
    return y


# ---------------------------------------------------------------------------
# OFFLINE: fit the weight-constrained model and persist parameters
# ---------------------------------------------------------------------------

def _fit_constrained_core(
    X_sc:  np.ndarray,
    y_tr:  np.ndarray,
    signs: np.ndarray,
    min_w: float,
    max_w: float,
    model_type: str = "logistic",
    unconstrained: bool = False,
) -> tuple[np.ndarray, float]:
    """
    Core fit. Returns (coef, intercept).

    unconstrained=True : plain MLE logistic — free signs, free weights
                         (used for the redeveloped Bear model).
    Otherwise weight-constrained, sign-fixed fit, with model_type:
      "logistic"   — calibrated logistic regression (binary 0/1 target)
      "fractional" — fractional logistic / quasi-binomial (continuous [0,1])
      "linear"     — OLS-style regression (continuous target)

    Requires sklearn + scipy (offline only).
    """
    if unconstrained:
        from sklearn.linear_model import LogisticRegression
        m = LogisticRegression(C=1e9, solver="lbfgs", max_iter=5000)
        m.fit(X_sc, y_tr)
        return m.coef_[0], float(m.intercept_[0])

    from scipy.optimize import minimize

    n = len(signs)

    if model_type == "linear":
        from sklearn.linear_model import LinearRegression
        m0 = LinearRegression().fit(X_sc, y_tr)
        gamma0 = np.abs(m0.coef_)
        b0 = float(m0.intercept_)

        def obj(p):
            g = p[:n]; b = p[n]
            resid = y_tr - (X_sc @ (signs * g) + b)
            return float(np.mean(resid ** 2)) + 1e-4 * float(np.sum(g ** 2))

    elif model_type == "fractional":
        # Fractional logistic: Bernoulli quasi-likelihood with continuous y in [0,1].
        # Warm start from a linear fit (LogisticRegression needs binary labels).
        from sklearn.linear_model import LinearRegression
        m0 = LinearRegression().fit(X_sc, y_tr)
        gamma0 = np.abs(m0.coef_)
        ybar = float(np.clip(np.mean(y_tr), 1e-4, 1 - 1e-4))
        b0 = float(np.log(ybar / (1 - ybar)))

        def obj(p):
            g = p[:n]; b = p[n]
            logits = np.clip(X_sc @ (signs * g) + b, -500, 500)
            s = _sigmoid(logits)
            nll = -(y_tr * np.log(s + 1e-12) + (1 - y_tr) * np.log(1 - s + 1e-12)).sum()
            return nll + 0.5 * float(np.sum(g ** 2))
    else:
        from sklearn.linear_model import LogisticRegression
        m0 = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=42)
        m0.fit(X_sc, y_tr)
        gamma0 = np.abs(m0.coef_[0])
        b0 = float(m0.intercept_[0])

        def obj(p):
            g = p[:n]; b = p[n]
            logits = np.clip(X_sc @ (signs * g) + b, -500, 500)
            s = _sigmoid(logits)
            nll = -(y_tr * np.log(s + 1e-12) + (1 - y_tr) * np.log(1 - s + 1e-12)).sum()
            return nll + 0.5 * float(np.sum(g ** 2))

    cons = []
    for i in range(n):
        cons.append({"type": "ineq", "fun": (lambda p, i=i: max_w * p[:n].sum() - p[i])})
        if min_w > 0.0:
            cons.append({"type": "ineq", "fun": (lambda p, i=i: p[i] - min_w * p[:n].sum())})

    if min_w > 0.0:
        g_start = np.full(n, float(np.mean(gamma0)) if np.mean(gamma0) > 0 else 1.0)
    else:
        g_start = gamma0
    res = minimize(obj, np.append(g_start, b0),
                   method="SLSQP", constraints=cons,
                   bounds=[(1e-8, None)] * n + [(None, None)],
                   options={"maxiter": 5000, "ftol": 1e-11})

    return signs * np.abs(res.x[:n]), float(res.x[n])


def _walk_forward_oos(
    features_df: pd.DataFrame,
    y: pd.Series,
    spec: dict,
) -> pd.Series:
    """
    Expanding-window out-of-sample probabilities for the final model.

    At each month t >= oos_start the constrained model is re-fit on all
    complete observations strictly before t (standardized on that window
    only), then used to predict month t. No look-ahead.
    """
    feats = spec["features"]
    signs = np.array([spec["signs"][f] for f in feats], dtype=float)
    min_w = float(spec.get("min_w", 0.0))
    max_w = float(spec.get("max_w", 1.0))
    model_type = spec.get("model_type", "logistic")
    unconstrained = bool(spec.get("unconstrained", False))
    oos_ts = pd.Timestamp(spec.get("oos_start"))
    train_start = pd.Timestamp(spec.get("train_start", "1900-01-01"))

    idx = features_df.index
    out = pd.Series(np.nan, index=idx, name="prob_oos")

    # Missing factors are mean-filled (0 after standardization); rows need a
    # valid target only.  Binary targets require >=5 positives to fit.
    def _enough(tr_mask) -> bool:
        if tr_mask.sum() < 30:
            return False
        if model_type == "logistic":
            return y.loc[tr_mask].sum() >= 5
        return True

    for t in idx:
        if t < oos_ts:
            continue
        tr = (idx < t) & y.notna() & (idx >= train_start)
        if not _enough(tr):
            continue

        X_tr  = features_df.loc[tr, feats]
        y_arr = y.loc[tr].values.astype(float)
        mu    = X_tr.mean()
        sig   = X_tr.std(ddof=1).replace(0.0, np.nan)
        X_sc  = ((X_tr - mu) / sig).fillna(0.0).values

        coef, b = _fit_constrained_core(X_sc, y_arr, signs, min_w, max_w,
                                        model_type, unconstrained)

        x_te = ((features_df.loc[[t], feats] - mu) / sig).fillna(0.0).values
        z = float((x_te @ coef + b)[0])
        use_sigmoid = model_type in ("logistic", "fractional")
        out.loc[t] = float(_sigmoid(np.array([z]))[0]) if use_sigmoid else z

    return out.dropna()


def fit_and_save(kind: Literal["bear", "correction"]) -> dict:
    """
    Fit the final model (sklearn + scipy required) and write its parameters
    to bear/<kind>_model_params.json so the dashboard can run without ML libs.
    Also computes walk-forward OOS probabilities and writes bear/<kind>_oos.csv.
    """
    spec  = _SPECS[kind]
    feats = spec["features"]

    features_df = pd.read_csv(_DATA_DIR / spec["features_csv"],
                              index_col=0, parse_dates=True)
    targets     = pd.read_csv(_DATA_DIR / "targets.csv",
                              index_col=0, parse_dates=True)
    y = _apply_target_transform(targets[spec["target_col"]],
                                spec.get("target_transform"))

    # Extend the sample to train_start; rows need only a valid target.
    # Missing factors (not yet published in the early years) are mean-filled:
    # after standardization a missing value becomes 0, i.e. the feature mean.
    train_start = pd.Timestamp(spec.get("train_start", "1900-01-01"))
    mask  = y.notna() & (features_df.index >= train_start)
    X_raw = features_df.loc[mask, feats]
    y_tr  = y.loc[mask].values
    mu    = X_raw.mean()                          # skips NaN
    sig   = X_raw.std(ddof=1).replace(0.0, np.nan)
    X_sc  = ((X_raw - mu) / sig).fillna(0.0).values   # mean-fill missing factors

    signs = np.array([spec["signs"][f] for f in feats], dtype=float)
    min_w = float(spec.get("min_w", 0.0))
    max_w = float(spec.get("max_w", 1.0))
    model_type = spec.get("model_type", "logistic")
    unconstrained = bool(spec.get("unconstrained", False))

    coef_arr, intercept = _fit_constrained_core(
        X_sc, y_tr, signs, min_w, max_w, model_type, unconstrained
    )
    coef = coef_arr.tolist()

    # Newey-West HAC p-values (lag = horizon) at the fitted estimates, to
    # correct for the autocorrelation from overlapping rolling-window targets.
    maxlags = int(spec["horizon"])
    X_const = np.column_stack([np.ones(len(y_tr)), X_sc])
    beta_full = np.concatenate([[intercept], coef_arr])
    try:
        pvals = _hac_pvalues(X_const, y_tr.astype(float), beta_full, maxlags)
        hac_pvalues = {f: float(pvals[i + 1]) for i, f in enumerate(feats)}
        hac_intercept_p = float(pvals[0])
    except Exception:
        hac_pvalues = {}
        hac_intercept_p = float("nan")

    # Walk-forward out-of-sample series (saved separately for the app)
    oos = _walk_forward_oos(features_df, y, spec)
    oos.to_csv(_oos_path(kind), header=True)

    params = {
        "kind":       kind,
        "title":      spec["title"],
        "subtitle":   spec["subtitle"],
        "horizon":    spec["horizon"],
        "features":   feats,
        "signs":      spec["signs"],
        "labels":     {f: list(spec["labels"][f]) for f in feats},
        "coef":       coef,
        "intercept":  intercept,
        "mu":         {f: float(mu[f]) for f in feats},
        "sigma":      {f: float(sig[f]) for f in feats},
        "base_rate":  float(np.nanmean(y_tr)),
        "min_w":      min_w,
        "max_w":      max_w,
        "model_type": model_type,
        "value_kind": spec.get("value_kind", "probability"),
        "unconstrained": unconstrained,
        "train_start": str(train_start.date()),
        "hac_pvalues":     hac_pvalues,
        "hac_intercept_p": hac_intercept_p,
        "hac_maxlags":     maxlags,
    }
    with open(_params_path(kind), "w") as fh:
        json.dump(params, fh, indent=2)
    return params


# ---------------------------------------------------------------------------
# RUNTIME: load params and build the assessment (numpy / pandas only)
# ---------------------------------------------------------------------------

def load_assessment(kind: Literal["bear", "correction"]) -> dict:
    """
    Return the current probability, factor table, and historical curve.

    Uses precomputed parameters from bear/<kind>_model_params.json (no ML
    libraries needed). If the JSON is missing, attempts an on-the-fly fit
    (which requires scikit-learn + scipy) and persists it.
    """
    path = _params_path(kind)
    if not path.exists():
        try:
            params = fit_and_save(kind)
        except ImportError as exc:
            raise RuntimeError(
                f"Model parameters not found at {path.name} and could not be "
                f"fitted ({exc}). Run `python -m bear.inference` in an "
                f"environment with scikit-learn + scipy to generate them."
            ) from exc
    else:
        with open(path) as fh:
            params = json.load(fh)

    spec  = _SPECS[kind]
    feats = params["features"]
    coef  = np.array(params["coef"], dtype=float)
    intercept = float(params["intercept"])
    mu    = pd.Series(params["mu"])
    sig   = pd.Series(params["sigma"])
    base  = float(params["base_rate"])
    model_type = params.get("model_type", "logistic")
    value_kind = params.get("value_kind", "probability")

    def _predict(Z: np.ndarray) -> np.ndarray:
        lin = Z @ coef + intercept
        return _sigmoid(lin) if model_type in ("logistic", "fractional") else lin

    features_df = pd.read_csv(_DATA_DIR / spec["features_csv"],
                              index_col=0, parse_dates=True)
    train_start = pd.Timestamp(params.get("train_start", "1900-01-01"))

    # Historical fitted value from train_start onward; missing factors are
    # mean-filled (0 after standardization), consistent with estimation.
    hrows   = features_df.index >= train_start
    X_all   = ((features_df.loc[hrows, feats] - mu) / sig).fillna(0.0)
    history = pd.Series(np.nan, index=features_df.index, name="value")
    history.loc[hrows] = _predict(X_all.values)

    # Walk-forward OOS series (precomputed offline)
    oos_path = _oos_path(kind)
    if oos_path.exists():
        oos_df = pd.read_csv(oos_path, index_col=0, parse_dates=True)
        history_oos = oos_df.iloc[:, 0].rename("value_oos").dropna()
    else:
        history_oos = pd.Series(dtype=float, name="value_oos")

    # Current reading = last complete row
    avail     = features_df[feats].dropna()
    as_of     = avail.index[-1]
    row_raw   = avail.loc[as_of]
    row_sc    = (row_raw - mu) / sig
    contribs  = row_sc.values * coef
    current_p = float(_predict(np.array([row_sc.values]))[0])

    weights = np.abs(coef) / np.abs(coef).sum() * 100

    # Direction of a factor's current push:
    #   probability target -> contribution>0 raises bear probability  -> Bearish
    #   drawdown target     -> contribution<0 deepens the drawdown      -> Bearish
    def _direction(ctr: float) -> str:
        if value_kind == "drawdown":
            return "Bearish" if ctr < 0 else "Bullish"
        return "Bearish" if ctr > 0 else "Bullish"

    hac_pvalues = params.get("hac_pvalues", {})
    factor_rows = []
    for f, c, w, vr, vz, ctr in zip(feats, coef, weights,
                                     row_raw.values, row_sc.values, contribs):
        label, category = params["labels"][f]
        pv = hac_pvalues.get(f, float("nan"))
        factor_rows.append({
            "Feature":      f,
            "Description":  label,
            "Category":     category,
            "Raw value":    round(float(vr), 4),
            "Z-score":      round(float(vz), 3),
            "Coefficient":  round(float(c), 4),
            "Weight %":     round(float(w), 1),
            "Contribution": round(float(ctr), 4),
            "P (HAC)":      pv,
            "Direction":    _direction(ctr),
        })
    factors = pd.DataFrame(factor_rows).sort_values(
        "Weight %", ascending=False
    ).reset_index(drop=True)

    return {
        "kind":         kind,
        "title":        params["title"],
        "subtitle":     params["subtitle"],
        "horizon":      params["horizon"],
        "current_prob": current_p,
        "as_of":        as_of,
        "intercept":    intercept,
        "factors":      factors,
        "history":      history.dropna(),
        "history_oos":  history_oos,
        "base_rate":    base,
        "model_type":   model_type,
        "value_kind":   value_kind,
        # Extras for the mathematical formulation
        "features":     feats,
        "coef":         coef.tolist(),
        "mu":           {f: float(mu[f]) for f in feats},
        "sigma":        {f: float(sig[f]) for f in feats},
        "labels":       params["labels"],
        "min_w":        float(params.get("min_w", 0.0)),
        "max_w":        float(params.get("max_w", 1.0)),
        "unconstrained": bool(params.get("unconstrained", False)),
        "hac_maxlags":  int(params.get("hac_maxlags", params["horizon"])),
    }


def dump_training_data(kind: Literal["bear", "correction"]) -> tuple[Path, int]:
    """
    Write the exact complete-case training dataset for a model to CSV.

    Columns: the lag-adjusted model features (raw values, as fed to the fit),
    the modeling target, and the underlying realized forward drawdown for
    reference. Rows are the months actually used in estimation (every feature
    and the target present). Returns (path, n_rows).
    """
    spec  = _SPECS[kind]
    feats = spec["features"]
    horizon = spec["horizon"]

    features_df = pd.read_csv(_DATA_DIR / spec["features_csv"],
                              index_col=0, parse_dates=True)
    targets     = pd.read_csv(_DATA_DIR / "targets.csv",
                              index_col=0, parse_dates=True)

    raw_target = targets[spec["target_col"]]
    y = _apply_target_transform(raw_target, spec.get("target_transform"))

    # Same sample as estimation: from train_start, rows with a valid target.
    # Missing factors are mean-filled with each factor's training mean.
    train_start = pd.Timestamp(spec.get("train_start", "1900-01-01"))
    mask  = y.notna() & (features_df.index >= train_start)
    X_raw = features_df.loc[mask, feats]
    mu    = X_raw.mean()

    out = X_raw.fillna(mu)                        # mean-fill missing factors
    target_name = ("target_severity" if spec.get("value_kind") == "severity"
                   else "target_event")
    out[target_name] = y.loc[mask]
    mdd_col = "mdd_12m" if horizon == 12 else "mdd_6m"
    out[mdd_col] = targets.loc[mask, mdd_col]   # underlying realized drawdown
    out.index.name = "date"

    path = _DATA_DIR / f"{kind}_training_data.csv"
    out.to_csv(path, date_format="%Y-%m-%d", float_format="%.6f")
    return path, int(mask.sum())


if __name__ == "__main__":
    for k in ("bear", "correction", "bearplus", "correctionplus"):
        p = fit_and_save(k)                      # (re)generate params + OOS
        a = load_assessment(k)
        unit = "%" if a["value_kind"] in ("probability", "drawdown") else ""
        print(f"\n{'='*60}\n  {a['title']} — {a['subtitle']}\n{'='*60}")
        print(f"  Params written: {_params_path(k).name}")
        label = "Current value" if a["value_kind"] == "drawdown" else "Current probability"
        print(f"  {label}: {a['current_prob']:.1%}  (as of {a['as_of'].date()})")
        base_label = "Historical mean" if a["value_kind"] == "drawdown" else "Base rate"
        print(f"  {base_label}: {a['base_rate']:.1%}")
        print(f"  History: {len(a['history'])} months  |  OOS: {len(a['history_oos'])} months")
        print(f"\n{a['factors'].to_string(index=False)}")
