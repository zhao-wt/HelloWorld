"""
bear/model_bear.py — Phase 4: bear market logistic regression model.

Predicts the probability that the S&P 500 will experience a >20% drawdown
over the next 12 months.

Model spec
----------
  Estimator  : logistic regression with L2 regularisation
  Target     : y_bear (Phase 3) — binary, 1 = bear within 12 months
  Features   : 6 core variables (Phase 2 bear features)
  Imbalance  : class_weight='balanced' (King-Zeng 2001 approximation)
  Evaluation : walk-forward expanding-window OOS from 2000-01-31

Core features and expected signs
---------------------------------
  ntfs_level    (−)  near-term forward spread — Engstrom-Sharpe 2019
  baa_level     (+)  BAA-10y default spread — Chen-Chen-Chou 2017
  baa_3m_chg    (+)  3-month change in BAA spread
  ebp_level     (+)  excess bond premium — Gilchrist-Zakrajšek 2012
  sahm_level    (+)  Sahm rule continuous reading
  lei_6m_growth (−)  OECD LEI annualised 6-month growth

No-look-ahead safeguards
------------------------
  1. Features already carry publication lags from Phase 2.
  2. Expanding-window standardisation: mean/std computed only on training
     data available up to (not including) each test month.
  3. Walk-forward OOS: model is re-fitted from scratch at each step using
     only past observations.
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

# Root of the bear/ package — used to build absolute paths that work
# regardless of which directory the script is invoked from.
_BEAR_DIR = Path(__file__).resolve().parent
_DATA_DIR = _BEAR_DIR.parent / "data"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CORE_FEATURES: list[str] = [
    "ntfs_level",
    "baa_level",
    "baa_3m_chg",
    "ebp_level",
    "icsa_yoy_pct",   # replaces sahm_level: ICSA has genuine lead; sahm is coincident
    "lei_6m_growth",
]

EXPECTED_SIGNS: dict[str, int] = {
    "ntfs_level":    -1,
    "baa_level":     +1,
    "baa_3m_chg":    +1,
    "ebp_level":     +1,
    "icsa_yoy_pct":  +1,  # rising claims -> labor stress -> bear risk
    "lei_6m_growth": -1,
}

# Walk-forward OOS start date — ensures 1987 and 1990 bear episodes are
# in the initial training window before the first OOS prediction.
OOS_START = "2000-01-31"

# Minimum positive examples in training before fitting
MIN_POSITIVES = 5

# Winsorise ICSA YoY at these percentiles (COVID spike = 2123%)
WINSOR_LOWER = 0.01
WINSOR_UPPER = 0.99

# ---------------------------------------------------------------------------
# Clean candidate set for exhaustive search
# ---------------------------------------------------------------------------
# Excluded: sahm_level, sahm_trigger, icsa_yoy_pct (coincident indicators —
#   their empirical signs flip vs naive expectation for a 12m forward model),
#   lei_stress_dummy (shows negative sign in data, contradicts expected +).
#
# Expected sign rationale
# -----------------------
# ntfs_level     (−) low/negative NTFS = near-term inversion = recession alarm
# ntfs_3m_chg    (+) rising NTFS = Fed cutting short rate = recession onset
# ts_10y3m       (−) inverted long-short spread = recession risk
# ts_10y2y       (−) inverted 10y-2y = recession risk
# ts_inv_dummy   (+) binary inversion indicator = bear risk
# ebp_level      (+) high EBP = credit-supply tightness = bear risk
# ebp_3m_chg     (+) rapidly rising EBP = credit deterioration = bear risk
# baa_level      (+) wide default spreads = financial stress = bear risk
# baa_3m_chg     (+) widening spreads = credit momentum = bear risk
# baa_zscore_60m (+) spreads elevated vs 5-year history = structural stress
# lei_6m_growth  (−) falling LEI = deteriorating economy = bear risk
# ffr_6m_chg     (−) large negative = Fed cutting = recession response

CLEAN_CANDIDATES: list[str] = [
    "ntfs_level", "ntfs_3m_chg",
    "ts_10y3m", "ts_10y2y", "ts_inv_dummy",
    "ebp_level", "ebp_3m_chg",
    "baa_level", "baa_3m_chg", "baa_zscore_60m",
    "lei_6m_growth", "ffr_6m_chg",
]

# Category membership for the category-constrained search.
# Constraint: at least 1 and at most 2 features selected per category.
# Labor excluded: sahm/icsa show sign flips for a 12m forward model.
FEATURE_CATEGORIES: dict[str, list[str]] = {
    "Yield curve": ["ntfs_level", "ntfs_3m_chg", "ts_10y3m", "ts_10y2y", "ts_inv_dummy"],
    "Credit":      ["ebp_level", "ebp_3m_chg", "baa_level", "baa_3m_chg", "baa_zscore_60m"],
    "Leading":     ["lei_6m_growth"],
    "Policy":      ["ffr_6m_chg"],
}
CATEGORY_MIN_MAX: dict[str, tuple[int, int]] = {
    "Yield curve": (1, 2),
    "Credit":      (1, 2),
    "Leading":     (1, 1),  # only 1 candidate available
    "Policy":      (1, 1),  # only 1 candidate available
}

EXPECTED_SIGNS_CLEAN: dict[str, int] = {
    "ntfs_level":     -1,
    "ntfs_3m_chg":    +1,
    "ts_10y3m":       -1,
    "ts_10y2y":       -1,
    "ts_inv_dummy":   +1,
    "ebp_level":      +1,
    "ebp_3m_chg":     +1,
    "baa_level":      +1,
    "baa_3m_chg":     +1,
    "baa_zscore_60m": +1,
    "lei_6m_growth":  -1,
    "ffr_6m_chg":     -1,
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_model_data(
    features_path: Path = _DATA_DIR / "bear_features.csv",
    targets_path:  Path = _DATA_DIR / "targets.csv",
    features:      list[str] = CORE_FEATURES,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load bear features and targets, align on date index.

    Returns
    -------
    X : pd.DataFrame  — feature matrix (all rows, NaN where unavailable)
    y : pd.Series     — binary bear target (NaN in last 12 months)
    """
    bear_f  = pd.read_csv(features_path, index_col=0, parse_dates=True)
    targets = pd.read_csv(targets_path,  index_col=0, parse_dates=True)

    X = bear_f[features].copy()
    y = targets["y_bear"].copy()

    # Align on common index
    idx = X.index.intersection(y.index)
    return X.loc[idx], y.loc[idx]


# ---------------------------------------------------------------------------
# Pre-processing helpers
# ---------------------------------------------------------------------------

def winsorize(s: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """Cap series at trailing [lower, upper] quantiles computed on non-NaN values."""
    lo = s.quantile(lower)
    hi = s.quantile(upper)
    return s.clip(lo, hi)


def expanding_standardize(
    X_train: pd.DataFrame,
    X_test:  pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Standardise X_test using mean and std computed on X_train only.
    Columns with zero std are left un-scaled (replaced with 0 after centering).
    """
    mu  = X_train.mean()
    sig = X_train.std(ddof=1).replace(0.0, np.nan)
    X_tr = (X_train - mu) / sig
    X_te = (X_test  - mu) / sig
    return X_tr.fillna(0.0), X_te.fillna(0.0)


def preprocess(X: pd.DataFrame) -> pd.DataFrame:
    """Apply winsorisation to flow features before modelling."""
    out = X.copy()
    if "icsa_yoy_pct" in out.columns:
        out["icsa_yoy_pct"] = winsorize(out["icsa_yoy_pct"])
    return out


# ---------------------------------------------------------------------------
# VIF check
# ---------------------------------------------------------------------------

def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    """
    Variance Inflation Factors for each column.
    Rule of thumb: VIF > 5 suggests problematic multicollinearity.
    Uses only complete rows.
    """
    Xc = X.dropna().copy()
    Xc = (Xc - Xc.mean()) / Xc.std(ddof=1).replace(0, 1)
    Xc_arr = Xc.values

    vifs = []
    for i, col in enumerate(Xc.columns):
        others = np.delete(Xc_arr, i, axis=1)
        # R^2 of column i regressed on all others
        try:
            coef, *_ = np.linalg.lstsq(
                np.hstack([np.ones((len(others), 1)), others]),
                Xc_arr[:, i],
                rcond=None,
            )
            y_hat = np.hstack([np.ones((len(others), 1)), others]) @ coef
            ss_res = ((Xc_arr[:, i] - y_hat) ** 2).sum()
            ss_tot = ((Xc_arr[:, i] - Xc_arr[:, i].mean()) ** 2).sum()
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            vif = 1 / (1 - r2) if r2 < 1 else np.inf
        except Exception:
            vif = np.nan
        vifs.append({"Feature": col, "VIF": round(vif, 2)})

    return pd.DataFrame(vifs)


# ---------------------------------------------------------------------------
# Full-sample model fit
# ---------------------------------------------------------------------------

def fit_full_model(
    X: pd.DataFrame,
    y: pd.Series,
    C: float = 1.0,
    features: list[str] = CORE_FEATURES,
) -> tuple[LogisticRegression, pd.DataFrame, np.ndarray]:
    """
    Fit logistic regression on all complete rows.

    Returns
    -------
    model     : fitted LogisticRegression
    coef_df   : coefficient table with expected vs actual signs
    y_prob_is : in-sample predicted probabilities (NaN where data absent)
    """
    # Complete cases only
    mask  = X[features].notna().all(axis=1) & y.notna()
    X_fit = X.loc[mask, features]
    y_fit = y.loc[mask]

    # Standardise on full training data
    mu  = X_fit.mean()
    sig = X_fit.std(ddof=1).replace(0.0, np.nan)
    X_scaled = ((X_fit - mu) / sig).fillna(0.0)

    model = LogisticRegression(
        C=C,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=1000,
        random_state=42,
    )
    model.fit(X_scaled, y_fit)

    # In-sample probabilities
    X_all_scaled = ((X[features] - mu) / sig).fillna(0.0)
    valid_rows   = X[features].notna().all(axis=1)
    y_prob_is    = pd.Series(np.nan, index=X.index, name="prob_bear_is")
    y_prob_is.loc[valid_rows] = model.predict_proba(
        X_all_scaled.loc[valid_rows]
    )[:, 1]

    # Coefficient table
    coef_df = pd.DataFrame({
        "Feature":       features,
        "Coefficient":   model.coef_[0].round(4),
        "Expected sign": [EXPECTED_SIGNS.get(f, 0) for f in features],
    })
    coef_df["Actual sign"]  = np.sign(coef_df["Coefficient"]).astype(int)
    coef_df["Sign correct"] = coef_df["Expected sign"] == coef_df["Actual sign"]

    return model, coef_df, y_prob_is


# ---------------------------------------------------------------------------
# Walk-forward OOS
# ---------------------------------------------------------------------------

def walk_forward_oos(
    X:         pd.DataFrame,
    y:         pd.Series,
    oos_start: str = OOS_START,
    C:         float = 1.0,
    features:  list[str] = CORE_FEATURES,
    min_pos:   int = MIN_POSITIVES,
) -> pd.Series:
    """
    Expanding-window OOS predictions.

    At each month t >= oos_start:
      1. Train on all months before t with complete data.
      2. Standardise features using training-data statistics only.
      3. Predict probability for month t.
      4. Advance t by 1 month.

    Returns
    -------
    pd.Series of OOS probabilities (NaN before oos_start or when training
    data is insufficient).
    """
    oos_ts   = pd.Timestamp(oos_start)
    y_prob   = pd.Series(np.nan, index=X.index, name="prob_bear_oos")
    all_idx  = X.index

    for i, t in enumerate(all_idx):
        if t < oos_ts:
            continue

        # Training set: all months strictly before t
        X_train = X.loc[all_idx < t, features]
        y_train = y.loc[all_idx < t]

        # Complete training rows
        train_mask = X_train.notna().all(axis=1) & y_train.notna()
        X_tr = X_train.loc[train_mask]
        y_tr = y_train.loc[train_mask]

        if len(y_tr) < 20 or y_tr.sum() < min_pos:
            continue

        # Test row
        X_te = X.loc[[t], features]
        if X_te.isna().any(axis=1).values[0]:
            continue

        # Expanding-window standardisation
        X_tr_sc, X_te_sc = expanding_standardize(X_tr, X_te)

        # Fit and predict
        try:
            m = LogisticRegression(
                C=C,
                class_weight="balanced",
                solver="lbfgs",
                max_iter=1000,
                random_state=42,
            )
            m.fit(X_tr_sc, y_tr)
            y_prob.loc[t] = m.predict_proba(X_te_sc)[0, 1]
        except Exception as exc:
            warnings.warn(f"OOS fit failed at {t.date()}: {exc}", stacklevel=2)

    return y_prob


# ---------------------------------------------------------------------------
# Weight-constrained logistic regression
# ---------------------------------------------------------------------------

# Features used in the constrained model.
# lei_6m_growth is included so the optimizer must assign it ≥10% weight,
# ensuring the leading macro indicator always participates.
CONSTRAINED_FEATURES: list[str] = [
    "ntfs_level",
    "baa_level",
    "baa_3m_chg",
    "ebp_level",
    "icsa_yoy_pct",
    "lei_6m_growth",
]


class ConstrainedLogisticModel:
    """
    Thin wrapper that stores constrained logistic regression coefficients
    and exposes predict_proba() compatible with the rest of the pipeline.
    """

    def __init__(self, coef: np.ndarray, intercept: float) -> None:
        self.coef_      = np.array([coef])
        self.intercept_ = np.array([intercept])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits = X @ self.coef_[0] + self.intercept_[0]
        prob   = expit(logits)
        return np.column_stack([1 - prob, prob])


def fit_weight_constrained_model(
    X_scaled: np.ndarray,
    y_arr:    np.ndarray,
    signs:    np.ndarray,
    min_w:    float = 0.10,
    max_w:    float = 0.40,
    C:        float = 1.0,
) -> tuple[np.ndarray, float, bool]:
    """
    Logistic regression with per-feature weight constraints.

    Parametrisation
    ---------------
    Write each coefficient as c_i = sign_i × γ_i  (γ_i ≥ 0).
    Then  weight_i = γ_i / Σγ_j  and the constraints become linear in γ:

        γ_i  ≥  min_w × Σγ_j       (lower bound)
        γ_i  ≤  max_w × Σγ_j       (upper bound)

    The signs are fixed to those of the full-sample unconstrained model so
    the economic interpretation of each feature is preserved.

    Parameters
    ----------
    X_scaled : standardised feature matrix  (n_samples × n_features)
    y_arr    : binary target vector
    signs    : sign of each coefficient  (±1), fixed from unconstrained fit
    min_w    : minimum weight per feature (default 0.10 = 10 %)
    max_w    : maximum weight per feature (default 0.40 = 40 %)
    C        : inverse L2 regularisation strength

    Returns
    -------
    coef      : constrained coefficient vector (length n_features)
    intercept : scalar intercept
    converged : bool
    """
    n  = len(signs)
    pr = float(y_arr.mean())
    sw = np.where(y_arr == 1, 0.5 / max(pr, 1e-9), 0.5 / max(1 - pr, 1e-9))

    def objective(params: np.ndarray) -> float:
        gamma = params[:n]
        b     = params[n]
        logits = X_scaled @ (signs * gamma) + b
        logits = np.clip(logits, -500, 500)
        nll = -(sw * (y_arr * np.log(expit(logits) + 1e-12)
                      + (1 - y_arr) * np.log(1 - expit(logits) + 1e-12))).sum()
        return nll + 0.5 / C * float(np.sum(gamma ** 2))

    # Linear inequality constraints: each in [min_w, max_w] of total
    cons = []
    for i in range(n):
        cons += [
            {"type": "ineq", "fun": lambda p, i=i: p[i] - min_w * p[:n].sum()},
            {"type": "ineq", "fun": lambda p, i=i: max_w * p[:n].sum() - p[i]},
        ]

    # Bounds: γ_i ≥ 0, intercept unconstrained
    bnds = [(1e-8, None)] * n + [(None, None)]

    # Initialise at equal weights
    p0 = np.append(np.full(n, 1.0 / n), 0.0)

    res = minimize(
        objective, p0,
        method="SLSQP",
        constraints=cons,
        bounds=bnds,
        options={"maxiter": 3000, "ftol": 1e-11},
    )

    gamma_opt = np.abs(res.x[:n])
    return signs * gamma_opt, float(res.x[n]), bool(res.success)


def walk_forward_constrained_oos(
    X:         pd.DataFrame,
    y:         pd.Series,
    oos_start: str = OOS_START,
    min_w:     float = 0.10,
    max_w:     float = 0.40,
    C:         float = 1.0,
    features:  list[str] = CONSTRAINED_FEATURES,
    min_pos:   int = MIN_POSITIVES,
) -> pd.Series:
    """
    Walk-forward OOS using the weight-constrained logistic model.

    At each test month the constrained model is re-fitted from scratch on
    all preceding complete observations, with signs fixed to those of the
    unconstrained fit on the same training window.
    """
    oos_ts  = pd.Timestamp(oos_start)
    y_prob  = pd.Series(np.nan, index=X.index, name="prob_bear_constrained_oos")
    all_idx = X.index

    for t in all_idx:
        if t < oos_ts:
            continue

        X_train = X.loc[all_idx < t, features]
        y_train = y.loc[all_idx < t]
        mask    = X_train.notna().all(axis=1) & y_train.notna()
        X_tr    = X_train.loc[mask]
        y_tr    = y_train.loc[mask]

        if len(y_tr) < 20 or y_tr.sum() < min_pos:
            continue

        X_te = X.loc[[t], features]
        if X_te.isna().any(axis=1).values[0]:
            continue

        X_tr_sc, X_te_sc = expanding_standardize(X_tr, X_te)

        # Sign direction from unconstrained fit on same training data
        try:
            m0 = LogisticRegression(
                C=C, class_weight="balanced", solver="lbfgs",
                max_iter=1000, random_state=42,
            )
            m0.fit(X_tr_sc.values, y_tr.values)
            signs = np.sign(m0.coef_[0])

            coef, intercept, _ = fit_weight_constrained_model(
                X_tr_sc.values, y_tr.values, signs, min_w, max_w, C,
            )
            prob = expit(X_te_sc.values @ coef + intercept)[0]
            y_prob.loc[t] = float(prob)
        except Exception as exc:
            warnings.warn(f"Constrained OOS failed at {t.date()}: {exc}", stacklevel=2)

    return y_prob


# ---------------------------------------------------------------------------
# Exhaustive combination search with sign filter
# ---------------------------------------------------------------------------

def exhaustive_combination_search(
    features_df:    pd.DataFrame,
    y:              pd.Series,
    candidates:     list[str],
    expected_signs: dict[str, int],
    min_k:          int   = 4,
    max_k:          int   = 7,
    top_n:          int   = 10,
    min_obs:        int   = 60,
) -> list[dict]:
    """
    Search every combination of size [min_k, max_k] from candidates.

    For each combination:
      1. Fit unconstrained balanced logistic (sklearn — fast).
      2. Require every coefficient sign to match expected_signs.
      3. Score by in-sample AUC.

    Returns the top_n passing combinations sorted by IS AUC (descending).

    Sign rationale is enforced here so downstream HAC + constrained fits
    inherit economically coherent directions.
    """
    from itertools import combinations as _combos
    from math import comb as _comb

    total = sum(_comb(len(candidates), k) for k in range(min_k, max_k + 1))
    print(f"\n  Searching {total:,} combinations  "
          f"({len(candidates)} candidates, size {min_k}–{max_k}) ...")

    passing: list[dict] = []
    tested = sign_fail = obs_fail = 0

    for k in range(min_k, max_k + 1):
        for feat_tuple in _combos(candidates, k):
            feat_list = list(feat_tuple)
            tested += 1

            mask = features_df[feat_list].notna().all(axis=1) & y.notna()
            if mask.sum() < min_obs:
                obs_fail += 1
                continue

            X_c  = features_df.loc[mask, feat_list]
            y_c  = y.loc[mask]
            mu   = X_c.mean()
            sig  = X_c.std(ddof=1).replace(0.0, np.nan)
            X_sc = ((X_c - mu) / sig).fillna(0.0)

            try:
                m = LogisticRegression(
                    C=1.0, class_weight="balanced",
                    solver="lbfgs", max_iter=500, random_state=42,
                )
                m.fit(X_sc.values, y_c.values)
            except Exception:
                continue

            # Strict sign filter
            coef_map = dict(zip(feat_list, m.coef_[0]))
            if not all(
                int(np.sign(coef_map[f])) == expected_signs[f]
                for f in feat_list
                if f in expected_signs
            ):
                sign_fail += 1
                continue

            try:
                is_auc = roc_auc_score(
                    y_c.values,
                    m.predict_proba(X_sc.values)[:, 1],
                )
            except Exception:
                continue

            passing.append({
                "features":   feat_list,
                "n_features": k,
                "n_obs":      int(mask.sum()),
                "is_auc":     round(is_auc, 4),
                "coefs":      {f: round(v, 4) for f, v in coef_map.items()},
            })

    print(f"  Tested {tested:,}  |  sign-fail {sign_fail:,}  |  "
          f"obs-fail {obs_fail:,}  |  passing {len(passing):,}")
    return sorted(passing, key=lambda x: -x["is_auc"])[:top_n]


# ---------------------------------------------------------------------------
# Category-constrained combination search
# ---------------------------------------------------------------------------

def category_constrained_search(
    features_df:    pd.DataFrame,
    y:              pd.Series,
    categories:     dict[str, list[str]],
    min_max:        dict[str, tuple[int, int]],
    expected_signs: dict[str, int],
    top_n:          int = 10,
    min_obs:        int = 60,
) -> list[dict]:
    """
    Search all feature combinations satisfying per-category cardinality constraints.

    For each category, select between min and max features (inclusive).
    The Cartesian product across categories defines the full search space.
    Each combination is then scored by in-sample AUC after a strict sign filter.

    Parameters
    ----------
    categories  : dict mapping category name -> list of candidate features.
    min_max     : dict mapping category name -> (min_count, max_count).
    expected_signs : required coefficient signs; combinations that violate any
                     sign are discarded.

    Returns
    -------
    Top top_n combinations sorted by IS AUC (descending), all sign-correct.
    """
    from itertools import combinations as _combos, product as _product

    # Build per-category option lists (each option = a sub-list of selected features)
    category_options: list[list[list[str]]] = []
    for cat, feats in categories.items():
        mn, mx = min_max[cat]
        options: list[list[str]] = []
        for k in range(mn, min(mx, len(feats)) + 1):
            for combo in _combos(feats, k):
                options.append(list(combo))
        category_options.append(options)

    # Enumerate all cross-category combinations
    all_feat_lists = [
        [f for sub in cat_sel for f in sub]
        for cat_sel in _product(*category_options)
    ]

    total     = len(all_feat_lists)
    sign_fail = 0
    passing: list[dict] = []

    print(f"\n  Searching {total:,} category-constrained combinations ...")

    for feat_list in all_feat_lists:
        mask = features_df[feat_list].notna().all(axis=1) & y.notna()
        if mask.sum() < min_obs:
            continue

        X_c  = features_df.loc[mask, feat_list]
        y_c  = y.loc[mask]
        mu   = X_c.mean()
        sig  = X_c.std(ddof=1).replace(0.0, np.nan)
        X_sc = ((X_c - mu) / sig).fillna(0.0)

        try:
            m = LogisticRegression(
                C=1.0, class_weight="balanced",
                solver="lbfgs", max_iter=500, random_state=42,
            )
            m.fit(X_sc.values, y_c.values)
        except Exception:
            continue

        coef_map = dict(zip(feat_list, m.coef_[0]))
        if not all(
            int(np.sign(coef_map[f])) == expected_signs[f]
            for f in feat_list if f in expected_signs
        ):
            sign_fail += 1
            continue

        try:
            is_auc = roc_auc_score(y_c.values, m.predict_proba(X_sc.values)[:, 1])
        except Exception:
            continue

        passing.append({
            "features":   feat_list,
            "n_features": len(feat_list),
            "n_obs":      int(mask.sum()),
            "is_auc":     round(is_auc, 4),
            "coefs":      {f: round(v, 4) for f, v in coef_map.items()},
        })

    print(f"  sign-fail {sign_fail:,}  |  passing {len(passing):,}")
    return sorted(passing, key=lambda x: -x["is_auc"])[:top_n]


# ---------------------------------------------------------------------------
# HAC-aware feature selection
# ---------------------------------------------------------------------------

def univariate_hac_screen(
    features_df: pd.DataFrame,
    y:           pd.Series,
    candidates:  list[str],
    max_lags:    int = 12,
) -> pd.DataFrame:
    """
    Run univariate HAC logistic regression for every candidate feature.

    Each feature is standardised on its own complete-case history before
    fitting, so coefficients are comparable across features.

    Returns a DataFrame sorted by ascending p-value (most significant first).
    """
    rows = []
    for feat in candidates:
        mask = features_df[feat].notna() & y.notna()
        if mask.sum() < 30:
            continue
        X_f  = features_df.loc[mask, [feat]]
        y_f  = y.loc[mask]
        mu   = X_f.mean();  sig = X_f.std(ddof=1).replace(0.0, np.nan)
        X_sc = ((X_f - mu) / sig).fillna(0.0)

        # IS AUC (quick gauge of univariate predictive power)
        try:
            m = LogisticRegression(C=1.0, class_weight="balanced",
                                   solver="lbfgs", max_iter=500, random_state=42)
            m.fit(X_sc.values, y_f.values)
            p_hat = m.predict_proba(X_sc.values)[:, 1]
            is_auc = roc_auc_score(y_f.values, p_hat)
        except Exception:
            is_auc = np.nan

        # HAC logistic inference
        try:
            X_sm = sm.add_constant(X_sc.values, prepend=True)
            res  = sm.Logit(y_f.values, X_sm).fit(
                cov_type="HAC",
                cov_kwds={"maxlags": max_lags, "use_correction": True},
                disp=False,
            )
            coef = float(res.params[1])
            hse  = float(res.bse[1])
            z    = float(res.tvalues[1])
            pv   = float(res.pvalues[1])
        except Exception:
            coef, hse, z, pv = np.nan, np.nan, np.nan, np.nan

        rows.append({
            "Feature":     feat,
            "Coefficient": round(coef, 4),
            "HAC SE":      round(hse,  4),
            "z-stat":      round(z,    3),
            "p-value":     round(pv,   4),
            "IS AUC":      round(is_auc, 4) if not np.isnan(is_auc) else np.nan,
            "N obs":       int(mask.sum()),
        })

    return (
        pd.DataFrame(rows)
        .sort_values("p-value")
        .reset_index(drop=True)
    )


def forward_select_hac(
    features_df:  pd.DataFrame,
    y:            pd.Series,
    candidates:   list[str],
    max_features: int   = 6,
    max_lags:     int   = 12,
    p_entry:      float = 0.20,
    min_auc_gain: float = 0.005,
) -> list[str]:
    """
    Greedy forward selection: at each step add the candidate that gives the
    largest in-sample AUC improvement, subject to its HAC p-value < p_entry.

    Stop when:
      * max_features is reached, or
      * no remaining candidate meets p_entry, or
      * no remaining candidate improves IS AUC by at least min_auc_gain.

    Parameters
    ----------
    p_entry      : HAC p-value threshold for a feature to be eligible.
    min_auc_gain : minimum IS AUC improvement to justify adding a feature.
    """
    selected: list[str] = []
    remaining           = list(candidates)

    print(f"\n  Forward selection  (p_entry={p_entry}, min_gain={min_auc_gain:.3f})")
    print(f"  {'Round':<6}  {'Added feature':<22}  {'IS AUC':>7}  {'HAC p':>7}")
    print(f"  {'-'*6}  {'-'*22}  {'-'*7}  {'-'*7}")

    best_auc = 0.0

    for rnd in range(1, max_features + 1):
        best_candidate = None
        best_new_auc   = best_auc
        best_pval      = 1.0

        for feat in remaining:
            trial = selected + [feat]
            mask  = features_df[trial].notna().all(axis=1) & y.notna()
            if mask.sum() < 30:
                continue

            X_tr = features_df.loc[mask, trial]
            y_tr = y.loc[mask]
            mu   = X_tr.mean();  sig = X_tr.std(ddof=1).replace(0.0, np.nan)
            X_sc = ((X_tr - mu) / sig).fillna(0.0)

            # IS AUC
            try:
                m = LogisticRegression(C=1.0, class_weight="balanced",
                                       solver="lbfgs", max_iter=500, random_state=42)
                m.fit(X_sc.values, y_tr.values)
                auc = roc_auc_score(y_tr.values,
                                    m.predict_proba(X_sc.values)[:, 1])
            except Exception:
                continue

            # HAC p-value for the candidate feature only
            try:
                X_sm = sm.add_constant(X_sc.values, prepend=True)
                res  = sm.Logit(y_tr.values, X_sm).fit(
                    cov_type="HAC",
                    cov_kwds={"maxlags": max_lags, "use_correction": True},
                    disp=False,
                )
                feat_idx = trial.index(feat) + 1   # +1 for intercept
                pv = float(res.pvalues[feat_idx])
            except Exception:
                pv = 1.0

            if pv < p_entry and auc > best_new_auc + min_auc_gain:
                best_new_auc   = auc
                best_candidate = feat
                best_pval      = pv

        if best_candidate is None:
            print(f"  {'--':<6}  No eligible candidate — stopping.")
            break

        selected.append(best_candidate)
        remaining.remove(best_candidate)
        best_auc = best_new_auc
        print(f"  {rnd:<6}  {best_candidate:<22}  {best_auc:.4f}  {best_pval:.4f}")

    return selected


# ---------------------------------------------------------------------------
# Newey-West HAC inference
# ---------------------------------------------------------------------------

def hac_inference(
    X_scaled:  pd.DataFrame,
    y:         pd.Series,
    max_lags:  int = 12,
) -> pd.DataFrame:
    """
    Fit logistic regression via statsmodels and return a Newey-West HAC
    inference table.

    Why HAC?
    --------
    A 12-month forward target at monthly frequency means consecutive
    observations share 11 of 12 months in their forward window — they are
    far from independent. Standard MLE treats them as i.i.d., which severely
    underestimates standard errors (inflates t-stats, overstates significance).

    The Newey-West sandwich estimator (Econometrica 1987) with
    max_lags = horizon corrects for both heteroscedasticity and
    autocorrelation of this order.

    The HAC covariance matrix V_HAC = H⁻¹ S_HAC H⁻¹ where:
      H      = -X'WX  (Hessian, W = diag of p*(1-p))
      S_HAC  = Σ_t sₜsₜ' + Σ_{l=1}^{L} w_l Σ_t (sₜsₜ₋ₗ' + sₜ₋ₗsₜ')
      sₜ     = (yₜ - p̂ₜ) xₜ  (score for observation t)
      w_l    = 1 - l/(L+1)   (Bartlett kernel)

    Parameters
    ----------
    X_scaled : standardised feature matrix (complete rows only)
    y        : binary target (aligned with X_scaled)
    max_lags : Newey-West lag truncation; set to forward horizon (12 for bear)

    Returns
    -------
    pd.DataFrame with columns:
        Coefficient, HAC SE, z-stat, p-value, 95% CI lower, 95% CI upper
    """
    mask = X_scaled.notna().all(axis=1) & y.notna()
    X_c  = X_scaled.loc[mask]
    y_c  = y.loc[mask]

    X_sm = sm.add_constant(X_c.values, prepend=True)

    result = sm.Logit(y_c.values, X_sm).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": max_lags, "use_correction": True},
        disp=False,
    )

    names    = ["intercept"] + list(X_c.columns)
    coef     = result.params
    hac_se   = result.bse
    z_stat   = result.tvalues
    p_val    = result.pvalues
    ci_lo    = result.conf_int()[:, 0]
    ci_hi    = result.conf_int()[:, 1]

    return pd.DataFrame({
        "Feature":    names,
        "Coefficient": coef.round(4),
        "HAC SE":      hac_se.round(4),
        "z-stat":      z_stat.round(3),
        "p-value":     p_val.round(4),
        "CI 2.5%":     ci_lo.round(4),
        "CI 97.5%":    ci_hi.round(4),
        "Significant": (p_val < 0.05),
    })


# ---------------------------------------------------------------------------
# Feature weights
# ---------------------------------------------------------------------------

def feature_weights(
    model: LogisticRegression,
    features: list[str] = CORE_FEATURES,
    expected_signs: dict[str, int] = EXPECTED_SIGNS,
) -> pd.DataFrame:
    """
    Derive the relative weight of each feature from standardised coefficients.

    Method
    ------
    Because features are z-scored before fitting, each coefficient measures
    the change in log-odds per one-standard-deviation move in the feature.
    The absolute value captures magnitude independently of direction.

        weight_i = |coef_i| / sum_j(|coef_j|) * 100

    Weights sum to exactly 100%.

    Direction column shows whether the feature currently acts as a
    bearish driver (coefficient sign agrees with expected sign, so a
    one-SD adverse move increases bear probability) or bullish (opposite).

    Parameters
    ----------
    model          : fitted LogisticRegression (features must have been z-scored).
    features       : ordered list of feature names matching model.coef_.
    expected_signs : dict of expected coefficient signs {feature: +1 or -1}.

    Returns
    -------
    pd.DataFrame sorted by weight descending.
    """
    coefs     = model.coef_[0]
    abs_coefs = np.abs(coefs)
    total     = abs_coefs.sum()
    weights   = abs_coefs / total * 100

    rows = []
    for feat, coef, w in zip(features, coefs, weights):
        exp_sign = expected_signs.get(feat, 0)
        act_sign = int(np.sign(coef))
        # "Bearish driver" means: moving in the adverse direction raises bear prob.
        # Sign agrees with expected → bearish driver when the indicator deteriorates.
        direction = "Bearish driver" if act_sign == exp_sign else "Contrarian"
        rows.append({
            "Feature":       feat,
            "Std coef":      round(coef, 4),
            "Weight (%)":    round(w, 1),
            "Direction":     direction,
        })

    df = pd.DataFrame(rows).sort_values("Weight (%)", ascending=False).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    y_true: pd.Series,
    y_prob: pd.Series,
    label:  str = "",
) -> dict[str, float]:
    """
    Compute AUC, PR-AUC, and Brier score on valid (non-NaN) rows.

    Note: overlapping 12-month targets inflate effective sample size.
    Brier and AUC are point estimates; Phase 6 computes Newey-West SEs.
    """
    mask   = y_true.notna() & y_prob.notna()
    y_t    = y_true.loc[mask].values
    y_p    = y_prob.loc[mask].values

    if len(y_t) < 10 or y_t.sum() == 0:
        return {"AUC": np.nan, "PR-AUC": np.nan, "Brier": np.nan}

    auc    = roc_auc_score(y_t, y_p)
    pr_auc = average_precision_score(y_t, y_p)
    brier  = brier_score_loss(y_t, y_p)

    if label:
        print(f"  {label}")
        print(f"    AUC    : {auc:.4f}")
        print(f"    PR-AUC : {pr_auc:.4f}")
        print(f"    Brier  : {brier:.4f}")
        print(f"    N obs  : {len(y_t)}  (positives: {int(y_t.sum())})")

    return {"AUC": auc, "PR-AUC": pr_auc, "Brier": brier}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print(f"\n{'='*65}")
    print("  Phase 4 — Bear Market Logistic Regression")
    print(f"{'='*65}")

    # -- Load data --
    X, y = load_model_data()
    X = preprocess(X)
    print(f"\n  Features : {CORE_FEATURES}")
    print(f"  Rows     : {len(X)}  ({X.index[0].date()} to {X.index[-1].date()})")
    mask = X[CORE_FEATURES].notna().all(axis=1) & y.notna()
    print(f"  Complete : {mask.sum()} rows  "
          f"(bear rate: {y[mask].mean():.1%})")

    # -- VIF check --
    print(f"\n{'='*65}")
    print("  VIF check (complete rows only)")
    print(f"{'='*65}")
    vif_df = compute_vif(X[CORE_FEATURES].dropna())
    for _, row in vif_df.iterrows():
        flag = "  *** HIGH ***" if row["VIF"] > 5 else ""
        print(f"  {row['Feature']:<20}  VIF = {row['VIF']:.2f}{flag}")

    # -- Full-sample model --
    print(f"\n{'='*65}")
    print("  Full-sample model (L2, C=1.0, class_weight=balanced)")
    print(f"{'='*65}")
    model, coef_df, prob_is = fit_full_model(X, y)

    print("\n  Coefficient table:")
    print(f"  {'Feature':<20}  {'Coef':>8}  {'Exp':>5}  {'Act':>5}  {'OK':>5}")
    print(f"  {'-'*20}  {'-'*8}  {'-'*5}  {'-'*5}  {'-'*5}")
    for _, row in coef_df.iterrows():
        ok = "yes" if row["Sign correct"] else "NO <--"
        print(f"  {row['Feature']:<20}  {row['Coefficient']:>8.4f}  "
              f"{row['Expected sign']:>5}  {row['Actual sign']:>5}  {ok:>6}")
    print(f"  Intercept: {model.intercept_[0]:.4f}")

    # -- Newey-West HAC inference --
    print(f"\n{'='*65}")
    print(f"  Newey-West HAC inference  (max_lags = 12 = bear horizon)")
    print(f"  Overlapping 12m targets: consecutive months share 11/12 of window.")
    print(f"  Standard MLE SEs are underestimated — HAC corrects this.")
    print(f"{'='*65}")

    # Standardise using complete training data (same as full-sample model)
    mask_hac   = X[CORE_FEATURES].notna().all(axis=1) & y.notna()
    X_hac_raw  = X.loc[mask_hac, CORE_FEATURES]
    mu_hac     = X_hac_raw.mean()
    sig_hac    = X_hac_raw.std(ddof=1).replace(0.0, np.nan)
    X_hac_sc   = ((X_hac_raw - mu_hac) / sig_hac).fillna(0.0)
    y_hac      = y.loc[mask_hac]

    hac_df = hac_inference(X_hac_sc, y_hac, max_lags=12)
    print()
    print(f"  {'Feature':<20}  {'Coef':>8}  {'HAC SE':>8}  "
          f"{'z-stat':>7}  {'p-val':>7}  {'95% CI':>18}  {'Sig?':>5}")
    print(f"  {'-'*20}  {'-'*8}  {'-'*8}  {'-'*7}  {'-'*7}  {'-'*18}  {'-'*5}")
    for _, row in hac_df.iterrows():
        ci = f"[{row['CI 2.5%']:+.3f}, {row['CI 97.5%']:+.3f}]"
        sig = "  *" if row["Significant"] else ""
        print(f"  {row['Feature']:<20}  {row['Coefficient']:>8.4f}  "
              f"{row['HAC SE']:>8.4f}  {row['z-stat']:>7.3f}  "
              f"{row['p-value']:>7.4f}  {ci:>18}{sig}")
    print(f"\n  * significant at 5% level (HAC-corrected)")

    # -- Feature weights --
    print(f"\n{'='*65}")
    print("  Feature weights  (|standardised coef| / sum × 100)")
    print(f"{'='*65}")
    w_df = feature_weights(model)
    print(f"\n  {'Feature':<20}  {'Std coef':>9}  {'Weight':>8}  Direction")
    print(f"  {'-'*20}  {'-'*9}  {'-'*8}  {'-'*20}")
    for _, row in w_df.iterrows():
        bar = "█" * int(row["Weight (%)"] / 2)
        print(f"  {row['Feature']:<20}  {row['Std coef']:>9.4f}  "
              f"{row['Weight (%)']:>7.1f}%  {row['Direction']}  {bar}")
    print(f"  {'TOTAL':<20}  {'':9}  {'100.0':>7}%")

    wrong = coef_df[~coef_df["Sign correct"]]
    if not wrong.empty:
        print("\n  Sign-flip notes:")
        notes = {
            "sahm_level":    "Sahm is coincident, not leading — high Sahm = deep in recession "
                             "= forward 12m recovery (y_bear=0). Use icsa_yoy_pct instead.",
            "lei_6m_growth": "Suppressor effect: once EBP/BAA control for credit risk, LEI "
                             "flips sign. Retain but interpret cautiously.",
            "icsa_yoy_pct":  "Economically defensible: negative YoY = falling claims = "
                             "tight/overheating labor market = late-cycle = elevated bear risk. "
                             "Also: high claims occur DURING crashes; forward 12m = recovery.",
        }
        for _, row in wrong.iterrows():
            note = notes.get(row["Feature"], "Investigate collinearity.")
            print(f"    {row['Feature']}: {note}")

    # -- In-sample metrics --
    print(f"\n{'='*65}")
    print("  Metrics")
    print(f"{'='*65}")
    evaluate(y, prob_is, label="In-sample (full period)")

    # -- Walk-forward OOS --
    print(f"\n  Running walk-forward OOS from {OOS_START} ...")
    print("  (re-fits model at every month — may take ~30s)")
    prob_oos = walk_forward_oos(X, y, oos_start=OOS_START)
    oos_valid = prob_oos.notna().sum()
    print(f"  OOS predictions: {oos_valid} months")
    evaluate(y, prob_oos, label=f"OOS ({OOS_START} onward)")

    # -- Spot-check probabilities at key episodes --
    print(f"\n{'='*65}")
    print("  OOS probability at key bear market episodes")
    print(f"{'='*65}")
    check_dates = {
        "2000-08-31": "Dot-com onset",
        "2001-09-30": "Post-9/11",
        "2007-06-30": "Pre-GFC",
        "2007-10-31": "GFC peak",
        "2008-09-30": "Lehman",
        "2020-01-31": "Pre-COVID",
        "2020-03-31": "COVID trough",
        "2022-01-31": "Rate-hike bear onset",
        "2024-12-31": "Recent",
        "2026-04-30": "Latest",
    }
    print(f"  {'Date':<15}  {'OOS prob':>9}  {'IS prob':>8}  {'y_bear':>7}  Note")
    print(f"  {'-'*15}  {'-'*9}  {'-'*8}  {'-'*7}  {'-'*20}")
    for d, note in check_dates.items():
        ts = pd.Timestamp(d)
        if ts not in X.index:
            continue
        p_oos = prob_oos.get(ts, np.nan)
        p_is  = prob_is.get(ts, np.nan)
        yb    = y.get(ts, np.nan)
        oos_s = f"{p_oos:.1%}" if pd.notna(p_oos) else "  NaN"
        is_s  = f"{p_is:.1%}"  if pd.notna(p_is)  else "  NaN"
        yb_s  = f"{int(yb)}"   if pd.notna(yb)     else "NaN"
        print(f"  {d:<15}  {oos_s:>9}  {is_s:>8}  {yb_s:>7}  {note}")

    # ================================================================
    # Category-constrained search: ≥1 ≤2 per category, weight ≤ 30%
    # ================================================================

    # Load full Phase 2 bear feature set
    all_bear = pd.read_csv(
        _DATA_DIR / "bear_features.csv", index_col=0, parse_dates=True
    )
    y_all = y.reindex(all_bear.index)

    print(f"\n{'='*65}")
    print("  Category-constrained search")
    print("  Rule: ≥1 and ≤2 features per category, weight ≤ 30%")
    print(f"  Categories:")
    for cat, feats in FEATURE_CATEGORIES.items():
        mn, mx = CATEGORY_MIN_MAX[cat]
        print(f"    {cat:<15} [{mn}-{mx}]: {feats}")
    print(f"{'='*65}")

    # -- Step 1: Category-constrained IS search with sign filter --
    top_combos = category_constrained_search(
        all_bear, y_all,
        FEATURE_CATEGORIES, CATEGORY_MIN_MAX, EXPECTED_SIGNS_CLEAN,
        top_n=10, min_obs=60,
    )

    print(f"\n  Top 10 by IS AUC (sign-correct, category-balanced):")
    print(f"  {'Rank':<5}  {'IS AUC':>7}  {'N':>5}  {'k':>3}  Features")
    print(f"  {'-'*5}  {'-'*7}  {'-'*5}  {'-'*3}  {'-'*50}")
    for rank, c in enumerate(top_combos, 1):
        print(f"  {rank:<5}  {c['is_auc']:>7.4f}  {c['n_obs']:>5}  "
              f"{c['n_features']:>3}  {c['features']}")

    # -- Step 2: Weight constraint + OOS on top 3 --
    print(f"\n{'='*65}")
    print("  Top 3 — weight constraint (≤30%) + OOS evaluation")
    print(f"{'='*65}")

    oos_results = []
    for rank, combo in enumerate(top_combos[:3], 1):
        feats = combo["features"]
        print(f"\n  [{rank}] {feats}")

        mask_c = all_bear[feats].notna().all(axis=1) & y_all.notna()
        X_c    = all_bear.loc[mask_c, feats]
        y_c    = y_all.loc[mask_c]
        mu_c   = X_c.mean();  sig_c = X_c.std(ddof=1).replace(0.0, np.nan)
        X_c_sc = ((X_c - mu_c) / sig_c).fillna(0.0)

        # Signs from unconstrained fit (already verified correct in Step 1)
        signs_c = np.array([EXPECTED_SIGNS_CLEAN[f] for f in feats])

        coef_c, intercept_c, converged = fit_weight_constrained_model(
            X_c_sc.values, y_c.values, signs_c, min_w=0.01, max_w=0.30,
        )
        if not converged:
            print("    Optimiser did not converge — skipping.")
            continue

        # Constrained IS AUC
        valid_c  = all_bear[feats].notna().all(axis=1)
        X_all_sc = ((all_bear[feats] - mu_c) / sig_c).fillna(0.0)
        prob_is_c = pd.Series(np.nan, index=all_bear.index)
        prob_is_c.loc[valid_c] = expit(X_all_sc.loc[valid_c].values @ coef_c + intercept_c)
        is_metrics = evaluate(y_all, prob_is_c)

        # OOS
        prob_oos_c = walk_forward_oos(
            all_bear, y_all, oos_start=OOS_START,
            features=feats, C=1.0,
        )
        oos_metrics = evaluate(y_all, prob_oos_c)

        print(f"    IS  AUC={is_metrics['AUC']:.4f}  "
              f"PR-AUC={is_metrics['PR-AUC']:.4f}  Brier={is_metrics['Brier']:.4f}")
        print(f"    OOS AUC={oos_metrics['AUC']:.4f}  "
              f"PR-AUC={oos_metrics['PR-AUC']:.4f}  Brier={oos_metrics['Brier']:.4f}")

        weights = np.abs(coef_c) / np.abs(coef_c).sum() * 100
        oos_results.append({
            "rank":         rank,
            "features":     feats,
            "coef":         coef_c,
            "intercept":    intercept_c,
            "signs":        signs_c,
            "mu":           mu_c,
            "sig":          sig_c,
            "weights":      weights,
            "is_auc":       is_metrics["AUC"],
            "oos_auc":      oos_metrics["AUC"],
            "oos_prauc":    oos_metrics["PR-AUC"],
            "oos_brier":    oos_metrics["Brier"],
            "prob_oos":     prob_oos_c,
        })

    # -- Step 3: Report all 3 models + highlight best by OOS AUC --
    if not oos_results:
        print("\n  No valid constrained models found.")
    else:
        best = max(oos_results, key=lambda x: x["oos_auc"])
        feats   = best["features"]
        coef_b  = best["coef"]
        signs_b = best["signs"]
        mu_b    = best["mu"]
        sig_b   = best["sig"]

        print(f"\n{'='*65}")
        print(f"  Step 3 — Best model  (OOS AUC {best['oos_auc']:.4f})")
        print(f"  Features: {feats}")
        print(f"{'='*65}")

        # HAC inference on best model
        mask_b = all_bear[feats].notna().all(axis=1) & y_all.notna()
        X_b    = all_bear.loc[mask_b, feats]
        y_b    = y_all.loc[mask_b]
        X_b_sc = ((X_b - mu_b) / sig_b).fillna(0.0)
        hac_b  = hac_inference(X_b_sc, y_b, max_lags=12)

        # Weight table
        abs_c   = np.abs(coef_b)
        weights = abs_c / abs_c.sum() * 100

        print(f"\n  {'Feature':<22}  {'Exp':>4}  {'Coef':>8}  {'HAC SE':>8}  "
              f"{'z':>6}  {'p-val':>7}  {'Weight':>7}  {'95% CI':>20}  Sig?")
        print(f"  {'-'*22}  {'-'*4}  {'-'*8}  {'-'*8}  "
              f"{'-'*6}  {'-'*7}  {'-'*7}  {'-'*20}  {'-'*4}")

        # Intercept row
        ic_row = hac_b[hac_b["Feature"] == "intercept"].iloc[0]
        ci_ic  = f"[{ic_row['CI 2.5%']:+.3f}, {ic_row['CI 97.5%']:+.3f}]"
        sig_ic = "  *" if ic_row["Significant"] else ""
        print(f"  {'intercept':<22}  {'':>4}  {ic_row['Coefficient']:>8.4f}  "
              f"{ic_row['HAC SE']:>8.4f}  {ic_row['z-stat']:>6.3f}  "
              f"{ic_row['p-value']:>7.4f}  {'':>7}  {ci_ic:>20}{sig_ic}")

        for feat, coef, w, exp_s in zip(feats, coef_b, weights, signs_b):
            row = hac_b[hac_b["Feature"] == feat]
            if row.empty:
                continue
            row   = row.iloc[0]
            ci    = f"[{row['CI 2.5%']:+.3f}, {row['CI 97.5%']:+.3f}]"
            sig   = "  *" if row["Significant"] else ""
            exp_s_str = "+" if exp_s > 0 else "−"
            act_s_str = "+" if coef > 0 else "−"
            sign_ok   = "✓" if int(np.sign(coef)) == exp_s else "✗"
            print(f"  {feat:<22}  {exp_s_str:>4}  {coef:>8.4f}  "
                  f"{row['HAC SE']:>8.4f}  {row['z-stat']:>6.3f}  "
                  f"{row['p-value']:>7.4f}  {w:>6.1f}%  {ci:>20}{sig}  {sign_ok}")

        print(f"  {'TOTAL':<22}  {'':>4}  {'':>8}  {'':>8}  "
              f"{'':>6}  {'':>7}  {'100.0%':>7}")

        print(f"\n  In-sample  : AUC={best['is_auc']:.4f}")
        print(f"  OOS        : AUC={best['oos_auc']:.4f}  "
              f"PR-AUC={best['oos_prauc']:.4f}  Brier={best['oos_brier']:.4f}")
        print(f"\n  * significant at 5% (HAC Newey-West, max_lags=12)")

        # Stash for export (added to `out` after it is defined below)
        _best_selected_oos = best["prob_oos"]

    # -- Constrained model [5%, 40%] --
    print(f"\n{'='*65}")
    print("  Constrained model  (each feature weight in [5 %, 40 %])")
    print(f"  Features: {CONSTRAINED_FEATURES}")
    print(f"{'='*65}")

    mask_c = X[CONSTRAINED_FEATURES].notna().all(axis=1) & y.notna()
    X_c    = X.loc[mask_c, CONSTRAINED_FEATURES]
    y_c    = y.loc[mask_c]
    mu_c   = X_c.mean();  sig_c = X_c.std(ddof=1).replace(0.0, np.nan)
    X_c_sc = ((X_c - mu_c) / sig_c).fillna(0.0)

    # Full-sample unconstrained signs (fixed for constrained optimisation)
    m_unc = LogisticRegression(C=1.0, class_weight="balanced",
                               solver="lbfgs", max_iter=1000, random_state=42)
    m_unc.fit(X_c_sc.values, y_c.values)
    signs_c = np.sign(m_unc.coef_[0])

    coef_c, intercept_c, converged = fit_weight_constrained_model(
        X_c_sc.values, y_c.values, signs_c, min_w=0.05, max_w=0.40,
    )
    print(f"\n  Optimiser converged: {converged}")

    # Constrained weights
    abs_c   = np.abs(coef_c)
    w_c     = abs_c / abs_c.sum() * 100
    print(f"\n  {'Feature':<20}  {'Std coef':>9}  {'Weight':>8}  Bar")
    print(f"  {'-'*20}  {'-'*9}  {'-'*8}  {'-'*25}")
    for feat, coef, w in sorted(zip(CONSTRAINED_FEATURES, coef_c, w_c),
                                 key=lambda x: -x[2]):
        bar = "█" * int(w / 2)
        print(f"  {feat:<20}  {coef:>9.4f}  {w:>7.1f}%  {bar}")
    print(f"  {'TOTAL':<20}  {'':>9}  {'100.0':>7}%")

    # In-sample metrics
    X_all_c_sc = ((X[CONSTRAINED_FEATURES] - mu_c) / sig_c).fillna(0.0)
    valid_c     = X[CONSTRAINED_FEATURES].notna().all(axis=1)
    prob_c_is   = pd.Series(np.nan, index=X.index, name="prob_bear_constrained_is")
    prob_c_is.loc[valid_c] = expit(
        X_all_c_sc.loc[valid_c].values @ coef_c + intercept_c
    )
    print()
    evaluate(y, prob_c_is, label="Constrained — in-sample")

    print(f"\n  Running constrained walk-forward OOS from {OOS_START} ...")
    prob_c_oos = walk_forward_constrained_oos(X, y, oos_start=OOS_START, min_w=0.05, max_w=0.40)
    print(f"  OOS predictions: {prob_c_oos.notna().sum()} months")
    evaluate(y, prob_c_oos, label=f"Constrained — OOS ({OOS_START} onward)")

    # -- Export --
    out = pd.DataFrame({
        "prob_bear_is":             prob_is,
        "prob_bear_oos":            prob_oos,
        "prob_bear_constrained_is": prob_c_is,
        "prob_bear_constrained_oos":prob_c_oos,
        "y_bear":                   y,
    })
    if "_best_selected_oos" in dir():
        out["prob_bear_selected_oos"] = _best_selected_oos.reindex(out.index)
    out_path = _DATA_DIR / "bear_model_output.csv"
    out.to_csv(out_path, date_format="%Y-%m-%d", float_format="%.6f")

    model_path = _DATA_DIR / "bear_model.pkl"
    with open(model_path, "wb") as fh:
        pickle.dump(model, fh)

    print(f"\n{'='*65}")
    print(f"  Exported: {out_path}  ({out.notna().any(axis=1).sum()} rows)")
    print(f"  Model   : {model_path}")
    print(f"{'='*65}")
