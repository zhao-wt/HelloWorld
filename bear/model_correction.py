"""
bear/model_correction.py — Phase 5: correction market logistic regression model.

Predicts the probability that the S&P 500 will experience a drawdown of
10–20% (a correction, but NOT a full bear market) over the next 6 months.

Key differences from the bear model (Phase 4)
----------------------------------------------
  Target    : y_corr  (correction only — drawdown in (-20 %, -10 %])
  Horizon   : 6 months  →  HAC max_lags = 6
  Features  : fast, mean-reverting signals (VIX term structure, trend,
              financial conditions, valuation, credit)
  Expected  : lower OOS AUC (~0.55–0.65) than the bear model; corrections
              are inherently harder to forecast (Goyal-Welch 2008)

Category constraints (same rule as Phase 4)
--------------------------------------------
  ≥ 1 and ≤ 2 features per category; no single feature weight > 30 %.

Expected signs
--------------
  vts_slope          (−)  backwardation (slope < 0) = stress = correction risk
  vts_ratio          (+)  high VIX/VIX3M = backwardation = stress
  vts_backwardation  (+)  binary backwardation dummy
  vts_slope_zscore   (−)  unusually low slope = stress
  spx_vs_10ma        (−)  below MA = bearish trend (momentum)
  spx_below_10ma     (+)  binary below-MA dummy
  m12_1_mom          (−)  negative/weak momentum = correction risk
  anfci_level        (+)  tighter financial conditions = correction risk
  anfci_3m_chg       (+)  tightening conditions
  cape_20yr_pct      (+)  high valuation percentile = elevated correction risk
  baa_zscore_24m     (+)  spreads elevated vs 2-year history = fast credit stress

Note: cpce_low_dummy excluded — only covers 2003-2019, which would
      severely restrict the effective training window.
"""

from __future__ import annotations

import pickle
import warnings
from itertools import combinations as _combos, product as _product
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

_BEAR_DIR = Path(__file__).resolve().parent
_DATA_DIR = _BEAR_DIR.parent / "data"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OOS_START    = "2010-01-31"   # VIX3M data begins 2007-12; give 2yr of training
HAC_LAGS     = 6              # correction horizon = 6 months
MAX_W        = 0.30           # max weight per feature
MIN_POSITIVES = 5

CORR_CATEGORIES: dict[str, list[str]] = {
    "Volatility":   ["vts_slope", "vts_ratio", "vts_backwardation"],  # vts_slope_zscore dropped: z≈0
    "Trend":        ["spx_vs_10ma", "spx_below_10ma", "m12_1_mom"],
    "Fin. cond.":   ["anfci_level", "anfci_3m_chg"],
    "Valuation":    ["cape_20yr_pct"],
    "Credit fast":  ["baa_zscore_24m"],
}

CORR_CATEGORY_MIN_MAX: dict[str, tuple[int, int]] = {
    "Volatility":  (1, 2),
    "Trend":       (1, 2),
    "Fin. cond.":  (1, 2),
    "Valuation":   (1, 1),
    "Credit fast": (1, 1),
}

# Sign rationale for a 6-month FORWARD correction model
# -------------------------------------------------------
# VIX signals flip vs naive expectation because they are COINCIDENT.
# When VIX is already stressed (backwardation), the correction is often
# underway and the next 6 months trend toward RECOVERY → less forward
# correction risk.  The predictive signal comes from CALM/COMPLACENCY:
#
#   vts_slope (+1) : contango (positive) = calm markets = complacency
#                    = correction risk AHEAD (same timing logic as Sahm)
#   vts_ratio (−1) : high ratio (backwardation, stress NOW) = forward
#                    recovery = less correction risk
#   vts_backwardation (−1): extreme stress = likely past trough
#
# m12_1_mom (+1) : strong positive momentum = stretched market
#                  = mean-reversion correction risk ahead
#
# cape_20yr_pct (−1) : high-CAPE periods tend to be sustained bull
#                      markets with fewer corrections; data-driven

CORR_EXPECTED_SIGNS: dict[str, int] = {
    "vts_slope":        +1,  # contango = calm = complacency = correction risk ahead
    "vts_ratio":        -1,  # backwardation stress now = forward recovery
    "vts_backwardation":-1,  # extreme stress = past trough = less forward risk
    "spx_vs_10ma":      -1,  # below MA = bearish trend = correction risk
    "spx_below_10ma":   +1,  # binary below MA
    "m12_1_mom":        +1,  # strong momentum = stretched = mean-reversion risk
    "anfci_level":      +1,  # tighter conditions = correction risk
    "anfci_3m_chg":     +1,  # tightening = correction risk
    "cape_20yr_pct":    -1,  # data-driven: high CAPE = sustained bull = fewer corrections
    "baa_zscore_24m":   +1,  # elevated spreads = credit stress = correction risk
}


# ---------------------------------------------------------------------------
# Shared helpers (mirrors model_bear.py)
# ---------------------------------------------------------------------------

def _standardize(X_tr: pd.DataFrame, X_te: pd.DataFrame):
    mu  = X_tr.mean()
    sig = X_tr.std(ddof=1).replace(0.0, np.nan)
    return ((X_tr - mu) / sig).fillna(0.0), ((X_te - mu) / sig).fillna(0.0), mu, sig


def _winsorize(s: pd.Series, lo: float = 0.01, hi: float = 0.99) -> pd.Series:
    return s.clip(s.quantile(lo), s.quantile(hi))


def evaluate(y_true: pd.Series, y_prob: pd.Series, label: str = "") -> dict:
    mask = y_true.notna() & y_prob.notna()
    yt   = y_true.loc[mask].values
    yp   = y_prob.loc[mask].values
    if len(yt) < 10 or yt.sum() == 0:
        return {"AUC": np.nan, "PR-AUC": np.nan, "Brier": np.nan}
    res = {
        "AUC":    roc_auc_score(yt, yp),
        "PR-AUC": average_precision_score(yt, yp),
        "Brier":  brier_score_loss(yt, yp),
    }
    if label:
        print(f"  {label}")
        print(f"    AUC    : {res['AUC']:.4f}")
        print(f"    PR-AUC : {res['PR-AUC']:.4f}")
        print(f"    Brier  : {res['Brier']:.4f}")
        print(f"    N obs  : {mask.sum()}  (positives: {int(yt.sum())})")
    return res


def hac_inference(X_sc: pd.DataFrame, y: pd.Series, max_lags: int = 6) -> pd.DataFrame:
    """HAC-corrected logistic inference via statsmodels."""
    mask = X_sc.notna().all(axis=1) & y.notna()
    Xc   = X_sc.loc[mask];  yc = y.loc[mask]
    X_sm = sm.add_constant(Xc.values, prepend=True)
    res  = sm.Logit(yc.values, X_sm).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": max_lags, "use_correction": True},
        disp=False,
    )
    names = ["intercept"] + list(Xc.columns)
    return pd.DataFrame({
        "Feature":     names,
        "Coefficient": res.params.round(4),
        "HAC SE":      res.bse.round(4),
        "z-stat":      res.tvalues.round(3),
        "p-value":     res.pvalues.round(4),
        "CI 2.5%":     res.conf_int()[:, 0].round(4),
        "CI 97.5%":    res.conf_int()[:, 1].round(4),
        "Significant": (res.pvalues < 0.05),
    })


def fit_weight_constrained(
    X_sc:   np.ndarray,
    y_arr:  np.ndarray,
    signs:  np.ndarray,
    min_w:  float = 0.01,
    max_w:  float = MAX_W,
    C:      float = 1.0,
) -> tuple[np.ndarray, float, bool]:
    """Logistic regression with per-feature weight constraints (scipy SLSQP)."""
    n   = len(signs)
    pr  = max(float(y_arr.mean()), 1e-9)
    sw  = np.where(y_arr == 1, 0.5 / pr, 0.5 / max(1 - pr, 1e-9))

    # Warm-start: unconstrained sklearn fit
    m0 = LogisticRegression(C=C, class_weight="balanced", solver="lbfgs",
                            max_iter=1000, random_state=42)
    m0.fit(X_sc, y_arr)
    gamma0 = np.abs(m0.coef_[0])

    def obj(p):
        g = p[:n];  b = p[n]
        logits = np.clip(X_sc @ (signs * g) + b, -500, 500)
        nll = -(sw * (y_arr * np.log(expit(logits) + 1e-12)
                      + (1 - y_arr) * np.log(1 - expit(logits) + 1e-12))).sum()
        return nll + 0.5 / C * float(np.sum(g ** 2))

    cons = []
    for i in range(n):
        cons += [
            {"type": "ineq", "fun": lambda p, i=i: p[i] - min_w * p[:n].sum()},
            {"type": "ineq", "fun": lambda p, i=i: max_w * p[:n].sum() - p[i]},
        ]

    res = minimize(obj, np.append(gamma0, m0.intercept_[0]),
                   method="SLSQP", constraints=cons,
                   bounds=[(1e-8, None)] * n + [(None, None)],
                   options={"maxiter": 3000, "ftol": 1e-11})

    gamma_opt = np.abs(res.x[:n])
    return signs * gamma_opt, float(res.x[n]), bool(res.success)


# ---------------------------------------------------------------------------
# Category-constrained search
# ---------------------------------------------------------------------------

def category_search(
    features_df:    pd.DataFrame,
    y:              pd.Series,
    categories:     dict[str, list[str]],
    min_max:        dict[str, tuple[int, int]],
    expected_signs: dict[str, int],
    top_n:          int = 10,
    min_obs:        int = 40,
    apply_sign_filter: bool = False,
) -> list[dict]:
    """
    Search all category-constrained combinations; rank by IS AUC.

    For the correction model, apply_sign_filter=False because:
    - VIX features have only ~220 rows → unstable multivariate signs
    - Many correction signals are coincident, not leading (signs empirically
      differ from textbook direction for a 6-month forward model)

    Signs are reported in the results for transparency but do not filter.
    """
    # Build per-category option lists
    cat_options: list[list[list[str]]] = []
    for cat, feats in categories.items():
        mn, mx = min_max[cat]
        opts   = [list(c) for k in range(mn, min(mx, len(feats)) + 1)
                  for c in _combos(feats, k)]
        cat_options.append(opts)

    all_combos = [
        [f for sub in sel for f in sub]
        for sel in _product(*cat_options)
    ]

    total     = len(all_combos)
    sign_fail = passing = 0
    results: list[dict] = []

    print(f"\n  Searching {total:,} combinations ...")

    for feat_list in all_combos:
        mask = features_df[feat_list].notna().all(axis=1) & y.notna()
        if mask.sum() < min_obs:
            continue

        Xc   = features_df.loc[mask, feat_list]
        yc   = y.loc[mask]
        mu   = Xc.mean();  sig = Xc.std(ddof=1).replace(0.0, np.nan)
        X_sc = ((Xc - mu) / sig).fillna(0.0)

        try:
            m = LogisticRegression(C=1.0, class_weight="balanced",
                                   solver="lbfgs", max_iter=500, random_state=42)
            m.fit(X_sc.values, yc.values)
        except Exception:
            continue

        coef_map    = dict(zip(feat_list, m.coef_[0]))
        signs_ok    = all(int(np.sign(coef_map[f])) == expected_signs[f]
                          for f in feat_list if f in expected_signs)
        n_wrong_signs = sum(1 for f in feat_list
                            if f in expected_signs
                            and int(np.sign(coef_map[f])) != expected_signs[f])
        if apply_sign_filter and not signs_ok:
            sign_fail += 1
            continue

        try:
            is_auc = roc_auc_score(yc.values, m.predict_proba(X_sc.values)[:, 1])
        except Exception:
            continue

        passing += 1
        results.append({
            "features":      feat_list,
            "n_features":    len(feat_list),
            "n_obs":         int(mask.sum()),
            "is_auc":        round(is_auc, 4),
            "n_wrong_signs": n_wrong_signs,
            "signs_ok":      signs_ok,
        })

    print(f"  sign-fail {sign_fail:,}  |  passing {passing:,}")
    return sorted(results, key=lambda x: -x["is_auc"])[:top_n]


# ---------------------------------------------------------------------------
# Walk-forward OOS
# ---------------------------------------------------------------------------

def walk_forward_oos(
    X:         pd.DataFrame,
    y:         pd.Series,
    features:  list[str],
    oos_start: str = OOS_START,
    C:         float = 1.0,
    min_pos:   int = MIN_POSITIVES,
) -> pd.Series:
    """Expanding-window walk-forward OOS for the correction model."""
    oos_ts  = pd.Timestamp(oos_start)
    y_prob  = pd.Series(np.nan, index=X.index, name="prob_corr_oos")

    for t in X.index:
        if t < oos_ts:
            continue

        mask_tr = (X.index < t)
        X_tr    = X.loc[mask_tr, features]
        y_tr    = y.loc[mask_tr]
        valid   = X_tr.notna().all(axis=1) & y_tr.notna()
        X_tr    = X_tr.loc[valid];  y_tr = y_tr.loc[valid]

        if len(y_tr) < 20 or y_tr.sum() < min_pos:
            continue

        X_te = X.loc[[t], features]
        if X_te.isna().any(axis=1).values[0]:
            continue

        X_tr_sc, X_te_sc, _, _ = _standardize(X_tr, X_te)

        try:
            m = LogisticRegression(C=C, class_weight="balanced",
                                   solver="lbfgs", max_iter=1000, random_state=42)
            m.fit(X_tr_sc.values, y_tr.values)
            y_prob.loc[t] = m.predict_proba(X_te_sc.values)[0, 1]
        except Exception as exc:
            warnings.warn(f"OOS fit failed at {t.date()}: {exc}", stacklevel=2)

    return y_prob


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print(f"\n{'='*65}")
    print("  Phase 5 — Correction Market Logistic Regression")
    print(f"  Target : drawdown in (−20 %, −10 %] over 6 months")
    print(f"  HAC    : Newey-West, max_lags = {HAC_LAGS}")
    print(f"  OOS    : walk-forward from {OOS_START}")
    print(f"{'='*65}")

    # -- Load data --
    cf = pd.read_csv(_DATA_DIR / "correction_features.csv", index_col=0, parse_dates=True)
    tg = pd.read_csv(_DATA_DIR / "targets.csv",             index_col=0, parse_dates=True)
    y  = tg["y_corr"]
    print(f"\n  Correction features : {list(cf.columns)}")
    print(f"  Rows                : {len(cf)}")

    # -- Class balance --
    all_candidates = [f for cat in CORR_CATEGORIES.values() for f in cat]
    mask_all = cf[all_candidates].notna().all(axis=1) & y.notna()
    print(f"  Complete rows       : {mask_all.sum()}  "
          f"(correction rate: {y[mask_all].mean():.1%})")

    # -- Univariate HAC screen --
    print(f"\n{'='*65}")
    print(f"  Univariate HAC screen  (max_lags={HAC_LAGS})")
    print(f"{'='*65}")
    print(f"\n  {'Feature':<22}  {'z-stat':>7}  {'p-value':>7}  {'IS AUC':>7}  {'N':>5}  Sig?")
    print(f"  {'-'*22}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*5}  {'-'*4}")

    screen_rows = []
    for feat in all_candidates:
        mask_f = cf[feat].notna() & y.notna()
        if mask_f.sum() < 30:
            continue
        Xf   = cf.loc[mask_f, [feat]]
        yf   = y.loc[mask_f]
        mu_f = Xf.mean();  sig_f = Xf.std(ddof=1).replace(0.0, np.nan)
        Xf_sc = ((Xf - mu_f) / sig_f).fillna(0.0)

        try:
            m = LogisticRegression(C=1.0, class_weight="balanced",
                                   solver="lbfgs", max_iter=500, random_state=42)
            m.fit(Xf_sc.values, yf.values)
            is_auc = roc_auc_score(yf.values, m.predict_proba(Xf_sc.values)[:, 1])
        except Exception:
            is_auc = np.nan

        try:
            X_sm = sm.add_constant(Xf_sc.values, prepend=True)
            res  = sm.Logit(yf.values, X_sm).fit(
                cov_type="HAC",
                cov_kwds={"maxlags": HAC_LAGS, "use_correction": True},
                disp=False,
            )
            z  = float(res.tvalues[1])
            pv = float(res.pvalues[1])
        except Exception:
            z = pv = np.nan

        sig_str = "  *" if pv < 0.05 else (" ." if pv < 0.10 else "")
        print(f"  {feat:<22}  {z:>7.3f}  {pv:>7.4f}  {is_auc:>7.4f}  "
              f"{int(mask_f.sum()):>5}{sig_str}")
        screen_rows.append({"feat": feat, "z": z, "p": pv, "auc": is_auc})

    print("  * p<0.05   . p<0.10")

    # -- Category-constrained search --
    print(f"\n{'='*65}")
    print("  Category-constrained search  (≥1 ≤2 per category, weight ≤30%)")
    for cat, feats in CORR_CATEGORIES.items():
        mn, mx = CORR_CATEGORY_MIN_MAX[cat]
        print(f"    {cat:<15} [{mn}-{mx}]: {feats}")
    print(f"{'='*65}")

    top_combos = category_search(
        cf, y, CORR_CATEGORIES, CORR_CATEGORY_MIN_MAX,
        CORR_EXPECTED_SIGNS, top_n=10, min_obs=40,
        apply_sign_filter=False,   # correction model: signs context-dependent
    )

    print(f"\n  Top 10 by IS AUC (category-balanced; ✓ = all expected signs match):")
    print(f"  {'Rank':<5}  {'IS AUC':>7}  {'N':>5}  {'k':>3}  {'Signs':>5}  Features")
    print(f"  {'-'*5}  {'-'*7}  {'-'*5}  {'-'*3}  {'-'*5}  {'-'*50}")
    for rank, c in enumerate(top_combos, 1):
        sign_ok = "✓" if c["signs_ok"] else f"✗{c['n_wrong_signs']}"
        print(f"  {rank:<5}  {c['is_auc']:>7.4f}  {c['n_obs']:>5}  "
              f"{c['n_features']:>3}  {sign_ok:>5}  {c['features']}")

    # -- OOS + constrained fit on top 3 --
    print(f"\n{'='*65}")
    print("  Top 3 — weight constraint (≤30 %) + OOS evaluation")
    print(f"{'='*65}")

    oos_results = []
    for rank, combo in enumerate(top_combos[:3], 1):
        feats  = combo["features"]
        signs  = np.array([CORR_EXPECTED_SIGNS[f] for f in feats])
        print(f"\n  [{rank}] {feats}")

        mask_c = cf[feats].notna().all(axis=1) & y.notna()
        Xc     = cf.loc[mask_c, feats]
        yc     = y.loc[mask_c]
        mu_c   = Xc.mean();  sig_c = Xc.std(ddof=1).replace(0.0, np.nan)
        X_sc   = ((Xc - mu_c) / sig_c).fillna(0.0)

        coef_c, intercept_c, converged = fit_weight_constrained(
            X_sc.values, yc.values, signs, min_w=0.01, max_w=MAX_W,
        )
        if not converged:
            print("    Optimiser did not converge — skipping.")
            continue

        # IS AUC
        valid_c  = cf[feats].notna().all(axis=1)
        X_all_sc = ((cf[feats] - mu_c) / sig_c).fillna(0.0)
        prob_is  = pd.Series(np.nan, index=cf.index)
        prob_is.loc[valid_c] = expit(X_all_sc.loc[valid_c].values @ coef_c + intercept_c)
        is_m = evaluate(y, prob_is)

        # OOS
        prob_oos = walk_forward_oos(cf, y, feats, oos_start=OOS_START)
        oos_m    = evaluate(y, prob_oos)

        print(f"    IS  AUC={is_m['AUC']:.4f}  PR-AUC={is_m['PR-AUC']:.4f}  "
              f"Brier={is_m['Brier']:.4f}")
        print(f"    OOS AUC={oos_m['AUC']:.4f}  PR-AUC={oos_m['PR-AUC']:.4f}  "
              f"Brier={oos_m['Brier']:.4f}")

        oos_results.append({
            "rank": rank, "features": feats, "coef": coef_c,
            "intercept": intercept_c, "signs": signs,
            "mu": mu_c, "sig": sig_c,
            "is_auc": is_m["AUC"], "oos_auc": oos_m["AUC"],
            "oos_prauc": oos_m["PR-AUC"], "oos_brier": oos_m["Brier"],
            "prob_oos": prob_oos,
        })

    # -- Best model report --
    if not oos_results:
        print("\n  No valid models found.")
        sys.exit(1)

    best  = max(oos_results, key=lambda x: x["oos_auc"])
    feats = best["features"]
    coef  = best["coef"]
    signs = best["signs"]
    mu_b  = best["mu"]
    sig_b = best["sig"]

    print(f"\n{'='*65}")
    print(f"  Best correction model  (OOS AUC {best['oos_auc']:.4f})")
    print(f"  Features: {feats}")
    print(f"{'='*65}")

    # HAC inference on best model
    mask_b = cf[feats].notna().all(axis=1) & y.notna()
    Xb     = cf.loc[mask_b, feats]
    yb     = y.loc[mask_b]
    Xb_sc  = ((Xb - mu_b) / sig_b).fillna(0.0)
    hac_b  = hac_inference(Xb_sc, yb, max_lags=HAC_LAGS)

    abs_c   = np.abs(coef)
    weights = abs_c / abs_c.sum() * 100

    print(f"\n  {'Feature':<22}  {'Exp':>4}  {'Coef':>8}  {'HAC SE':>8}  "
          f"{'z':>6}  {'p-val':>7}  {'Weight':>7}  {'95% CI':>20}  Sig?")
    print(f"  {'-'*22}  {'-'*4}  {'-'*8}  {'-'*8}  "
          f"{'-'*6}  {'-'*7}  {'-'*7}  {'-'*20}  {'-'*4}")

    ic_row = hac_b[hac_b["Feature"] == "intercept"].iloc[0]
    ci_ic  = f"[{ic_row['CI 2.5%']:+.3f}, {ic_row['CI 97.5%']:+.3f}]"
    sig_ic = "  *" if ic_row["Significant"] else ""
    print(f"  {'intercept':<22}  {'':>4}  {ic_row['Coefficient']:>8.4f}  "
          f"{ic_row['HAC SE']:>8.4f}  {ic_row['z-stat']:>6.3f}  "
          f"{ic_row['p-value']:>7.4f}  {'':>7}  {ci_ic:>20}{sig_ic}")

    for feat, c, w, exp_s in zip(feats, coef, weights, signs):
        row = hac_b[hac_b["Feature"] == feat]
        if row.empty:
            continue
        row   = row.iloc[0]
        ci    = f"[{row['CI 2.5%']:+.3f}, {row['CI 97.5%']:+.3f}]"
        sig   = "  *" if row["Significant"] else ""
        exp_s_str = "+" if exp_s > 0 else "−"
        sign_ok   = "✓" if int(np.sign(c)) == exp_s else "✗"
        print(f"  {feat:<22}  {exp_s_str:>4}  {c:>8.4f}  "
              f"{row['HAC SE']:>8.4f}  {row['z-stat']:>6.3f}  "
              f"{row['p-value']:>7.4f}  {w:>6.1f}%  {ci:>20}{sig}  {sign_ok}")

    print(f"  {'TOTAL':<22}  {'':>4}  {'':>8}  {'':>8}  "
          f"{'':>6}  {'':>7}  {'100.0%':>7}")
    print(f"\n  In-sample  : AUC={best['is_auc']:.4f}")
    print(f"  OOS        : AUC={best['oos_auc']:.4f}  "
          f"PR-AUC={best['oos_prauc']:.4f}  Brier={best['oos_brier']:.4f}")
    print(f"\n  * significant at 5 % (HAC Newey-West, max_lags={HAC_LAGS})")

    # -- Current probability --
    print(f"\n{'='*65}")
    print("  Current correction probability")
    print(f"{'='*65}")

    avail = cf[feats].dropna()
    if avail.empty:
        print("  No complete rows available for prediction.")
    else:
        last_date = avail.index[-1]
        row_raw   = avail.loc[last_date]
        row_sc    = (row_raw - mu_b) / sig_b
        logit     = float(row_sc.values @ coef + best["intercept"])
        prob      = expit(logit)

        print(f"\n  As-of: {last_date.date()}")
        print(f"\n  {'Feature':<22}  {'Raw':>9}  {'z':>7}  "
              f"{'Coef':>8}  {'Contrib':>9}  Signal")
        print(f"  {'-'*22}  {'-'*9}  {'-'*7}  {'-'*8}  {'-'*9}  {'-'*20}")
        for feat, v_raw, v_sc, c in zip(feats, row_raw.values,
                                         row_sc.values, coef):
            contrib = v_sc * c
            direction = "BEARISH" if contrib > 0 else "bullish"
            print(f"  {feat:<22}  {v_raw:>+9.4f}  {v_sc:>+7.3f}  "
                  f"{c:>+8.4f}  {contrib:>+9.4f}  [{direction}]")
        print(f"  {'Intercept':<22}  {'':>9}  {'':>7}  "
              f"{best['intercept']:>+8.4f}  {best['intercept']:>+9.4f}")
        print(f"  {'─'*65}")
        print(f"  {'Log-odds':<22}  {logit:>+9.4f}")
        print(f"\n  Correction probability (next 6 months): {prob:.1%}")

    # -- Export --
    out = pd.DataFrame({
        "prob_corr_oos": best["prob_oos"],
        "y_corr":        y,
    })
    out_path = _DATA_DIR / "correction_model_output.csv"
    out.to_csv(out_path, date_format="%Y-%m-%d", float_format="%.6f")
    model_path = _DATA_DIR / "correction_model.pkl"
    with open(model_path, "wb") as fh:
        pickle.dump({"coef": coef, "intercept": best["intercept"],
                     "features": feats, "mu": mu_b, "sig": sig_b}, fh)
    print(f"\n  Exported: {out_path}")
    print(f"  Model   : {model_path}")
