"""
bear/search_model_c.py — feature search for ENSEMBLE MODEL C (1960s).

Third member of the bear-market ensemble. Trains from 1962, so beyond Models A/B
it may use the 1960s-era families:
    NTFS   (1961)  -> near-term forward spread (Engstrom-Sharpe)
    SAHMREALTIME (1959), ICSA (1968) -> real-time labor
    USALOLITOAASTSAM (1955) -> OECD leading indicator
    DFF    (1954)  -> fed funds policy

Method mirrors search_model_a/b: best univariate representative per category,
exhaustive subset fit (unconstrained logistic) ranked by in-sample AUC, then
walk-forward OOS AUC AND HAC max p-value on the finalists, so the committed
model can satisfy the hard rule that every coefficient is HAC-significant.

Target: 1{ mdd_12m <= -0.20 }  (>20% drawdown over the next 12 months).
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from bear.inference import (
    _DATA_DIR,
    _apply_target_transform,
    _fit_constrained_core,
    _hac_pvalues,
    _sigmoid,
    _walk_forward_oos,
)

TRAIN_START = pd.Timestamp("1962-01-31")
OOS_START   = pd.Timestamp("1985-01-31")   # ~23y initial training; OOS 1985-2026
HORIZON     = 12

CATEGORIES: dict[str, dict[str, int]] = {
    "Trend":     {"spx_vs_10ma": -1, "spx_12m_mom": -1},
    "Inflation": {"infl_yoy": +1, "infl_zscore_120m": +1},
    "Valuation": {"cape_z_120m": +1, "cape_20yr_pct": +1},
    "Credit":    {"baa_aaa_spread": +1, "baa_aaa_z24": +1, "baa_aaa_z60": +1,
                  "baa_10y_spread": +1, "baa_10y_z24": +1, "baa_10y_z60": +1,
                  "baa_yield_chg6": +1},
    "Real":      {"indpro_yoy": -1, "indpro_6m_growth": -1},
    "Rates":     {"dgs10_12m_chg": +1},
    "Term":      {"ts_10y3m_level": -1, "ts_10y3m_inv_dummy": +1},
    "NTFS":      {"ntfs_level": -1, "ntfs_3m_chg": -1},
    "Policy":    {"ffr_6m_chg": -1},
    "Leading":   {"lei_6m_growth": -1, "lei_stress_dummy": +1},
    "Labor":     {"sahm_level": +1, "sahm_trigger": +1, "unrate_sahm": +1,
                  "unrate_12m_chg": +1, "icsa_yoy_pct": +1},
}

ALL_SIGNS = {f: s for cat in CATEGORIES.values() for f, s in cat.items()}
FEAT_CAT  = {f: c for c, d in CATEGORIES.items() for f in d}


def _load():
    feats = pd.read_csv(_DATA_DIR / "bear_features.csv", index_col=0, parse_dates=True)
    tg    = pd.read_csv(_DATA_DIR / "targets.csv", index_col=0, parse_dates=True)
    y = _apply_target_transform(tg["mdd_12m"], "exceeds_20")
    return feats, y


def _fit(feats, y, cols):
    mask = y.notna() & (feats.index >= TRAIN_START)
    X = feats.loc[mask, cols]; yy = y.loc[mask].values.astype(float)
    mu, sig = X.mean(), X.std(ddof=1).replace(0.0, np.nan)
    Xs = ((X - mu) / sig).fillna(0.0).values
    signs = np.array([ALL_SIGNS[c] for c in cols], dtype=float)
    coef, b = _fit_constrained_core(Xs, yy, signs, 0.0, 1.0, "logistic", True)
    return Xs, yy, coef, b


def _insample_auc(feats, y, cols):
    Xs, yy, coef, b = _fit(feats, y, cols)
    return float(roc_auc_score(yy, _sigmoid(Xs @ coef + b)))


def _hac_maxp(feats, y, cols):
    Xs, yy, coef, b = _fit(feats, y, cols)
    Xc = np.column_stack([np.ones(len(yy)), Xs])
    pv = _hac_pvalues(Xc, yy, np.concatenate([[b], coef]), HORIZON)[1:]
    return float(pv.max()), pv


def _oos_auc(feats, y, cols):
    spec = {"features": cols, "signs": {c: ALL_SIGNS[c] for c in cols},
            "min_w": 0.0, "max_w": 1.0, "unconstrained": True,
            "model_type": "logistic", "oos_start": OOS_START,
            "train_start": TRAIN_START, "horizon": HORIZON}
    oos = _walk_forward_oos(feats, y, spec)
    yy = y.reindex(oos.index); ok = yy.notna()
    return float(roc_auc_score(yy[ok].values, oos[ok].values)), int(ok.sum())


def main():
    feats, y = _load()

    print("=" * 70)
    print("Univariate in-sample AUC (1962+) — best representative per category")
    print("=" * 70)
    reps = {}
    for cat, d in CATEGORIES.items():
        scored = sorted(((c, _insample_auc(feats, y, [c])) for c in d),
                        key=lambda kv: kv[1], reverse=True)
        reps[cat] = scored[0][0]
        print(f"  {cat:10s} -> {reps[cat]:20s} (AUC={scored[0][1]:.4f})   "
              f"others: {', '.join(f'{c}={a:.3f}' for c, a in scored[1:])}")
    rep_list = list(reps.values())
    print("\nRepresentatives:", rep_list, "\n")

    print("=" * 70)
    print("Subset search (in-sample AUC, one feature per category)")
    print("=" * 70)
    results = []
    for k in (3, 4, 5, 6):
        for combo in combinations(rep_list, k):
            results.append((list(combo), _insample_auc(feats, y, list(combo))))
    results.sort(key=lambda r: r[1], reverse=True)
    for cols, auc in results[:12]:
        print(f"  IS_AUC={auc:.4f}  [{', '.join(FEAT_CAT[c] for c in cols)}]")

    print("\n" + "=" * 70)
    print("Finalists: OOS AUC (1985+) + HAC significance (all p<0.05?)")
    print("=" * 70)
    finalists = [c for c, _ in results[:14]]
    scored = []
    for cols in finalists:
        oa, n = _oos_auc(feats, y, cols)
        maxp, pv = _hac_maxp(feats, y, cols)
        allsig = maxp < 0.05
        scored.append((cols, _insample_auc(feats, y, cols), oa, maxp, allsig))
        tag = "ALL SIG" if allsig else f"maxp={maxp:.3f}"
        print(f"  OOS={oa:.4f} IS={scored[-1][1]:.4f}  {tag:14s} "
              f"[{', '.join(FEAT_CAT[c] for c in cols)}]  {cols}")

    sig_models = [s for s in scored if s[4]]
    pool = sig_models if sig_models else scored
    pool.sort(key=lambda r: r[2], reverse=True)
    best = pool[0]
    print("\n" + "=" * 70)
    print("BEST MODEL C (max OOS AUC among all-HAC-significant sets):")
    print(f"  features : {best[0]}")
    print(f"  signs    : {{{', '.join(f'{c!r}: {ALL_SIGNS[c]:+d}' for c in best[0])}}}")
    print(f"  in-sample AUC : {best[1]:.4f}")
    print(f"  OOS AUC       : {best[2]:.4f}")
    print(f"  HAC max p     : {best[3]:.4g}  (all significant: {best[4]})")
    print("=" * 70)


if __name__ == "__main__":
    main()
