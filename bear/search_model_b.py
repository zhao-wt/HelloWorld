"""
bear/search_model_b.py — feature search for ENSEMBLE MODEL B (post-war, 1940s).

Model B is the second member of the bear-market ensemble. It trains from 1950,
so in addition to Model A's pre-1920 series it may use post-war data:
    UNRATE (1948)  -> labor signals
    TB3MS  (1920)  -> a real 10y-3m term spread (DGS10 - TB3MS)

Eligible raw series (all begin by the 1940s):
    SPX, CPI, DGS10            (1871)
    SHILLER_CAPE               (1900)
    BAA, AAA, INDPRO, TB3MS    (1919-1920)
    UNRATE                     (1948)

Method mirrors search_model_a: best univariate representative per category,
then exhaustive subset fit (unconstrained logistic) ranked by in-sample AUC,
then walk-forward OOS AUC on the finalists.

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
    _sigmoid,
    _walk_forward_oos,
)

TRAIN_START = pd.Timestamp("1950-01-31")   # UNRATE-based labor available from 1949
OOS_START   = pd.Timestamp("1970-01-31")   # 20y initial training; OOS 1970-2026
HORIZON     = 12

CATEGORIES: dict[str, dict[str, int]] = {
    "Trend":     {"spx_vs_10ma": -1, "spx_12m_mom": -1},
    "Inflation": {"infl_yoy": +1, "infl_zscore_120m": +1},
    "Valuation": {"cape_z_120m": +1, "cape_20yr_pct": +1},
    "Credit":    {"baa_aaa_spread": +1, "baa_aaa_chg6": +1, "baa_aaa_z24": +1,
                  "baa_aaa_z60": +1, "baa_10y_spread": +1, "baa_10y_z24": +1,
                  "baa_10y_z60": +1, "baa_yield_chg6": +1},
    "Real":      {"indpro_yoy": -1, "indpro_6m_growth": -1},
    "Term":      {"ts_10y3m_level": -1, "ts_10y3m_inv_dummy": +1, "dgs10_12m_chg": +1},
    "Labor":     {"unrate_12m_chg": +1, "unrate_sahm": +1},
}

ALL_SIGNS = {f: s for cat in CATEGORIES.values() for f, s in cat.items()}
FEAT_CAT  = {f: c for c, d in CATEGORIES.items() for f in d}


def _load():
    feats = pd.read_csv(_DATA_DIR / "bear_features.csv", index_col=0, parse_dates=True)
    tg    = pd.read_csv(_DATA_DIR / "targets.csv", index_col=0, parse_dates=True)
    y = _apply_target_transform(tg["mdd_12m"], "exceeds_20")
    return feats, y


def _insample_auc(feats, y, cols):
    mask = y.notna() & (feats.index >= TRAIN_START)
    X_raw = feats.loc[mask, cols]
    y_tr  = y.loc[mask].values.astype(float)
    mu, sig = X_raw.mean(), X_raw.std(ddof=1).replace(0.0, np.nan)
    X_sc = ((X_raw - mu) / sig).fillna(0.0).values
    signs = np.array([ALL_SIGNS[c] for c in cols], dtype=float)
    coef, b = _fit_constrained_core(X_sc, y_tr, signs, 0.0, 1.0, "logistic", True)
    return float(roc_auc_score(y_tr, _sigmoid(X_sc @ coef + b)))


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

    print("=" * 64)
    print("Univariate in-sample AUC (1950+)")
    print("=" * 64)
    reps = {}
    for cat, d in CATEGORIES.items():
        scored = sorted(((c, _insample_auc(feats, y, [c])) for c in d),
                        key=lambda kv: kv[1], reverse=True)
        for c, a in scored:
            print(f"  {cat:10s} {c:20s} AUC={a:.4f}")
        reps[cat] = scored[0][0]
        print(f"    -> representative: {reps[cat]}\n")

    rep_list = list(reps.values())
    print("Category representatives:", rep_list, "\n")

    print("=" * 64)
    print("Subset search (in-sample AUC, one feature per category)")
    print("=" * 64)
    results = []
    for k in (3, 4, 5, 6):
        for combo in combinations(rep_list, k):
            results.append((list(combo), _insample_auc(feats, y, list(combo))))
    results.sort(key=lambda r: r[1], reverse=True)
    for cols, auc in results[:15]:
        print(f"  AUC={auc:.4f}  [{', '.join(FEAT_CAT[c] for c in cols)}]  {cols}")

    print("\n" + "=" * 64)
    print("Walk-forward OOS AUC (1970+) on top finalists")
    print("=" * 64)
    finalists = [c for c, _ in results[:10]]
    scored = []
    for cols in finalists:
        oa, n = _oos_auc(feats, y, cols)
        ins = _insample_auc(feats, y, cols)
        scored.append((cols, ins, oa, n))
        print(f"  OOS={oa:.4f} (n={n})  IS={ins:.4f}  "
              f"[{', '.join(FEAT_CAT[c] for c in cols)}]  {cols}")

    scored.sort(key=lambda r: r[2], reverse=True)
    best = scored[0]
    print("\n" + "=" * 64)
    print("BEST MODEL B (by OOS AUC):")
    print(f"  features : {best[0]}")
    print(f"  signs    : {{{', '.join(f'{c!r}: {ALL_SIGNS[c]:+d}' for c in best[0])}}}")
    print(f"  in-sample AUC : {best[1]:.4f}")
    print(f"  OOS AUC       : {best[2]:.4f}  (n={best[3]})")
    print("=" * 64)


if __name__ == "__main__":
    main()
