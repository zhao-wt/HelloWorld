"""
bear/search_model_a.py — feature search for ENSEMBLE MODEL A (long history).

Model A is the longest-history member of the bear-market ensemble. Constraint:
every feature must be built from raw data that STARTS BEFORE 1920, so the model
can be trained from the 1920s and learn pre-WWII regime dynamics (incl. the 1929
crash and the Great Depression).

Eligible raw series (all begin before 1920):
    SPX, CPI, DGS10            (1871)
    SHILLER_CAPE               (1900)
    BAA, AAA, INDPRO           (1919)

Method
------
1. Group candidate features by economic category.
2. Pick the best univariate (in-sample AUC) representative per category.
3. Exhaustively fit unconstrained logistic over subsets (size 3-5) of those
   representatives; rank by in-sample AUC.
4. Run expanding-window walk-forward OOS on the strongest subsets; rank by OOS
   AUC (the honest metric).

Target: 1{ mdd_12m <= -0.20 }  (>20% drawdown over the next 12 months).
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

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

TRAIN_START = pd.Timestamp("1922-01-31")   # full credit-z suite available
OOS_START   = pd.Timestamp("1950-01-31")   # long walk-forward window (1950-2026)
HORIZON     = 12

# Candidate features by category, with economic prior sign (effect on bear risk)
CATEGORIES: dict[str, dict[str, int]] = {
    "Trend":     {"spx_vs_10ma": -1, "spx_12m_mom": -1},
    "Inflation": {"infl_yoy": +1, "infl_zscore_120m": +1},
    "Valuation": {"cape_z_120m": +1, "cape_20yr_pct": +1},
    "Credit":    {"baa_aaa_spread": +1, "baa_aaa_chg6": +1, "baa_aaa_z24": +1,
                  "baa_aaa_z60": +1, "baa_10y_spread": +1, "baa_10y_z24": +1,
                  "baa_10y_z60": +1, "baa_yield_chg6": +1},
    "Real":      {"indpro_yoy": -1, "indpro_6m_growth": -1},
    "Rates":     {"dgs10_12m_chg": +1},
}

ALL_SIGNS: dict[str, int] = {f: s for cat in CATEGORIES.values() for f, s in cat.items()}
FEAT_CAT:  dict[str, str] = {f: c for c, d in CATEGORIES.items() for f in d}


def _load() -> tuple[pd.DataFrame, pd.Series]:
    feats = pd.read_csv(_DATA_DIR / "bear_features.csv", index_col=0, parse_dates=True)
    tg    = pd.read_csv(_DATA_DIR / "targets.csv", index_col=0, parse_dates=True)
    y = _apply_target_transform(tg["mdd_12m"], "exceeds_20")
    return feats, y


def _insample_auc(feats: pd.DataFrame, y: pd.Series, cols: list[str]) -> float:
    """Unconstrained-logistic in-sample AUC on the TRAIN_START+ sample."""
    mask = y.notna() & (feats.index >= TRAIN_START)
    X_raw = feats.loc[mask, cols]
    y_tr  = y.loc[mask].values.astype(float)
    mu, sig = X_raw.mean(), X_raw.std(ddof=1).replace(0.0, np.nan)
    X_sc = ((X_raw - mu) / sig).fillna(0.0).values
    signs = np.array([ALL_SIGNS[c] for c in cols], dtype=float)
    coef, b = _fit_constrained_core(X_sc, y_tr, signs, 0.0, 1.0, "logistic", True)
    p = _sigmoid(X_sc @ coef + b)
    return float(roc_auc_score(y_tr, p))


def _oos_auc(feats: pd.DataFrame, y: pd.Series, cols: list[str]) -> tuple[float, int]:
    spec = {"features": cols, "signs": {c: ALL_SIGNS[c] for c in cols},
            "min_w": 0.0, "max_w": 1.0, "unconstrained": True,
            "model_type": "logistic", "oos_start": OOS_START,
            "train_start": TRAIN_START, "horizon": HORIZON}
    oos = _walk_forward_oos(feats, y, spec)
    yy = y.reindex(oos.index)
    ok = yy.notna()
    return float(roc_auc_score(yy[ok].values, oos[ok].values)), int(ok.sum())


def main() -> None:
    feats, y = _load()

    # ---- 1. Univariate screen → best representative per category ----
    print("=" * 64)
    print("Univariate in-sample AUC (1922+)")
    print("=" * 64)
    reps: dict[str, str] = {}
    for cat, d in CATEGORIES.items():
        scored = sorted(((c, _insample_auc(feats, y, [c])) for c in d),
                        key=lambda kv: kv[1], reverse=True)
        for c, a in scored:
            print(f"  {cat:10s} {c:20s} AUC={a:.4f}")
        reps[cat] = scored[0][0]
        print(f"    -> representative: {reps[cat]}\n")

    rep_list = list(reps.values())
    print("Category representatives:", rep_list, "\n")

    # ---- 2. Exhaustive subsets (size 3-5) of representatives, in-sample AUC ----
    print("=" * 64)
    print("Subset search (in-sample AUC, one feature per category)")
    print("=" * 64)
    results = []
    for k in (3, 4, 5):
        for combo in combinations(rep_list, k):
            auc = _insample_auc(feats, y, list(combo))
            results.append((list(combo), auc))
    results.sort(key=lambda r: r[1], reverse=True)
    for cols, auc in results[:12]:
        cats = [FEAT_CAT[c] for c in cols]
        print(f"  AUC={auc:.4f}  [{', '.join(cats)}]  {cols}")

    # ---- 3. Walk-forward OOS on top finalists ----
    print("\n" + "=" * 64)
    print("Walk-forward OOS AUC (1950+) on top finalists")
    print("=" * 64)
    finalists = [c for c, _ in results[:8]]
    scored = []
    for cols in finalists:
        oos_auc, n = _oos_auc(feats, y, cols)
        ins = _insample_auc(feats, y, cols)
        scored.append((cols, ins, oos_auc, n))
        print(f"  OOS={oos_auc:.4f} (n={n})  IS={ins:.4f}  "
              f"[{', '.join(FEAT_CAT[c] for c in cols)}]  {cols}")

    scored.sort(key=lambda r: r[2], reverse=True)
    best = scored[0]
    print("\n" + "=" * 64)
    print("BEST MODEL A (by OOS AUC):")
    print(f"  features : {best[0]}")
    print(f"  signs    : {{{', '.join(f'{c!r}: {ALL_SIGNS[c]:+d}' for c in best[0])}}}")
    print(f"  in-sample AUC : {best[1]:.4f}")
    print(f"  OOS AUC       : {best[2]:.4f}  (n={best[3]})")
    print("=" * 64)


if __name__ == "__main__":
    main()
