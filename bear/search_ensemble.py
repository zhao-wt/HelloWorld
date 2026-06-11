"""
bear/search_ensemble.py — generic era-search for ensemble members.

Significance-constrained forward selection (the method behind the bear ensemble),
parameterized by target and era so it serves both the bear and correction
ensembles. For a given era (train/oos start + a feature pool of series available
by that era), it starts empty and repeatedly adds the factor — one per category —
that keeps EVERY coefficient Newey-West HAC-significant (p<0.05) while maximizing
walk-forward OOS AUC.

Usage:
    python -m bear.search_ensemble bear        # re-derive the bear members
    python -m bear.search_ensemble correction  # derive the correction members
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from bear.inference import (
    _DATA_DIR,
    _apply_target_transform,
    _fit_constrained_core,
    _hac_pvalues,
    _walk_forward_oos,
)
from bear.univariate import FACTORS   # feature -> (label, category)

# Cumulative per-era feature pools (series available by that era's start).
POOL_A = [
    "spx_vs_10ma", "spx_12m_mom", "m12_1_mom",
    "infl_yoy", "infl_zscore_120m", "cape_z_120m", "cape_20yr_pct",
    "baa_aaa_spread", "baa_aaa_chg6", "baa_aaa_z24", "baa_aaa_z60",
    "baa_10y_spread", "baa_10y_z24", "baa_10y_z60", "baa_yield_chg6",
    "indpro_yoy", "indpro_6m_growth", "dgs10_12m_chg",
]
POOL_B = POOL_A + ["ts_10y3m_level", "ts_10y3m_inv_dummy",
                   "unrate_12m_chg", "unrate_sahm"]
POOL_C = POOL_B + ["ntfs_level", "ntfs_3m_chg", "ffr_6m_chg",
                   "lei_6m_growth", "lei_stress_dummy",
                   "sahm_level", "sahm_trigger", "icsa_yoy_pct"]
POOL_D = POOL_C + ["baa_level", "baa_3m_chg", "baa_zscore_60m", "baa_zscore_24m",
                   "ebp_level", "ebp_3m_chg", "nfci_level", "nfci_3m_chg",
                   "anfci_level", "anfci_3m_chg", "vix_level", "vix_zscore_24m",
                   "ts_10y3m", "ts_inv_dummy", "ts_10y2y",
                   "vts_slope", "vts_ratio", "vts_backwardation",
                   "vts_slope_zscore", "spx_below_10ma", "cpce_low_dummy"]

ERAS = {
    "a": {"train": "1920-01-31", "oos": "1950-01-31", "pool": POOL_A},
    "b": {"train": "1950-01-31", "oos": "1970-01-31", "pool": POOL_B},
    "c": {"train": "1962-01-31", "oos": "1985-01-31", "pool": POOL_C},
    "d": {"train": "1986-01-31", "oos": "2005-01-31", "pool": POOL_D},
}

TARGETS = {
    "bear":       {"target_col": "mdd_12m", "transform": "exceeds_20", "horizon": 12},
    "correction": {"target_col": "mdd_6m",  "transform": "exceeds_10", "horizon": 6},
}

FEAT_CAT = {f: FACTORS[f][1] for f in FACTORS}


def _load(target_col, transform):
    feats = pd.read_csv(_DATA_DIR / "all_features.csv", index_col=0, parse_dates=True)
    tg = pd.read_csv(_DATA_DIR / "targets.csv", index_col=0, parse_dates=True)
    y = _apply_target_transform(tg[target_col], transform)
    return feats, y


def _hac_pvals(feats, y, cols, train_start, horizon):
    mask = y.notna() & (feats.index >= train_start)
    X = feats.loc[mask, cols]
    yy = y.loc[mask].values.astype(float)
    mu, sd = X.mean(), X.std(ddof=1).replace(0.0, np.nan)
    Xs = ((X - mu) / sd).fillna(0.0).values
    coef, b = _fit_constrained_core(Xs, yy, np.ones(len(cols)), 0.0, 1.0, "logistic", True)
    Xc = np.column_stack([np.ones(len(yy)), Xs])
    pv = _hac_pvalues(Xc, yy, np.concatenate([[b], coef]), horizon)[1:]
    return pv, coef


def _oos_auc(feats, y, cols, train_start, oos_start, horizon):
    spec = {"features": cols, "signs": {c: 1 for c in cols}, "min_w": 0.0,
            "max_w": 1.0, "unconstrained": True, "model_type": "logistic",
            "oos_start": oos_start, "train_start": train_start, "horizon": horizon}
    o = _walk_forward_oos(feats, y, spec)
    yy = y.reindex(o.index); ok = yy.dropna().index.intersection(o.dropna().index)
    return float(roc_auc_score(yy.loc[ok].values, o.loc[ok].values))


def forward_select(family, era_key):
    cfg = ERAS[era_key]
    spec = TARGETS[family]
    feats, y = _load(spec["target_col"], spec["transform"])
    train_start = pd.Timestamp(cfg["train"]); oos_start = pd.Timestamp(cfg["oos"])
    horizon = spec["horizon"]
    pool = [c for c in cfg["pool"] if c in feats.columns]

    chosen, used_cats = [], set()
    while True:
        best = None
        for c in pool:
            if c in chosen or FEAT_CAT[c] in used_cats:
                continue
            cols = chosen + [c]
            pv, _ = _hac_pvals(feats, y, cols, train_start, horizon)
            if (pv < 0.05).all():
                auc = _oos_auc(feats, y, cols, train_start, oos_start, horizon)
                if best is None or auc > best[1]:
                    best = (c, auc)
        if best is None:
            break
        chosen.append(best[0]); used_cats.add(FEAT_CAT[best[0]])

    pv, coef = _hac_pvals(feats, y, chosen, train_start, horizon)
    oos = _oos_auc(feats, y, chosen, train_start, oos_start, horizon)
    return {"era": era_key, "train": cfg["train"], "oos": cfg["oos"],
            "features": chosen,
            "signs": {c: (1 if k > 0 else -1) for c, k in zip(chosen, coef)},
            "pvals": {c: float(p) for c, p in zip(chosen, pv)},
            "oos_auc": oos}


def main():
    family = sys.argv[1] if len(sys.argv) > 1 else "correction"
    print(f"Significance-constrained forward selection — {family.upper()} ensemble")
    print(f"target={TARGETS[family]['target_col']} ({TARGETS[family]['transform']}), "
          f"horizon={TARGETS[family]['horizon']}\n")
    for era_key in ("a", "b", "c", "d"):
        r = forward_select(family, era_key)
        print("=" * 72)
        print(f"MODEL {era_key.upper()}  (train {r['train'][:4]}+, OOS {r['oos'][:4]}+)  "
              f"OOS AUC={r['oos_auc']:.4f}")
        for c in r["features"]:
            print(f"    {c:20s} sign {r['signs'][c]:+d}  cat {FEAT_CAT[c]:14s} "
                  f"HAC p={r['pvals'][c]:.4f}")
        print(f"  signs dict: {r['signs']}")


if __name__ == "__main__":
    main()
