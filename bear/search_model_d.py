"""
bear/search_model_d.py — feature search for ENSEMBLE MODEL D (1980s, modern).

Fourth and final member of the bear-market ensemble. Trains from 1986, so it may
use the full modern toolkit on top of Models A/B/C:
    NFCI / ANFCI (1971)   -> financial conditions (Chicago Fed)
    EBP          (1973)   -> excess bond premium (Gilchrist-Zakrajsek)
    T10Y2Y (1976), T10Y3M (1982) -> native term spreads / inversion
    BAA10Y       (1986)   -> default spread
    VIXCLS       (1990)   -> implied volatility (minor early mean-fill)

Because the modern sample is short and the 12-month targets overlap, a plain OOS
ranking yields no fully HAC-significant multi-factor set. So selection is by
SIGNIFICANCE-CONSTRAINED FORWARD SELECTION: start empty, repeatedly add the
factor (one per category) that keeps every coefficient HAC-significant
(Newey-West, lag 12, p<0.05) while maximizing walk-forward OOS AUC.

Target: 1{ mdd_12m <= -0.20 }  (>20% drawdown over the next 12 months).
"""

from __future__ import annotations

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

TRAIN_START = pd.Timestamp("1986-01-31")
OOS_START   = pd.Timestamp("2005-01-31")   # OOS spans GFC, COVID, 2022
HORIZON     = 12

CATEGORIES: dict[str, dict[str, int]] = {
    "Trend":     {"spx_vs_10ma": -1, "spx_12m_mom": -1},
    "Inflation": {"infl_yoy": +1, "infl_zscore_120m": +1},
    "Valuation": {"cape_z_120m": +1, "cape_20yr_pct": +1},
    "Credit":    {"baa_10y_spread": +1, "baa_10y_z60": +1, "baa_aaa_z60": +1,
                  "baa_aaa_spread": +1, "baa_zscore_60m": +1, "baa_level": +1,
                  "ebp_level": +1, "ebp_3m_chg": +1},
    "Real":      {"indpro_yoy": -1, "indpro_6m_growth": -1},
    "Rates":     {"dgs10_12m_chg": +1},
    "Term":      {"ts_10y3m": -1, "ts_inv_dummy": +1, "ts_10y2y": -1,
                  "ts_10y3m_level": -1, "ts_10y3m_inv_dummy": +1},
    "NTFS":      {"ntfs_level": -1, "ntfs_3m_chg": -1},
    "FinCond":   {"nfci_level": +1, "nfci_3m_chg": +1, "anfci_level": +1, "anfci_3m_chg": +1},
    "Vol":       {"vix_level": +1, "vix_zscore_24m": +1},
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


def _hac_pv(feats, y, cols):
    mask = y.notna() & (feats.index >= TRAIN_START)
    X = feats.loc[mask, cols]; yy = y.loc[mask].values.astype(float)
    mu, sig = X.mean(), X.std(ddof=1).replace(0.0, np.nan)
    Xs = ((X - mu) / sig).fillna(0.0).values
    signs = np.array([ALL_SIGNS[c] for c in cols], dtype=float)
    coef, b = _fit_constrained_core(Xs, yy, signs, 0.0, 1.0, "logistic", True)
    Xc = np.column_stack([np.ones(len(yy)), Xs])
    return _hac_pvalues(Xc, yy, np.concatenate([[b], coef]), HORIZON)[1:], coef


def _oos_auc(feats, y, cols):
    spec = {"features": cols, "signs": {c: ALL_SIGNS[c] for c in cols},
            "min_w": 0.0, "max_w": 1.0, "unconstrained": True,
            "model_type": "logistic", "oos_start": OOS_START,
            "train_start": TRAIN_START, "horizon": HORIZON}
    oos = _walk_forward_oos(feats, y, spec)
    yy = y.reindex(oos.index); ok = yy.notna()
    return float(roc_auc_score(yy[ok].values, oos[ok].values))


def main():
    feats, y = _load()

    print("=" * 70)
    print("Univariate HAC significance (1986+), p<0.10 shown")
    print("=" * 70)
    for c in ALL_SIGNS:
        pv = _hac_pv(feats, y, [c])[0][0]
        if pv < 0.10:
            print(f"  {c:20s} ({FEAT_CAT[c]:9s}) p={pv:.4g}")

    print("\n" + "=" * 70)
    print("Significance-constrained forward selection (all HAC p<0.05, max OOS)")
    print("=" * 70)
    chosen: list[str] = []
    used_cats: set[str] = set()
    while True:
        best = None
        for c in ALL_SIGNS:
            if FEAT_CAT[c] in used_cats:
                continue
            cols = chosen + [c]
            pv, _ = _hac_pv(feats, y, cols)
            if (pv < 0.05).all():
                oa = _oos_auc(feats, y, cols)
                if best is None or oa > best[1]:
                    best = (c, oa)
        if best is None:
            break
        chosen.append(best[0]); used_cats.add(FEAT_CAT[best[0]])
        print(f"  + {best[0]:20s} ({FEAT_CAT[best[0]]:9s})  OOS={best[1]:.4f}  set={chosen}")

    print("\n" + "=" * 70)
    print("BEST MODEL D:")
    pv, coef = _hac_pv(feats, y, chosen)
    print(f"  features : {chosen}")
    print(f"  signs    : {{{', '.join(f'{c!r}: {1 if k>0 else -1:+d}' for c, k in zip(chosen, coef))}}}")
    for c, k, p in zip(chosen, coef, pv):
        print(f"    {c:20s} coef={k:+.3f}  HAC p={p:.4g}  [{FEAT_CAT[c]}]")
    print(f"  OOS AUC  : {_oos_auc(feats, y, chosen):.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
