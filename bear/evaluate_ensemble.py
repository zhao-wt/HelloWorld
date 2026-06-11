"""
bear/evaluate_ensemble.py — Phase E5: evaluation & calibration of the bear
ensemble (and its members), all on honest walk-forward OOS predictions.

Sections
--------
1. Discrimination : OOS AUC and Brier score over the long span and the common
   2005+ window; ensemble vs best single member vs the legacy bearplus model.
2. Threshold skill: confusion matrix / precision / recall / specificity / F1 at
   the Youden-optimal threshold and at fixed operating points.
3. Calibration    : reliability table, Brier decomposition, and the Cox
   calibration regression (intercept = calibration-in-the-large, slope).
4. Recalibration  : walk-forward Platt scaling of the equal-weight ensemble,
   to see whether averaging probabilities leaves the output mis-calibrated.
5. Event check    : the ensemble's signal around the known post-1950 bear onsets.

No look-ahead: every probability scored here is a member's expanding-window OOS
prediction; the recalibration map at month t is fit only on data before t.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bear.ensemble import (
    MEMBERS, COMMON_START, member_oos, ensemble_equal, _target,
    _logit, platt_walkforward,
)
from bear.inference import _oos_path


def _align(p: pd.Series, y: pd.Series):
    idx = p.dropna().index.intersection(y.dropna().index)
    return p.loc[idx].values, y.loc[idx].values.astype(float)


def _auc(p, y):
    from sklearn.metrics import roc_auc_score
    pa, ya = _align(p, y)
    return float(roc_auc_score(ya, pa)) if len(set(ya)) > 1 else float("nan")


def _brier(p, y):
    pa, ya = _align(p, y)
    return float(np.mean((pa - ya) ** 2))


def _cox_calibration(p, y):
    """Fit logit(y) ~ a + b*logit(p). Well-calibrated: a=0, b=1."""
    from sklearn.linear_model import LogisticRegression
    pa, ya = _align(p, y)
    X = _logit(pa).reshape(-1, 1)
    m = LogisticRegression(C=1e6, max_iter=1000).fit(X, ya)
    return float(m.intercept_[0]), float(m.coef_[0, 0])


def _reliability(p, y, edges=None):
    pa, ya = _align(p, y)
    if edges is None:
        edges = np.array([0, .05, .10, .15, .20, .30, .50, 1.0])
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (pa >= lo) & (pa < hi if hi < 1.0 else pa <= hi)
        if m.sum() == 0:
            continue
        rows.append((f"[{lo:.2f},{hi:.2f})", int(m.sum()),
                     float(pa[m].mean()), float(ya[m].mean())))
    return pd.DataFrame(rows, columns=["bin", "n", "mean_pred", "obs_freq"])


def _threshold_table(p, y, thr):
    pa, ya = _align(p, y)
    yh = (pa >= thr).astype(int)
    tp = int(((yh == 1) & (ya == 1)).sum()); fp = int(((yh == 1) & (ya == 0)).sum())
    fn = int(((yh == 0) & (ya == 1)).sum()); tn = int(((yh == 0) & (ya == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if prec and rec and not np.isnan(prec) and not np.isnan(rec) else float("nan")
    return dict(thr=thr, TP=tp, FP=fp, FN=fn, TN=tn, precision=prec,
                recall=rec, specificity=spec, F1=f1)


def _youden_threshold(p, y):
    from sklearn.metrics import roc_curve
    pa, ya = _align(p, y)
    fpr, tpr, thr = roc_curve(ya, pa)
    j = tpr - fpr
    return float(thr[int(np.argmax(j))])


BEAR_ONSETS = ["1969-01", "1973-01", "1980-01", "1987-08", "2000-09",
               "2007-10", "2020-02", "2022-01"]


def main():
    y = _target()
    oos = member_oos()
    ens = ensemble_equal(oos)

    print("=" * 74)
    print("1. DISCRIMINATION (walk-forward OOS)")
    print("=" * 74)
    base_full = float(y.reindex(ens.dropna().index).dropna().mean())
    print(f"  realized base rate (ensemble span): {base_full:.3f}")
    print(f"  {'series':24s}{'AUC(2005+)':>12s}{'Brier(2005+)':>14s}{'AUC(full)':>12s}")
    ens_c = ens.loc[ens.index >= COMMON_START]
    for name, s in [("ensemble (equal-wt)", ens),
                    ("  member modeld", oos["modeld"]),
                    ("  member modelb", oos["modelb"])]:
        sc = s.loc[s.index >= COMMON_START]
        print(f"  {name:24s}{_auc(sc,y):>12.4f}{_brier(sc,y):>14.4f}{_auc(s,y):>12.4f}")
    try:
        bp = pd.read_csv(_oos_path("bearplus"), index_col=0, parse_dates=True).iloc[:, 0]
        bpc = bp.loc[bp.index >= COMMON_START]
        print(f"  {'  legacy bearplus':24s}{_auc(bpc,y):>12.4f}{_brier(bpc,y):>14.4f}{_auc(bp,y):>12.4f}")
    except Exception:
        pass

    print("\n" + "=" * 74)
    print("2. THRESHOLD SKILL — ensemble, common window 2005+")
    print("=" * 74)
    thr_y = _youden_threshold(ens_c, y)
    for label, thr in [(f"Youden-opt ({thr_y:.3f})", thr_y),
                       ("fixed 0.20", 0.20), ("fixed 0.30", 0.30)]:
        r = _threshold_table(ens_c, y, thr)
        print(f"  {label:22s} TP={r['TP']:>2} FP={r['FP']:>2} FN={r['FN']:>2} TN={r['TN']:>3} "
              f"| precision={r['precision']:.2f} recall={r['recall']:.2f} "
              f"spec={r['specificity']:.2f} F1={r['F1']:.2f}")

    print("\n" + "=" * 74)
    print("3. CALIBRATION — raw equal-weight ensemble")
    print("=" * 74)
    for label, s in [("common 2005+", ens_c), ("full span", ens)]:
        a, b = _cox_calibration(s, y)
        print(f"  {label:14s} Brier={_brier(s,y):.4f}  "
              f"Cox intercept={a:+.3f} slope={b:.3f}  "
              f"(ideal 0 / 1; slope>1 => under-confident)")
    print("\n  Reliability table (full OOS span):")
    rel = _reliability(ens, y)
    print(rel.to_string(index=False,
          formatters={"mean_pred": "{:.3f}".format, "obs_freq": "{:.3f}".format}))

    print("\n" + "=" * 74)
    print("4. RECALIBRATION — walk-forward Platt scaling of the ensemble")
    print("=" * 74)
    recal = platt_walkforward(ens, y, COMMON_START)
    rc = recal.loc[recal.index >= COMMON_START]
    a0, b0 = _cox_calibration(ens_c.reindex(rc.index), y)
    a1, b1 = _cox_calibration(rc, y)
    print(f"  Brier  raw={_brier(ens_c.reindex(rc.index),y):.4f}  ->  recal={_brier(rc,y):.4f}")
    print(f"  AUC    raw={_auc(ens_c.reindex(rc.index),y):.4f}  ->  recal={_auc(rc,y):.4f} (rank-preserving)")
    print(f"  Cox    raw=({a0:+.2f},{b0:.2f})  ->  recal=({a1:+.2f},{b1:.2f})")

    print("\n" + "=" * 74)
    print("5. EVENT CHECK — ensemble OOS prob in the 6 months BEFORE each bear onset")
    print("=" * 74)
    for onset in BEAR_ONSETS:
        t = pd.Timestamp(onset) + pd.offsets.MonthEnd(0)
        win = ens.loc[(ens.index <= t) & (ens.index > t - pd.DateOffset(months=6))]
        if len(win) == 0:
            print(f"  {onset}: (no ensemble OOS yet)")
            continue
        print(f"  {onset}: max P={win.max():.3f}  at onset P={ens.reindex([t]).iloc[0]:.3f}"
              if not np.isnan(ens.reindex([t]).iloc[0]) else f"  {onset}: max P={win.max():.3f}")


if __name__ == "__main__":
    main()
