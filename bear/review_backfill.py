"""
bear/review_backfill.py — quality review of the cross-section backfill.

The backfill imputes 1960-1975 values that have no actuals to check against,
so we validate by PSEUDO-HOLDOUT: for each backfilled factor, hold out its
EARLIEST real data (the regime closest to what we actually extrapolate into),
fit the anchor regression on the LATER data only, predict the held-out early
window, and compare to the truth. We also compare against the naive mean-fill
baseline.

Run:  python -m bear.review_backfill
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bear.build_bear_training import ANCHORS

_BEAR_DIR = Path(__file__).resolve().parent
_DATA_DIR = _BEAR_DIR.parent / "data"
_DOCS_DIR = _BEAR_DIR / "docs"

BACKFILLED = ["ebp_level", "ebp_3m_chg", "ntfs_level", "ntfs_3m_chg", "icsa_yoy_pct"]
HOLDOUT_FRAC = 0.30          # earliest 30% of each factor's real history


def _fit_predict(Xtr, ytr, Xte):
    Xc = np.column_stack([np.ones(len(Xtr)), Xtr])
    beta, *_ = np.linalg.lstsq(Xc, ytr, rcond=None)
    return np.column_stack([np.ones(len(Xte)), Xte]) @ beta


def review() -> pd.DataFrame:
    feats = pd.read_csv(_DATA_DIR / "bear_features.csv", index_col=0, parse_dates=True)
    rows = []
    panels = {}
    for f in BACKFILLED:
        df = feats[[f] + ANCHORS].dropna()
        if len(df) < 80:
            continue
        df = df.sort_index()
        k = int(len(df) * HOLDOUT_FRAC)
        hold, fit = df.iloc[:k], df.iloc[k:]      # earliest = holdout

        pred = _fit_predict(fit[ANCHORS].values, fit[f].values, hold[ANCHORS].values)
        actual = hold[f].values

        # metrics
        corr = float(np.corrcoef(actual, pred)[0, 1])
        rmse = float(np.sqrt(np.mean((actual - pred) ** 2)))
        sst  = float(np.sum((actual - actual.mean()) ** 2))
        r2_oos = 1 - float(np.sum((actual - pred) ** 2)) / sst if sst > 0 else np.nan
        # naive mean-fill baseline (predict the fit-period mean)
        mean_pred = np.full_like(actual, fit[f].mean())
        rmse_mean = float(np.sqrt(np.mean((actual - mean_pred) ** 2)))
        skill = 1 - rmse / rmse_mean if rmse_mean > 0 else np.nan   # vs mean-fill
        # sign agreement (relevant for change/level around 0)
        sign_agree = float(np.mean(np.sign(actual) == np.sign(pred)))

        rows.append({
            "factor": f,
            "holdout": f"{hold.index[0].strftime('%Y-%m')}–{hold.index[-1].strftime('%Y-%m')}",
            "n": len(hold),
            "corr": round(corr, 2),
            "R2_oos": round(r2_oos, 2),
            "RMSE": round(rmse, 3),
            "RMSE_mean": round(rmse_mean, 3),
            "skill_vs_mean": round(skill, 2),
            "sign_agree": round(sign_agree, 2),
        })
        panels[f] = (hold.index, actual, pred)

    summary = pd.DataFrame(rows)

    # --- chart: actual vs backfill on the earliest-window holdout ---
    _DOCS_DIR.mkdir(parents=True, exist_ok=True)
    n = len(panels)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.3 * n), squeeze=False)
    for ax, (f, (idx, actual, pred)) in zip(axes[:, 0], panels.items()):
        ax.plot(idx, actual, color="#173f2a", lw=1.6, label="actual")
        ax.plot(idx, pred, color="#b68a35", lw=1.6, ls="--", label="backfill (anchors)")
        ax.set_title(f"{f} — earliest-window holdout (fit on later data)",
                     fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.25); ax.legend(fontsize=8)
    fig.tight_layout()
    out = _DOCS_DIR / "backfill_review.png"
    fig.savefig(out, dpi=120); plt.close(fig)
    return summary


if __name__ == "__main__":
    s = review()
    print("Pseudo-holdout backfill quality (hold out earliest 30%, fit on later):\n")
    print(s.to_string(index=False))
    print("\nGuide: skill_vs_mean > 0 means the regression backfill beats naive "
          "mean-fill; R2_oos > 0 means it explains early-period variance.")
    print(f"Chart: {(_DOCS_DIR / 'backfill_review.png')}")
