"""
bear/build_bear_training.py — assemble the LONG-HISTORY bear training set.

Rules (per spec):
  * Keep only bear factors whose real data starts BEFORE 1975. Factors that
    start later (ts_10y3m 1982, T10Y2Y 1976, baa_* 1986) are dropped; the long
    term spread (ts_10y3m_level = DGS10 - TB3MS, 1934) and EBP (1973) replace them.
  * Training sample starts 1960.
  * Missing values before 1975 are backfilled by CROSS-SECTION REGRESSION:
    each gap factor is regressed on the always-available "anchor" factors over
    their overlap, and the fitted relationship imputes the early months.

Output: bear/bear_training_data.csv  (factors + target, no NaN in 1960+).
Stops here — does not train a model.

Run:  python -m bear.build_bear_training
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_BEAR_DIR = Path(__file__).resolve().parent
_DATA_DIR = _BEAR_DIR.parent / "data"

TRAIN_START = pd.Timestamp("1960-01-31")
CUTOFF      = pd.Timestamp("1975-01-01")   # a factor must start before this

# Long-history bear factor pool — option 1: keep only factors that are REAL
# back to ~1960 or that backfill reliably (validated in review_backfill.py).
# Dropped: ebp_level/ebp_3m_chg and icsa_yoy_pct (regression backfill was worse
# than mean-fill on the pseudo-holdout); ntfs_3m_chg (only marginal skill).
FACTORS = [
    "ntfs_level",                              # 1961, real + excellent backfill (R2_oos 0.73)
    "ts_10y3m_level", "ts_10y3m_inv_dummy",    # 1934, real (DGS10 - TB3MS)
    "sahm_level", "sahm_trigger",              # 1960, real
    "lei_6m_growth", "lei_stress_dummy",       # 1955, real
    "ffr_6m_chg",                              # 1954, real
    "cape_20yr_pct",                           # 1910, real (Shiller CAPE percentile)
]

# Continuous factors with full coverage from 1960 — used as imputation anchors.
ANCHORS = ["ts_10y3m_level", "sahm_level", "lei_6m_growth", "ffr_6m_chg"]


def cross_section_backfill(
    df: pd.DataFrame,
    anchors: list[str],
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Backfill missing factor values from the cross-section of anchor factors.

    For each factor with gaps, fit OLS  factor ~ 1 + anchors  over the months
    where the factor and all anchors are observed, then predict the missing
    early months from the anchors. Anchors themselves are never imputed.
    Returns (filled_df, report).
    """
    out = df.copy()
    report = []
    for col in df.columns:
        if col in anchors:
            continue
        obs = out[col].notna()
        n_missing = int((~obs & (out.index >= TRAIN_START)).sum())
        if n_missing == 0:
            continue
        train = obs & out[anchors].notna().all(axis=1)
        if train.sum() < 30:
            report.append({"factor": col, "filled": 0, "note": "insufficient overlap"})
            continue
        X = out.loc[train, anchors].values
        y = out.loc[train, col].values
        Xc = np.column_stack([np.ones(len(X)), X])
        beta, *_ = np.linalg.lstsq(Xc, y, rcond=None)
        # R^2 for the report
        yhat = Xc @ beta
        ss_res = float(((y - yhat) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        miss = (~obs) & out[anchors].notna().all(axis=1) & (out.index >= TRAIN_START)
        if miss.any():
            Xm = np.column_stack([np.ones(int(miss.sum())), out.loc[miss, anchors].values])
            out.loc[miss, col] = Xm @ beta
        report.append({"factor": col, "filled": int(miss.sum()), "r2": round(r2, 3)})
    return out, report


def build() -> Path:
    feats   = pd.read_csv(_DATA_DIR / "bear_features.csv", index_col=0, parse_dates=True)
    targets = pd.read_csv(_DATA_DIR / "targets.csv", index_col=0, parse_dates=True)

    missing = [c for c in FACTORS if c not in feats.columns]
    if missing:
        raise ValueError(f"Missing factor columns in bear_features.csv: {missing}")

    panel = feats[FACTORS].copy()

    # --- enforce the "starts before 1975" rule ---
    print(f"{'Factor':<20}{'First real':>12}{'Keep (<1975)':>14}")
    print("-" * 46)
    keep = []
    for c in FACTORS:
        s = panel[c].dropna()
        first = s.index[0] if len(s) else None
        ok = first is not None and first < CUTOFF
        print(f"{c:<20}{(first.strftime('%Y-%m') if first is not None else 'EMPTY'):>12}"
              f"{('yes' if ok else 'NO — drop'):>14}")
        if ok:
            keep.append(c)
    panel = panel[keep]

    # --- restrict to 1960+ and backfill pre-1975 gaps ---
    panel = panel[panel.index >= TRAIN_START]
    filled, report = cross_section_backfill(panel, [a for a in ANCHORS if a in keep])

    print("\nCross-section backfill (factor ~ anchors OLS):")
    for r in report:
        if "r2" in r:
            print(f"  {r['factor']:<20} filled {r['filled']:>3} months   (R^2={r['r2']})")
        else:
            print(f"  {r['factor']:<20} {r['note']}")

    # --- attach target ---
    mdd = targets["mdd_12m"]
    filled["target_event"] = (mdd <= -0.20).astype(float).where(mdd.notna()).reindex(filled.index)
    filled["mdd_12m"] = mdd.reindex(filled.index)

    # Training rows = target resolved AND every factor present. The only
    # residual gap is the most recent month(s) where a coincident anchor
    # (Sahm real-time) is not yet published — those rows are dropped.
    complete = filled["target_event"].notna() & filled[keep].notna().all(axis=1)
    out = filled[complete].copy()
    out.index.name = "date"

    path = _DATA_DIR / "bear_training_data.csv"
    out.to_csv(path, date_format="%Y-%m-%d", float_format="%.6f")
    return path


if __name__ == "__main__":
    path = build()
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    feat_cols = [c for c in df.columns if c not in ("target_event", "mdd_12m")]
    print(f"\n{'='*60}")
    print(f"  bear_training_data.csv  —  {len(df)} rows x {len(df.columns)} cols")
    print(f"  Range : {df.index[0].date()} -> {df.index[-1].date()}")
    print(f"  Factors ({len(feat_cols)}): {feat_cols}")
    print(f"  Events: {int(df['target_event'].sum())} / {len(df)} "
          f"({df['target_event'].mean():.1%})")
    print(f"  Any NaN in factors? {df[feat_cols].isna().any().any()}")
    print(f"{'='*60}")
    # show the backfilled early window + a recent row
    print("\nEarly (backfilled) rows:")
    print(df[feat_cols].head(3).round(3).to_string())
    print("\nReal-data row (1975):")
    print(df.loc["1975-01":"1975-01", feat_cols].round(3).to_string())
