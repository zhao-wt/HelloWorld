"""
bear/targets.py — Phase 3: target construction.

Builds two binary dependent variables from daily SPX prices:

  y_bear_t = 1  if the maximum drawdown from any intra-window peak
               over the next 12 months is  ≤ −20%

  y_corr_t = 1  if the maximum drawdown over the next 6 months
               falls in (−20%, −10%]  (correction only — no bear breach)

Why daily prices?
-----------------
Monthly month-end prices systematically understate intra-month drawdowns.
The 2020 COVID crash reached −34% intra-month but only −12.5% on a
month-end-to-month-end basis — well below the −20% bear threshold.
Using daily prices for the MDD window and assigning the result back to
each month-end date gives accurate, literature-consistent labels.

No look-ahead:
--------------
The target at month t uses only prices strictly after t (i.e. P_{t+1}
through P_{t+h}).  The month-t price itself is not in the forward window.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Long-history price chain (Shiller monthly + Yahoo daily)
# ---------------------------------------------------------------------------

def load_chained_prices(cache_dir: Path) -> pd.Series:
    """
    Build the long-history price series used for drawdown targets:

      * Yahoo ^GSPC DAILY from its start (1927) onward — full daily resolution
        so modern drawdowns are accurate (e.g. COVID -34% intra-month).
      * Shiller composite MONTHLY for 1871-1926, ratio-adjusted at the splice
        so the level is continuous. Pre-1927 drawdowns are monthly-resolution.

    Returns a price Series (mixed monthly-then-daily) suitable for
    compute_forward_mdd.
    """
    from datetime import date
    from bear.data import fetch_spx, fetch_shiller_spx

    yahoo = fetch_spx(date(1900, 1, 1), date.today(), cache_dir)
    try:
        shiller = fetch_shiller_spx(cache_dir)
    except Exception:
        return yahoo

    ym = yahoo.resample("ME").last().dropna()
    if ym.empty:
        return shiller
    splice = ym.index[0]
    factor = 1.0
    if splice in shiller.index and float(shiller.loc[splice]) != 0:
        factor = float(ym.iloc[0]) / float(shiller.loc[splice])
    pre = shiller[shiller.index < splice] * factor      # monthly, pre-1927
    combined = pd.concat([pre, yahoo]).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    combined.name = "SPX"
    return combined


# ---------------------------------------------------------------------------
# Core MDD computation
# ---------------------------------------------------------------------------

def _mdd_in_window(arr: np.ndarray) -> float:
    """
    Maximum drawdown in a price array (the most negative value of
    P_s / running_peak_s − 1 over all s in the array).

    Returns np.nan if the array is empty or contains NaN.
    """
    if len(arr) == 0 or np.isnan(arr).any():
        return np.nan
    running_peak = np.maximum.accumulate(arr)
    drawdowns    = arr / running_peak - 1
    return float(drawdowns.min())


def compute_forward_mdd(
    daily_prices: pd.Series,
    monthly_dates: pd.DatetimeIndex,
    horizon_months: int,
) -> pd.Series:
    """
    For each month-end date t in monthly_dates, compute the forward
    maximum drawdown using daily prices over the next horizon_months.

    MDD(t, h) = min_{t < s ≤ t+h} ( P_s / max_{t < u ≤ s} P_u  −  1 )

    The forward window uses daily prices strictly after t, so the
    month-t price is excluded (no look-ahead).

    Parameters
    ----------
    daily_prices   : daily SPX adjusted-close prices (from fetch_spx).
    monthly_dates  : month-end dates for which to compute the target.
    horizon_months : number of calendar months in the forward window.

    Returns
    -------
    pd.Series indexed by monthly_dates, values in (−1, 0].
    Last horizon_months entries will be NaN (forward window incomplete).
    """
    daily_prices = daily_prices.sort_index().dropna()
    results: dict[pd.Timestamp, float] = {}

    for t in monthly_dates:
        end = t + pd.DateOffset(months=horizon_months)
        mask = (daily_prices.index > t) & (daily_prices.index <= end)
        window = daily_prices.loc[mask]
        results[t] = _mdd_in_window(window.values)

    return pd.Series(results, name=f"mdd_{horizon_months}m").sort_index()


# ---------------------------------------------------------------------------
# Binary target builder
# ---------------------------------------------------------------------------

def build_targets(
    daily_prices: pd.Series,
    monthly_dates: pd.DatetimeIndex,
    bear_horizon: int   = 12,
    corr_horizon: int   = 6,
    bear_threshold: float = -0.20,
    corr_lo: float        = -0.20,
    corr_hi: float        = -0.10,
) -> pd.DataFrame:
    """
    Build bear and correction binary targets, plus their underlying MDDs.

    Parameters
    ----------
    daily_prices   : daily SPX prices (from Phase 1 fetch_spx cache).
    monthly_dates  : month-end dates at which to assign labels.
    bear_horizon   : forward window in months for bear target (default 12).
    corr_horizon   : forward window in months for correction target (default 6).
    bear_threshold : MDD threshold for bear label (default −0.20 = −20%).
    corr_lo / hi   : MDD band for correction label (default −20% to −10%).

    Returns
    -------
    pd.DataFrame with columns:
        mdd_12m   forward max drawdown over bear_horizon months (raw value)
        mdd_6m    forward max drawdown over corr_horizon months (raw value)
        y_bear    1 if mdd_12m <= bear_threshold, 0 otherwise, NaN if unknown
        y_corr    1 if corr_lo < mdd_6m <= corr_hi, 0 otherwise, NaN if unknown
    """
    print(f"  Computing MDD ({bear_horizon}m horizon) for {len(monthly_dates)} months...")
    mdd_bear = compute_forward_mdd(daily_prices, monthly_dates, bear_horizon)

    print(f"  Computing MDD ({corr_horizon}m horizon) for {len(monthly_dates)} months...")
    mdd_corr = compute_forward_mdd(daily_prices, monthly_dates, corr_horizon)

    y_bear = (mdd_bear <= bear_threshold).astype(float).where(mdd_bear.notna())
    y_corr = (
        (mdd_corr > corr_lo) & (mdd_corr <= corr_hi)
    ).astype(float).where(mdd_corr.notna())

    return pd.DataFrame({
        "mdd_12m": mdd_bear,
        "mdd_6m":  mdd_corr,
        "y_bear":  y_bear,
        "y_corr":  y_corr,
    }, index=monthly_dates)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def summarize_targets(targets: pd.DataFrame) -> None:
    """Print class balance and historical episode table."""
    for col, label in [("y_bear", "BEAR (>20% drawdown, 12m)"),
                       ("y_corr", "CORRECTION (10-20%, 6m)")]:
        s = targets[col].dropna()
        n_pos  = int(s.sum())
        n_neg  = int((s == 0).sum())
        n_tot  = len(s)
        rate   = n_pos / n_tot * 100
        print(f"\n  {label}")
        print(f"    Positive  : {n_pos:>5}  ({rate:.1f}%)")
        print(f"    Negative  : {n_neg:>5}  ({100-rate:.1f}%)")
        print(f"    Total obs : {n_tot:>5}  "
              f"({s.index[0].date()} to {s.index[-1].date()})")

        # Identify distinct episodes (consecutive runs of 1s)
        episodes = _find_episodes(s)
        if episodes:
            print(f"    Distinct episodes: {len(episodes)}")
            for start, end, n_months in episodes:
                mdd_col = "mdd_12m" if "bear" in col else "mdd_6m"
                worst   = targets.loc[start:end, mdd_col].min()
                print(f"      {start.strftime('%Y-%m')} to {end.strftime('%Y-%m')} "
                      f"({n_months}m)  worst MDD={worst:.1%}")


def _find_episodes(s: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp, int]]:
    """Return list of (start, end, n_months) for consecutive runs of 1s."""
    episodes = []
    in_ep    = False
    ep_start: Optional[pd.Timestamp] = None

    for date, val in s.items():
        if val == 1.0 and not in_ep:
            in_ep    = True
            ep_start = date
        elif val != 1.0 and in_ep:
            n = len(s.loc[ep_start:date]) - 1
            episodes.append((ep_start, s.loc[:date].index[-2], n))
            in_ep = False

    if in_ep and ep_start is not None:
        n = len(s.loc[ep_start:])
        episodes.append((ep_start, s.index[-1], n))

    return episodes


def non_overlapping_sample(
    targets: pd.DataFrame,
    step: int = 12,
) -> pd.DataFrame:
    """
    Return every step-th row to create a non-overlapping subsample.

    For the 12-month bear target, consecutive monthly observations share
    11/12 of their forward window — they are far from independent.
    Subsetting every 12th month gives approximately independent labels.
    Useful for robustness checks in Phase 6 (validation.py).
    """
    idx = targets.dropna(subset=["y_bear", "y_corr"], how="all").index
    keep = idx[::step]
    return targets.loc[keep]


# ---------------------------------------------------------------------------
# Entry point  (python -m bear.targets)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    _bear_dir = Path(__file__).resolve().parent
    _data_dir = _bear_dir.parent / "data"
    cache_dir  = _data_dir / "cache"

    # -- Long-history chained prices: Shiller monthly (1871+) + Yahoo daily --
    spx_daily = load_chained_prices(cache_dir)
    print(f"Chained prices: {len(spx_daily)} obs  "
          f"({spx_daily.index[0].date()} to {spx_daily.index[-1].date()})  "
          f"[Shiller monthly pre-1927, Yahoo daily after]")

    # -- Load monthly dates from Phase 1 output --
    raw_path = _data_dir / "raw_monthly.csv"
    if not raw_path.exists():
        print(f"Raw panel not found at {raw_path}. Run 'python -m bear.data' first.")
        sys.exit(1)

    raw = pd.read_csv(raw_path, index_col=0, parse_dates=True)
    monthly_dates = raw.index

    print(f"Monthly dates  : {len(monthly_dates)} months  "
          f"({monthly_dates[0].date()} to {monthly_dates[-1].date()})")

    # -- Build targets --
    print(f"\n{'='*60}")
    print("  Building targets")
    print(f"{'='*60}")
    targets = build_targets(spx_daily, monthly_dates)

    # -- Summaries --
    print(f"\n{'='*60}")
    print("  Class balance")
    print(f"{'='*60}")
    summarize_targets(targets)

    # -- Non-overlapping subsample check --
    nol = non_overlapping_sample(targets, step=12)
    print(f"\n{'='*60}")
    print("  Non-overlapping subsample (every 12th month)")
    print(f"{'='*60}")
    for col, label in [("y_bear", "Bear"), ("y_corr", "Correction")]:
        s = nol[col].dropna()
        n_pos = int(s.sum())
        print(f"  {label:12s}: {n_pos}/{len(s)} positive  ({n_pos/len(s)*100:.1f}%)")

    # -- Sanity check: key episodes --
    print(f"\n{'='*60}")
    print("  Spot-check: MDD and labels at key dates")
    print(f"{'='*60}")
    check_dates = [
        "1929-09-30", "1937-02-28", "1973-10-31", "1980-02-29",
        "1987-08-31", "1990-06-30", "2000-08-31", "2007-10-31",
        "2020-01-31", "2021-10-31", "2022-01-31",
    ]
    rows = []
    for d in check_dates:
        try:
            ts = pd.Timestamp(d)
            if ts not in targets.index:
                continue
            row = targets.loc[ts]
            rows.append({
                "Date":    d,
                "MDD-12m": f"{row['mdd_12m']:.1%}" if pd.notna(row["mdd_12m"]) else "NaN",
                "MDD-6m":  f"{row['mdd_6m']:.1%}"  if pd.notna(row["mdd_6m"])  else "NaN",
                "y_bear":  int(row["y_bear"]) if pd.notna(row["y_bear"]) else "-",
                "y_corr":  int(row["y_corr"]) if pd.notna(row["y_corr"]) else "-",
            })
        except Exception:
            pass
    print(pd.DataFrame(rows).to_string(index=False))

    # -- Export --
    out_path = _data_dir / "targets.csv"
    targets.to_csv(out_path, date_format="%Y-%m-%d", float_format="%.6f")
    print(f"\nExported: {out_path}  ({len(targets)} rows x {len(targets.columns)} cols)")
