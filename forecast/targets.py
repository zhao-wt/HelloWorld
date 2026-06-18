"""
forecast/targets.py — forward total-return targets.

Builds the dependent variables for the Market Forecast models: the forward
total price return of the S&P 500 over the next h months, for h in {1,3,6,12}.

    ret_h(t) = P_{t+h} / P_t - 1

where P is the month-end chained S&P 500 price (Shiller composite 1871-1926,
Yahoo ^GSPC month-end after). The forward window uses only prices strictly
after t, so the last h rows of each column are NaN (no look-ahead) — the same
discipline used for the drawdown targets in bear/targets.py.

We reuse the long-history price chain from bear.targets.load_chained_prices so
the forecast targets are consistent with the drawdown work.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from bear.targets import load_chained_prices

_FORECAST_DIR = Path(__file__).resolve().parent
_DATA_DIR = _FORECAST_DIR.parent / "data"

HORIZONS = [1, 3, 6, 12]


def monthly_price(cache_dir: Path) -> pd.Series:
    """Month-end chained S&P 500 price (Shiller monthly + Yahoo daily→ME)."""
    daily = load_chained_prices(cache_dir)
    px = daily.sort_index().resample("ME").last().dropna()
    px.name = "SPX"
    return px


def compute_forward_return(px: pd.Series, horizon_months: int) -> pd.Series:
    """
    Forward total price return over horizon_months, assigned to month-end t.

    ret(t) = P_{t+h} / P_t - 1, using the month-end price h months after t.
    NaN where the forward price is unavailable (last h rows).
    """
    fwd = px.shift(-horizon_months)
    return (fwd / px - 1.0).rename(f"ret_{horizon_months}m")


def build_targets(px: pd.Series, horizons: list[int] = HORIZONS) -> pd.DataFrame:
    """Forward returns for every horizon, aligned on the monthly price index."""
    cols = {f"ret_{h}m": compute_forward_return(px, h) for h in horizons}
    return pd.DataFrame(cols, index=px.index)


def summarize(targets: pd.DataFrame) -> None:
    print(f"\n{'='*64}\n  Forward total-return targets\n{'='*64}")
    for col in targets.columns:
        s = targets[col].dropna()
        ann = (1 + s.mean()) ** (12 / int(col.split("_")[1].rstrip("m"))) - 1
        print(f"  {col:<8}  n={len(s):>5}  mean={s.mean():+.2%}  "
              f"std={s.std():.2%}  min={s.min():+.1%}  max={s.max():+.1%}  "
              f"(ann. mean {ann:+.1%})")
    print(f"\n  Span: {targets.index[0].date()} → {targets.index[-1].date()}")


if __name__ == "__main__":
    cache_dir = _DATA_DIR / "cache"
    px = monthly_price(cache_dir)
    print(f"Monthly price: {len(px)} obs  "
          f"({px.index[0].date()} → {px.index[-1].date()})")

    targets = build_targets(px)
    summarize(targets)

    # Spot-check a few episodes (forward returns from key dates)
    print(f"\n{'='*64}\n  Spot-check: forward returns at key dates\n{'='*64}")
    for d in ["2007-10-31", "2009-02-28", "2020-02-29", "2020-03-31", "2022-09-30"]:
        ts = pd.Timestamp(d)
        if ts in targets.index:
            row = targets.loc[ts]
            vals = "  ".join(
                f"{c}={row[c]:+.1%}" if pd.notna(row[c]) else f"{c}=NaN"
                for c in targets.columns
            )
            print(f"  {d}:  {vals}")

    out = _DATA_DIR / "forecast_targets.csv"
    targets.to_csv(out, date_format="%Y-%m-%d", float_format="%.6f")
    print(f"\nExported: {out}  ({len(targets)} rows x {len(targets.columns)} cols)")
