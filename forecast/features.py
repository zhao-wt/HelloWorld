"""
forecast/features.py — predictor panel for the Market Forecast models.

Most return predictors already exist in data/all_features.csv (built by
bear/features.py from the lag-adjusted monthly panel): valuation (CAPE z /
percentile), term spread, credit spreads, trend, momentum, inflation,
industrial production, unemployment, financial conditions, volatility.

This module REUSES all_features.csv and adds the few return-specific predictors
from the equity-premium literature (Welch-Goyal 2008; Campbell-Thompson 2008)
that are not already there:

    ep_yield        earnings yield  E/P = 1 / Shiller CAPE          (valuation)
    tbill_level     3-month T-bill rate level (TB3MS)               (rates)
    ltr_36m         36-month trailing price return (long-term       (reversal)
                    reversal — DeBondt-Thaler)
    rvol_12m        trailing 12-month realized vol of monthly       (volatility)
                    returns (annualized)

All new predictors are built from the lag-adjusted raw panel (publication lags
applied via bear.features.apply_publication_lags) so the no-look-ahead
discipline matches the rest of the dashboard.

The curated predictor set used by the models is PREDICTORS (grouped by family),
chosen for long history and literature support. Members standardize on the
training window and mean-fill any not-yet-published factor (0 after
standardization), exactly like bear/inference.py, so short-history predictors
(e.g. NFCI from 1971) do not truncate the early sample.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from bear.features import apply_publication_lags, _trailing_zscore  # noqa: F401

_FORECAST_DIR = Path(__file__).resolve().parent
_DATA_DIR = _FORECAST_DIR.parent / "data"

# Predictor catalogue grouped by family (label, family) for the UI / leaderboard.
# Keys must exist in the merged feature frame (all_features ∪ new columns).
PREDICTOR_INFO: dict[str, tuple[str, str]] = {
    # Valuation
    "cape_z_120m":   ("Shiller CAPE, 10yr z-score",        "Valuation"),
    "cape_20yr_pct": ("Shiller CAPE, 20yr percentile",     "Valuation"),
    "ep_yield":      ("Earnings yield (1 / CAPE)",         "Valuation"),
    # Rates / term structure
    "ts_10y3m_level": ("10y-3m term spread",               "Rates/Term"),
    "dgs10_12m_chg":  ("10y Treasury yield, 12m change",   "Rates/Term"),
    "tbill_level":    ("3-month T-bill rate, level",       "Rates/Term"),
    # Credit
    "baa_aaa_spread": ("BAA-AAA quality spread",           "Credit"),
    "baa_10y_spread": ("BAA - 10y Treasury spread",        "Credit"),
    # Trend / momentum
    "spx_vs_10ma":  ("S&P 500 vs 10-month MA",             "Trend/Momentum"),
    "spx_12m_mom":  ("S&P 500 12-month momentum",          "Trend/Momentum"),
    "ltr_36m":      ("36-month trailing return (reversal)", "Trend/Momentum"),
    # Macro
    "infl_yoy":      ("CPI inflation, YoY %",              "Macro"),
    "indpro_yoy":    ("Industrial production, YoY %",      "Macro"),
    "unrate_12m_chg": ("Unemployment rate, 12m change",    "Macro"),
    # Volatility / conditions
    "rvol_12m":     ("Realized volatility, trailing 12m",  "Volatility/Conditions"),
    "nfci_level":   ("NFCI financial conditions, level",   "Volatility/Conditions"),
}

PREDICTORS: list[str] = list(PREDICTOR_INFO.keys())


def build_extra_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Compute the return-specific predictors not in all_features.csv."""
    p = apply_publication_lags(raw)          # publication lags first (no look-ahead)
    f = pd.DataFrame(index=p.index)

    # Valuation: earnings yield = 1 / CAPE
    if "SHILLER_CAPE" in p.columns:
        cape = p["SHILLER_CAPE"].replace(0.0, np.nan)
        f["ep_yield"] = 1.0 / cape

    # Rates: short-rate level (long-history 3m bill, TB3MS extended to 1920)
    if "TB3MS" in p.columns:
        f["tbill_level"] = p["TB3MS"]

    # Trend: 36-month trailing price return (long-term reversal)
    if "SPX" in p.columns:
        spx = p["SPX"]
        f["ltr_36m"] = spx / spx.shift(36) - 1.0
        # Volatility: trailing 12m realized vol of monthly returns (annualized)
        mret = spx.pct_change()
        f["rvol_12m"] = mret.rolling(12, min_periods=6).std(ddof=1) * np.sqrt(12)

    return f


def build_features() -> pd.DataFrame:
    """Merge all_features.csv with the new return-specific predictors."""
    all_f = pd.read_csv(_DATA_DIR / "all_features.csv", index_col=0, parse_dates=True)
    raw = pd.read_csv(_DATA_DIR / "raw_monthly.csv", index_col=0, parse_dates=True)
    extra = build_extra_features(raw)

    # Use all_features columns that we need + add the new ones.
    have = [c for c in PREDICTORS if c in all_f.columns]
    merged = all_f[have].join(extra, how="outer").sort_index()
    # Keep only the catalogue columns, in catalogue order.
    cols = [c for c in PREDICTORS if c in merged.columns]
    return merged[cols]


def summarize(df: pd.DataFrame) -> None:
    print(f"\n{'='*78}\n  Forecast predictors  ({len(df)} rows x {len(df.columns)} cols)\n{'='*78}")
    print(f"  {'Predictor':<16}  {'Family':<22}  {'First':<11}  {'NaN%':>6}  {'Mean':>10}")
    print(f"  {'-'*16}  {'-'*22}  {'-'*11}  {'-'*6}  {'-'*10}")
    for c in df.columns:
        s = df[c].dropna()
        fam = PREDICTOR_INFO[c][1]
        first = str(s.index[0].date()) if len(s) else "-"
        nanpct = df[c].isna().mean() * 100
        mean = s.mean() if len(s) else float("nan")
        print(f"  {c:<16}  {fam:<22}  {first:<11}  {nanpct:>5.1f}%  {mean:>10.4f}")


if __name__ == "__main__":
    feats = build_features()
    summarize(feats)
    out = _DATA_DIR / "forecast_features.csv"
    feats.to_csv(out, date_format="%Y-%m-%d", float_format="%.6f")
    print(f"\nExported: {out}  ({len(feats)} rows x {len(feats.columns)} cols)")
