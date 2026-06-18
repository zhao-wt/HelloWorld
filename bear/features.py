"""
bear/features.py — Phase 2: feature engineering.

Takes the monthly raw panel from Phase 1, applies publication lags,
and computes all model-ready features for both the bear and correction models.

No-look-ahead disciplines
--------------------------
1. Publication lags  : every raw series is shifted forward by its real-world
   publication delay (in months) before any transformation is computed.
2. Rolling windows   : all z-scores and percentiles use strictly trailing
   windows — no centered or full-sample statistics.
3. Differencing      : all diff/pct_change/momentum terms reference only
   past values of the lag-adjusted series.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Publication lags: raw series name -> months to shift forward
# ---------------------------------------------------------------------------
# Conservative translation of day-level lags (data.py) to monthly model lags.
# Data observed in month t is usable no earlier than month t + LAG_MONTHS[s].
#   ceil(1  day  / 30) = 1   (daily market data)
#   ceil(35 days / 30) = 2   (monthly macro, BLS ~5 weeks) -> use 1 (fits in same calendar month+1)
#   ceil(42 days / 30) = 2   (OECD LEI, GZ EBP ~6 weeks)
# For clarity: lag=1 means "month t data is available to use in month t+1 model".

LAG_MONTHS: dict[str, int] = {
    # --- daily market/financial data ---
    "DGS3MO":           1,
    "DGS10":            1,
    "T10Y3M":           1,
    "T10Y2Y":           1,
    "BAMLH0A0HYM2":     1,
    "DFF":              1,
    "NTFS":             1,
    "SPX":              1,
    "SP500":            1,
    "VIXCLS":           1,
    "VXVCLS":           1,
    "CPCE":             1,
    # --- weekly data ---
    "ICSA":             1,
    "ANFCI":            1,
    "NFCI":             1,
    # --- monthly macro (BLS ~35d) ---
    "UNRATE":           1,
    "SAHMREALTIME":     1,
    "BAA10Y":           1,
    "SHILLER_CAPE":     1,
    # --- monthly with ~6-week lag ---
    "EBP":              2,
    "USALOLITOAASTSAM": 2,
}


# ---------------------------------------------------------------------------
# Step 1 — Apply publication lags
# ---------------------------------------------------------------------------

def apply_publication_lags(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Shift each column forward by its real-world publication lag (in months).

    Value observed at month t -> appears at month t + LAG_MONTHS[col].
    All downstream feature computations MUST use the returned DataFrame
    to avoid look-ahead bias.

    Columns not in LAG_MONTHS default to a 1-month lag.
    """
    out: dict[str, pd.Series] = {}
    for col in panel.columns:
        lag = LAG_MONTHS.get(col, 1)
        out[col] = panel[col].shift(lag)
    return pd.DataFrame(out, index=panel.index)


# ---------------------------------------------------------------------------
# Step 2 — Trailing-window helpers (strictly no full-sample leakage)
# ---------------------------------------------------------------------------

def _trailing_zscore(
    s: pd.Series,
    window: int,
    min_periods: Optional[int] = None,
) -> pd.Series:
    """
    Trailing z-score computed on a rolling window.
    Returns NaN until min_periods of data are present.
    All statistics use only data up to and including the current row.
    """
    mp = min_periods if min_periods is not None else window
    mu  = s.rolling(window, min_periods=mp).mean()
    sig = s.rolling(window, min_periods=mp).std(ddof=1)
    return ((s - mu) / sig.replace(0.0, np.nan)).rename(s.name)


def _trailing_percentile(
    s: pd.Series,
    window: int,
    min_periods: Optional[int] = None,
) -> pd.Series:
    """
    Trailing empirical percentile rank in [0, 1].

    At each t: fraction of the (window-1) most-recent past values <= s[t].
    Uses only data up to and including the current row — no look-ahead.
    """
    mp = min_periods if min_periods is not None else max(window // 2, 2)

    def _pct(x: np.ndarray) -> float:
        return float((x[:-1] <= x[-1]).mean()) if len(x) > 1 else float("nan")

    return s.rolling(window, min_periods=mp).apply(_pct, raw=True).rename(s.name)


# ---------------------------------------------------------------------------
# Step 3a — Bear model features
# ---------------------------------------------------------------------------

def build_bear_features(p: pd.DataFrame) -> pd.DataFrame:
    """
    Compute bear model features from a lag-adjusted panel.

    All series in p must already be shifted by their publication lags
    (call apply_publication_lags() first).

    Target: forward max-drawdown <= -20% over 12 months.
    Informative horizon: 6-18 months (slow, persistent macro variables).

    Feature catalogue
    -----------------
    Yield curve  (Estrella-Mishkin 1998; Engstrom-Sharpe 2019)
      ntfs_level        Near-term forward spread level            sign: -
      ntfs_3m_chg       NTFS 3-month change                       sign: -
      ts_10y3m          10y-3m term spread level                  sign: -
      ts_inv_dummy      Inversion dummy (T10Y3M < 0)              sign: +
      ts_10y2y          10y-2y term spread (secondary)            sig

    Credit  (Gilchrist-Zakrajšek 2012; Tokic-Jackson 2023)
      ebp_level         Excess bond premium level                 sign: +
      ebp_3m_chg        EBP 3-month change                        sign: +
      baa_level         BAA-10y default spread level              sign: +
      baa_3m_chg        BAA-10y 3-month change                    sign: +
      baa_zscore_60m    BAA-10y trailing 60-month z-score         sign: +

    Labor  (Sahm 2019)
      sahm_level        Sahm rule continuous reading              sign: +
      sahm_trigger      Dummy: Sahm >= 0.5                        sign: +
      icsa_yoy_pct      Initial claims YoY % change               sign: +

    Leading indicator  (Conference Board / OECD)
      lei_6m_growth     Annualized 6-month LEI growth             sign: -
      lei_stress_dummy  Dummy: lei_6m_growth < -4%                sign: +

    Policy  (Tokic-Jackson 2023)
      ffr_6m_chg        6-month change in fed funds (easing flag) sign: -
    """
    f = pd.DataFrame(index=p.index)

    # ---- Yield curve ----
    if "NTFS" in p.columns:
        f["ntfs_level"]  = p["NTFS"]
        f["ntfs_3m_chg"] = p["NTFS"].diff(3)

    if "T10Y3M" in p.columns:
        f["ts_10y3m"]     = p["T10Y3M"]
        f["ts_inv_dummy"] = (p["T10Y3M"] < 0).astype(float).where(p["T10Y3M"].notna())

    if "T10Y2Y" in p.columns:
        f["ts_10y2y"] = p["T10Y2Y"]

    # Long-history 10y-3m term spread: 10y (DGS10, Shiller-extended to 1871)
    # minus the 3m bill (TB3MS, 1934). Real back to 1934 — replaces the FRED
    # T10Y3M-based ts_10y3m (1982) for the long-history bear model.
    if "DGS10" in p.columns and "TB3MS" in p.columns:
        ts_10y3m_long = p["DGS10"] - p["TB3MS"]
        f["ts_10y3m_level"]     = ts_10y3m_long
        f["ts_10y3m_inv_dummy"] = (ts_10y3m_long < 0).astype(float).where(ts_10y3m_long.notna())

    # ---- Credit ----
    if "EBP" in p.columns:
        f["ebp_level"]  = p["EBP"]
        f["ebp_3m_chg"] = p["EBP"].diff(3)

    if "BAA10Y" in p.columns:
        f["baa_level"]      = p["BAA10Y"]
        f["baa_3m_chg"]     = p["BAA10Y"].diff(3)
        f["baa_zscore_60m"] = _trailing_zscore(p["BAA10Y"], window=60, min_periods=36)

    # ---- Credit (long history, raw Moody's yields back to 1919) ----
    # Built from raw BAA / AAA corporate yields (1919) and the 10y Treasury
    # (DGS10, 1871) so the credit signal is available for the long-history
    # ensemble Model A. Wider spreads = funding stress = elevated bear risk.
    if "BAA" in p.columns and "AAA" in p.columns:
        baa_aaa = p["BAA"] - p["AAA"]                     # quality (default) spread
        f["baa_aaa_spread"] = baa_aaa
        f["baa_aaa_chg6"]   = baa_aaa.diff(6)
        f["baa_aaa_z24"]    = _trailing_zscore(baa_aaa, window=24, min_periods=12)
        f["baa_aaa_z60"]    = _trailing_zscore(baa_aaa, window=60, min_periods=36)

    if "BAA" in p.columns and "DGS10" in p.columns:
        baa_10y = p["BAA"] - p["DGS10"]                   # corporate default spread
        f["baa_10y_spread"] = baa_10y
        f["baa_10y_z24"]    = _trailing_zscore(baa_10y, window=24, min_periods=12)
        f["baa_10y_z60"]    = _trailing_zscore(baa_10y, window=60, min_periods=36)

    if "BAA" in p.columns:
        f["baa_yield_chg6"] = p["BAA"].diff(6)            # corporate funding-cost momentum

    # ---- Real economy (industrial production, 1919) ----
    # Falling industrial production leads recessions and equity drawdowns.
    if "INDPRO" in p.columns:
        indpro = p["INDPRO"]
        f["indpro_yoy"]      = indpro.pct_change(12) * 100
        # Annualized 6-month growth: (IP_t / IP_{t-6})^2 - 1, in %
        f["indpro_6m_growth"] = ((indpro / indpro.shift(6)) ** 2 - 1) * 100

    # ---- Long rates (10y Treasury, 1871) ----
    if "DGS10" in p.columns:
        f["dgs10_12m_chg"] = p["DGS10"].diff(12)          # rising long rates = headwind

    # ---- Labor (long history, raw unemployment rate back to 1948) ----
    # Built from UNRATE (1948) so the labor signal is available for the
    # ensemble Model B (1940s). Rising unemployment leads equity drawdowns.
    if "UNRATE" in p.columns:
        u = p["UNRATE"]
        f["unrate_12m_chg"] = u.diff(12)                  # YoY change in unemployment
        # Real-time Sahm rule: 3m-avg unemployment minus its trailing 12m low.
        f["unrate_sahm"]    = u.rolling(3, min_periods=3).mean() - u.rolling(12, min_periods=12).min()

    # ---- Labor ----
    if "SAHMREALTIME" in p.columns:
        f["sahm_level"]   = p["SAHMREALTIME"]
        f["sahm_trigger"] = (p["SAHMREALTIME"] >= 0.5).astype(float).where(p["SAHMREALTIME"].notna())

    if "ICSA" in p.columns:
        f["icsa_yoy_pct"] = p["ICSA"].pct_change(12) * 100

    # ---- Leading indicator ----
    if "USALOLITOAASTSAM" in p.columns:
        lei = p["USALOLITOAASTSAM"]
        # Annualized 6-month growth: (LEI_t / LEI_{t-6})^2 - 1
        lei_ratio          = lei / lei.shift(6)
        f["lei_6m_growth"] = lei_ratio ** 2 - 1
        # 3Ds dummy: sustained decline; approximate with growth < -4%
        # (official diffusion index not freely available)
        f["lei_stress_dummy"] = (f["lei_6m_growth"] < -0.04).astype(float).where(f["lei_6m_growth"].notna())

    # ---- Financial conditions (Chicago Fed; modern, 1971) ----
    # Tighter financial conditions precede equity drawdowns (for Model D).
    if "NFCI" in p.columns:
        f["nfci_level"]  = p["NFCI"]
        f["nfci_3m_chg"] = p["NFCI"].diff(3)
    if "ANFCI" in p.columns:
        f["anfci_level"]  = p["ANFCI"]
        f["anfci_3m_chg"] = p["ANFCI"].diff(3)

    # ---- Volatility (CBOE VIX; modern, 1990) ----
    # Elevated implied volatility flags stress regimes (for Model D).
    if "VIXCLS" in p.columns:
        f["vix_level"]      = p["VIXCLS"]
        f["vix_zscore_24m"] = _trailing_zscore(p["VIXCLS"], window=24, min_periods=12)

    # ---- Policy ----
    if "DFF" in p.columns:
        f["ffr_6m_chg"] = p["DFF"].diff(6)

    # ---- Trend (SPX, long history) ----
    if "SPX" in p.columns:
        f["spx_vs_10ma"] = p["SPX"] / p["SPX"].rolling(10, min_periods=6).mean() - 1
        f["spx_12m_mom"] = p["SPX"] / p["SPX"].shift(12) - 1   # 12-month price momentum

    # ---- Inflation (CPI; Chen 2009 — a top bear predictor) ----
    if "CPI" in p.columns:
        infl = p["CPI"].pct_change(12) * 100
        f["infl_yoy"]          = infl
        f["infl_zscore_120m"]  = _trailing_zscore(infl, window=120, min_periods=60)

    # ---- Valuation (long history; conditions bear-market severity) ----
    if "SHILLER_CAPE" in p.columns:
        f["cape_20yr_pct"] = _trailing_percentile(
            p["SHILLER_CAPE"], window=240, min_periods=120
        )
        f["cape_z_120m"] = _trailing_zscore(
            p["SHILLER_CAPE"], window=120, min_periods=60
        )

    return f


# ---------------------------------------------------------------------------
# Step 3b — Correction model features
# ---------------------------------------------------------------------------

def build_correction_features(p: pd.DataFrame) -> pd.DataFrame:
    """
    Compute correction model features from a lag-adjusted panel.

    All series in p must already be shifted by their publication lags.

    Target: forward max-drawdown in (-20%, -10%] over 6 months.
    Informative horizon: weeks to 3 months (fast, mean-reverting signals).

    Feature catalogue
    -----------------
    Volatility term structure  (CBOE VIX; backwardation = stress)
      vts_slope          VIX3M - VIX (positive = contango, normal)  sign: -
      vts_ratio          VIX / VIX3M                                sign: +
      vts_backwardation  Dummy: VIX > VIX3M                         sign: +
      vts_slope_zscore   VTS slope trailing 24-month z-score        sign: -

    Trend & momentum  (Moskowitz-Ooi-Pedersen 2012)
      spx_vs_10ma        SPX % dev from 10-month MA (≈ 200-day MA)  sign: -
      spx_below_10ma     Dummy: SPX < 10-month MA                   sign: +
      m12_1_mom          12-1 momentum (SPX_{t}/SPX_{t-12} - 1)     sign: -

    Financial conditions  (Chicago Fed)
      anfci_level        ANFCI level                                sign: +
      anfci_3m_chg       ANFCI 3-month change                       sign: +

    Valuation  (conditioner/amplifier — weak standalone timer)
      cape_20yr_pct      CAPE trailing 240-month percentile [0,1]   sign: +

    Credit (fast, z-score form — Tokic-Jackson bridge feature)
      baa_zscore_24m     BAA-10y trailing 24-month z-score          sign: +

    Sentiment  (Pan-Poteshman 2006; contrarian, nonlinear)
      cpce_low_dummy     Dummy: CPCE at 24m trailing 10th pct (complacency)  sign: +
    """
    f = pd.DataFrame(index=p.index)

    # ---- VIX term structure ----
    if "VIXCLS" in p.columns and "VXVCLS" in p.columns:
        vix   = p["VIXCLS"]
        vxv   = p["VXVCLS"]
        slope = vxv - vix
        f["vts_slope"]         = slope
        f["vts_ratio"]         = vix / vxv.replace(0.0, np.nan)
        f["vts_backwardation"] = (vix > vxv).astype(float).where(vix.notna() & vxv.notna())
        f["vts_slope_zscore"]  = _trailing_zscore(slope, window=24, min_periods=12)

    # ---- Trend & momentum ----
    if "SPX" in p.columns:
        spx  = p["SPX"]
        ma10 = spx.rolling(10, min_periods=6).mean()
        f["spx_vs_10ma"]    = (spx / ma10.replace(0.0, np.nan) - 1) * 100
        f["spx_below_10ma"] = (spx < ma10).astype(float).where(spx.notna() & ma10.notna())
        # Lag-adjusted SPX at t = raw SPX at t-1, so this gives
        # raw SPX_{t-1} / raw SPX_{t-13} - 1  (12-1 momentum)
        f["m12_1_mom"]      = spx / spx.shift(12) - 1

    # ---- Financial conditions ----
    if "ANFCI" in p.columns:
        f["anfci_level"]  = p["ANFCI"]
        f["anfci_3m_chg"] = p["ANFCI"].diff(3)

    # ---- Valuation (conditioner) ----
    if "SHILLER_CAPE" in p.columns:
        f["cape_20yr_pct"] = _trailing_percentile(
            p["SHILLER_CAPE"], window=240, min_periods=120
        )

    # ---- Credit — fast z-score form ----
    if "BAA10Y" in p.columns:
        f["baa_zscore_24m"] = _trailing_zscore(p["BAA10Y"], window=24, min_periods=12)

    # ---- Sentiment ----
    if "CPCE" in p.columns:
        cpce_pct = _trailing_percentile(p["CPCE"], window=24, min_periods=12)
        # Low put/call (complacency) -> elevated correction risk
        f["cpce_low_dummy"] = (cpce_pct <= 0.10).astype(float).where(cpce_pct.notna())

    return f


# ---------------------------------------------------------------------------
# Master builder
# ---------------------------------------------------------------------------

def build_all_features(
    raw_panel: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full Phase 2 pipeline: lag adjustment -> bear features + correction features.

    Parameters
    ----------
    raw_panel : pd.DataFrame
        Month-end indexed raw panel from Phase 1 (bear/raw_monthly.csv).

    Returns
    -------
    bear_features : pd.DataFrame
        Feature matrix for the bear model.
    correction_features : pd.DataFrame
        Feature matrix for the correction model.
    """
    # 1. Apply publication lags — fundamental no-look-ahead safeguard
    lagged = apply_publication_lags(raw_panel)

    # 2. Compute feature sets
    bear_f = build_bear_features(lagged)
    corr_f = build_correction_features(lagged)

    return bear_f, corr_f


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def summarize_features(df: pd.DataFrame, label: str = "") -> pd.DataFrame:
    """
    Return a summary table: count, first non-NaN date, NaN %, mean, std, min, max.
    """
    rows = []
    for col in df.columns:
        s = df[col].dropna()
        rows.append({
            "Feature":     col,
            "Non-NaN obs": len(s),
            "First obs":   str(s.index[0].date()) if len(s) else "-",
            "Last obs":    str(s.index[-1].date()) if len(s) else "-",
            "NaN %":       f"{df[col].isna().mean() * 100:.1f}%",
            "Mean":        round(s.mean(), 4) if len(s) else float("nan"),
            "Std":         round(s.std(),  4) if len(s) else float("nan"),
            "Min":         round(s.min(),  4) if len(s) else float("nan"),
            "Max":         round(s.max(),  4) if len(s) else float("nan"),
        })
    out = pd.DataFrame(rows)
    if label:
        print(f"\n{'='*70}")
        print(f"  {label}  ({len(df)} rows x {len(df.columns)} features)")
        print(f"{'='*70}")
        print(out.to_string(index=False))
    return out


# ---------------------------------------------------------------------------
# Entry point  (python -m bear.features)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    _bear_dir = Path(__file__).resolve().parent
    _data_dir = _bear_dir.parent / "data"
    csv_path  = _data_dir / "raw_monthly.csv"
    if not csv_path.exists():
        print(f"Raw panel not found at {csv_path}. Run 'python -m bear.data' first.")
        sys.exit(1)

    print(f"Loading raw panel from {csv_path}...")
    raw = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    print(f"  {len(raw)} rows  x  {len(raw.columns)} columns  "
          f"({raw.index[0].date()} to {raw.index[-1].date()})")

    print("\nApplying publication lags and computing features...")
    bear_f, corr_f = build_all_features(raw)

    # Show summaries
    summarize_features(bear_f,  label="BEAR MODEL features")
    summarize_features(corr_f, label="CORRECTION MODEL features")

    # Export to CSV for validation
    bear_out = _data_dir / "bear_features.csv"
    corr_out = _data_dir / "correction_features.csv"
    bear_f.to_csv(bear_out, date_format="%Y-%m-%d", float_format="%.6f")
    corr_f.to_csv(corr_out, date_format="%Y-%m-%d", float_format="%.6f")

    # Combined panel (union of all engineered features) — used by the ensemble
    # members and univariate leaderboards, which may mix bear- and correction-
    # only factors in a single model.
    all_out = _data_dir / "all_features.csv"
    extra = [c for c in corr_f.columns if c not in bear_f.columns]
    all_f = bear_f.join(corr_f[extra], how="outer").sort_index()
    all_f.to_csv(all_out, date_format="%Y-%m-%d", float_format="%.6f")

    print(f"\nExported:")
    print(f"  {bear_out}  ({len(bear_f)} rows x {len(bear_f.columns)} features)")
    print(f"  {corr_out}  ({len(corr_f)} rows x {len(corr_f.columns)} features)")
    print(f"  {all_out}  ({len(all_f)} rows x {len(all_f.columns)} features)")

    # Quick sanity check: known signal values at key bear market episodes
    print(f"\n{'='*70}")
    print("  Sanity check — bear features at key episodes")
    print(f"{'='*70}")
    episodes = {
        "1982-08 recession":  "1982-08-31",
        "1990-08 recession":  "1990-08-31",
        "2001-09 dot-com":    "2001-09-30",
        "2009-02 GFC":        "2009-02-28",
        "2020-03 COVID":      "2020-03-31",
        "2022-09 bear":       "2022-09-30",
        "2026-05 current":    "2026-05-31",
    }
    cols = ["ntfs_level", "ts_10y3m", "ebp_level", "baa_level",
            "sahm_level", "lei_6m_growth", "ffr_6m_chg"]
    check_cols = [c for c in cols if c in bear_f.columns]
    rows_list = []
    for label, date_str in episodes.items():
        try:
            row = bear_f.loc[date_str, check_cols].round(3)
            row.name = label
            rows_list.append(row)
        except KeyError:
            pass
    if rows_list:
        print(pd.DataFrame(rows_list).to_string())
