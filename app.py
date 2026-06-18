import os
import sys
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any, Optional

import altair as alt
import pandas as pd
import streamlit as st

# Make the bear/ package importable when running `streamlit run app.py`
sys.path.insert(0, str(Path(__file__).resolve().parent))

st.set_page_config(page_title="Piney Woods Advisory", layout="wide")

ALL_INDICATORS: list[dict[str, Any]] = [
    {
        "name": "10Y Treasury minus 2Y Treasury",
        "category": "Yield curve",
        "description": (
            "Spread between 10-year and 2-year Treasury yields. "
            "Inversions (negative spread) often precede recessions."
        ),
        "source_url": "https://fred.stlouisfed.org/series/T10Y2Y",
        "fred_id": "T10Y2Y",
        "format": "percent",
    },
    {
        "name": "10Y Treasury minus 3M Treasury",
        "category": "Yield curve",
        "description": (
            "Spread between 10-year Treasury yield and 3-month Treasury bill. "
            "A closely watched recession indicator when inverted."
        ),
        "source_url": "https://fred.stlouisfed.org/series/T10Y3M",
        "fred_id": "T10Y3M",
        "format": "percent",
    },
    {
        "name": "Initial jobless claims",
        "category": "Labor market",
        "description": (
            "Weekly count of new unemployment insurance claims. "
            "Rising claims can signal weakening labor demand."
        ),
        "source_url": "https://fred.stlouisfed.org/series/ICSA",
        "fred_id": "ICSA",
        "format": "count",
    },
    {
        "name": "Building permits",
        "category": "Housing",
        "description": (
            "New private housing units authorized by building permits. "
            "A leading indicator for residential construction activity."
        ),
        "source_url": "https://fred.stlouisfed.org/series/PERMIT",
        "fred_id": "PERMIT",
        "format": "count",
    },
    {
        "name": "Consumer sentiment",
        "category": "Sentiment",
        "description": (
            "University of Michigan Consumer Sentiment Index. "
            "Reflects household expectations about the economy."
        ),
        "source_url": "https://fred.stlouisfed.org/series/UMCSENT",
        "fred_id": "UMCSENT",
        "format": "index",
    },
    {
        "name": "HY Credit Spreads",
        "category": "Credit",
        "description": (
            "ICE BofA US High Yield Option-Adjusted Spread. "
            "Wider spreads indicate higher perceived credit risk."
        ),
        "source_url": "https://fred.stlouisfed.org/series/BAMLH0A0HYM2",
        "fred_id": "BAMLH0A0HYM2",
        "format": "percent",
    },
    {
        "name": "Chicago Fed National Financial Conditions Index",
        "category": "Financial conditions",
        "description": (
            "NFCI summarizes financial conditions; positive values indicate "
            "tighter-than-average conditions."
        ),
        "source_url": "https://fred.stlouisfed.org/series/NFCI",
        "fred_id": "NFCI",
        "format": "index",
    },
    {
        "name": "S&P 500 vs 200DMA",
        "category": "Equities",
        "description": (
            "S&P 500 percent above or below its 200-day moving average. "
            "Positive values indicate price above the long-run trend."
        ),
        "source_url": "https://fred.stlouisfed.org/series/SP500",
        "fred_id": "SP500_VS_200DMA",
        "computed": "sp500_vs_200dma",
        "format": "percent",
    },
    {
        "name": "Conference Board LEI",
        "category": "Composite",
        "description": (
            "OECD composite leading indicator (amplitude adjusted) for the U.S. "
            "Used as a FRED-available proxy; the Conference Board LEI is not "
            "published directly on FRED."
        ),
        "source_url": "https://fred.stlouisfed.org/series/USALOLITOAASTSAM",
        "fred_id": "USALOLITOAASTSAM",
        "format": "index",
    },
    {
        "name": "Unemployment Trend (Current - 3YMA)",
        "category": "Labor market",
        "description": (
            "UNRATE minus its 3-year moving average. Positive values mean "
            "unemployment is above its longer trend."
        ),
        "source_url": "https://fred.stlouisfed.org/series/UNRATE",
        "fred_id": "UNRATE_VS_3YMA",
        "computed": "unemployment_vs_3yma",
        "format": "percent",
    },
    {
        "name": "Real GDP Growth",
        "category": "Growth",
        "description": (
            "Percent change in real gross domestic product at a quarterly "
            "annual rate (BEA)."
        ),
        "source_url": "https://fred.stlouisfed.org/series/A191RL1Q225SBEA",
        "fred_id": "A191RL1Q225SBEA",
        "format": "percent",
    },
    {
        "name": "Shiller CAPE",
        "category": "Valuation",
        "description": (
            "Cyclically adjusted P/E (P/E10) for the S&P 500 from Robert "
            "Shiller's historical data."
        ),
        "source_url": "https://shillerdata.com",
        "fred_id": "SHILLER_CAPE",
        "computed": "shiller_cape",
        "format": "ratio",
    },
    {
        "name": "S&P 500 IT sector weight",
        "category": "Equities",
        "description": (
            "Proxy for information-technology weight: NASDAQ Composite vs "
            "S&P 500 ratio mapped to an estimated market-cap share range. "
            "Official GICS IT weights are not available on FRED."
        ),
        "source_url": "https://fred.stlouisfed.org/series/SP500",
        "fred_id": "SP500_IT_WEIGHT",
        "computed": "sp500_it_weight",
        "format": "percent",
    },
    {
        "name": "NBER Recession",
        "category": "Cycle",
        "description": (
            "NBER recession indicator (1 = in recession, 0 = expansion) from "
            "the period following the peak through the trough."
        ),
        "source_url": "https://fred.stlouisfed.org/series/USREC",
        "fred_id": "USREC",
        "format": "binary",
    },
    {
        "name": "S&P 500 Index",
        "category": "Equities",
        "description": "S&P 500 index level used as the market-price base series.",
        "source_url": "https://fred.stlouisfed.org/series/SP500",
        "fred_id": "SP500",
        "computed": "raw_monthly",
        "raw_column": "SPX",
        "format": "index",
    },
    {
        "name": "Near-term forward spread raw",
        "category": "Yield curve",
        "description": "Raw near-term forward Treasury spread used by the bear-market feature set.",
        "source_url": "https://www.federalreserve.gov/data/yield-curve-tables/feds200628.csv",
        "fred_id": "NTFS",
        "computed": "raw_monthly",
        "raw_column": "NTFS",
        "format": "percent",
    },
    {
        "name": "3M Treasury yield",
        "category": "Yield curve",
        "description": ("3-month Treasury bill rate (secondary market, discount "
                        "basis). Uses TB3MS for long history back to 1934."),
        "source_url": "https://fred.stlouisfed.org/series/TB3MS",
        "fred_id": "TB3MS",
        "format": "percent",
    },
    {
        "name": "10Y Treasury yield",
        "category": "Yield curve",
        "description": "10-year Treasury constant maturity yield.",
        "source_url": "https://fred.stlouisfed.org/series/DGS10",
        "fred_id": "DGS10",
        "format": "percent",
    },
    {
        "name": "BAA-10Y credit spread raw",
        "category": "Credit",
        "description": "Moody's BAA corporate yield relative to the 10-year Treasury yield.",
        "source_url": "https://fred.stlouisfed.org/series/BAA10Y",
        "fred_id": "BAA10Y",
        "format": "percent",
    },
    {
        "name": "Excess bond premium raw",
        "category": "Credit",
        "description": "Gilchrist-Zakrajsek excess bond premium raw series.",
        "source_url": "https://www.federalreserve.gov/econresdata/notes/feds-notes/2016/files/ebp_csv.csv",
        "fred_id": "EBP",
        "computed": "raw_monthly",
        "raw_column": "EBP",
        "format": "percent",
    },
    {
        "name": "Sahm rule real-time",
        "category": "Labor market",
        "description": "Real-time Sahm rule recession indicator.",
        "source_url": "https://fred.stlouisfed.org/series/SAHMREALTIME",
        "fred_id": "SAHMREALTIME",
        "format": "percent",
    },
    {
        "name": "Unemployment rate",
        "category": "Labor market",
        "description": "Civilian unemployment rate.",
        "source_url": "https://fred.stlouisfed.org/series/UNRATE",
        "fred_id": "UNRATE",
        "format": "percent",
    },
    {
        "name": "Effective fed funds rate",
        "category": "Policy",
        "description": "Effective federal funds rate.",
        "source_url": "https://fred.stlouisfed.org/series/DFF",
        "fred_id": "DFF",
        "format": "percent",
    },
    {
        "name": "VIX",
        "category": "Volatility",
        "description": "CBOE Volatility Index.",
        "source_url": "https://fred.stlouisfed.org/series/VIXCLS",
        "fred_id": "VIXCLS",
        "format": "index",
    },
    {
        "name": "VIX 3M",
        "category": "Volatility",
        "description": "CBOE 3-month volatility index.",
        "source_url": "https://fred.stlouisfed.org/series/VXVCLS",
        "fred_id": "VXVCLS",
        "format": "index",
    },
    {
        "name": "Adjusted NFCI",
        "category": "Financial conditions",
        "description": "Chicago Fed Adjusted National Financial Conditions Index.",
        "source_url": "https://fred.stlouisfed.org/series/ANFCI",
        "fred_id": "ANFCI",
        "format": "index",
    },
    {
        "name": "Put/call ratio",
        "category": "Sentiment",
        "description": "Equity put/call ratio raw series used by the correction feature set.",
        "source_url": "https://www.cboe.com/us/options/market_statistics/daily/",
        "fred_id": "CPCE",
        "computed": "raw_monthly",
        "raw_column": "CPCE",
        "format": "ratio",
    },
    {
        "name": "Near-term forward spread",
        "category": "Yield curve",
        "description": (
            "Near-term forward Treasury spread used by the bear-market catalog; "
            "lower or negative readings are recession and bear-risk signals."
        ),
        "source_url": "https://www.federalreserve.gov/data/yield-curve-tables/feds200628.csv",
        "fred_id": "NTFS_LEVEL",
        "computed": "bear_feature",
        "feature_column": "ntfs_level",
        "format": "percent",
    },
    {
        "name": "Near-term forward spread 3M change",
        "category": "Yield curve",
        "description": (
            "Three-month change in the near-term forward spread. Rapid policy "
            "and curve shifts can warn of deteriorating cycle conditions."
        ),
        "source_url": "https://www.federalreserve.gov/data/yield-curve-tables/feds200628.csv",
        "fred_id": "NTFS_3M_CHG",
        "computed": "bear_feature",
        "feature_column": "ntfs_3m_chg",
        "format": "percent",
    },
    {
        "name": "10Y-3M inversion flag",
        "category": "Yield curve",
        "description": "Binary flag for an inverted 10-year minus 3-month Treasury spread.",
        "source_url": "SPX_Drawdown_Feature_Engineering_Catalog.pdf",
        "fred_id": "TS_INV_DUMMY",
        "computed": "bear_feature",
        "feature_column": "ts_inv_dummy",
        "format": "binary",
    },
    {
        "name": "Excess bond premium",
        "category": "Credit",
        "description": (
            "Gilchrist-Zakrajsek excess bond premium, a credit-supply and "
            "risk-appetite measure used in the bear-market feature set."
        ),
        "source_url": "SPX_Drawdown_Feature_Engineering_Catalog.pdf",
        "fred_id": "EBP_LEVEL",
        "computed": "bear_feature",
        "feature_column": "ebp_level",
        "format": "percent",
    },
    {
        "name": "Excess bond premium 3M change",
        "category": "Credit",
        "description": "Three-month change in the excess bond premium.",
        "source_url": "SPX_Drawdown_Feature_Engineering_Catalog.pdf",
        "fred_id": "EBP_3M_CHG",
        "computed": "bear_feature",
        "feature_column": "ebp_3m_chg",
        "format": "percent",
    },
    {
        "name": "BAA-10Y credit spread",
        "category": "Credit",
        "description": "Moody's BAA corporate yield minus 10-year Treasury yield.",
        "source_url": "SPX_Drawdown_Feature_Engineering_Catalog.pdf",
        "fred_id": "BAA_LEVEL",
        "computed": "bear_feature",
        "feature_column": "baa_level",
        "format": "percent",
    },
    {
        "name": "BAA-10Y spread 3M change",
        "category": "Credit",
        "description": "Three-month change in the BAA minus 10-year Treasury spread.",
        "source_url": "SPX_Drawdown_Feature_Engineering_Catalog.pdf",
        "fred_id": "BAA_3M_CHG",
        "computed": "bear_feature",
        "feature_column": "baa_3m_chg",
        "format": "percent",
    },
    {
        "name": "BAA spread 60M z-score",
        "category": "Credit",
        "description": "BAA spread normalized against its trailing 60-month history.",
        "source_url": "SPX_Drawdown_Feature_Engineering_Catalog.pdf",
        "fred_id": "BAA_ZSCORE_60M",
        "computed": "bear_feature",
        "feature_column": "baa_zscore_60m",
        "format": "index",
    },
    {
        "name": "Sahm rule level",
        "category": "Labor market",
        "description": "Continuous Sahm-rule labor-market deterioration signal.",
        "source_url": "SPX_Drawdown_Feature_Engineering_Catalog.pdf",
        "fred_id": "SAHM_LEVEL",
        "computed": "bear_feature",
        "feature_column": "sahm_level",
        "format": "percent",
    },
    {
        "name": "Sahm rule trigger",
        "category": "Labor market",
        "description": "Binary Sahm-rule trigger flag.",
        "source_url": "SPX_Drawdown_Feature_Engineering_Catalog.pdf",
        "fred_id": "SAHM_TRIGGER",
        "computed": "bear_feature",
        "feature_column": "sahm_trigger",
        "format": "binary",
    },
    {
        "name": "Initial claims YoY change",
        "category": "Labor market",
        "description": "Year-over-year percentage change in initial jobless claims.",
        "source_url": "SPX_Drawdown_Feature_Engineering_Catalog.pdf",
        "fred_id": "ICSA_YOY_PCT",
        "computed": "bear_feature",
        "feature_column": "icsa_yoy_pct",
        "format": "percent",
    },
    {
        "name": "LEI 6M growth",
        "category": "Composite",
        "description": "Six-month growth rate of the leading-index proxy used in the bear model.",
        "source_url": "SPX_Drawdown_Feature_Engineering_Catalog.pdf",
        "fred_id": "LEI_6M_GROWTH",
        "computed": "bear_feature",
        "feature_column": "lei_6m_growth",
        "format": "percent",
    },
    {
        "name": "LEI stress flag",
        "category": "Composite",
        "description": "Binary leading-index stress flag from the engineered bear feature set.",
        "source_url": "SPX_Drawdown_Feature_Engineering_Catalog.pdf",
        "fred_id": "LEI_STRESS_DUMMY",
        "computed": "bear_feature",
        "feature_column": "lei_stress_dummy",
        "format": "binary",
    },
    {
        "name": "Fed funds 6M change",
        "category": "Policy",
        "description": "Six-month change in the effective federal funds rate.",
        "source_url": "SPX_Drawdown_Feature_Engineering_Catalog.pdf",
        "fred_id": "FFR_6M_CHG",
        "computed": "bear_feature",
        "feature_column": "ffr_6m_chg",
        "format": "percent",
    },
    {
        "name": "VIX term-structure slope",
        "category": "Volatility",
        "description": "VIX3M minus VIX term-structure slope used for correction-risk timing.",
        "source_url": "SPX_Drawdown_Feature_Engineering_Catalog.pdf",
        "fred_id": "VTS_SLOPE",
        "computed": "correction_feature",
        "feature_column": "vts_slope",
        "format": "percent",
    },
    {
        "name": "VIX/VIX3M ratio",
        "category": "Volatility",
        "description": "VIX divided by VIX3M, a fast volatility stress and backwardation signal.",
        "source_url": "SPX_Drawdown_Feature_Engineering_Catalog.pdf",
        "fred_id": "VTS_RATIO",
        "computed": "correction_feature",
        "feature_column": "vts_ratio",
        "format": "ratio",
    },
    {
        "name": "VIX backwardation flag",
        "category": "Volatility",
        "description": "Binary flag for VIX term-structure backwardation.",
        "source_url": "SPX_Drawdown_Feature_Engineering_Catalog.pdf",
        "fred_id": "VTS_BACKWARDATION",
        "computed": "correction_feature",
        "feature_column": "vts_backwardation",
        "format": "binary",
    },
    {
        "name": "VIX term-structure z-score",
        "category": "Volatility",
        "description": "VIX term-structure slope normalized against recent history.",
        "source_url": "SPX_Drawdown_Feature_Engineering_Catalog.pdf",
        "fred_id": "VTS_SLOPE_ZSCORE",
        "computed": "correction_feature",
        "feature_column": "vts_slope_zscore",
        "format": "index",
    },
    {
        "name": "S&P 500 vs 10M average",
        "category": "Equities",
        "description": "S&P 500 percent above or below its 10-month moving average.",
        "source_url": "SPX_Drawdown_Feature_Engineering_Catalog.pdf",
        "fred_id": "SPX_VS_10MA",
        "computed": "correction_feature",
        "feature_column": "spx_vs_10ma",
        "format": "percent",
    },
    {
        "name": "S&P 500 below 10M average flag",
        "category": "Equities",
        "description": "Binary flag for S&P 500 below its 10-month moving average.",
        "source_url": "SPX_Drawdown_Feature_Engineering_Catalog.pdf",
        "fred_id": "SPX_BELOW_10MA",
        "computed": "correction_feature",
        "feature_column": "spx_below_10ma",
        "format": "binary",
    },
    {
        "name": "S&P 500 12-1 momentum",
        "category": "Equities",
        "description": "S&P 500 12-month minus 1-month momentum feature.",
        "source_url": "SPX_Drawdown_Feature_Engineering_Catalog.pdf",
        "fred_id": "M12_1_MOM",
        "computed": "correction_feature",
        "feature_column": "m12_1_mom",
        "format": "percent",
    },
    {
        "name": "Adjusted NFCI level",
        "category": "Financial conditions",
        "description": "Chicago Fed Adjusted NFCI level used in the correction model.",
        "source_url": "SPX_Drawdown_Feature_Engineering_Catalog.pdf",
        "fred_id": "ANFCI_LEVEL",
        "computed": "correction_feature",
        "feature_column": "anfci_level",
        "format": "index",
    },
    {
        "name": "Adjusted NFCI 3M change",
        "category": "Financial conditions",
        "description": "Three-month change in the Chicago Fed Adjusted NFCI.",
        "source_url": "SPX_Drawdown_Feature_Engineering_Catalog.pdf",
        "fred_id": "ANFCI_3M_CHG",
        "computed": "correction_feature",
        "feature_column": "anfci_3m_chg",
        "format": "index",
    },
    {
        "name": "CAPE 20Y percentile",
        "category": "Valuation",
        "description": "Shiller CAPE percentile versus its trailing 20-year history.",
        "source_url": "SPX_Drawdown_Feature_Engineering_Catalog.pdf",
        "fred_id": "CAPE_20YR_PCT",
        "computed": "correction_feature",
        "feature_column": "cape_20yr_pct",
        "format": "percent",
    },
    {
        "name": "BAA spread 24M z-score",
        "category": "Credit fast",
        "description": "BAA credit spread normalized against its trailing 24-month history.",
        "source_url": "SPX_Drawdown_Feature_Engineering_Catalog.pdf",
        "fred_id": "BAA_ZSCORE_24M",
        "computed": "correction_feature",
        "feature_column": "baa_zscore_24m",
        "format": "index",
    },
    {
        "name": "Put/call low extreme flag",
        "category": "Sentiment",
        "description": "Contrarian low put/call extreme flag from the correction feature set.",
        "source_url": "SPX_Drawdown_Feature_Engineering_Catalog.pdf",
        "fred_id": "CPCE_LOW_DUMMY",
        "computed": "correction_feature",
        "feature_column": "cpce_low_dummy",
        "format": "binary",
    },
]

RAW_INDICATOR_NAMES = {
    "10Y Treasury minus 2Y Treasury",
    "10Y Treasury minus 3M Treasury",
    "Initial jobless claims",
    "Building permits",
    "Consumer sentiment",
    "HY Credit Spreads",
    "Chicago Fed National Financial Conditions Index",
    "Conference Board LEI",
    "Real GDP Growth",
    "Shiller CAPE",
    "NBER Recession",
    "S&P 500 Index",
    "Near-term forward spread raw",
    "3M Treasury yield",
    "10Y Treasury yield",
    "BAA-10Y credit spread raw",
    "Excess bond premium raw",
    "Sahm rule real-time",
    "Unemployment rate",
    "Effective fed funds rate",
    "VIX",
    "VIX 3M",
    "Adjusted NFCI",
    "Put/call ratio",
}


def is_flag_indicator(indicator: dict[str, Any]) -> bool:
    feature_column = str(indicator.get("feature_column", "")).lower()
    fred_id = str(indicator.get("fred_id", "")).lower()
    name = str(indicator.get("name", "")).lower()
    flag_terms = ("flag", "dummy", "trigger")
    return any(term in feature_column or term in fred_id or term in name for term in flag_terms)


INDICATORS = [item for item in ALL_INDICATORS if item["name"] in RAW_INDICATOR_NAMES]
DERIVED_INDICATORS = [
    item
    for item in ALL_INDICATORS
    if item["name"] not in RAW_INDICATOR_NAMES and not is_flag_indicator(item)
]
INDICATOR_BY_NAME = {item["name"]: item for item in INDICATORS}

SHILLER_CAPE_URL = (
    "https://img1.wsimg.com/blobby/go/e5e77e0b-59d1-44d9-ab25-4763ac982e53/"
    "downloads/441f0d2c-37e4-4803-b4e2-8fe10407fbf6/ie_data.xls"
)
SHILLER_CAPE_LOCAL_PATH = Path(__file__).resolve().parent / "data" / "shiller_ie_data.xls"
LOGO_PATH = Path(__file__).resolve().parent / "assets" / "piney-woods-logo.png"
BEAR_DIR = Path(__file__).resolve().parent / "bear"
DATA_DIR = Path(__file__).resolve().parent / "data"

CYCLE_STAGES = ("Early", "Mid", "Late", "Recession")
STAGE_COLORS = {
    "Early": "#22c55e",
    "Mid": "#eab308",
    "Late": "#f97316",
    "Recession": "#ef4444",
}

BEAR_MODEL_FEATURES = [
    "ntfs_level",
    "baa_level",
    "baa_3m_chg",
    "ebp_level",
    "icsa_yoy_pct",
    "lei_6m_growth",
]
CORRECTION_MODEL_FEATURES = [
    "vts_slope",
    "spx_vs_10ma",
    "m12_1_mom",
    "anfci_3m_chg",
    "cape_20yr_pct",
    "baa_zscore_24m",
]


def get_fred_api_key() -> Optional[str]:
    key = os.environ.get("FRED_API_KEY")
    if key and key.strip():
        return key.strip()
    try:
        if "FRED_API_KEY" in st.secrets:
            return str(st.secrets["FRED_API_KEY"]).strip()
        if "fred" in st.secrets and "api_key" in st.secrets["fred"]:
            return str(st.secrets["fred"]["api_key"]).strip()
    except Exception:
        pass
    return None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_latest_from_fred(series_id: str, api_key: str) -> dict[str, Any]:
    """Latest observation with FRED vintage (publication) date."""
    import requests

    resp = requests.get(
        "https://api.stlouisfed.org/fred/series/observations",
        params={
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": 1,
        },
        timeout=30,
    )
    resp.raise_for_status()
    observations = resp.json().get("observations", [])
    if not observations:
        raise ValueError(f"No observations for {series_id}")

    obs = observations[0]
    raw_value = obs.get("value")
    if raw_value in (None, "."):
        raise ValueError(f"Missing latest value for {series_id}")

    publication = obs.get("realtime_start") or obs.get("date")
    return {
        "value": float(raw_value),
        "observation_date": obs["date"],
        "publication_date": publication,
    }


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fred_series(series_id: str, api_key: Optional[str]) -> pd.Series:
    if not api_key:
        raise ValueError("FRED API key required to fetch historical series")

    import requests

    resp = requests.get(
        "https://api.stlouisfed.org/fred/series/observations",
        params={
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "sort_order": "asc",
            "limit": 100000,
        },
        timeout=60,
    )
    resp.raise_for_status()
    observations = resp.json().get("observations", [])
    dates: list[pd.Timestamp] = []
    values: list[float] = []
    for obs in observations:
        if obs.get("value") in (None, "."):
            continue
        dates.append(pd.Timestamp(obs["date"]))
        values.append(float(obs["value"]))
    if not values:
        raise ValueError(f"No observations for {series_id}")
    return pd.Series(values, index=dates).sort_index()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_observations_series(
    series_id: str, api_key: str, limit: int = 500
) -> pd.Series:
    import requests

    resp = requests.get(
        "https://api.stlouisfed.org/fred/series/observations",
        params={
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": limit,
        },
        timeout=30,
    )
    resp.raise_for_status()
    observations = resp.json().get("observations", [])
    observations.reverse()
    dates: list[pd.Timestamp] = []
    values: list[float] = []
    for obs in observations:
        if obs.get("value") in (None, "."):
            continue
        dates.append(pd.Timestamp(obs["date"]))
        values.append(float(obs["value"]))
    if not values:
        raise ValueError(f"No observations for {series_id}")
    return pd.Series(values, index=dates).sort_index()


def ensure_local_shiller_file() -> Path:
    import requests

    resp = requests.get(
        SHILLER_CAPE_URL,
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=60,
    )
    resp.raise_for_status()
    remote_bytes = resp.content

    if SHILLER_CAPE_LOCAL_PATH.exists():
        local_bytes = SHILLER_CAPE_LOCAL_PATH.read_bytes()
        if local_bytes == remote_bytes:
            return SHILLER_CAPE_LOCAL_PATH

    SHILLER_CAPE_LOCAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    SHILLER_CAPE_LOCAL_PATH.write_bytes(remote_bytes)
    return SHILLER_CAPE_LOCAL_PATH


def fetch_shiller_cape_series() -> pd.Series:
    source_path = ensure_local_shiller_file()
    df = pd.read_excel(source_path, sheet_name="Data", header=6)
    date_col = df.columns[0]
    cape_candidates = ["P/E10 or", "P/E10", "CAPE"]
    cape_col = next((col for col in cape_candidates if col in df.columns), None)
    if cape_col not in df.columns:
        raise ValueError("Shiller CAPE column not found in source file")

    def _decimal_year_to_timestamp(year_value: float) -> pd.Timestamp:
        year = int(year_value)
        month = int(round((year_value - year) * 100))
        month = min(max(month, 1), 12)
        return pd.Timestamp(year=year, month=month, day=1)

    numeric_dates = pd.to_numeric(df[date_col], errors="coerce")
    numeric_cape = pd.to_numeric(df[cape_col], errors="coerce")
    rows = df[numeric_dates.between(1900, 2030) & numeric_cape.notna()]
    numeric_dates = numeric_dates.loc[rows.index]
    numeric_cape = numeric_cape.loc[rows.index]
    dates = numeric_dates.apply(_decimal_year_to_timestamp)
    series = pd.Series(numeric_cape.astype(float).values, index=dates, name="Value")
    series.index.name = "Date"
    series = series.sort_index()
    return series.dropna()


@st.cache_data(ttl=3600, show_spinner=False)
def compute_sp500_vs_200dma(api_key: str, limit: int = 500) -> pd.Series:
    prices = fetch_observations_series("SP500", api_key, limit=limit)
    ma200 = prices.rolling(200, min_periods=200).mean()
    return ((prices / ma200) - 1.0) * 100.0


@st.cache_data(ttl=3600, show_spinner=False)
def compute_unemployment_vs_3yma(api_key: str, limit: int = 500) -> pd.Series:
    components = compute_unemployment_trend_components(api_key, limit=limit)
    return components["Difference"]


@st.cache_data(ttl=3600, show_spinner=False)
def compute_unemployment_trend_components(
    api_key: str, limit: int = 500
) -> pd.DataFrame:
    unrate = fetch_observations_series("UNRATE", api_key, limit=limit)
    ma36 = unrate.rolling(36, min_periods=36).mean()
    components = pd.concat(
        [
            unrate.rename("Unemployment rate"),
            ma36.rename("3-year moving average"),
        ],
        axis=1,
    )
    components["Difference"] = (
        components["Unemployment rate"] - components["3-year moving average"]
    )
    components.index.name = "Date"
    return components.dropna()


@st.cache_data(ttl=3600, show_spinner=False)
def compute_sp500_it_weight(api_key: str, limit: int = 500) -> pd.Series:
    sp500 = fetch_observations_series("SP500", api_key, limit=limit)
    nasdaq = fetch_observations_series("NASDAQCOM", api_key, limit=limit)
    combined = pd.concat([sp500, nasdaq], axis=1, join="inner")
    combined.columns = ["sp500", "nasdaq"]
    ratio = combined["nasdaq"] / combined["sp500"]
    median_ratio = float(ratio.median())
    if median_ratio <= 0:
        raise ValueError("Invalid NASDAQ/SP500 ratio")
    # Map ratio vs history to an estimated IT market-cap weight range (~15–45%).
    return (ratio / median_ratio) * 30.0


@st.cache_data(show_spinner=False)
def load_engineered_feature_series(features_file: str, column: str) -> pd.Series:
    df = pd.read_csv(DATA_DIR / features_file, parse_dates=["date"])
    if column not in df.columns:
        raise ValueError(f"{column} not found in {features_file}")
    series = pd.to_numeric(df[column], errors="coerce")
    series.index = pd.to_datetime(df["date"])
    series.index.name = "Date"
    return series.dropna().sort_index()


@st.cache_data(show_spinner=False)
def load_raw_monthly_series(column: str) -> pd.Series:
    df = pd.read_csv(DATA_DIR / "raw_monthly.csv", parse_dates=["date"])
    if column not in df.columns:
        raise ValueError(f"{column} not found in raw_monthly.csv")
    series = pd.to_numeric(df[column], errors="coerce")
    series.index = pd.to_datetime(df["date"])
    series.index.name = "Date"
    return series.dropna().sort_index()


def compute_indicator_series(indicator: dict[str, Any], api_key: str) -> pd.Series:
    computed = indicator.get("computed")
    if computed == "raw_monthly":
        return load_raw_monthly_series(indicator["raw_column"])
    if computed == "sp500_vs_200dma":
        return compute_sp500_vs_200dma(api_key).dropna()
    if computed == "unemployment_vs_3yma":
        return compute_unemployment_vs_3yma(api_key).dropna()
    if computed == "shiller_cape":
        return fetch_shiller_cape_series().dropna()
    if computed == "sp500_it_weight":
        return compute_sp500_it_weight(api_key).dropna()
    if computed == "bear_feature":
        return load_engineered_feature_series(
            "bear_features.csv", indicator["feature_column"]
        )
    if computed == "correction_feature":
        return load_engineered_feature_series(
            "correction_features.csv", indicator["feature_column"]
        )
    return fetch_fred_series(indicator["fred_id"], api_key)


def fetch_indicator_history_for_stage(indicator: dict[str, Any], api_key: str) -> pd.Series:
    computed = indicator.get("computed")
    if computed == "raw_monthly":
        return load_raw_monthly_series(indicator["raw_column"])
    if computed == "sp500_vs_200dma":
        return compute_sp500_vs_200dma(api_key).dropna()
    if computed == "unemployment_vs_3yma":
        return compute_unemployment_vs_3yma(api_key).dropna()
    if computed == "shiller_cape":
        return fetch_shiller_cape_series().dropna()
    if computed == "sp500_it_weight":
        return compute_sp500_it_weight(api_key).dropna()
    if computed == "bear_feature":
        return load_engineered_feature_series(
            "bear_features.csv", indicator["feature_column"]
        )
    if computed == "correction_feature":
        return load_engineered_feature_series(
            "correction_features.csv", indicator["feature_column"]
        )
    return fetch_fred_series_for_stage(indicator["fred_id"], api_key)


def fetch_indicator_history_for_stats(indicator: dict[str, Any], api_key: str) -> pd.Series:
    """History used for distribution stats (median / p10 / p90)."""
    computed = indicator.get("computed")
    if computed == "raw_monthly":
        return load_raw_monthly_series(indicator["raw_column"])
    # For computed series, use a larger recent window to get stable percentiles.
    if computed == "sp500_vs_200dma":
        return compute_sp500_vs_200dma(api_key, limit=2000).dropna()
    if computed == "unemployment_vs_3yma":
        return compute_unemployment_vs_3yma(api_key, limit=600).dropna()
    if computed == "shiller_cape":
        return fetch_shiller_cape_series().dropna()
    if computed == "sp500_it_weight":
        return compute_sp500_it_weight(api_key, limit=2000).dropna()
    if computed == "bear_feature":
        return load_engineered_feature_series(
            "bear_features.csv", indicator["feature_column"]
        )
    if computed == "correction_feature":
        return load_engineered_feature_series(
            "correction_features.csv", indicator["feature_column"]
        )
    # For FRED-native series, pull a longer history (25y) via API.
    return fetch_fred_series(indicator["fred_id"], api_key)


def _quantiles(series: pd.Series) -> tuple[float, float, float]:
    clean = series.dropna().astype(float)
    if clean.empty:
        raise ValueError("Empty series")
    q10, q50, q90 = clean.quantile([0.1, 0.5, 0.9]).tolist()
    return float(q50), float(q10), float(q90)


def fetch_indicator_latest(indicator: dict[str, Any], api_key: str) -> dict[str, Any]:
    computed = indicator.get("computed")
    if computed:
        series = compute_indicator_series(indicator, api_key).dropna()
        if series.empty:
            raise ValueError(f"No data for {indicator['name']}")
        last_date = pd.Timestamp(series.index[-1])
        return {
            "value": float(series.iloc[-1]),
            "observation_date": last_date.strftime("%Y-%m-%d"),
            "publication_date": last_date.strftime("%Y-%m-%d"),
        }
    return fetch_latest_from_fred(indicator["fred_id"], api_key)


def indicator_stage_key(indicator: dict[str, Any]) -> str:
    if indicator.get("feature_column"):
        return indicator["feature_column"]
    return indicator.get("computed") or indicator["fred_id"]


def format_date(date_str: str) -> str:
    return pd.Timestamp(date_str).strftime("%Y-%m-%d")


def _percentile_rank(series: pd.Series, value: float) -> float:
    return float((series <= value).mean() * 100)


def _momentum(series: pd.Series, periods: int = 12) -> float:
    if len(series) <= periods:
        periods = max(len(series) - 1, 1)
    return float(series.iloc[-1]) - float(series.iloc[-1 - periods])


def classify_cycle_stage(stage_key: str, series: pd.Series) -> str:
    """Heuristic cycle stage from level, history, and recent momentum."""
    history = series.dropna().sort_index()
    if history.empty:
        return "Mid"

    latest = float(history.iloc[-1])
    pct = _percentile_rank(history, latest)
    mom = _momentum(history)

    high_risk_high_values = {
        "ts_inv_dummy",
        "ebp_level",
        "ebp_3m_chg",
        "baa_level",
        "baa_3m_chg",
        "baa_zscore_60m",
        "sahm_level",
        "sahm_trigger",
        "icsa_yoy_pct",
        "lei_stress_dummy",
        "vts_ratio",
        "vts_backwardation",
        "spx_below_10ma",
        "anfci_level",
        "anfci_3m_chg",
        "cape_20yr_pct",
        "baa_zscore_24m",
    }
    high_risk_low_values = {
        "ntfs_level",
        "ts_10y3m",
        "ts_10y2y",
        "lei_6m_growth",
        "ffr_6m_chg",
        "vts_slope",
        "vts_slope_zscore",
        "spx_vs_10ma",
    }
    high_risk_extreme_values = {
        "ntfs_3m_chg",
        "m12_1_mom",
        "cpce_low_dummy",
    }

    if stage_key in high_risk_high_values:
        if latest >= 1 and (
            stage_key.endswith("dummy")
            or stage_key.endswith("trigger")
            or stage_key.endswith("flag")
            or stage_key in {"ts_inv_dummy", "vts_backwardation", "spx_below_10ma"}
        ):
            return "Late"
        if pct >= 90:
            return "Recession"
        if pct >= 70 or mom > 0:
            return "Late"
        if pct <= 30 and mom <= 0:
            return "Early"
        return "Mid"

    if stage_key in high_risk_low_values:
        if pct <= 10:
            return "Recession"
        if pct <= 30 or mom < 0:
            return "Late"
        if pct >= 70 and mom >= 0:
            return "Early"
        return "Mid"

    if stage_key in high_risk_extreme_values:
        if abs(pct - 50) >= 40:
            return "Late"
        if abs(pct - 50) <= 15:
            return "Mid"
        return "Early"

    if stage_key in ("T10Y2Y", "T10Y3M"):
        if latest < 0:
            return "Recession"
        if latest < 0.3 or mom < -0.1:
            return "Late"
        if latest > 0.8 and mom >= 0:
            return "Early"
        return "Mid"

    if stage_key == "ICSA":
        if latest >= 350_000 or pct >= 80:
            return "Recession"
        if latest >= 280_000 or (mom > 15_000 and pct >= 55):
            return "Late"
        if mom < -5_000 and pct <= 45:
            return "Early"
        return "Mid"

    if stage_key == "PERMIT":
        if pct <= 25:
            return "Recession"
        if mom < 0 and pct <= 45:
            return "Late"
        if mom > 0 and pct <= 55:
            return "Early"
        return "Mid"

    if stage_key == "UMCSENT":
        if latest < 65 or pct <= 20:
            return "Recession"
        if mom < -2 or (pct <= 40 and mom <= 0):
            return "Late"
        if mom > 2 and pct <= 50:
            return "Early"
        return "Mid"

    if stage_key == "BAMLH0A0HYM2":
        if latest >= 6 or pct >= 85:
            return "Recession"
        if latest >= 4.5 or (mom > 0.3 and pct >= 60):
            return "Late"
        if mom < -0.2 and pct <= 45:
            return "Early"
        return "Mid"

    if stage_key == "NFCI":
        if latest > 0.5 or pct >= 85:
            return "Recession"
        if latest > 0 or (mom > 0.05 and pct >= 55):
            return "Late"
        if mom < -0.05 and pct <= 45:
            return "Early"
        return "Mid"

    if stage_key == "sp500_vs_200dma":
        if latest < -10:
            return "Recession"
        if latest < 0 or mom < -1:
            return "Late"
        if latest > 5 and mom >= 0:
            return "Early"
        return "Mid"

    if stage_key == "USALOLITOAASTSAM":
        if mom < -0.5 and pct <= 35:
            return "Late"
        if mom > 0.3 and pct >= 55:
            return "Early"
        if pct <= 25:
            return "Recession"
        return "Mid"

    if stage_key == "unemployment_vs_3yma":
        if latest > 0.3 or (mom > 0.1 and pct >= 70):
            return "Recession"
        if latest > 0.1 or mom > 0.05:
            return "Late"
        if latest < -0.05 and mom < 0:
            return "Early"
        return "Mid"

    if stage_key == "A191RL1Q225SBEA":
        if latest < 0:
            return "Recession"
        if latest < 1.5 or mom < -0.5:
            return "Late"
        if latest > 3 and mom >= 0:
            return "Early"
        return "Mid"

    if stage_key == "shiller_cape":
        if latest >= 35 or pct >= 90:
            return "Late"
        if latest <= 18 or pct <= 15:
            return "Early"
        if latest >= 30:
            return "Late"
        return "Mid"

    if stage_key == "sp500_it_weight":
        if latest >= 38 or pct >= 85:
            return "Late"
        if latest <= 22 or pct <= 20:
            return "Early"
        if mom > 0.5 and pct >= 60:
            return "Late"
        return "Mid"

    if stage_key == "USREC":
        if latest >= 1:
            return "Recession"
        if mom < 0 and pct <= 30:
            return "Early"
        return "Mid"

    return "Mid"


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fred_series_for_stage(series_id: str, api_key: str) -> pd.Series:
    """Recent history for cycle-stage rules (lighter than full 25Y pull)."""
    import requests

    resp = requests.get(
        "https://api.stlouisfed.org/fred/series/observations",
        params={
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": 500,
        },
        timeout=30,
    )
    resp.raise_for_status()
    observations = resp.json().get("observations", [])
    observations.reverse()
    dates: list[pd.Timestamp] = []
    values: list[float] = []
    for obs in observations:
        if obs.get("value") in (None, "."):
            continue
        dates.append(pd.Timestamp(obs["date"]))
        values.append(float(obs["value"]))
    if not values:
        raise ValueError(f"No observations for {series_id}")
    return pd.Series(values, index=dates).sort_index()


def render_colored_snapshot_table(df: pd.DataFrame) -> None:
    """HTML table — avoids Streamlit + pandas Styler rendering as blank."""
    if df.empty:
        st.info("No latest data available.")
        return

    header_cells = "".join(f"<th>{col}</th>" for col in df.columns)
    body_rows: list[str] = []
    for _, row in df.iterrows():
        cells: list[str] = []
        for col in df.columns:
            value = row[col]
            if col == "Cycle stage" and value in STAGE_COLORS:
                cells.append(
                    f'<td style="background:{STAGE_COLORS[value]};color:white;'
                    f'font-weight:600;text-align:center;">{value}</td>'
                )
            else:
                cells.append(f"<td>{value}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")

    table_html = f"""
    <table style="width:100%;border-collapse:collapse;font-size:0.95rem;">
      <thead>
        <tr style="background:#f3f4f6;text-align:left;">{header_cells}</tr>
      </thead>
      <tbody>
        {"".join(body_rows)}
      </tbody>
    </table>
    <style>
      table td, table th {{
        border: 1px solid #e5e7eb;
        padding: 0.5rem 0.75rem;
      }}
    </style>
    """
    st.markdown(table_html, unsafe_allow_html=True)


def build_latest_snapshot_table(api_key: str) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for item in INDICATORS:
        try:
            latest = fetch_indicator_latest(item, api_key)
            row = {
                "Indicator": item["name"],
                "Category": item["category"],
                "Latest value": format_value(latest["value"], item["format"]),
                "Observation date": format_date(latest["observation_date"]),
                "Publication date": format_date(latest["publication_date"]),
                "Median": "—",
                "10th pct": "—",
                "90th pct": "—",
                "Cycle stage": "—",
            }
            try:
                hist_for_stats = fetch_indicator_history_for_stats(item, api_key)
                med, p10, p90 = _quantiles(hist_for_stats)
                row["Median"] = format_value(med, item["format"])
                row["10th pct"] = format_value(p10, item["format"])
                row["90th pct"] = format_value(p90, item["format"])

                hist_for_stage = fetch_indicator_history_for_stage(item, api_key)
                row["Cycle stage"] = classify_cycle_stage(
                    indicator_stage_key(item), hist_for_stage
                )
            except Exception:
                pass
            rows.append(row)
        except Exception:
            rows.append(
                {
                    "Indicator": item["name"],
                    "Category": item["category"],
                    "Latest value": "—",
                    "Observation date": "—",
                    "Publication date": "—",
                    "Median": "—",
                    "10th pct": "—",
                    "90th pct": "—",
                    "Cycle stage": "—",
                }
            )
    return pd.DataFrame(rows)


def format_value(value: float, fmt: str) -> str:
    if fmt == "percent":
        return f"{value:.2f}%"
    if fmt == "count":
        return f"{value:,.0f}"
    if fmt == "index":
        return f"{value:,.2f}"
    if fmt == "ratio":
        return f"{value:.2f}"
    if fmt == "binary":
        return "Recession" if value >= 1 else "Expansion"
    return f"{value:,.2f}"


def interpret_indicator(indicator: dict[str, Any], series: pd.Series) -> str:
    latest = float(series.iloc[-1])
    name = indicator["name"]
    fmt = indicator["format"]

    parts: list[str] = [f"Latest reading for **{name}** is **{format_value(latest, fmt)}**."]

    if len(series) >= 2:
        prior = float(series.iloc[-2])
        change = latest - prior
        direction = "up" if change > 0 else "down" if change < 0 else "unchanged"
        parts.append(
            f"Versus the prior observation, the series moved **{direction}** "
            f"({format_value(abs(change), fmt)} in level terms)."
        )

    if len(series) >= 13:
        ref = float(series.iloc[-13])
        yoy_style = latest - ref
        parts.append(
            f"Roughly one year ago (≈12 observations back), the level was "
            f"**{format_value(ref, fmt)}**; the change since then is "
            f"**{format_value(yoy_style, fmt)}**."
        )

    stage_key = indicator_stage_key(indicator)
    if stage_key in ("T10Y2Y", "T10Y3M"):
        if latest < 0:
            parts.append(
                "The yield curve is **inverted**, which historically has been "
                "associated with tighter financial conditions and elevated recession risk."
            )
        else:
            parts.append(
                "The yield curve is **positively sloped**, which is more typical "
                "of expansion phases, though the level of the spread still matters."
            )
    elif stage_key == "ICSA":
        if latest > 300_000:
            parts.append(
                "Claims are **elevated** relative to tight labor-market norms, "
                "suggesting possible cooling in hiring."
            )
        else:
            parts.append(
                "Claims remain **relatively low** by historical standards, "
                "consistent with a still-resilient labor market."
            )
    elif stage_key == "PERMIT":
        parts.append(
            "Permits lead housing starts; sustained declines often foreshadow "
            "weaker residential investment."
        )
    elif stage_key == "UMCSENT":
        if latest < 70:
            parts.append("Sentiment is **subdued**, which can weigh on consumer spending.")
        elif latest > 90:
            parts.append("Sentiment is **relatively strong**, supportive of consumption.")
        else:
            parts.append("Sentiment is in a **middle range**—neither euphoric nor deeply pessimistic.")
    elif stage_key == "BAMLH0A0HYM2":
        if latest > 5:
            parts.append(
                "The high-yield spread is **wide**, signaling stressed credit markets."
            )
        elif latest < 3.5:
            parts.append(
                "The spread is **tight**, consistent with benign credit conditions."
            )
        else:
            parts.append("Spreads are in a **moderate** range.")
    elif stage_key == "NFCI":
        if latest > 0:
            parts.append(
                "Financial conditions are **tighter than average** (positive NFCI)."
            )
        else:
            parts.append(
                "Financial conditions are **looser than average** (negative NFCI)."
            )
    elif stage_key == "sp500_vs_200dma":
        if latest < 0:
            parts.append("Price is **below** the 200-day average—often a late-cycle or risk-off signal.")
        else:
            parts.append("Price is **above** the 200-day average—typical of expansion phases.")
    elif stage_key == "USALOLITOAASTSAM":
        parts.append("A declining leading index often precedes slower growth over the next few quarters.")
    elif stage_key == "unemployment_vs_3yma":
        if latest > 0:
            parts.append("Unemployment is **above** its 3-year moving average.")
        else:
            parts.append("Unemployment is **at or below** its 3-year moving average.")
    elif stage_key == "A191RL1Q225SBEA":
        if latest < 2:
            parts.append("Growth is **subdued** on a quarterly annualized basis.")
        else:
            parts.append("Growth looks **solid** relative to typical expansion rates.")
    elif stage_key == "shiller_cape":
        if latest > 30:
            parts.append("Valuation is **rich** vs long-run CAPE history.")
        else:
            parts.append("Valuation is **moderate or below** long-run CAPE extremes.")
    elif stage_key == "sp500_it_weight":
        parts.append("Higher tech concentration can amplify late-cycle equity vulnerability.")
    elif stage_key == "USREC":
        if latest >= 1:
            parts.append("NBER flags the economy as in **recession** for this period.")
        else:
            parts.append("NBER flags the economy in **expansion** for this period.")

    return " ".join(parts)


def model_use_form(indicator: dict[str, Any]) -> str:
    name = indicator["name"]
    fred_id = indicator["fred_id"]

    if fred_id in ("T10Y2Y", "T10Y3M"):
        return (
            "Use the raw spread level directly; inversion and low-percentile regimes "
            "can be derived as secondary features."
        )
    if name == "Near-term forward spread raw":
        return "Use the level and short-window changes, especially 3-month change."
    if fred_id in ("DGS3MO", "DGS10", "DFF"):
        return "Use the level and policy-rate changes over 3- to 6-month windows."
    if fred_id in ("BAMLH0A0HYM2", "BAA10Y") or name == "Excess bond premium raw":
        return "Use the level, 3-month change, and trailing z-score for credit stress."
    if fred_id in ("ICSA", "UNRATE", "SAHMREALTIME"):
        return "Use the level plus labor deterioration transforms such as YoY change or Sahm gap."
    if fred_id in ("PERMIT", "UMCSENT", "USALOLITOAASTSAM", "A191RL1Q225SBEA"):
        return "Use the level and recent growth/momentum change as cycle indicators."
    if fred_id in ("SP500",):
        return "Use trend and momentum transforms such as distance from moving average and 12-1 momentum."
    if fred_id in ("VIXCLS", "VXVCLS"):
        return "Use VIX level and VIX/VIX3M term-structure transforms for correction risk."
    if fred_id in ("NFCI", "ANFCI"):
        return "Use level and 3-month change to capture tightening financial conditions."
    if fred_id == "SHILLER_CAPE":
        return "Use level and trailing percentile as valuation conditioning features."
    if fred_id == "CPCE":
        return "Use z-scores or extreme percentile thresholds as contrarian sentiment features."
    if fred_id == "USREC":
        return "Use as a historical cycle label/context series, not as a forward-looking predictor."
    return "Use level and recent change; confirm transformations during model validation."


def risk_direction_description(indicator: dict[str, Any]) -> str:
    name = indicator["name"]
    fred_id = indicator["fred_id"]
    stage_key = indicator_stage_key(indicator)

    higher_risk = {
        "ICSA",
        "UNRATE",
        "SAHMREALTIME",
        "BAMLH0A0HYM2",
        "BAA10Y",
        "EBP",
        "NFCI",
        "ANFCI",
        "VIXCLS",
        "VXVCLS",
        "SHILLER_CAPE",
        "A191RL1Q225SBEA",
        "USREC",
        "ebp_level",
        "ebp_3m_chg",
        "baa_level",
        "baa_3m_chg",
        "baa_zscore_60m",
        "sahm_level",
        "icsa_yoy_pct",
        "anfci_level",
        "anfci_3m_chg",
        "cape_20yr_pct",
        "baa_zscore_24m",
        "vts_ratio",
    }
    lower_risk = {
        "T10Y2Y",
        "T10Y3M",
        "NTFS",
        "DGS10",
        "PERMIT",
        "UMCSENT",
        "USALOLITOAASTSAM",
        "SP500",
        "ntfs_level",
        "lei_6m_growth",
        "vts_slope",
        "vts_slope_zscore",
        "spx_vs_10ma",
    }
    context_dependent = {
        "DGS3MO",
        "DFF",
        "CPCE",
        "ntfs_3m_chg",
        "ffr_6m_chg",
        "m12_1_mom",
    }

    if fred_id in higher_risk or stage_key in higher_risk:
        return "Higher readings generally indicate greater pending correction or bear-market risk."
    if fred_id in lower_risk or stage_key in lower_risk:
        return "Lower readings generally indicate greater pending correction or bear-market risk."
    if fred_id in context_dependent or stage_key in context_dependent:
        return (
            "Extremes and rapid changes matter more than the level alone; "
            "interpret with the model-use transform."
        )
    if name == "Near-term forward spread 3M change":
        return "Sharp declines generally indicate greater pending bear-market risk."
    return "Use recent changes and percentile context; the level alone is not a one-way risk signal."


def indicator_raw_sources(indicator: dict[str, Any]) -> list[tuple[str, str]]:
    name = indicator["name"]
    fred_id = indicator["fred_id"]
    source_url = indicator["source_url"]

    if name in {
        "Near-term forward spread raw",
        "Near-term forward spread",
        "Near-term forward spread 3M change",
    }:
        return [
            (
                "Fed GSW zero-coupon yield curve",
                "https://www.federalreserve.gov/data/yield-curve-tables/feds200628.csv",
            ),
            ("3M Treasury yield", "https://fred.stlouisfed.org/series/DGS3MO"),
        ]
    if name == "Excess bond premium raw" or fred_id.startswith("EBP"):
        return [
            (
                "Fed excess bond premium CSV",
                "https://www.federalreserve.gov/econresdata/notes/feds-notes/2016/files/ebp_csv.csv",
            )
        ]
    if fred_id == "CPCE":
        return [("CBOE equity put/call files", "https://www.cboe.com/us/options/market_statistics/daily/")]
    if fred_id == "SHILLER_CAPE":
        return [("Robert Shiller data", "https://shillerdata.com")]
    if source_url.startswith("http"):
        return [("Primary data series", source_url)]
    return [(source_url, "")]


def indicator_functional_form(indicator: dict[str, Any]) -> Optional[str]:
    name = indicator["name"]
    computed = indicator.get("computed")

    if name == "Near-term forward spread raw":
        return (
            "NTFS = f(18m, 21m) - 3M Treasury yield, where f(18m, 21m) is derived "
            "from interpolated Fed GSW 1Y/2Y zero-coupon yields."
        )
    if name == "Near-term forward spread":
        return "ntfs_level = NTFS."
    if name == "Near-term forward spread 3M change":
        return "ntfs_3m_chg = NTFS_t - NTFS_{t-3 months}."
    if computed == "raw_monthly":
        return "Daily or higher-frequency source series resampled to month-end observations."
    if computed == "shiller_cape":
        return "CAPE = real S&P 500 price divided by trailing 10-year average real earnings."
    if computed == "sp500_vs_200dma":
        return "SPX vs 200DMA = (S&P 500 / trailing 200-day moving average - 1) x 100."
    if computed == "unemployment_vs_3yma":
        return "Unemployment trend = UNRATE - trailing 36-month moving average."
    if computed == "sp500_it_weight":
        return "Estimated IT weight = NASDAQ/S&P 500 relative ratio mapped to a market-share range."
    if computed == "bear_feature" or computed == "correction_feature":
        return f"{indicator.get('feature_column', 'feature')} from the local engineered feature set."
    return None


def key_reference_links(indicator: dict[str, Any]) -> list[tuple[str, str]]:
    category = indicator["category"]
    fred_id = indicator["fred_id"]
    refs: list[tuple[str, str]] = []

    if category == "Yield curve":
        refs.extend(
            [
                ("Estrella & Mishkin yield curve recession model", "https://www.newyorkfed.org/research/capital_markets/ycfaq.html"),
                ("Engstrom-Sharpe near-term forward spread", "https://www.federalreserve.gov/econres/feds/files/2018055pap.pdf"),
            ]
        )
    elif category in ("Credit", "Credit fast"):
        refs.extend(
            [
                ("Gilchrist-Zakrajsek credit spreads and EBP", "https://doi.org/10.1257/aer.102.4.1692"),
                ("Fed EBP data", "https://www.federalreserve.gov/econres/notes/feds-notes/recession-risk-and-the-excess-bond-premium-20160408.html"),
            ]
        )
    elif category == "Labor market":
        refs.extend(
            [
                ("Sahm rule real-time series", "https://fred.stlouisfed.org/series/SAHMREALTIME"),
                ("Initial claims FRED series", "https://fred.stlouisfed.org/series/ICSA"),
            ]
        )
    elif category in ("Equities", "Volatility", "Sentiment"):
        refs.extend(
            [
                ("Time-series momentum evidence", "https://doi.org/10.1016/j.jfineco.2011.11.003"),
                ("CBOE VIX methodology", "https://www.cboe.com/tradable_products/vix/"),
            ]
        )
    elif category == "Valuation":
        refs.extend(
            [
                ("Shiller data", "https://shillerdata.com"),
                ("Goyal-Welch return predictability", "https://doi.org/10.1093/rfs/hhm014"),
            ]
        )
    elif category == "Financial conditions":
        refs.append(("Chicago Fed NFCI/ANFCI", "https://www.chicagofed.org/research/data/nfci/current-data"))
    elif category == "Composite":
        refs.append(("OECD leading indicators", "https://fred.stlouisfed.org/series/USALOLITOAASTSAM"))
    elif category == "Growth":
        refs.append(("Real GDP growth FRED", "https://fred.stlouisfed.org/series/A191RL1Q225SBEA"))
    elif category == "Housing":
        refs.append(("Building permits FRED", "https://fred.stlouisfed.org/series/PERMIT"))
    elif category == "Cycle":
        refs.append(("NBER recession indicator FRED", "https://fred.stlouisfed.org/series/USREC"))

    if fred_id and indicator["source_url"].startswith("https://fred.stlouisfed.org"):
        refs.insert(0, ("Primary data series", indicator["source_url"]))
    return refs[:3]


def render_indicator_metadata(indicator: dict[str, Any]) -> None:
    refs = key_reference_links(indicator)
    raw_source_items = indicator_raw_sources(indicator)
    raw_source_html = "".join(
        f'<li><a href="{escape(url)}" target="_blank" rel="noopener">{escape(label)}</a></li>'
        if url
        else f"<li>{escape(label)}</li>"
        for label, url in raw_source_items
    )
    functional_form = indicator_functional_form(indicator)
    functional_form_html = (
        f"<div><strong>Functional form:</strong> {escape(functional_form)}</div>"
        if functional_form
        else ""
    )
    ref_html = (
        "".join(
            f'<li><a href="{escape(url)}" target="_blank" rel="noopener">'
            f"{escape(label)}</a></li>"
            for label, url in refs
        )
        or "<li>No reference links configured.</li>"
    )
    description_html = (
        f"<p>{escape(indicator['description'])}</p>"
        f"<p><strong>Risk direction:</strong> {escape(risk_direction_description(indicator))}</p>"
        f"<p><strong>Model use:</strong> {escape(model_use_form(indicator))}</p>"
    )

    st.markdown(
        f"""
        <div class="pwa-meta-grid">
          <div class="pwa-meta-card">
            <div class="pwa-meta-title">Identity</div>
            <div><strong>Series ID:</strong> <code>{escape(indicator['fred_id'])}</code></div>
            <div><strong>Raw data source:</strong></div>
            <ul>{raw_source_html}</ul>
            {functional_form_html}
          </div>
          <div class="pwa-meta-card">
            <div class="pwa-meta-title">Indicator Description</div>
            {description_html}
          </div>
          <div class="pwa-meta-card">
            <div class="pwa-meta-title">Key References</div>
            <ul>{ref_html}</ul>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_reference_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Indicator": item["name"],
                "Category": item["category"],
                "Description": item["description"],
                "Source": item["source_url"],
                "FRED series ID": item["fred_id"],
            }
            for item in INDICATORS
        ]
    )


def build_derived_indicator_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Indicator": item["name"],
                "Category": item["category"],
                "Description": item["description"],
                "Source": item["source_url"],
                "Feature column": item.get("feature_column", "—"),
                "Base series ID": item["fred_id"],
            }
            for item in DERIVED_INDICATORS
        ]
    )


def render_app_styles() -> None:
    st.markdown(
        """
        <style>
          :root {
            --pwa-ink: #18211b;
            --pwa-muted: #5d675f;
            --pwa-line: #dce3dd;
            --pwa-green: #173f2a;
            --pwa-green-2: #24563a;
            --pwa-gold: #b68a35;
            --pwa-bg: #f7f8f6;
          }
          .main .block-container {
            max-width: 1240px;
            padding-top: 1.5rem;
            padding-bottom: 3rem;
          }
          h1, h2, h3 {
            color: var(--pwa-ink);
            letter-spacing: 0;
          }
          div[data-testid="stTabs"] button {
            font-weight: 650;
            color: var(--pwa-muted);
          }
          div[data-testid="stTabs"] button[aria-selected="true"] {
            color: var(--pwa-green);
            border-bottom-color: var(--pwa-gold);
          }
          .pwa-header {
            border: 1px solid var(--pwa-line);
            border-radius: 8px;
            padding: 1.1rem 1.25rem;
            background: linear-gradient(135deg, #ffffff 0%, var(--pwa-bg) 100%);
            margin-bottom: 1rem;
          }
          .pwa-kicker {
            color: var(--pwa-gold);
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-bottom: 0.25rem;
          }
          .pwa-title {
            color: var(--pwa-ink);
            font-size: 2.15rem;
            font-weight: 760;
            line-height: 1.08;
            margin: 0;
          }
          .pwa-subtitle {
            color: var(--pwa-muted);
            font-size: 0.98rem;
            margin-top: 0.45rem;
            max-width: 780px;
          }
          .pwa-asof {
            color: var(--pwa-muted);
            font-size: 0.86rem;
            text-align: right;
            margin-top: 0.35rem;
          }
          .pwa-section {
            border-top: 1px solid var(--pwa-line);
            padding-top: 1rem;
            margin-top: 1.25rem;
          }
          .pwa-section h3 {
            margin-bottom: 0.15rem;
          }
          .pwa-section p {
            color: var(--pwa-muted);
            margin-top: 0;
          }
          .pwa-panel {
            border: 1px solid var(--pwa-line);
            border-radius: 8px;
            padding: 1rem 1.15rem;
            background: #ffffff;
          }
          .pwa-panel-title {
            color: var(--pwa-ink);
            font-weight: 700;
            margin-bottom: 0.25rem;
          }
          .pwa-panel-copy {
            color: var(--pwa-muted);
            margin: 0;
          }
          .pwa-meta-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.75rem;
            align-items: stretch;
            margin: 0.75rem 0 0.25rem;
          }
          .pwa-meta-card {
            border: 1px solid var(--pwa-line);
            border-radius: 8px;
            background: #ffffff;
            padding: 0.9rem 1rem;
            min-height: 190px;
            height: 100%;
            color: var(--pwa-muted);
            font-size: 0.92rem;
            line-height: 1.45;
            overflow-wrap: anywhere;
          }
          .pwa-meta-card p {
            margin: 0 0 0.6rem;
          }
          .pwa-meta-card ul {
            margin: 0;
            padding-left: 1.05rem;
          }
          .pwa-meta-card li {
            margin-bottom: 0.35rem;
          }
          .pwa-meta-card code {
            white-space: normal;
          }
          .pwa-meta-title {
            color: var(--pwa-ink);
            font-weight: 700;
            margin-bottom: 0.5rem;
          }
          @media (max-width: 900px) {
            .pwa-meta-grid {
              grid-template-columns: 1fr;
            }
          }
          div[data-testid="stMetric"] {
            border: 1px solid var(--pwa-line);
            border-radius: 8px;
            padding: 0.85rem 0.9rem;
            background: #ffffff;
          }
          div[data-testid="stMetricLabel"] p {
            color: var(--pwa-muted);
            font-size: 0.84rem;
          }
          table td, table th {
            border-color: var(--pwa-line) !important;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(title: str, caption: str) -> None:
    st.markdown(
        f"""
        <div class="pwa-section">
          <h3>{title}</h3>
          <p>{caption}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_panel(title: str, copy: str) -> None:
    st.markdown(
        f"""
        <div class="pwa-panel">
          <div class="pwa-panel-title">{title}</div>
          <p class="pwa-panel-copy">{copy}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_model_artifacts(features_file: str, output_file: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = pd.read_csv(DATA_DIR / features_file, parse_dates=["date"])
    outputs = pd.read_csv(DATA_DIR / output_file, parse_dates=["date"])
    features = features.set_index("date").sort_index()
    outputs = outputs.set_index("date").sort_index()
    return features, outputs


@st.cache_data(show_spinner=False)
def load_raw_spx() -> pd.Series:
    raw = pd.read_csv(DATA_DIR / "raw_monthly.csv", parse_dates=["date"])
    raw = raw.set_index("date").sort_index()
    spx_col = "SPX" if "SPX" in raw.columns else "SP500"
    return pd.to_numeric(raw[spx_col], errors="coerce").dropna()


def _latest_complete_row(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    available_columns = [col for col in columns if col in df.columns]
    if not available_columns:
        return pd.Series(dtype=float)
    complete = df[available_columns].dropna(how="any")
    if complete.empty:
        return pd.Series(dtype=float)
    return complete.iloc[-1]


def _latest_output_row(outputs: pd.DataFrame) -> pd.Series:
    usable = outputs.dropna(how="all")
    if usable.empty:
        return pd.Series(dtype=float)
    return usable.iloc[-1]


def _format_model_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.index, pd.DatetimeIndex):
        out = out.reset_index()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.strftime("%Y-%m-%d")
        elif pd.api.types.is_numeric_dtype(out[col]):
            if col.startswith("prob_"):
                out[col] = out[col].map(lambda x: "—" if pd.isna(x) else f"{x:.1%}")
            else:
                out[col] = out[col].map(lambda x: "—" if pd.isna(x) else f"{x:,.4f}")
    return out


def _model_output_history(outputs: pd.DataFrame) -> pd.DataFrame:
    usable = outputs.dropna(how="all")
    if usable.empty:
        return usable
    end_date = usable.index.max()
    start_date = end_date - pd.DateOffset(years=10)
    return usable.loc[usable.index >= start_date]


def _boolean_intervals(s: pd.Series) -> pd.DataFrame:
    clean = s.fillna(False).astype(bool)
    intervals: list[dict[str, pd.Timestamp]] = []
    start: Optional[pd.Timestamp] = None
    last_date: Optional[pd.Timestamp] = None
    for date, active in clean.items():
        if active and start is None:
            start = date
        elif not active and start is not None:
            intervals.append(
                {"start": start, "end": (last_date or date) + pd.DateOffset(months=1)}
            )
            start = None
        last_date = date
    if start is not None:
        intervals.append(
            {"start": start, "end": (last_date or start) + pd.DateOffset(months=1)}
        )
    return pd.DataFrame(intervals)


def render_probability_spx_chart(
    history: pd.DataFrame,
    probability_column: str,
    event_type: str,
) -> None:
    spx = load_raw_spx().rename("SPX")
    chart_data = history[[probability_column]].join(spx, how="left")
    chart_data = chart_data.dropna(subset=[probability_column, "SPX"], how="all")
    if chart_data.empty:
        st.info("No probability or SPX history is available.")
        return

    event_title = (
        "Correction probability OOS" if event_type == "correction"
        else "Bear probability OOS"
    )
    chart_reset = chart_data.reset_index()
    xmin, xmax = chart_reset["date"].min(), chart_reset["date"].max()

    base = alt.Chart(chart_reset).encode(
        x=alt.X(
            "date:T",
            title="Date",
            axis=alt.Axis(format="%Y-%m", labelAngle=-45),
        )
    )
    prob_line = base.mark_line(color="#173f2a", strokeWidth=2.5).encode(
        y=alt.Y(
            f"{probability_column}:Q",
            title=event_title,
            axis=alt.Axis(format="%"),
        ),
        tooltip=[
            alt.Tooltip("date:T", title="Date", format="%Y-%m"),
            alt.Tooltip(
                f"{probability_column}:Q",
                title=event_title,
                format=".1%",
            ),
            alt.Tooltip("SPX:Q", title="SPX", format=",.0f"),
        ],
    )
    spx_line = base.mark_line(color="#b68a35", strokeWidth=2).encode(
        y=alt.Y(
            "SPX:Q",
            title="SPX",
            axis=alt.Axis(orient="right", format=",.0f"),
        )
    )

    episodes = load_drawdown_episodes(get_fred_api_key())
    layers: list[alt.Chart] = build_drawdown_band_layers(episodes, xmin, xmax)
    layers.extend([prob_line, spx_line])
    chart = (
        alt.layer(*layers)
        .resolve_scale(y="independent", color="independent")
        .properties(height=340)
    )
    st.altair_chart(chart, use_container_width=True)


def render_model_page(
    title: str,
    caption: str,
    features_file: str,
    output_file: str,
    model_features: list[str],
    primary_probability: str,
    target_column: str,
    output_columns: Optional[list[str]] = None,
    overlay_event: Optional[str] = None,
) -> None:
    render_section_header(title, caption)

    try:
        features, outputs = load_model_artifacts(features_file, output_file)
    except Exception as exc:
        st.error(f"Could not load model artifacts: {exc}")
        return

    current_inputs = _latest_complete_row(features, model_features)
    current_outputs = _latest_output_row(outputs)

    metric_cols = st.columns(4)
    if not current_inputs.empty:
        metric_cols[0].metric("Input as-of", current_inputs.name.strftime("%Y-%m-%d"))
    else:
        metric_cols[0].metric("Input as-of", "—")

    if not current_outputs.empty:
        metric_cols[1].metric("Output as-of", current_outputs.name.strftime("%Y-%m-%d"))
        probability = current_outputs.get(primary_probability)
        metric_cols[2].metric(
            "Current probability",
            "—" if pd.isna(probability) else f"{float(probability):.1%}",
        )
        realized = current_outputs.get(target_column)
        metric_cols[3].metric(
            "Current target",
            "Pending" if pd.isna(realized) else f"{int(realized)}",
            help="Targets can be blank while the forward outcome window is still unresolved.",
        )
    else:
        metric_cols[1].metric("Output as-of", "—")
        metric_cols[2].metric("Current probability", "—")
        metric_cols[3].metric("Current target", "—")

    input_col, output_col = st.columns(2, vertical_alignment="top")
    with input_col:
        st.markdown("#### Current model inputs")
        if current_inputs.empty:
            st.info("No complete input row is available for this model.")
        else:
            input_table = current_inputs.rename("Value").to_frame()
            input_table.index.name = "Feature"
            st.dataframe(
                _format_model_table(input_table),
                hide_index=False,
                use_container_width=True,
            )

    with output_col:
        st.markdown("#### Current model outputs")
        if current_outputs.empty:
            st.info("No model output row is available.")
        else:
            output_table = current_outputs.to_frame().T
            if output_columns:
                output_table = output_table[
                    [col for col in output_columns if col in output_table.columns]
                ]
            output_table.index.name = "date"
            st.dataframe(
                _format_model_table(output_table),
                hide_index=True,
                use_container_width=True,
            )

    render_section_header(
        "Sample Model Output",
        "Saved model output history filtered to the latest 10 years available in the artifact.",
    )
    history = _model_output_history(outputs)
    if history.empty:
        st.info("No historical model output is available.")
        return

    if output_columns:
        visible_columns = [col for col in output_columns if col in history.columns]
    else:
        visible_columns = list(history.columns)
    probability_columns = [col for col in visible_columns if col.startswith("prob_")]
    chart_data = history[probability_columns].dropna(how="all").reset_index()
    if overlay_event and primary_probability in history.columns:
        render_probability_spx_chart(history, primary_probability, overlay_event)
    elif probability_columns and not chart_data.empty:
        chart_long = chart_data.melt(
            id_vars="date",
            value_vars=probability_columns,
            var_name="Series",
            value_name="Probability",
        ).dropna(subset=["Probability"])
        chart = (
            alt.Chart(chart_long)
            .mark_line()
            .encode(
                x=alt.X(
                    "date:T",
                    title="Date",
                    axis=alt.Axis(format="%Y-%m", labelAngle=-45),
                ),
                y=alt.Y(
                    "Probability:Q",
                    title="Probability",
                    axis=alt.Axis(format="%"),
                ),
                color=alt.Color("Series:N", title="Output"),
                tooltip=[
                    alt.Tooltip("date:T", title="Date", format="%Y-%m"),
                    alt.Tooltip("Series:N", title="Output"),
                    alt.Tooltip("Probability:Q", title="Probability", format=".1%"),
                ],
            )
            .properties(height=320)
        )
        st.altair_chart(chart, use_container_width=True)

    display_history = history[[col for col in visible_columns if col in history.columns]]
    st.dataframe(
        _format_model_table(display_history.tail(120)),
        hide_index=True,
        use_container_width=True,
    )


# ---------------------------------------------------------------------------
# Calibrated final-model assessment (uses bear.inference)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def _load_assessment_cached(kind: str) -> dict:
    """Fit the final calibrated model and return its assessment dict."""
    from bear.inference import load_assessment
    return load_assessment(kind)


@st.cache_data(ttl=3600, show_spinner=False)
def _load_ensemble_cached(params_file: str, oos_file: str,
                          mtime: float = 0.0) -> tuple[dict, pd.DataFrame]:
    """Load a precomputed ensemble summary + OOS series (no ML libs needed).

    Keyed on the params-file mtime so regenerated data busts the cache.
    """
    import json
    with open(DATA_DIR / params_file) as fh:
        params = json.load(fh)
    oos = pd.read_csv(DATA_DIR / oos_file, index_col=0, parse_dates=True)
    return params, oos


@st.cache_data(ttl=3600, show_spinner=False)
def _load_univariate_cached(csv_file: str, mtime: float = 0.0) -> pd.DataFrame:
    """Load a precomputed univariate leaderboard (no ML at runtime).

    Keyed on the CSV mtime so regenerating the file refreshes the table.
    """
    return pd.read_csv(DATA_DIR / csv_file)


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _risk_level(value: float, base: float) -> tuple[str, str]:
    """
    Map a reading vs its historical base to a (label, hex-color) risk level.

    Works for probabilities (base > 0) and for drawdowns (both negative):
    in either case ratio > 1 means the current reading is worse than average.
    """
    ratio = value / base if base != 0 else 0.0
    if ratio < 1.0:
        return "Low", "#22c55e"
    if ratio < 1.8:
        return "Elevated", "#eab308"
    return "High", "#ef4444"


def render_factor_table(factors: pd.DataFrame) -> None:
    """HTML factor-reading table with colored Direction cells."""
    has_hac = "P (HAC)" in factors.columns
    headers = ["Factor", "Category", "Raw value", "Z-score",
               "Weight", "Contribution"]
    if has_hac:
        headers.append("p (HAC)")
    headers.append("Direction")
    header_html = "".join(f"<th>{h}</th>" for h in headers)

    rows_html: list[str] = []
    for _, r in factors.iterrows():
        dir_color = "#ef4444" if r["Direction"] == "Bearish" else "#22c55e"
        cells = [
            f'<td>{escape(str(r["Description"]))}</td>',
            f'<td>{escape(str(r["Category"]))}</td>',
            f'<td style="text-align:right;">{r["Raw value"]:+.3f}</td>',
            f'<td style="text-align:right;">{r["Z-score"]:+.2f}</td>',
            f'<td style="text-align:right;">{r["Weight %"]:.1f}%</td>',
            f'<td style="text-align:right;">{r["Contribution"]:+.3f}</td>',
        ]
        if has_hac:
            pv = r["P (HAC)"]
            if pd.isna(pv):
                pcell = "—"
            else:
                star = " *" if pv < 0.05 else (" ." if pv < 0.10 else "")
                pcell = f"{pv:.3f}{star}"
            weight = "600" if (not pd.isna(pv) and pv < 0.05) else "400"
            cells.append(
                f'<td style="text-align:right;font-weight:{weight};">{pcell}</td>'
            )
        cells.append(
            f'<td style="background:{dir_color};color:white;font-weight:600;'
            f'text-align:center;">{escape(str(r["Direction"]))}</td>'
        )
        rows_html.append("<tr>" + "".join(cells) + "</tr>")

    table_html = f"""
    <table style="width:100%;border-collapse:collapse;font-size:0.92rem;">
      <thead><tr style="background:#f3f4f6;text-align:left;">{header_html}</tr></thead>
      <tbody>{"".join(rows_html)}</tbody>
    </table>
    <style>
      table td, table th {{ border:1px solid #e5e7eb; padding:0.45rem 0.7rem; }}
    </style>
    """
    st.markdown(table_html, unsafe_allow_html=True)


def render_probability_history_chart(
    history: pd.Series,
    base_rate: float,
    color: str,
    years: int = 10,
    value_kind: str = "probability",
    band_mode: str = "episode",
) -> None:
    """
    Altair area chart of the fitted value over the last `years` years.

    value_kind:
      "probability" — y in [0, 1], reference line = base rate
      "drawdown"    — y negative (expected drawdown), reference = historical mean
    """
    cutoff = pd.Timestamp.today() - pd.DateOffset(years=years)
    hist = history[history.index >= cutoff]
    if hist.empty:
        st.info("No history available for the selected window.")
        return

    df = hist.rename("Probability").reset_index()
    df.columns = ["date", "Probability"]

    if value_kind == "drawdown":
        value_title = "Expected drawdown"
        ymin = min(float(df["Probability"].min()) * 1.1, base_rate * 1.5)
        y_scale = alt.Scale(domain=[ymin, 0.05])
        ref_label = f"Historical mean {base_rate:.0%}"
    elif value_kind == "severity":
        value_title = "Expected drawdown"
        ymax = max(float(df["Probability"].max()) * 1.15, base_rate * 2)
        y_scale = alt.Scale(domain=[0, min(ymax, 1.0)])
        ref_label = f"Historical mean {base_rate:.0%}"
    else:
        value_title = "Probability"
        y_scale = alt.Scale(domain=[0, 1])
        ref_label = f"Base rate {base_rate:.0%}"

    line = (
        alt.Chart(df)
        .mark_area(
            line={"color": color},
            color=alt.Gradient(
                gradient="linear",
                stops=[
                    alt.GradientStop(color="#ffffff", offset=0),
                    alt.GradientStop(color=color, offset=1),
                ],
                x1=1, x2=1, y1=1, y2=0,
            ),
            opacity=0.35,
        )
        .encode(
            x=alt.X("date:T", title="Date",
                    axis=alt.Axis(format="%Y", labelAngle=0)),
            y=alt.Y("Probability:Q", title=value_title,
                    axis=alt.Axis(format="%"),
                    scale=y_scale),
            tooltip=[
                alt.Tooltip("date:T", title="Date", format="%Y-%m"),
                alt.Tooltip("Probability:Q", title=value_title, format=".1%"),
            ],
        )
    )
    rule = (
        alt.Chart(pd.DataFrame({"y": [base_rate]}))
        .mark_rule(strokeDash=[5, 4], color="#6b7280")
        .encode(y="y:Q")
    )
    rule_label = (
        alt.Chart(pd.DataFrame({"y": [base_rate], "label": [ref_label]}))
        .mark_text(align="left", dx=5, dy=-6, color="#6b7280", fontSize=11)
        .encode(y="y:Q", text="label:N")
    )
    if band_mode == "rolling12":
        band_layers = build_rolling12_band_layers(df["date"].min(), df["date"].max())
    elif band_mode == "rolling6":
        band_layers = build_rolling6_band_layers(df["date"].min(), df["date"].max())
    else:
        band_layers = build_drawdown_band_layers(
            load_drawdown_episodes(get_fred_api_key()),
            df["date"].min(),
            df["date"].max(),
        )
    chart = alt.layer(*band_layers, line, rule, rule_label).properties(height=300)
    st.altair_chart(chart, use_container_width=True)


def render_member_breakdown_table(params: dict) -> None:
    """Compact table of the four era-trained members and their OOS skill."""
    meta = params["members_meta"]
    realized_base = params["metrics"]["realized_base_rate"]
    rows_html = []
    for k in params["members"]:
        m = meta[k]
        # Era label from the title, e.g. "Bear — Model A (1920s)" -> "A · 1920s"
        letter = k[-1].upper()
        era = m["title"].split("(")[-1].rstrip(")") if "(" in m["title"] else ""
        cats = ", ".join(m["categories"])
        cur = m["current_prob"]
        cal = m.get("current_prob_calibrated", cur)
        aucc = m["oos_auc_common"]; aucn = m["oos_auc_native"]
        raw_color = "#ef4444" if cur >= m["base_rate"] else "#22c55e"
        cal_color = "#ef4444" if cal >= realized_base else "#22c55e"
        rows_html.append(
            "<tr>"
            f'<td style="font-weight:600;">Model {letter} <span style="color:#5d675f;">· {escape(era)}</span></td>'
            f'<td>train {escape(str(m["train_start"])[:4])}+</td>'
            f'<td style="text-align:center;">{m["n_factors"]}</td>'
            f'<td style="font-size:0.86rem;">{escape(cats)}</td>'
            f'<td style="text-align:right;color:#8a948c;">{cur:.1%}</td>'
            f'<td style="text-align:right;color:{cal_color};font-weight:600;">{cal:.1%}</td>'
            f'<td style="text-align:right;">{aucn:.3f}</td>'
            f'<td style="text-align:right;">{aucc:.3f}</td>'
            "</tr>"
        )
    # Footer: the ensemble = equal-weight average, then Platt-calibrated.
    ens_raw = params["current_ensemble_prob"]
    ens_cal = params["current_ensemble_prob_calibrated"]
    ens_color = "#ef4444" if ens_cal >= realized_base else "#22c55e"
    rows_html.append(
        '<tr style="background:#eef2f0;font-weight:700;">'
        '<td colspan="4">Ensemble (equal-weight average → Platt-calibrated)</td>'
        f'<td style="text-align:right;color:#8a948c;">{ens_raw:.1%}</td>'
        f'<td style="text-align:right;color:{ens_color};">{ens_cal:.1%}</td>'
        '<td colspan="2"></td>'
        "</tr>"
    )
    table_html = f"""
    <table style="width:100%;border-collapse:collapse;font-size:0.92rem;">
      <thead><tr style="background:#f3f4f6;text-align:left;">
        <th>Member</th><th>Trained</th><th>#</th><th>Factor categories</th>
        <th style="text-align:right;">Current P<br>(raw)</th>
        <th style="text-align:right;">Current P<br>(calibrated)</th>
        <th style="text-align:right;">OOS AUC<br>(native)</th>
        <th style="text-align:right;">OOS AUC<br>(2005+)</th>
      </tr></thead>
      <tbody>{"".join(rows_html)}</tbody>
    </table>
    <style>table td, table th {{ border:1px solid #e5e7eb; padding:0.45rem 0.7rem; }}</style>
    """
    st.markdown(table_html, unsafe_allow_html=True)
    st.caption(
        "Raw = each member's own logistic output. Calibrated = the same Platt map "
        "applied to every member and to the ensemble, so all sit on one comparable, "
        "history-matched scale. Color: red if calibrated probability ≥ the "
        f"{realized_base:.1%} realized base rate, green if below."
    )


def render_univariate_table(df: pd.DataFrame, base_rate: float | None = None) -> None:
    """Leaderboard of single-factor bear models, grouped by category.

    Rows whose current probability exceeds ``base_rate`` are tinted light red.
    """
    headers = ["Factor", "Data since", "Raw value", "Z-score",
               "p (HAC)", "AUC", "Probability", "Direction", "Basis for direction"]
    ncol = len(headers)
    header_html = "".join(f"<th>{h}</th>" for h in headers)
    rows_html: list[str] = []
    current_cat = None
    for _, r in df.iterrows():
        if r["Category"] != current_cat:
            current_cat = r["Category"]
            rows_html.append(
                f'<tr><td colspan="{ncol}" style="background:#eef2f0;'
                f'font-weight:700;color:#243b2f;">{escape(str(current_cat))}</td></tr>'
            )
        dir_color = "#ef4444" if r["Direction"] == "Bearish" else "#22c55e"
        # Light-red tint when the factor's probability is above the base rate.
        hot = base_rate is not None and r["Probability"] > base_rate
        bg = "background:#fdecea;" if hot else ""
        pv = r["P (HAC)"]
        if pd.isna(pv):
            pcell, weight = "—", "400"
        else:
            star = " *" if pv < 0.05 else (" ." if pv < 0.10 else "")
            pcell = f"{pv:.3f}{star}"
            weight = "600" if pv < 0.05 else "400"
        cells = [
            f'<td style="{bg}">{escape(str(r["Factor"]))}</td>',
            f'<td style="{bg}text-align:center;color:#5d675f;">{escape(str(r["Start"]))}</td>',
            f'<td style="{bg}text-align:right;">{r["Raw value"]:+.3f}</td>',
            f'<td style="{bg}text-align:right;">{r["Z-score"]:+.2f}</td>',
            f'<td style="{bg}text-align:right;font-weight:{weight};">{pcell}</td>',
            f'<td style="{bg}text-align:right;">{r["AUC"]:.3f}</td>',
            f'<td style="{bg}text-align:right;font-weight:600;">{r["Probability"]:.1%}</td>',
            f'<td style="background:{dir_color};color:white;font-weight:600;'
            f'text-align:center;">{escape(str(r["Direction"]))}</td>',
            f'<td style="{bg}font-size:0.74rem;color:#5d675f;">'
            f'{escape(str(r.get("Basis", "")))}</td>',
        ]
        rows_html.append("<tr>" + "".join(cells) + "</tr>")
    table_html = f"""
    <table style="width:100%;border-collapse:collapse;font-size:0.9rem;">
      <thead><tr style="background:#f3f4f6;text-align:left;">{header_html}</tr></thead>
      <tbody>{"".join(rows_html)}</tbody>
    </table>
    <style>table td, table th {{ border:1px solid #e5e7eb; padding:0.4rem 0.65rem; }}</style>
    """
    st.markdown(table_html, unsafe_allow_html=True)


ENSEMBLE_UI = {
    "bear": {
        "noun": "Bear",
        "params_file": "ensemble_params.json",
        "oos_file": "ensemble_oos.csv",
        "univariate_file": "univariate_bear.csv",
        "band_mode": "rolling12",
        "event": ">20% drawdown over the next 12 months",
        "uni_event": ">20% / 12-month bear",
        "band_note": "12-month rolling drawdown (correction 10–20%, bear >20%)",
        "hac_lag": 12,
        "build_cmd": "python -m bear.ensemble bear",
        "uni_cmd": "python -m bear.univariate bear",
    },
    "correction": {
        "noun": "Correction",
        "params_file": "correction_ensemble_params.json",
        "oos_file": "correction_ensemble_oos.csv",
        "univariate_file": "univariate_correction.csv",
        "band_mode": "rolling6",
        "event": ">10% drawdown within a 6-month rolling window",
        "uni_event": ">10% / 6-month correction",
        "band_note": "6-month rolling correction (index >10% below its trailing 6-month high)",
        "hac_lag": 6,
        "build_cmd": "python -m bear.ensemble correction",
        "uni_cmd": "python -m bear.univariate correction",
    },
}


FORECAST_HORIZONS = ["1", "3", "6", "12"]
FORECAST_HORIZON_LABELS = {"1": "1 Month", "3": "3 Months",
                           "6": "6 Months", "12": "1 Year"}


def render_forecast_history_chart(ens: pd.Series, realized: pd.Series,
                                  horizon_label: str, years: int = 50) -> None:
    """Line chart: ensemble forecast vs realized forward return over time."""
    cutoff = pd.Timestamp.today() - pd.DateOffset(years=years)
    df = pd.DataFrame({"Forecast": ens, "Realized": realized})
    df = df[df.index >= cutoff].dropna(how="all").reset_index()
    df.columns = ["date", "Forecast", "Realized"]
    if df.empty:
        st.info("No history available for the selected window.")
        return
    long = df.melt("date", value_vars=["Forecast", "Realized"],
                   var_name="Series", value_name="Return")
    color = alt.Color("Series:N",
                      scale=alt.Scale(domain=["Forecast", "Realized"],
                                      range=["#2563eb", "#9ca3af"]),
                      legend=alt.Legend(title=None, orient="top"))
    line = (
        alt.Chart(long)
        .mark_line(opacity=0.9)
        .encode(
            x=alt.X("date:T", title="Date",
                    axis=alt.Axis(format="%Y", labelAngle=0)),
            y=alt.Y("Return:Q", title=f"{horizon_label} return",
                    axis=alt.Axis(format="%")),
            color=color,
            tooltip=[alt.Tooltip("date:T", title="Date", format="%Y-%m"),
                     alt.Tooltip("Series:N"),
                     alt.Tooltip("Return:Q", format=".1%")],
        )
    )
    zero = (alt.Chart(pd.DataFrame({"y": [0.0]}))
            .mark_rule(color="#6b7280", strokeDash=[4, 4]).encode(y="y:Q"))
    st.altair_chart((zero + line).properties(height=300),
                    use_container_width=True)


def render_forecast_member_table(members: dict, ens_metrics: dict,
                                 ens_current: float) -> None:
    """HTML table: each model family's current forecast + OOS skill at a horizon."""
    order = ["enet", "knn", "rf", "mlp", "mean"]
    headers = ["Model", "Current forecast", "R² (OOS)", "Hit rate", "Corr", "n"]
    header_html = "".join(f"<th>{h}</th>" for h in headers)
    rows_html = []

    def _row(label, cur, m, bold=False, shade=None):
        cur_color = "#16a34a" if cur >= 0 else "#dc2626"
        r2 = m.get("r2_oos", float("nan"))
        r2cell = "—" if pd.isna(r2) else f"{r2:+.3f}"
        r2color = "#16a34a" if (not pd.isna(r2) and r2 > 0) else "#6b7280"
        hit = m.get("hit_rate", float("nan"))
        hitcell = "—" if pd.isna(hit) else f"{hit:.0%}"
        corr = m.get("corr", float("nan"))
        corrcell = "—" if pd.isna(corr) else f"{corr:+.2f}"
        bg = f"background:{shade};" if shade else ""
        fw = "700" if bold else "400"
        return (
            f'<tr style="{bg}font-weight:{fw};">'
            f'<td>{escape(label)}</td>'
            f'<td style="text-align:right;color:{cur_color};font-weight:600;">{cur:+.2%}</td>'
            f'<td style="text-align:right;color:{r2color};">{r2cell}</td>'
            f'<td style="text-align:right;">{hitcell}</td>'
            f'<td style="text-align:right;">{corrcell}</td>'
            f'<td style="text-align:right;">{m.get("n", "—")}</td>'
            f'</tr>'
        )

    rows_html.append(_row("Ensemble (equal-weight)", ens_current, ens_metrics,
                          bold=True, shade="#eef2ff"))
    for mk in order:
        if mk not in members:
            continue
        m = members[mk]
        label = m["label"] + (" — benchmark" if mk == "mean" else "")
        rows_html.append(_row(label, m["current"], m))

    table_html = f"""
    <table style="width:100%;border-collapse:collapse;font-size:0.92rem;">
      <thead><tr style="background:#f3f4f6;text-align:left;">{header_html}</tr></thead>
      <tbody>{"".join(rows_html)}</tbody>
    </table>
    <style>table td, table th {{ border:1px solid #e5e7eb; padding:0.45rem 0.7rem; }}</style>
    """
    st.markdown(table_html, unsafe_allow_html=True)


def render_forecast_predictor_table(uni: pd.DataFrame) -> None:
    """Formatted predictor leaderboard (univariate OOS skill by horizon)."""
    disp = pd.DataFrame({
        "Predictor": uni["Description"],
        "Family": uni["Family"],
        "Data since": uni["Data since"],
    })
    for h in FORECAST_HORIZONS:
        disp[f"R² {h}m"] = uni[f"R2_OS_{h}m"].map(
            lambda v: "—" if pd.isna(v) else f"{v:+.3f}")
        disp[f"Hit {h}m"] = uni[f"hit_{h}m"].map(
            lambda v: "—" if pd.isna(v) else f"{v:.0%}")
    st.dataframe(disp, hide_index=True, use_container_width=True)


def render_forecast_page() -> None:
    """Market Forecast tab: ensemble point-forecast of S&P 500 return at
    1 / 3 / 6 / 12 months, with member breakdown and predictor leaderboard."""
    render_section_header(
        "Market Forecast — Return Ensemble",
        "Point forecast of the S&P 500 total price return over the next 1, 3, 6, "
        "and 12 months. Each horizon is an equal-weight combination of four "
        "model families (elastic net, k-nearest-neighbor, random forest, neural "
        "net) trained on a shared macro / valuation / trend predictor set. "
        "Forecasts are walk-forward out-of-sample; skill is measured against the "
        "prevailing historical mean (Campbell-Thompson R²_OS).",
    )

    try:
        params, oos = _load_ensemble_cached(
            "forecast_params.json", "forecast_ensemble_oos.csv",
            _mtime(DATA_DIR / "forecast_params.json"))
    except Exception as exc:
        st.error(
            f"Could not load the market-forecast ensemble: {exc}\n\n"
            "Run `python -m forecast.models` then `python -m forecast.ensemble` "
            "to generate the artifacts."
        )
        return

    by_h = params["by_horizon"]
    as_of = params.get("as_of", "—")

    # -- Headline: four horizon cards (cumulative forecast, annualized in help) --
    cols = st.columns(4)
    for i, h in enumerate(FORECAST_HORIZONS):
        m = by_h[h]
        f = m["current_forecast"]
        ann = m["current_forecast_annualized"]
        bench = m.get("benchmark_mean")
        delta = (f - bench) if bench is not None else None
        cols[i].metric(
            f"{FORECAST_HORIZON_LABELS[h]} expected return",
            f"{f:+.1%}",
            delta=(f"{delta:+.1%} vs avg" if delta is not None else None),
            help=f"Cumulative {h}-month forecast (annualized {ann:+.1%}). "
                 f"Equal-weight ensemble of the four model families.",
        )

    st.caption(
        "As of **" + str(as_of) + "**.  Out-of-sample R²_OS vs the prevailing mean: "
        + " · ".join(
            f"{FORECAST_HORIZON_LABELS[h]} **{by_h[h]['ensemble_metrics']['r2_oos']:+.3f}**"
            for h in FORECAST_HORIZONS)
        + ".  Positive R²_OS means the ensemble beats simply forecasting the "
        "historical average. As the equity-premium literature (Welch-Goyal 2008) "
        "shows, short-horizon return predictability is weak — value concentrates "
        "at longer horizons and in combination, so read the 1-month number with "
        "caution."
    )

    # -- Forecast vs realized chart (per-horizon selector) --
    render_section_header(
        "Forecast vs Realized — Out-of-Sample, Walk-Forward",
        "Blue = the ensemble's expanding-window forecast (each model re-fit on "
        "prior data only); grey = the return that subsequently realized.",
    )
    sel = st.radio("Horizon", options=FORECAST_HORIZONS,
                   format_func=lambda h: FORECAST_HORIZON_LABELS[h],
                   horizontal=True, key="forecast_horizon")
    if f"ens_{sel}m" in oos.columns:
        render_forecast_history_chart(
            oos[f"ens_{sel}m"].dropna(), oos[f"real_{sel}m"],
            FORECAST_HORIZON_LABELS[sel])

    # -- Member breakdown for the selected horizon --
    render_section_header(
        "Ensemble Members",
        "Current forecast and out-of-sample skill of each model family at the "
        "selected horizon. The ensemble is their equal-weight average; the "
        "prevailing-mean row is the benchmark each member is scored against.",
    )
    msel = by_h[sel]
    render_forecast_member_table(
        msel["members"], msel["ensemble_metrics"], msel["current_forecast"])

    # -- Predictor leaderboard --
    render_section_header(
        "Predictor Leaderboard",
        "Each row is a single-predictor linear forecast, walk-forward "
        "out-of-sample. R² is the Campbell-Thompson out-of-sample R²_OS vs the "
        "prevailing mean; Hit is the directional hit-rate. Grouped by family. "
        "Return predictability is intrinsically weak, so members earn their place "
        "in the ensemble by out-of-sample skill, not statistical significance.",
    )
    try:
        uni = _load_univariate_cached(
            "univariate_forecast.csv", _mtime(DATA_DIR / "univariate_forecast.csv"))
        render_forecast_predictor_table(uni)
    except Exception as exc:
        st.info("Predictor leaderboard unavailable — run "
                f"`python -m forecast.univariate` to generate it. ({exc})")


@st.cache_data(ttl=3600, show_spinner=False)
def _load_json_cached(filename: str, mtime: float = 0.0) -> dict:
    import json
    with open(DATA_DIR / filename) as fh:
        return json.load(fh)


def _tilt_color(tilt: str) -> str:
    return {"Overweight": "#16a34a", "Underweight": "#dc2626"}.get(tilt, "#6b7280")


def _render_enet_betas(rows: list) -> None:
    """One Elastic Net beta table (· = factor shrunk to exactly zero)."""
    macro = ["oil", "dur", "unemp", "infl"]
    cmax = {c: max((abs(r[c]) for r in rows), default=1.0) or 1.0 for c in macro}
    hdr = "".join(f"<th>{h}</th>" for h in
                  ["Sector", "Mkt β", "Oil", "Dur", "Unemp", "Infl", "R²"])

    def cell(v, cm):
        if abs(v) < 1e-9:
            return "<td style='color:#c8c8c8;'>·</td>"
        a = min(abs(v) / cm, 1.0) * 0.5
        bg = f"rgba(34,197,94,{a:.2f})" if v > 0 else f"rgba(239,68,68,{a:.2f})"
        return f"<td style='background:{bg};'>{v:+.2f}</td>"

    body = []
    for r in rows:
        cells = "".join(cell(r[c], cmax[c]) for c in macro)
        body.append(
            f"<tr><td>{escape(r['sector'])}</td>"
            f"<td style='font-weight:600;'>{r['mkt']:.2f}</td>"
            f"{cells}<td>{r['r2']*100:.0f}%</td></tr>")
    st.markdown(
        f"<table style='width:100%;border-collapse:collapse;font-size:0.82rem;'>"
        f"<thead><tr style='background:#f3f4f6;'>{hdr}</tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table>"
        f"<style>table td, table th {{ border:1px solid #e5e7eb; padding:0.3rem 0.45rem; text-align:center; }}</style>",
        unsafe_allow_html=True)


def render_sector_rotation_page() -> None:
    """Sector Rotation tab: (1) cross-sectional momentum tilt — the one model that
    survived every leak-free test — and (2) descriptive factor-risk betas."""
    render_section_header(
        "Sector Rotation",
        "Two independent views of the 11 GICS sectors. The tilt is the only "
        "return signal that held up out-of-sample in testing — 12-month "
        "cross-sectional price momentum (best at a ~6-month horizon). The factor "
        "panel below is descriptive RISK analysis (current betas), not a forecast.",
    )
    try:
        p = _load_json_cached("sector_rotation_params.json",
                              _mtime(DATA_DIR / "sector_rotation_params.json"))
    except Exception as exc:
        st.error(f"Could not load sector-rotation artifacts: {exc}\n\n"
                 "Run `python -m forecast.sector_rotation` to generate them.")
        return

    mo = p["momentum"]
    st.caption(f"As of **{p['as_of']}**.  Tilt signal: {mo['signal']} "
               f"(forecast horizon {mo['forecast_horizon']}).")

    # ---- Momentum ranking + tilt ----
    headers = ["Rank", "Sector", "ETF", "12m return", "Momentum z", "Tilt", "Weight"]
    hrow = "".join(f"<th>{h}</th>" for h in headers)
    body = []
    for r in mo["ranking"]:
        c = _tilt_color(r["tilt"])
        body.append(
            f"<tr><td>{r['rank']}</td>"
            f"<td>{escape(r['sector'])}</td><td>{escape(r['ticker'])}</td>"
            f"<td>{r['mom_12m']:+.1%}</td>"
            f"<td>{r['score']:+.2f}</td>"
            f"<td style='background:{c};color:white;font-weight:600;'>{r['tilt']}</td>"
            f"<td>{r['weight']:+.2f}</td></tr>")
    st.markdown(
        f"<table style='width:100%;border-collapse:collapse;font-size:0.92rem;'>"
        f"<thead><tr style='background:#f3f4f6;'>{hrow}</tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table>"
        f"<style>table td, table th {{ border:1px solid #e5e7eb; padding:0.4rem 0.65rem; text-align:center; }}</style>",
        unsafe_allow_html=True)

    s = mo.get("stats", {})
    if s:
        st.caption(
            f"Signal skill at {s['horizon']} (walk-forward, no look-ahead): rank "
            f"**IC {s['ic_full']:+.3f}** full-period · **{s['ic_test']:+.3f}** since 2020; "
            f"rank-weighted long/short info ratio **{s['ls_ir_full']:.2f}**. A real but "
            f"**modest** edge — best used as an overweight/underweight overlay, not a "
            f"standalone strategy. Weight = rank-weighted dollar-neutral tilt "
            f"(positive = overweight, negative = underweight)."
        )

    # ---- Factor-risk betas (Elastic Net, two windows) ----
    render_section_header(
        "Factor Risk — Sector Betas (Elastic Net, descriptive)",
        "Each sector's monthly return regressed on five factors with Elastic Net, "
        "which shrinks weak exposures to exactly zero (· = dropped). Natural units: "
        "Market β is dimensionless (~1 = moves with the S&P); the others are % sector "
        "return per +1% crude oil (Oil), +1pp in the 10y yield (Dur), +1pp in the "
        "unemployment rate (Unemp), and +1pp in CPI inflation (Infl). The two windows "
        "show how exposures migrate over time.",
    )
    try:
        lasso = _load_json_cached("factor_lasso_params.json",
                                  _mtime(DATA_DIR / "factor_lasso_params.json"))
    except Exception as exc:
        st.info("Regularized factor betas unavailable — run "
                f"`python -m forecast.factor_lasso`. ({exc})")
        return
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Full history** (inception → today)")
        _render_enet_betas(lasso["full"]["enet"])
    with c2:
        st.markdown("**Post-pandemic** (2020 → today)")
        _render_enet_betas(lasso["postcovid"]["enet"])
    st.caption(
        "Migration: **Market β** and **Energy↔oil** are stable across both windows "
        "(structural). The macro factors mostly switched on after 2020 — growth "
        "(**Technology, Semiconductors**) picked up negative **Duration** and "
        "**Inflation** betas (long-duration behavior, absent full-history); "
        "**Energy/Financials** strengthened positive rate betas; **Real Estate / "
        "Utilities** stay rate-negative in both (true bond proxies). · = the "
        "regularizer judged that exposure too weak to keep. Descriptive risk only — "
        "independent of the momentum tilt above."
    )


def render_ensemble_page(family: str, title: str, caption: str) -> None:
    """Primary ensemble page (bear or correction): calibrated probability +
    per-member breakdown + walk-forward history + univariate leaderboard."""
    ui = ENSEMBLE_UI[family]
    noun = ui["noun"]
    render_section_header(title, caption)

    try:
        params, oos = _load_ensemble_cached(
            ui["params_file"], ui["oos_file"],
            _mtime(DATA_DIR / ui["params_file"]))
        members = {k: _load_assessment_cached(k) for k in params["members"]}
    except Exception as exc:
        st.error(
            f"Could not load the {noun.lower()} ensemble assessment: {exc}\n\n"
            f"Run `{ui['build_cmd']}` to generate the ensemble artifacts."
        )
        return

    prob_raw = params["current_ensemble_prob"]
    prob     = params["current_ensemble_prob_calibrated"]
    base     = params["metrics"]["realized_base_rate"]
    as_of    = pd.Timestamp(params["as_of"])
    level, color = _risk_level(prob, base)

    # -- Top metric row (calibrated probability is the headline) --
    m = st.columns(4)
    m[0].metric(f"{noun} probability (ensemble)", f"{prob:.1%}",
                help="Equal-weight average of the four era-trained members, "
                     "Platt-recalibrated to the realized base rate.")
    m[1].metric("As of", as_of.strftime("%Y-%m-%d"))
    m[2].metric("Historical base rate", f"{base:.1%}",
                help=f"Realized frequency of a {ui['event']} over the OOS span.")
    m[3].markdown(
        f"""
        <div style="border:1px solid #dce3dd;border-radius:8px;
                    padding:0.85rem 0.9rem;background:#ffffff;">
          <div style="color:#5d675f;font-size:0.84rem;">Risk level</div>
          <div style="background:{color};color:white;font-weight:700;
                      text-align:center;border-radius:6px;padding:0.3rem;
                      margin-top:0.35rem;font-size:1.05rem;">{level}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        f"Raw equal-weight average **{prob_raw:.1%}** → Platt-calibrated **{prob:.1%}** "
        f"(averaging is under-confident; calibration map fit on realized OOS history). "
        f"Ensemble OOS AUC **{params['metrics']['ensemble_auc_common']:.3f}** on the common "
        f"2005+ window, **{params['metrics']['ensemble_auc_full']:.3f}** across the full "
        f"1950→ span."
    )

    # -- Member breakdown --
    render_section_header(
        "Ensemble Members",
        "Four logistic models, each trained on the longest history its feature set "
        "permits, every coefficient Newey-West HAC-significant. Each member votes "
        "once it is out-of-sample; the ensemble is their equal-weight average.",
    )
    render_member_breakdown_table(params)

    # -- Ensemble probability history (walk-forward OOS, full coverage) --
    render_section_header(
        f"{noun} Probability History — Out-of-Sample, Walk-Forward (Last 50 Years)",
        "Equal-weight ensemble of the members' honest expanding-window predictions "
        "(each member re-fit on prior data only). Growing membership: Model A from "
        f"1950, B from 1970, C from 1985, D from 2005. Shaded bands mark the {ui['band_note']}.",
    )
    render_probability_history_chart(
        oos["ensemble"].dropna(), base, color, years=50,
        value_kind="probability", band_mode=ui["band_mode"])

    # -- Per-member factor detail (expanders) --
    render_section_header(
        "Member Factor Readings",
        "Current factor values, weights, and HAC p-values for each member model.",
    )
    for k in params["members"]:
        a = members[k]
        meta = params["members_meta"][k]
        with st.expander(
            f"{a['title']}  —  current P {meta['current_prob']:.1%}  ·  "
            f"OOS AUC {meta['oos_auc_native']:.3f}"
        ):
            render_factor_table(a["factors"])
            render_model_formula(a)

    # -- Univariate factor leaderboard --
    render_section_header(
        "Univariate Factor Models",
        f"Each row is a single-factor logistic model of the {ui['uni_event']} "
        "event, fit on all available history. Columns show the latest raw value, "
        "data-history start, standardized z-score, Newey-West HAC p-value, in-sample "
        f"AUC, the model's current {noun.lower()} probability, and the direction of its "
        "current push. Grouped by category (strongest category first; AUC-ranked within).",
    )
    try:
        uni = _load_univariate_cached(
            ui["univariate_file"], _mtime(DATA_DIR / ui["univariate_file"]))
        render_univariate_table(uni, base_rate=base)
        st.caption(
            f"`*` p<0.05  `.` p<0.10 (HAC, max lag {ui['hac_lag']}). Factors include "
            f"combinations (e.g. VIX/VIX3M ratio, 10y−3m spread, price vs 10-month MA). "
            f"Rows tinted light red have a current probability above the {base:.1%} "
            f"historical base rate. A single factor's probability is naturally "
            f"noisier than the ensemble's."
        )
        st.markdown(
            f"""
<div style="font-size:0.8rem;color:#5d675f;border-top:1px solid #e5e7eb;
            margin-top:0.6rem;padding-top:0.5rem;">
<b>How the Direction is determined.</b> The method is the same for every factor.
Each factor is fit as a single-variable <i>unconstrained</i> logistic regression
on the {noun.lower()} event ({ui['event']}), so the model <i>learns from history</i>
the factor's relationship to {noun.lower()} risk. The small-font
<b>Basis for direction</b> column states the call when the factor sits <i>above</i>
its historical average: <i>Bearish above avg</i> (positive coefficient) or
<i>Bullish above avg</i> (negative coefficient); the opposite call applies when it
is below average. <b>Direction</b> combines that rule with the <b>Z-score</b>
(where the latest reading sits): it is the sign of (learned coefficient) ×
(z-score) — <b style="color:#ef4444;">Bearish</b> when the current reading pushes
the modeled probability above what the factor's mean implies,
<b style="color:#22c55e;">Bullish</b> otherwise. So a <i>"Bearish above avg"</i>
factor reads <b style="color:#22c55e;">Bullish</b> today whenever its z-score is
negative (currently below average), and vice-versa. The <b>Probability</b> is that
one-factor model's fitted {noun.lower()} probability at the latest reading;
<b>z-score</b> standardizes the raw value over the factor's full history
(<b>Data since</b>).
</div>
""",
            unsafe_allow_html=True,
        )
    except Exception as exc:
        st.info(f"Univariate leaderboard unavailable — run "
                f"`{ui['uni_cmd']}` to generate it. ({exc})")


def render_calibrated_model_page(kind: str, title: str, caption: str) -> None:
    """Full assessment page for a model: probability, factors, 10yr history."""
    render_section_header(title, caption)

    try:
        a = _load_assessment_cached(kind)
    except Exception as exc:
        st.error(
            f"Could not load the {title} assessment: {exc}\n\n"
            "Ensure the bear pipeline artifacts exist "
            "(run phases 1–5 to generate the feature and target CSVs)."
        )
        return

    prob       = a["current_prob"]
    base       = a["base_rate"]
    as_of      = a["as_of"]
    value_kind = a.get("value_kind", "probability")
    is_dd      = value_kind == "drawdown"
    is_sev     = value_kind == "severity"
    dd_like    = is_dd or is_sev          # drawdown-flavored labels
    level, color = _risk_level(prob, base)

    metric_label = "Expected drawdown" if dd_like else f"{a['title']} probability"
    base_label   = "Historical mean drawdown" if dd_like else "Historical base rate"
    base_help    = ("Average realized 12-month forward drawdown severity in the sample."
                    if dd_like else
                    "Unconditional frequency of the event in the training sample.")

    # -- Top metric row --
    m = st.columns(4)
    m[0].metric(metric_label, f"{prob:.1%}", help=a["subtitle"])
    m[1].metric("As of", as_of.strftime("%Y-%m-%d"))
    m[2].metric(base_label, f"{base:.1%}", help=base_help)
    m[3].markdown(
        f"""
        <div style="border:1px solid #dce3dd;border-radius:8px;
                    padding:0.85rem 0.9rem;background:#ffffff;">
          <div style="color:#5d675f;font-size:0.84rem;">Risk level</div>
          <div style="background:{color};color:white;font-weight:700;
                      text-align:center;border-radius:6px;padding:0.3rem;
                      margin-top:0.35rem;font-size:1.05rem;">{level}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if is_sev:
        st.caption(
            f"**Expected drawdown severity** over the next {a['horizon']} months, "
            f"from a weight-constrained *fractional logistic* regression on the "
            f"rolling 12-month drawdown (each factor 10–40% weight; signs fixed to "
            f"economic priors). Output $\\sigma(z)\\in[0,1]$ is the expected "
            f"drawdown as a fraction of the prior peak."
        )
    elif is_dd:
        st.caption(
            f"**Expected max drawdown** over the next {a['horizon']} months, from a "
            f"weight-constrained linear regression on the rolling 12-month drawdown "
            f"(each factor 10–40% weight; signs fixed to economic priors)."
        )
    elif a.get("unconstrained"):
        st.caption(
            f"Probability of a **{a['subtitle']}**, from an *unconstrained* logistic "
            f"regression (free signs and weights, maximum likelihood) on the long "
            f"1960+ sample. Factors were chosen by exhaustive search to maximize "
            f"in-sample AUC; inference uses Newey-West HAC."
        )
    else:
        st.caption(
            f"Probability of a **{a['subtitle']}**, estimated by a weight-constrained "
            f"logistic model (signs fixed to economic priors; probabilities calibrated "
            f"to the true base rate)."
        )

    # -- Factor readings --
    if dd_like:
        contrib_desc = ("contribution to the expected drawdown (log-odds of "
                        "severity). Bearish = larger drawdown.")
    else:
        contrib_desc = ("contribution to the current log-odds. "
                        "Bearish = pushes probability up.")
    render_section_header(
        "Current Factor Readings",
        "Each factor's latest value, standardized score, model weight, and its "
        + contrib_desc,
    )
    render_factor_table(a["factors"])
    st.caption(
        f"**p (HAC)** is the Newey-West heteroskedasticity- and "
        f"autocorrelation-consistent p-value (max lag = {a.get('hac_maxlags', a['horizon'])} "
        f"months), correcting for the serial correlation from overlapping rolling "
        f"windows.  `*` p<0.05  `.` p<0.10."
    )

    # -- Historical curves: in-sample (top row) and out-of-sample (bottom row) --
    # Bear+ shades the 12-month rolling drawdown classification (its dependent
    # variable); the other tabs shade realized peak-to-recovery episodes.
    if a["kind"] == "bearplus":
        band_mode = "rolling12"
    elif a["kind"] in ("correctionplus", "correction"):
        band_mode = "rolling6"
    else:
        band_mode = "episode"

    if band_mode == "rolling12":
        band_note = (" Shaded bands mark the 12-month rolling drawdown (index vs its "
                     "trailing 12-month high): correction (10–20%) or bear (>20%).")
    elif band_mode == "rolling6":
        band_note = (" Shaded bands mark the 6-month rolling correction (index >10% "
                     "below its trailing 6-month high).")
    else:
        band_note = ""
    noun = "Drawdown" if dd_like else "Probability"
    render_section_header(
        f"{noun} History — In-Sample Fit (Last 50 Years)",
        "Final-model fitted values applied across all history "
        "(parameters estimated on the full sample). "
        "Dashed line marks the historical mean." + band_note,
    )
    render_probability_history_chart(a["history"], base, color, years=50,
                                     value_kind=value_kind, band_mode=band_mode)

    history_oos = a.get("history_oos")
    if history_oos is not None and not history_oos.empty:
        render_section_header(
            f"{noun} History — Out-of-Sample, Walk-Forward (Last 50 Years)",
            "Honest expanding-window estimate: at each month the model is "
            "re-fit on prior data only, then predicts that month. No look-ahead.",
        )
        render_probability_history_chart(history_oos, base, color, years=50,
                                         value_kind=value_kind, band_mode=band_mode)
    else:
        st.info("Out-of-sample series unavailable — run `python -m bear.inference` "
                "to generate the walk-forward history.")

    # -- Mathematical formulation --
    render_model_formula(a)


def render_model_formula(a: dict) -> None:
    """Render the fitted model in LaTeX at the bottom of the page."""
    feats     = a["features"]
    coef      = a["coef"]
    intercept = a["intercept"]
    mu        = a["mu"]
    sigma     = a["sigma"]
    labels    = a["labels"]
    min_w     = a.get("min_w", 0.0)
    max_w     = a.get("max_w", 1.0)
    value_kind = a.get("value_kind", "probability")
    is_dd     = value_kind == "drawdown"
    is_sev    = value_kind == "severity"
    n         = len(feats)

    if is_sev:
        subtitle_math = ("Weight-constrained fractional logistic regression on "
                         "standardized factors (quasi-binomial), signs fixed to "
                         "economic priors.")
    elif is_dd:
        subtitle_math = ("Weight-constrained linear regression on standardized "
                         "factors, fitted by least squares with signs fixed to "
                         "economic priors.")
    else:
        subtitle_math = ("Weight-constrained logistic regression on standardized "
                         "factors, fitted by maximum likelihood with signs fixed "
                         "to economic priors.")
    render_section_header("Model — Mathematical Formulation", subtitle_math)

    if is_sev:
        # Fractional logistic: severity s_t = -MDD_t in [0,1], E[s_t]=sigma(z_t)
        st.latex(
            r"s_t \;=\; -\,\mathrm{MDD}_{t} \;=\; "
            r"-\min_{t < u \le t+" + str(a["horizon"]) + r"}"
            r"\left(\frac{P_u}{\max_{t<v\le u}P_v} - 1\right) \;\in\; [0,1]"
        )
        st.latex(
            r"\hat{s}_t \;=\; \mathbb{E}[s_t] \;=\; \sigma(z_t) "
            r"\;=\; \frac{1}{1 + e^{-z_t}}"
        )
        st.latex(
            r"z_t \;=\; \beta_0 + \sum_{i=1}^{" + str(n) + r"} \beta_i\,\tilde{x}_{i,t},"
            r"\qquad \tilde{x}_{i,t} \;=\; \frac{x_{i,t} - \mu_i}{\sigma_i}"
        )
        parts = [f"{intercept:+.3f}"]
        for i, c in enumerate(coef, start=1):
            parts.append(f"{c:+.3f}\\,\\tilde{{x}}_{{{i}}}")
        st.latex(r"z_t \;=\; " + " ".join(parts))
    elif is_dd:
        # Linear (identity link): predicted forward max drawdown
        st.latex(
            r"\widehat{\mathrm{MDD}}_{t} \;=\; "
            r"\mathbb{E}\!\left[\min_{t < s \le t+" + str(a["horizon"]) + r"}"
            r"\left(\frac{P_s}{\max_{t<u\le s}P_u} - 1\right)\right]"
            r" \;=\; \hat{y}_t"
        )
        st.latex(
            r"\hat{y}_t \;=\; \beta_0 + \sum_{i=1}^{" + str(n) + r"} \beta_i\,\tilde{x}_{i,t},"
            r"\qquad \tilde{x}_{i,t} \;=\; \frac{x_{i,t} - \mu_i}{\sigma_i}"
        )
        parts = [f"{intercept:+.4f}"]
        for i, c in enumerate(coef, start=1):
            parts.append(f"{c:+.4f}\\,\\tilde{{x}}_{{{i}}}")
        st.latex(r"\hat{y}_t \;=\; " + " ".join(parts))
    else:
        # Logistic link
        thresh = "20\\%" if a["horizon"] == 12 else "10\\%"
        st.latex(
            r"\hat{p}_t \;=\; \Pr\!\left(\text{drawdown} > " + thresh
            + r"\ \text{within " + str(a["horizon"]) + r"m}\right)"
            r" \;=\; \sigma(z_t) \;=\; \frac{1}{1 + e^{-z_t}}"
        )
        st.latex(
            r"z_t \;=\; \beta_0 + \sum_{i=1}^{" + str(n) + r"} \beta_i\,\tilde{x}_{i,t},"
            r"\qquad \tilde{x}_{i,t} \;=\; \frac{x_{i,t} - \mu_i}{\sigma_i}"
        )
        parts = [f"{intercept:+.3f}"]
        for i, c in enumerate(coef, start=1):
            parts.append(f"{c:+.3f}\\,\\tilde{{x}}_{{{i}}}")
        st.latex(r"z_t \;=\; " + " ".join(parts))

    # 4) Weight definition (and constraints, unless unconstrained)
    if a.get("unconstrained"):
        st.latex(
            r"w_i \;=\; \frac{|\beta_i|}{\sum_{j=1}^{" + str(n) + r"} |\beta_j|}"
            r"\quad\text{(relative importance; signs and weights unconstrained)}"
        )
    else:
        st.latex(
            r"w_i \;=\; \frac{|\beta_i|}{\sum_{j=1}^{" + str(n) + r"} |\beta_j|}"
            r"\,, \qquad " + f"{min_w*100:.0f}\\% \\le w_i \\le {max_w*100:.0f}\\%"
        )

    # 5) Variable legend (symbols in LaTeX)
    header = "| Symbol | Factor | $\\mu_i$ | $\\sigma_i$ | $\\beta_i$ | $w_i$ |\n"
    sep    = "|---|---|---:|---:|---:|---:|\n"
    rows = []
    for i, f in enumerate(feats, start=1):
        desc = labels[f][0]
        w_i = abs(coef[i - 1]) / sum(abs(c) for c in coef) * 100
        rows.append(
            f"| $x_{{{i}}}$ | {desc} | {mu[f]:.3f} | {sigma[f]:.3f} | "
            f"{coef[i-1]:+.3f} | {w_i:.1f}% |"
        )
    st.markdown(header + sep + "\n".join(rows))

    # 6) Rolling-window definition of the dependent variable
    h = a["horizon"]
    st.latex(
        r"\mathrm{MDD}_{t} \;=\; \min_{t < u \le t+" + str(h) + r"}"
        r"\left(\frac{P_u}{\max_{t < v \le u} P_v} - 1\right)"
        r"\quad\text{(rolling " + str(h) + r"-month forward window)}"
    )
    if is_sev:
        rolling_note = (
            f"The dependent variable is the **{h}-month rolling drawdown severity** "
            f"$s_t=-\\mathrm{{MDD}}_t$ (worst peak-to-trough decline over the rolling "
            f"{h}-month forward window)."
        )
    else:
        thr_pct = "20" if h == 12 else "10"
        evt_word = "bear market" if h == 12 else "correction"
        rolling_note = (
            f"The dependent variable is a binary **{h}-month rolling {evt_word}**: 1 "
            f"when the rolling {h}-month forward drawdown exceeds {thr_pct}% "
            f"($\\mathrm{{MDD}}_t \\le -{thr_pct}\\%$)."
        )

    st.caption(
        rolling_note
        + f"  Factors are standardized using their training-sample mean $\\mu_i$ and "
        f"standard deviation $\\sigma_i$. Coefficient inference uses Newey-West "
        f"**HAC** standard errors (max lag = {a.get('hac_maxlags', h)} months) to "
        f"correct for the autocorrelation induced by overlapping rolling windows."
    )


# ---------------------------------------------------------------------------
# Drawdown episode shading shared by all charts
# ---------------------------------------------------------------------------

CORRECTION_SHADE = "#f9a8d4"  # pink
BEAR_SHADE       = "#ef4444"  # red


@st.cache_data(ttl=3600, show_spinner=False)
def load_spx_drawdown_series(api_key: Optional[str] = None) -> pd.Series:
    """
    Build the SPX history used for all drawdown labels.

    Monthly local SPX is used for the long pre-daily history. When a FRED API
    key is available, daily SP500 is used from its first available date forward.
    """
    path = Path(__file__).resolve().parent / "data" / "raw_monthly.csv"
    if not path.exists():
        return pd.Series(dtype=float)

    raw = pd.read_csv(path, index_col=0, parse_dates=True)
    monthly_col = "SPX" if "SPX" in raw.columns else "SP500"
    monthly = pd.to_numeric(raw[monthly_col], errors="coerce").dropna().sort_index()
    monthly.index.name = "date"

    if not api_key:
        return monthly

    try:
        daily = fetch_fred_series("SP500", api_key).dropna().sort_index()
    except Exception:
        return monthly

    if daily.empty:
        return monthly

    daily.index.name = "date"
    monthly_prefix = monthly[monthly.index < daily.index.min()]
    combined = pd.concat([monthly_prefix, daily])
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    return combined.dropna()


def _spx_frequency_for_date(date: pd.Timestamp, daily_start: Optional[pd.Timestamp]) -> str:
    if daily_start is not None and pd.Timestamp(date) >= daily_start:
        return "daily"
    return "monthly"


def _format_drawdown_date(date: pd.Timestamp, frequency: str) -> str:
    fmt = "%Y-%m-%d" if frequency == "daily" else "%Y-%m"
    return pd.Timestamp(date).strftime(fmt)


def _month_duration_label(start_date: pd.Timestamp, end_date: pd.Timestamp) -> str:
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    months = (end.year - start.year) * 12 + (end.month - start.month)
    if end.day > start.day:
        months += 1
    return f"{max(int(months), 0)} months"


@st.cache_data(ttl=3600, show_spinner=False)
def load_drawdown_episodes(api_key: Optional[str] = None) -> list[dict[str, Any]]:
    """
    Identify historical S&P 500 drawdown bands from the shared SPX series.

    Correction bands start when drawdown first reaches -10% and end at full
    recovery to the prior high. Bear bands start when drawdown first reaches
    -20% and end at that same full-recovery point. A bear episode therefore
    overlaps the preceding correction episode.

    Returns dictionaries with event start, peak, trough, recovery, drawdown,
    and duration fields. Uses daily SP500 where available and monthly SPX
    for the older history.
    """
    spx = load_spx_drawdown_series(api_key)
    if spx.empty:
        return []

    daily_start: Optional[pd.Timestamp] = None
    if api_key:
        try:
            daily = fetch_fred_series("SP500", api_key).dropna().sort_index()
            if not daily.empty:
                daily_start = pd.Timestamp(daily.index.min())
        except Exception:
            daily_start = None

    episodes: list[dict[str, Any]] = []
    peak_date = pd.Timestamp(spx.index[0])
    peak_value = float(spx.iloc[0])
    correction_start: Optional[pd.Timestamp] = None
    bear_start: Optional[pd.Timestamp] = None
    trough_date: Optional[pd.Timestamp] = None
    trough_drawdown = 0.0

    for raw_date, raw_value in spx.items():
        date = pd.Timestamp(raw_date)
        value = float(raw_value)
        if peak_value <= 0:
            peak_date = date
            peak_value = value
            continue

        drawdown = value / peak_value - 1.0

        if drawdown >= -1e-9:
            if correction_start is not None:
                frequency = _spx_frequency_for_date(correction_start, daily_start)
                episodes.append(
                    {
                        "kind": "Correction",
                        "frequency": frequency,
                        "daily_start": daily_start,
                        "start_date": correction_start,
                        "peak_date": peak_date,
                        "trough_date": trough_date if trough_date is not None else correction_start,
                        "recovery_date": date,
                        "drawdown": trough_drawdown,
                        "recovered": True,
                    }
                )
            if bear_start is not None:
                frequency = _spx_frequency_for_date(bear_start, daily_start)
                episodes.append(
                    {
                        "kind": "Bear",
                        "frequency": frequency,
                        "daily_start": daily_start,
                        "start_date": bear_start,
                        "peak_date": peak_date,
                        "trough_date": trough_date if trough_date is not None else bear_start,
                        "recovery_date": date,
                        "drawdown": trough_drawdown,
                        "recovered": True,
                    }
                )
            correction_start = None
            bear_start = None
            trough_date = None
            trough_drawdown = 0.0
            peak_date = date
            peak_value = value
        else:
            if drawdown < trough_drawdown:
                trough_drawdown = drawdown
                trough_date = date
            if correction_start is None and drawdown <= -0.10:
                correction_start = date
            if bear_start is None and drawdown <= -0.20:
                bear_start = date

    if correction_start is not None:
        last_date = pd.Timestamp(spx.index[-1])
        frequency = _spx_frequency_for_date(correction_start, daily_start)
        episodes.append(
            {
                "kind": "Correction",
                "frequency": frequency,
                "daily_start": daily_start,
                "start_date": correction_start,
                "peak_date": peak_date,
                "trough_date": trough_date if trough_date is not None else correction_start,
                "recovery_date": last_date,
                "drawdown": trough_drawdown,
                "recovered": False,
            }
        )
    if bear_start is not None:
        last_date = pd.Timestamp(spx.index[-1])
        frequency = _spx_frequency_for_date(bear_start, daily_start)
        episodes.append(
            {
                "kind": "Bear",
                "frequency": frequency,
                "daily_start": daily_start,
                "start_date": bear_start,
                "peak_date": peak_date,
                "trough_date": trough_date if trough_date is not None else bear_start,
                "recovery_date": last_date,
                "drawdown": trough_drawdown,
                "recovered": False,
            }
        )

    return episodes


def build_drawdown_band_layers(
    episodes: list[dict[str, Any]],
    xmin: pd.Timestamp,
    xmax: pd.Timestamp,
) -> list[alt.Chart]:
    """
    Build shared SPX drawdown bands for every chart.

    Correction is drawn first and bear is drawn second so bear-market portions
    are visible inside the broader correction-to-recovery interval.
    """
    xmin = pd.Timestamp(xmin)
    xmax = pd.Timestamp(xmax)
    layers: list[alt.Chart] = []

    for kind, color, opacity in (
        ("Correction", CORRECTION_SHADE, 0.20),
        ("Bear", BEAR_SHADE, 0.26),
    ):
        bands = []
        for episode in episodes:
            if episode["kind"] != kind:
                continue
            s = max(pd.Timestamp(episode["start_date"]), xmin)
            e = min(pd.Timestamp(episode["recovery_date"]), xmax)
            if s < e:
                bands.append({"start": s, "end": e})
        if bands:
            layers.append(
                alt.Chart(pd.DataFrame(bands))
                .mark_rect(color=color, opacity=opacity)
                .encode(x=alt.X("start:T"), x2="end:T")
            )

    return layers


@st.cache_data(ttl=3600, show_spinner=False)
def load_rolling12_bands() -> list[dict[str, Any]]:
    """
    Month-level 12-month rolling-drawdown bands for the Bear+ tab.

    Computes the TRAILING 12-month rolling drawdown of the S&P 500 — how far
    the index currently sits below its highest close in the trailing 12 months:
        rolling_dd_t = P_t / max(P_{t-11..t}) - 1
    and classifies each month:
        bear        : rolling_dd <= -20%
        correction  : -20% < rolling_dd <= -10%
    Contiguous months of the same class are merged into a single band, so the
    shading lines up with the actual market declines on the time axis.
    Returns dicts with 'kind', 'start_date', 'end_date'.
    """
    path = Path(__file__).resolve().parent / "data" / "raw_monthly.csv"
    if not path.exists():
        return []
    spx = pd.read_csv(path, index_col=0, parse_dates=True)["SPX"].dropna()
    if spx.empty:
        return []

    rolling_peak = spx.rolling(12, min_periods=1).max()
    rolling_dd   = spx / rolling_peak - 1.0

    def _classify(v: float) -> Optional[str]:
        if v <= -0.20:
            return "Bear"
        if v <= -0.10:
            return "Correction"
        return None

    bands: list[dict[str, Any]] = []
    run_kind: Optional[str] = None
    run_start: Optional[pd.Timestamp] = None
    prev_date: Optional[pd.Timestamp] = None

    for raw_date, value in rolling_dd.items():
        date = pd.Timestamp(raw_date)
        kind = _classify(float(value))   # None for shallow / no drawdown
        if kind != run_kind:
            if run_kind is not None and run_start is not None:
                bands.append({"kind": run_kind, "start_date": run_start,
                              "end_date": prev_date + pd.DateOffset(months=1)})
            run_kind = kind
            run_start = date if kind is not None else None
        prev_date = date
    if run_kind is not None and run_start is not None:
        bands.append({"kind": run_kind, "start_date": run_start,
                      "end_date": prev_date + pd.DateOffset(months=1)})

    return bands


def build_rolling12_band_layers(
    xmin: pd.Timestamp,
    xmax: pd.Timestamp,
) -> list[alt.Chart]:
    """Altair rect layers for the 12-month rolling correction/bear bands."""
    xmin = pd.Timestamp(xmin)
    xmax = pd.Timestamp(xmax)
    bands_all = load_rolling12_bands()
    layers: list[alt.Chart] = []

    for kind, color, opacity in (
        ("Correction", CORRECTION_SHADE, 0.20),
        ("Bear", BEAR_SHADE, 0.26),
    ):
        bands = []
        for b in bands_all:
            if b["kind"] != kind:
                continue
            s = max(pd.Timestamp(b["start_date"]), xmin)
            e = min(pd.Timestamp(b["end_date"]), xmax)
            if s < e:
                bands.append({"start": s, "end": e})
        if bands:
            layers.append(
                alt.Chart(pd.DataFrame(bands))
                .mark_rect(color=color, opacity=opacity)
                .encode(x=alt.X("start:T"), x2="end:T")
            )
    return layers


@st.cache_data(ttl=3600, show_spinner=False)
def load_rolling6_correction_bands() -> list[dict[str, Any]]:
    """
    Month-level 6-month rolling-correction bands for the Correction+ tab.

    Computes the TRAILING 6-month rolling drawdown of the S&P 500:
        rolling_dd_t = P_t / max(P_{t-5..t}) - 1
    and marks every month where it is deeper than 10% as a correction. There is
    NO bear distinction here — any >10% drawdown is a single correction band.
    Contiguous months are merged. Returns dicts with 'start_date', 'end_date'.
    """
    path = Path(__file__).resolve().parent / "data" / "raw_monthly.csv"
    if not path.exists():
        return []
    spx = pd.read_csv(path, index_col=0, parse_dates=True)["SPX"].dropna()
    if spx.empty:
        return []

    rolling_peak = spx.rolling(6, min_periods=1).max()
    rolling_dd   = spx / rolling_peak - 1.0
    in_corr      = rolling_dd <= -0.10

    bands: list[dict[str, Any]] = []
    run_start: Optional[pd.Timestamp] = None
    prev_date: Optional[pd.Timestamp] = None
    for raw_date, flag in in_corr.items():
        date = pd.Timestamp(raw_date)
        if flag and run_start is None:
            run_start = date
        elif not flag and run_start is not None:
            bands.append({"start_date": run_start,
                          "end_date": prev_date + pd.DateOffset(months=1)})
            run_start = None
        prev_date = date
    if run_start is not None:
        bands.append({"start_date": run_start,
                      "end_date": prev_date + pd.DateOffset(months=1)})
    return bands


def build_rolling6_band_layers(
    xmin: pd.Timestamp,
    xmax: pd.Timestamp,
) -> list[alt.Chart]:
    """Altair rect layers for the 6-month rolling correction bands (no bear)."""
    xmin = pd.Timestamp(xmin)
    xmax = pd.Timestamp(xmax)
    bands = []
    for b in load_rolling6_correction_bands():
        s = max(pd.Timestamp(b["start_date"]), xmin)
        e = min(pd.Timestamp(b["end_date"]), xmax)
        if s < e:
            bands.append({"start": s, "end": e})
    if not bands:
        return []
    return [
        alt.Chart(pd.DataFrame(bands))
        .mark_rect(color=CORRECTION_SHADE, opacity=0.20)
        .encode(x=alt.X("start:T"), x2="end:T")
    ]


def build_drawdown_episode_table(episodes: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    corrections = [episode for episode in episodes if episode["kind"] == "Correction"]
    bears = [episode for episode in episodes if episode["kind"] == "Bear"]

    for correction in corrections:
        daily_start = correction.get("daily_start")
        peak_date = pd.Timestamp(correction["peak_date"])
        correction_date = pd.Timestamp(correction["start_date"])
        trough_date = pd.Timestamp(correction["trough_date"])
        recovery_date = pd.Timestamp(correction["recovery_date"])
        row_frequency = _spx_frequency_for_date(correction_date, daily_start)
        matching_bear = next(
            (
                bear
                for bear in bears
                if pd.Timestamp(bear["peak_date"]) == peak_date
                and pd.Timestamp(bear["recovery_date"]) == recovery_date
            ),
            None,
        )
        bear_date = (
            _format_drawdown_date(
                matching_bear["start_date"],
                row_frequency,
            )
            if matching_bear is not None
            else ""
        )
        rows.append(
            {
                "Peak date": _format_drawdown_date(
                    peak_date,
                    row_frequency,
                ),
                "Correction Date (10% DD)": _format_drawdown_date(
                    correction_date,
                    row_frequency,
                ),
                "Bear Date (20% DD)": bear_date,
                "Trough date": _format_drawdown_date(
                    trough_date,
                    row_frequency,
                ),
                "Recovery date": (
                    _format_drawdown_date(
                        recovery_date,
                        row_frequency,
                    )
                    if correction["recovered"]
                    else "Pending"
                ),
                "Drawdown": f"{correction['drawdown']:.1%}",
                "Peak->trough period": _month_duration_label(peak_date, trough_date),
                "Trough->recovery period": _month_duration_label(trough_date, recovery_date),
                "Lasting period": _month_duration_label(peak_date, recovery_date),
            }
        )
    return pd.DataFrame(rows)


def build_shaded_line_chart(
    df: pd.DataFrame,
    x: str,
    y_cols,
    episodes: list[dict[str, Any]],
    y_title: str = "Value",
) -> "alt.LayerChart":
    """
    Altair line chart with correction (pink) / bear (red) drawdown bands.

    Bands are clipped to the chart's visible date range so they never
    expand the x-axis beyond the plotted data.
    """
    if isinstance(y_cols, str):
        y_cols = [y_cols]

    xmin, xmax = df[x].min(), df[x].max()
    layers = build_drawdown_band_layers(episodes, xmin, xmax)

    long_df = df.melt(id_vars=[x], value_vars=list(y_cols),
                      var_name="Series", value_name="MetricValue").dropna(subset=["MetricValue"])

    line_enc = dict(
        x=alt.X(f"{x}:T", title="Date", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("MetricValue:Q", title=y_title),
        tooltip=[
            alt.Tooltip(f"{x}:T", title="Date", format="%Y-%m"),
            alt.Tooltip("Series:N", title="Series"),
            alt.Tooltip("MetricValue:Q", title="Value", format=".2f"),
        ],
    )
    if len(y_cols) > 1:
        line_enc["color"] = alt.Color("Series:N", title="Series")
        line = alt.Chart(long_df).mark_line().encode(**line_enc)
    else:
        line = alt.Chart(long_df).mark_line(color="#173f2a").encode(**line_enc)
    layers.append(line)

    return (
        alt.layer(*layers)
        .resolve_scale(color="independent")
        .properties(height=340)
    )


render_app_styles()

header_logo, header_title = st.columns([1, 7], vertical_alignment="center")
with header_logo:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=104)
with header_title:
    st.markdown(
        f"""
        <div class="pwa-header">
          <div class="pwa-kicker">Investment Regime Dashboard</div>
          <div class="pwa-title">Piney Woods Advisory, all rights reserved.</div>
          <div class="pwa-subtitle">
            A disciplined view of macro cycle conditions, valuation, credit, labor,
            housing, sentiment, and market trend indicators.
          </div>
          <div class="pwa-asof">
            As of {datetime.today().strftime("%B %d, %Y").replace(" 0", " ")}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

(dashboard_tab, tracker_tab, allocation_tab, performance_tab,
 forecast_tab, rotation_tab, correction_tab, bear_tab) = st.tabs(
    [
        "Dashboard",
        "Indicator Tracker",
        "Conditional Asset Allocation",
        "Historical Performance",
        "Market Forecast",
        "Sector Rotation",
        "Correction Model",
        "Bear Model",
    ]
)

with dashboard_tab:
    api_key = get_fred_api_key()
    if not api_key:
        st.warning(
            "Set a **FRED API key** to load latest data and charts. "
            "Use environment variable `FRED_API_KEY` or Streamlit secrets "
            "(`FRED_API_KEY` or `fred.api_key`). "
            "Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html"
        )
    else:
        render_section_header(
            "Cycle Stage Snapshot",
            (
                "Observation date is the period the value refers to. Publication date "
                "is when FRED first published that vintage. Cycle stage is a heuristic "
                "read from each series' level and recent momentum."
            ),
        )
        with st.spinner("Loading latest readings from FRED…"):
            snapshot = build_latest_snapshot_table(api_key)

        stage_counts = snapshot["Cycle stage"].value_counts()
        early_count = int(stage_counts.get("Early", 0))
        mid_count = int(stage_counts.get("Mid", 0))
        late_count = int(stage_counts.get("Late", 0))
        recession_count = int(stage_counts.get("Recession", 0))
        breadth_cols = st.columns(4)
        breadth_cols[0].metric("Early", early_count)
        breadth_cols[1].metric("Mid", mid_count)
        breadth_cols[2].metric("Late", late_count)
        breadth_cols[3].metric("Recession", recession_count)

        render_colored_snapshot_table(snapshot)
        legend_html = " ".join(
            f'<span style="background:{STAGE_COLORS[stage]};color:white;'
            f'padding:0.2rem 0.6rem;border-radius:0.25rem;margin-right:0.5rem;">'
            f"{stage}</span>"
            for stage in CYCLE_STAGES
        )
        st.markdown(legend_html, unsafe_allow_html=True)

    render_section_header(
        "Historical S&P 500 Corrections And Bear Markets",
        "Drawdown episodes are calculated from the same SPX history used for all chart shadows. "
        "Daily SP500 is used when available; older periods use monthly SPX. "
        "Duration columns are reported in months.",
    )
    episodes = load_drawdown_episodes(api_key)
    episode_table = build_drawdown_episode_table(episodes)
    if episode_table.empty:
        st.info("No SPX drawdown history is available.")
    else:
        st.dataframe(
            episode_table.sort_values("Peak date", ascending=False),
            hide_index=True,
            use_container_width=True,
        )

with tracker_tab:
    render_section_header(
        "Indicator Tracker",
        "Select a series to review source details, latest reading, summary statistics, and history.",
    )

    selected_name = st.selectbox(
        "Select indicator",
        options=[item["name"] for item in INDICATORS],
    )
    selected = INDICATOR_BY_NAME[selected_name]

    render_indicator_metadata(selected)

    api_key = get_fred_api_key()
    if api_key:
        try:
            series = compute_indicator_series(selected, api_key)
        except Exception as exc:
            st.error(f"Could not load {selected['name']}: {exc}")
            st.stop()

        if series.empty:
            st.error("No data returned for this series.")
            st.stop()

        try:
            latest = fetch_indicator_latest(selected, api_key)
        except Exception as exc:
            st.warning(f"Could not load publication metadata: {exc}")
            latest = {
                "value": float(series.iloc[-1]),
                "observation_date": pd.Timestamp(series.index[-1]).strftime("%Y-%m-%d"),
                "publication_date": pd.Timestamp(series.index[-1]).strftime("%Y-%m-%d"),
            }

        stage = classify_cycle_stage(indicator_stage_key(selected), series)

        render_section_header(
            "Latest Reading",
            "Current value, vintage date, distribution context, and cycle classification.",
        )
        latest_row = pd.DataFrame(
            [
                {
                    "Indicator": selected["name"],
                    "FRED series ID": selected["fred_id"],
                    "Latest value": format_value(latest["value"], selected["format"]),
                    "Observation date": format_date(latest["observation_date"]),
                    "Publication date": format_date(latest["publication_date"]),
                    "Median": "—",
                    "10th pct": "—",
                    "90th pct": "—",
                    "Cycle stage": stage,
                }
            ]
        )
        try:
            hist_for_stats = fetch_indicator_history_for_stats(selected, api_key)
            med, p10, p90 = _quantiles(hist_for_stats)
            latest_row.loc[0, "Median"] = format_value(med, selected["format"])
            latest_row.loc[0, "10th pct"] = format_value(p10, selected["format"])
            latest_row.loc[0, "90th pct"] = format_value(p90, selected["format"])
        except Exception:
            pass
        render_colored_snapshot_table(latest_row)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Latest value", format_value(latest["value"], selected["format"]))
        col2.metric(
            "Observation date",
            format_date(latest["observation_date"]),
            help="The period this data point refers to.",
        )
        col3.metric(
            "Publication date",
            format_date(latest["publication_date"]),
            help="When FRED first published this vintage of the reading.",
        )
        col4.metric(
            "Cycle stage",
            stage,
            help="Heuristic early / mid / late / recession signal for this indicator.",
        )

        render_section_header(
            "Historical Series",
            "Trend view for the selected indicator. Shaded bands mark realized "
            "S&P 500 drawdowns: pink starts at -10% and red starts at -20%; "
            "both end at full recovery.",
        )
        episodes = load_drawdown_episodes(api_key)
        if indicator_stage_key(selected) == "unemployment_vs_3yma":
            trend_components = compute_unemployment_trend_components(
                api_key, limit=600
            )
            line_df = trend_components[
                ["Unemployment rate", "3-year moving average"]
            ].reset_index()
            line_df["Date"] = pd.to_datetime(line_df["Date"])
            st.altair_chart(
                build_shaded_line_chart(
                    line_df,
                    x="Date",
                    y_cols=["Unemployment rate", "3-year moving average"],
                    episodes=episodes,
                    y_title="Percent",
                ),
                use_container_width=True,
            )

            spread_df = trend_components[["Difference"]].reset_index()
            spread_df["Date"] = pd.to_datetime(spread_df["Date"])
            st.markdown("#### Difference: unemployment rate minus 3-year moving average")
            st.bar_chart(
                spread_df,
                x="Date",
                y="Difference",
                use_container_width=True,
            )
        else:
            chart_df = series.rename("Value").to_frame().reset_index()
            chart_df.columns = ["Date", "Value"]
            chart_df["Date"] = pd.to_datetime(chart_df["Date"])
            chart_df["Value"] = pd.to_numeric(chart_df["Value"], errors="coerce")
            chart_df = chart_df.dropna(subset=["Date", "Value"])
            st.altair_chart(
                build_shaded_line_chart(
                    chart_df,
                    x="Date",
                    y_cols="Value",
                    episodes=episodes,
                    y_title=selected["name"],
                ),
                use_container_width=True,
            )

        render_section_header(
            "Research Note",
            "Plain-language interpretation for the selected indicator.",
        )
        render_panel("Interpretation", interpret_indicator(selected, series))
    else:
        st.info("Charts and metrics appear once a FRED API key is configured.")

    render_section_header(
        "Raw Indicator Library",
        "Directly downloaded source series used by the dashboard, grouped by macro and market channel.",
    )
    st.dataframe(
        build_reference_table(),
        hide_index=True,
        use_container_width=True,
    )

    render_section_header(
        "Derived Indicator Catalog",
        "Engineered non-flag factors reserved for indicator-tracker detail views and model context.",
    )
    st.dataframe(
        build_derived_indicator_table(),
        hide_index=True,
        use_container_width=True,
    )

with allocation_tab:
    render_section_header(
        "Conditional Asset Allocation",
        "Portfolio policy views organized around the current cycle regime.",
    )
    alloc_cols = st.columns(3)
    with alloc_cols[0]:
        render_panel(
            "Strategic Anchor",
            "Long-term policy weights and rebalancing bands for the core allocation.",
        )
    with alloc_cols[1]:
        render_panel(
            "Regime Tilt",
            "Conditional overweights and underweights derived from cycle-stage signals.",
        )
    with alloc_cols[2]:
        render_panel(
            "Risk Controls",
            "Drawdown, concentration, liquidity, and volatility checks for implementation.",
        )

with performance_tab:
    render_section_header(
        "Historical Performance",
        "Performance attribution and risk analytics for strategy review.",
    )
    perf_cols = st.columns(3)
    with perf_cols[0]:
        render_panel("Return History", "Trailing and calendar-period performance views.")
    with perf_cols[1]:
        render_panel("Risk Profile", "Volatility, drawdown, and downside-capture analytics.")
    with perf_cols[2]:
        render_panel("Attribution", "Cycle-stage and asset-class contribution analysis.")

with forecast_tab:
    render_forecast_page()

with rotation_tab:
    render_sector_rotation_page()

with correction_tab:
    render_ensemble_page(
        family="correction",
        title="Correction Model — Ensemble",
        caption=(
            "Probability of a >10% S&P 500 drawdown within a rolling 6-month window, "
            "from an equal-weight ensemble of four era-trained logistic models "
            "(1920s / 1950s / 1960s / 1980s), Platt-recalibrated. Every factor is "
            "Newey-West HAC-significant; predictions are walk-forward out-of-sample."
        ),
    )

with bear_tab:
    render_ensemble_page(
        family="bear",
        title="Bear Model — Ensemble",
        caption=(
            "Probability of a >20% S&P 500 drawdown within a rolling 12-month window, "
            "from an equal-weight ensemble of four era-trained logistic models "
            "(1920s / 1950s / 1960s / 1980s), Platt-recalibrated. Every factor is "
            "Newey-West HAC-significant; predictions are walk-forward out-of-sample."
        ),
    )
