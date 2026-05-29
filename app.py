import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Piney Woods Advisory", layout="wide")

INDICATORS: list[dict[str, Any]] = [
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
]

INDICATOR_BY_NAME = {item["name"]: item for item in INDICATORS}

SHILLER_CAPE_URL = (
    "https://img1.wsimg.com/blobby/go/e5e77e0b-59d1-44d9-ab25-4763ac982e53/"
    "downloads/441f0d2c-37e4-4803-b4e2-8fe10407fbf6/ie_data.xls"
)
SHILLER_CAPE_LOCAL_PATH = Path(__file__).resolve().parent / "data" / "shiller_ie_data.xls"
LOGO_PATH = Path(__file__).resolve().parent / "assets" / "piney-woods-logo.png"

CYCLE_STAGES = ("Early", "Mid", "Late", "Recession")
STAGE_COLORS = {
    "Early": "#22c55e",
    "Mid": "#eab308",
    "Late": "#f97316",
    "Recession": "#ef4444",
}


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

    observation_start = (datetime.today() - timedelta(days=365 * 25)).strftime(
        "%Y-%m-%d"
    )
    resp = requests.get(
        "https://api.stlouisfed.org/fred/series/observations",
        params={
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "observation_start": observation_start,
            "sort_order": "asc",
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


def compute_indicator_series(indicator: dict[str, Any], api_key: str) -> pd.Series:
    computed = indicator.get("computed")
    if computed == "sp500_vs_200dma":
        return compute_sp500_vs_200dma(api_key).dropna()
    if computed == "unemployment_vs_3yma":
        return compute_unemployment_vs_3yma(api_key).dropna()
    if computed == "shiller_cape":
        return fetch_shiller_cape_series().dropna()
    if computed == "sp500_it_weight":
        return compute_sp500_it_weight(api_key).dropna()
    return fetch_fred_series(indicator["fred_id"], api_key)


def fetch_indicator_history_for_stage(indicator: dict[str, Any], api_key: str) -> pd.Series:
    computed = indicator.get("computed")
    if computed == "sp500_vs_200dma":
        return compute_sp500_vs_200dma(api_key).dropna()
    if computed == "unemployment_vs_3yma":
        return compute_unemployment_vs_3yma(api_key).dropna()
    if computed == "shiller_cape":
        return fetch_shiller_cape_series().dropna()
    if computed == "sp500_it_weight":
        return compute_sp500_it_weight(api_key).dropna()
    return fetch_fred_series_for_stage(indicator["fred_id"], api_key)


def fetch_indicator_history_for_stats(indicator: dict[str, Any], api_key: str) -> pd.Series:
    """History used for distribution stats (median / p10 / p90)."""
    computed = indicator.get("computed")
    # For computed series, use a larger recent window to get stable percentiles.
    if computed == "sp500_vs_200dma":
        return compute_sp500_vs_200dma(api_key, limit=2000).dropna()
    if computed == "unemployment_vs_3yma":
        return compute_unemployment_vs_3yma(api_key, limit=600).dropna()
    if computed == "shiller_cape":
        return fetch_shiller_cape_series().dropna()
    if computed == "sp500_it_weight":
        return compute_sp500_it_weight(api_key, limit=2000).dropna()
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

cycle_tab, allocation_tab, performance_tab = st.tabs(
    ["Cycle Stage Monitor", "Conditional Asset Allocation", "Historical Performance"]
)

with cycle_tab:
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
        "Indicator Library",
        "Reference set used by the cycle monitor, grouped by macro and market channel.",
    )
    st.dataframe(
        build_reference_table(),
        column_config={
            "Source": st.column_config.LinkColumn("Source", display_text="FRED ↗"),
        },
        hide_index=True,
        use_container_width=True,
    )

    render_section_header(
        "Indicator Drilldown",
        "Select a series to review its latest reading, history, and interpretation.",
    )

    select_col, context_col = st.columns([1, 2], vertical_alignment="top")
    with select_col:
        selected_name = st.selectbox(
            "Select indicator",
            options=[item["name"] for item in INDICATORS],
        )

    selected = INDICATOR_BY_NAME[selected_name]
    with context_col:
        render_panel(
            selected["category"],
            f"{selected['description']} Source: {selected['source_url']}",
        )

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
            "Trend view for the selected indicator across the available observation window.",
        )
        if indicator_stage_key(selected) == "unemployment_vs_3yma":
            trend_components = compute_unemployment_trend_components(
                api_key, limit=600
            )
            line_df = trend_components[
                ["Unemployment rate", "3-year moving average"]
            ].reset_index()
            line_df["Date"] = pd.to_datetime(line_df["Date"])
            st.line_chart(
                line_df,
                x="Date",
                y=["Unemployment rate", "3-year moving average"],
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
            st.line_chart(chart_df, x="Date", y="Value", use_container_width=True)

        render_section_header(
            "Research Note",
            "Plain-language interpretation for the selected indicator.",
        )
        render_panel("Interpretation", interpret_indicator(selected, series))
    else:
        st.info("Charts and metrics appear once a FRED API key is configured.")

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
