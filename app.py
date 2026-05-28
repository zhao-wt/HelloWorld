import os
from datetime import datetime, timedelta
from typing import Any, Optional

import pandas as pd
import streamlit as st

st.set_page_config(page_title="U.S. Leading Indicators", layout="wide")

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
            "UNRATE minus its 3-month moving average. Positive values mean "
            "unemployment is rising faster than its very recent trend."
        ),
        "source_url": "https://fred.stlouisfed.org/series/UNRATE",
        "fred_id": "UNRATE_VS_3MMA",
        "computed": "unemployment_vs_3mma",
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


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_shiller_cape_series() -> pd.Series:
    import io
    import requests

    resp = requests.get(
        "https://img1.wsimg.com/blobby/go/e5e77e0b-59d1-44d9-ab25-4763ac982e53/downloads/441f0d2c-37e4-4803-b4e2-8fe10407fbf6/ie_data.xls",
        timeout=30,
    )
    resp.raise_for_status()
    df = pd.read_excel(io.BytesIO(resp.content), sheet_name="Data", header=6)
    date_col = df.columns[0]
    cape_col = "P/E10 or"
    if cape_col not in df.columns:
        raise ValueError("Shiller CAPE column not found in source file")
    def _decimal_year_to_timestamp(year_value: float) -> pd.Timestamp:
        year = int(year_value)
        month = int(round((year_value - year) * 100))
        month = min(max(month, 1), 12)
        return pd.Timestamp(year=year, month=month, day=1)

    rows = df[df[date_col].apply(lambda x: isinstance(x, (int, float)) and 1900 < x < 2030)]
    dates = rows[date_col].apply(_decimal_year_to_timestamp)
    series = pd.Series(rows[cape_col].astype(float).values, index=dates).sort_index()
    return series.dropna()


@st.cache_data(ttl=3600, show_spinner=False)
def compute_sp500_vs_200dma(api_key: str, limit: int = 500) -> pd.Series:
    prices = fetch_observations_series("SP500", api_key, limit=limit)
    ma200 = prices.rolling(200, min_periods=200).mean()
    return ((prices / ma200) - 1.0) * 100.0


@st.cache_data(ttl=3600, show_spinner=False)
def compute_unemployment_vs_3mma(api_key: str, limit: int = 500) -> pd.Series:
    unrate = fetch_observations_series("UNRATE", api_key, limit=limit)
    ma3 = unrate.rolling(3, min_periods=3).mean()
    return unrate - ma3


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
    if computed == "unemployment_vs_3mma":
        return compute_unemployment_vs_3mma(api_key).dropna()
    if computed == "shiller_cape":
        return fetch_shiller_cape_series().dropna()
    if computed == "sp500_it_weight":
        return compute_sp500_it_weight(api_key).dropna()
    return fetch_fred_series(indicator["fred_id"], api_key)


def fetch_indicator_history_for_stage(indicator: dict[str, Any], api_key: str) -> pd.Series:
    computed = indicator.get("computed")
    if computed == "sp500_vs_200dma":
        return compute_sp500_vs_200dma(api_key).dropna()
    if computed == "unemployment_vs_3mma":
        return compute_unemployment_vs_3mma(api_key).dropna()
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
    if computed == "unemployment_vs_3mma":
        return compute_unemployment_vs_3mma(api_key, limit=600).dropna()
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

    if stage_key == "unemployment_vs_3mma":
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
    elif stage_key == "unemployment_vs_3mma":
        if latest > 0:
            parts.append("Unemployment is **rising faster** than its 3-month trend.")
        else:
            parts.append("Unemployment is **not deteriorating** vs its 3-month trend.")
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


st.title("U.S. Leading Indicators Dashboard")
st.caption(
    "Monitor key macro and market indicators that tend to move ahead of the business cycle."
)

api_key = get_fred_api_key()
if not api_key:
    st.warning(
        "Set a **FRED API key** to load latest data and charts. "
        "Use environment variable `FRED_API_KEY` or Streamlit secrets "
        "(`FRED_API_KEY` or `fred.api_key`). "
        "Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html"
    )
else:
    st.subheader("Latest data")
    st.caption(
        "Observation date is the period the value refers to. "
        "Publication date is when FRED first published that vintage. "
        "Cycle stage is a heuristic read (Early / Mid / Late / Recession) from each "
        "series' level and recent momentum—not an official NBER classification."
    )
    with st.spinner("Loading latest readings from FRED…"):
        snapshot = build_latest_snapshot_table(api_key)
    render_colored_snapshot_table(snapshot)
    legend_html = " ".join(
        f'<span style="background:{STAGE_COLORS[stage]};color:white;'
        f'padding:0.2rem 0.6rem;border-radius:0.25rem;margin-right:0.5rem;">'
        f"{stage}</span>"
        for stage in CYCLE_STAGES
    )
    st.markdown(legend_html, unsafe_allow_html=True)

st.subheader("Indicator reference")
st.dataframe(
    build_reference_table(),
    column_config={
        "Source": st.column_config.LinkColumn("Source", display_text="FRED ↗"),
    },
    hide_index=True,
    use_container_width=True,
)

st.divider()
st.subheader("Explore an indicator")

selected_name = st.selectbox(
    "Select indicator",
    options=[item["name"] for item in INDICATORS],
)

selected = INDICATOR_BY_NAME[selected_name]
st.markdown(f"**Category:** {selected['category']}")
st.markdown(selected["description"])
st.markdown(f"[View source]({selected['source_url']})")

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

    st.subheader("Latest reading")
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

    chart_df = series.rename("value").to_frame()
    chart_df.index = pd.to_datetime(chart_df.index)
    st.line_chart(chart_df, use_container_width=True)

    st.markdown("#### Interpretation")
    st.markdown(interpret_indicator(selected, series))
else:
    st.info("Charts and metrics appear once a FRED API key is configured.")
