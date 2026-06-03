"""
bear/data.py — Phase 1: raw data pipeline.

Fetches every raw series needed by the bear and correction models.
All results are cached locally under config.cache_dir.

Publication-lag calendar (conservative; from observation month-end):
  Series               Freq      Lag (calendar days)   Rationale
  -------------------  --------  --------------------  -------------------------
  DGS3MO / DGS10       daily     1                     H.15, next business day
  T10Y3M / T10Y2Y      daily     1                     FRED computed, next day
  BAMLH0A0HYM2         daily     1                     ICE BofA, next day
  VIXCLS / VXVCLS      daily     1                     CBOE, next day
  DFF                  daily     1                     Fed, next day
  CPCE                 daily     1                     CBOE, next day
  SP500 (FRED)         daily     1                     S&P, next day
  NTFS                 daily     1                     derived from GSW
  ICSA                 weekly    5                     BLS: Thursday for prior week
  ANFCI / NFCI         weekly    7                     Chicago Fed: ~1 week
  UNRATE               monthly   35                    BLS jobs: first Fri of t+1
  SAHMREALTIME         monthly   35                    released with UNRATE
  USALOLITOAASTSAM     monthly   42                    OECD: ~3rd-4th week of t+1
  BAA10Y               monthly   35                    Moody's / FRED monthly
  EBP                  monthly   45                    GZ: ~6 weeks
  SHILLER_CAPE         monthly   30                    Shiller: end of month

No look-ahead is applied here — that is done in Phase 2 (features.py) when each
series is shifted forward by its publication lag before joining to the model panel.
"""

from __future__ import annotations

import hashlib
import io
import os
import pickle
import time
import warnings
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd
import requests


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    """Master configuration for the bear/correction data pipeline."""

    fred_api_key: str
    start: date = field(default_factory=lambda: date(1900, 1, 1))
    end: date = field(default_factory=date.today)
    cache_dir: Path = field(default_factory=lambda: Path("bear/cache"))

    # Gilchrist-Zakrajšek EBP — Fed FEDS-Note monthly update (correct URL confirmed).
    # Manual download page:
    # https://www.federalreserve.gov/econresdata/notes/feds-notes/2016/
    #   updating-the-recession-risk-and-the-excess-bond-premium-20161006.html
    ebp_url: str = (
        "https://www.federalreserve.gov/econresdata/notes/feds-notes/2016/files/ebp_csv.csv"
    )
    ebp_local_path: Optional[Path] = None

    # CBOE equity put/call ratio — not on FRED or Yahoo; must be supplied manually.
    # Download from: https://www.cboe.com/us/options/market_statistics/daily/
    cpce_local_path: Optional[Path] = None

    # Gurkaynak-Sack-Wright (2006) zero-coupon yield curve — used for NTFS.
    gsw_url: str = (
        "https://www.federalreserve.gov/data/yield-curve-tables/feds200628.csv"
    )

    request_timeout: int = 60

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Publication-lag calendar
# ---------------------------------------------------------------------------

PUBLICATION_LAGS: dict[str, int] = {
    # Daily
    "DGS3MO":            1,
    "DGS10":             1,
    "T10Y3M":            1,
    "T10Y2Y":            1,
    "BAMLH0A0HYM2":      1,
    "VIXCLS":            1,
    "VXVCLS":            1,
    "DFF":               1,
    "CPCE":              1,
    "SP500":             1,
    "NTFS":              1,
    "SPX":               1,
    # Weekly
    "ICSA":              5,
    "ANFCI":             7,
    "NFCI":              7,
    # Monthly
    "UNRATE":           35,
    "SAHMREALTIME":     35,
    "USALOLITOAASTSAM": 42,
    "BAA10Y":           35,
    "EBP":              45,
    "SHILLER_CAPE":     30,
}


def publication_lag(series_id: str) -> int:
    """Return conservative publication lag in calendar days for a series."""
    return PUBLICATION_LAGS.get(series_id, 30)


# ---------------------------------------------------------------------------
# Local cache (pickle, keyed by series + date range)
# ---------------------------------------------------------------------------

def _cache_path(cache_dir: Path, series_id: str, start: date, end: date) -> Path:
    key = hashlib.md5(f"{series_id}|{start}|{end}".encode()).hexdigest()[:10]
    return cache_dir / f"{series_id}_{key}.pkl"


def _load_cache(path: Path) -> Optional[pd.Series]:
    if path.exists():
        try:
            with open(path, "rb") as fh:
                return pickle.load(fh)  # type: ignore[no-any-return]
        except Exception:
            path.unlink(missing_ok=True)
    return None


def _save_cache(path: Path, obj: pd.Series) -> None:
    with open(path, "wb") as fh:
        pickle.dump(obj, fh)


# ---------------------------------------------------------------------------
# FRED fetcher
# ---------------------------------------------------------------------------

def fetch_fred_series(
    series_id: str,
    start: date,
    end: date,
    api_key: str,
    cache_dir: Path,
    *,
    frequency: Optional[str] = None,
    aggregation_method: str = "avg",
    max_retries: int = 5,
    retry_delay: float = 2.0,
) -> pd.Series:
    """
    Fetch a FRED series as pd.Series indexed by observation date.

    Retries up to max_retries times on HTTP 429 (rate-limit) with
    exponential backoff. Results are cached locally.

    Parameters
    ----------
    series_id : str
        FRED series identifier, e.g. 'T10Y3M', 'SAHMREALTIME'.
    frequency : str, optional
        FRED frequency conversion: 'd' daily, 'w' weekly, 'm' monthly.
        None = native frequency (no conversion).
    aggregation_method : str
        FRED aggregation when converting frequency: 'avg', 'sum', 'eop'.
    max_retries : int
        Maximum retry attempts on 429 errors.
    retry_delay : float
        Base delay in seconds; doubled on each retry.
    """
    cache_path = _cache_path(cache_dir, series_id, start, end)
    cached = _load_cache(cache_path)
    if cached is not None:
        return cached

    params: dict[str, str] = {
        "series_id":         series_id,
        "api_key":           api_key,
        "file_type":         "json",
        "observation_start": str(start),
        "observation_end":   str(end),
        "sort_order":        "asc",
    }
    if frequency:
        params["frequency"]          = frequency
        params["aggregation_method"] = aggregation_method

    last_exc: Exception = RuntimeError("No attempts made")
    delay = retry_delay
    resp = None
    for attempt in range(max_retries):
        try:
            resp = requests.get(
                "https://api.stlouisfed.org/fred/series/observations",
                params=params,
                timeout=60,
            )
            if resp.status_code == 429:
                time.sleep(delay)
                delay *= 2
                continue
            resp.raise_for_status()
            break
        except requests.HTTPError as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2
    else:
        raise last_exc

    if resp is None:
        raise RuntimeError(f"No response received for FRED series '{series_id}'")

    observations = resp.json().get("observations", [])
    dates:  list[pd.Timestamp] = []
    values: list[float] = []
    for obs in observations:
        if obs.get("value") in (None, "."):
            continue
        try:
            dates.append(pd.Timestamp(obs["date"]))
            values.append(float(obs["value"]))
        except (ValueError, TypeError):
            continue

    if not values:
        raise ValueError(
            f"No observations returned by FRED for series '{series_id}' "
            f"({start} to {end})"
        )

    series = pd.Series(values, index=dates, name=series_id).sort_index()
    _save_cache(cache_path, series)
    return series


# ---------------------------------------------------------------------------
# S&P 500 daily prices — Yahoo Finance
# ---------------------------------------------------------------------------

def fetch_spx(start: date, end: date, cache_dir: Path, symbol: str = "^GSPC") -> pd.Series:
    """
    Daily S&P 500 adjusted-close prices from Yahoo Finance.

    Used for: target construction (forward max drawdown), 200-DMA, momentum.
    """
    cache_path = _cache_path(cache_dir, "SPX", start, end)
    cached = _load_cache(cache_path)
    if cached is not None:
        return cached

    import yfinance as yf

    df = yf.download(symbol, start=str(start), end=str(end), auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No price data returned for {symbol}")

    close = df["Close"].squeeze()
    close.index = pd.to_datetime(close.index)
    close.name = "SPX"
    _save_cache(cache_path, close)
    return close  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Shiller CAPE
# ---------------------------------------------------------------------------

_SHILLER_URL = (
    "https://img1.wsimg.com/blobby/go/e5e77e0b-59d1-44d9-ab25-4763ac982e53/"
    "downloads/441f0d2c-37e4-4803-b4e2-8fe10407fbf6/ie_data.xls"
)
_SHILLER_LOCAL = Path(__file__).resolve().parent.parent / "data" / "shiller_ie_data.xls"


def fetch_shiller_cape(cache_dir: Path) -> pd.Series:
    """
    Monthly Shiller CAPE from Robert Shiller's historical dataset.

    Uses the local data/shiller_ie_data.xls if present; otherwise downloads.
    """
    cache_path = cache_dir / "shiller_cape.pkl"
    cached = _load_cache(cache_path)
    if cached is not None:
        return cached

    source = _SHILLER_LOCAL if _SHILLER_LOCAL.exists() else cache_dir / "shiller_ie_data.xls"
    if not source.exists():
        resp = requests.get(_SHILLER_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
        resp.raise_for_status()
        source.write_bytes(resp.content)

    df = pd.read_excel(source, sheet_name="Data", header=6)
    date_col = df.columns[0]
    cape_col = next(
        (c for c in ["P/E10 or", "P/E10", "CAPE"] if c in df.columns), None
    )
    if cape_col is None:
        raise ValueError("Shiller CAPE column not found in source file")

    def _decimal_year_to_ts(v: float) -> pd.Timestamp:
        yr = int(v)
        mo = min(max(int(round((v - yr) * 100)), 1), 12)
        return pd.Timestamp(year=yr, month=mo, day=1)

    num_dates = pd.to_numeric(df[date_col], errors="coerce")
    num_cape  = pd.to_numeric(df[cape_col], errors="coerce")
    mask = num_dates.between(1900, 2030) & num_cape.notna()
    series = pd.Series(
        num_cape[mask].astype(float).values,
        index=num_dates[mask].apply(_decimal_year_to_ts),
        name="SHILLER_CAPE",
    ).sort_index().dropna()

    _save_cache(cache_path, series)
    return series


# ---------------------------------------------------------------------------
# GSW zero-coupon yield curve → Near-Term Forward Spread (NTFS)
# ---------------------------------------------------------------------------

def fetch_gsw_zero_coupon(cache_dir: Path, gsw_url: str, timeout: int = 60) -> pd.DataFrame:
    """
    Download the Gurkaynak-Sack-Wright (2006) fitted zero-coupon yield curve.

    Returns a DataFrame indexed by date with columns SVENY01..SVENY30
    (continuously compounded yields, percent per year).
    """
    local = cache_dir / "feds200628.csv"
    if not local.exists():
        resp = requests.get(gsw_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=timeout)
        resp.raise_for_status()
        local.write_bytes(resp.content)

    raw = local.read_text(encoding="utf-8", errors="replace")
    lines = raw.splitlines()

    header_idx = next(
        (i for i, ln in enumerate(lines) if ln.strip().upper().startswith("DATE")),
        None,
    )
    if header_idx is None:
        raise ValueError(
            f"Cannot locate header row in GSW CSV — expected a line starting with 'Date'. "
            f"Check the downloaded file at {local}."
        )

    df = pd.read_csv(
        io.StringIO("\n".join(lines[header_idx:])),
        parse_dates=[0],
        index_col=0,
    )
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    df.columns = [c.strip().upper() for c in df.columns]

    sven_cols = [c for c in df.columns if c.startswith("SVENY")]
    if not sven_cols:
        raise ValueError(
            "No SVENY columns found in the GSW CSV. "
            "The file format may have changed — check the Fed data page."
        )

    return df[sven_cols].apply(pd.to_numeric, errors="coerce").dropna(how="all")


def compute_ntfs(gsw_df: pd.DataFrame, tbill_3m: pd.Series) -> pd.Series:
    """
    Near-Term Forward Spread (Engstrom & Sharpe, FAJ 2019).

    NTFS_t = f(t; 18m -> 21m) - y_t(3m)

    f(18m, 21m) is the forward 3-month rate starting 18 months from now,
    derived from GSW yields via linear interpolation:

        y(1.50yr) = 0.50 * SVENY01 + 0.50 * SVENY02
        y(1.75yr) = 0.25 * SVENY01 + 0.75 * SVENY02
        f(18m,21m) = [1.75 * y(1.75) - 1.50 * y(1.50)] / 0.25

    All yields in percent per year (continuously compounded).
    Negative NTFS signals markets pricing a near-term rate drop (recession risk).
    A 1-SD (~80bp) fall raises 4-quarter recession prob ~35pp (Engstrom-Sharpe 2019).
    """
    if "SVENY01" not in gsw_df.columns or "SVENY02" not in gsw_df.columns:
        raise ValueError("GSW DataFrame must contain columns SVENY01 and SVENY02.")

    y1 = gsw_df["SVENY01"]
    y2 = gsw_df["SVENY02"]

    y_18m = 0.50 * y1 + 0.50 * y2
    y_21m = 0.25 * y1 + 0.75 * y2
    f_18_21 = (1.75 * y_21m - 1.50 * y_18m) / 0.25

    tbill_aligned = tbill_3m.reindex(f_18_21.index, method="ffill", limit=5)
    return (f_18_21 - tbill_aligned).rename("NTFS").dropna()


# ---------------------------------------------------------------------------
# Gilchrist-Zakrajšek Excess Bond Premium (EBP)
# ---------------------------------------------------------------------------

def fetch_ebp(
    cache_dir: Path,
    ebp_url: str,
    local_path: Optional[Path] = None,
    timeout: int = 60,
) -> pd.Series:
    """
    Gilchrist-Zakrajšek (2012) Excess Bond Premium, monthly.

    The EBP isolates credit-supply / risk-appetite beyond default risk.
    It is among the strongest medium-horizon bear-market predictors
    (Tokic & Jackson 2023; GZ 2012 AER).

    Sourcing priority
    -----------------
    1. User-supplied local CSV  (ebp_local_path in DataConfig).
    2. Download from ebp_url    (Fed FEDS-Note monthly update, confirmed URL).
    3. Returns empty Series with a warning if both fail.

    Manual download page
    --------------------
    https://www.federalreserve.gov/econresdata/notes/feds-notes/2016/
      updating-the-recession-risk-and-the-excess-bond-premium-20161006.html
    """
    cache_path = cache_dir / "ebp.pkl"
    cached = _load_cache(cache_path)
    if cached is not None:
        return cached

    if local_path and Path(local_path).exists():
        return _parse_ebp_csv(Path(local_path), cache_path)

    try:
        resp = requests.get(ebp_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=timeout)
        resp.raise_for_status()
        raw_path = cache_dir / "ebp_raw.csv"
        raw_path.write_bytes(resp.content)
        return _parse_ebp_csv(raw_path, cache_path)
    except Exception as exc:
        warnings.warn(
            f"EBP download failed ({exc}). EBP will be absent from the raw data dict.\n"
            "To supply EBP manually:\n"
            "  1. Download ebp_csv.csv from:\n"
            "     https://www.federalreserve.gov/econresdata/notes/feds-notes/2016/"
            "files/ebp_csv.csv\n"
            "  2. Pass: DataConfig(ebp_local_path=Path('/path/to/ebp_csv.csv'))",
            stacklevel=3,
        )
        return pd.Series(dtype=float, name="EBP")


def _parse_ebp_csv(path: Path, cache_path: Path) -> pd.Series:
    """Parse a Gilchrist-Zakrajšek EBP CSV; handles common column conventions."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    date_col = next(
        (c for c in df.columns if c in {"date", "yyyymm", "yyyy-mm", "yearmo", "month"}),
        df.columns[0],
    )
    ebp_col = next((c for c in df.columns if "ebp" in c), None)
    if ebp_col is None:
        raise ValueError(
            f"No 'ebp' column found in {path}. "
            f"Available columns: {list(df.columns)}"
        )

    df[date_col] = pd.to_datetime(df[date_col])
    series = pd.Series(
        pd.to_numeric(df[ebp_col], errors="coerce").values,
        index=df[date_col],
        name="EBP",
    ).sort_index().dropna()

    _save_cache(cache_path, series)
    return series


# ---------------------------------------------------------------------------
# CBOE equity put/call ratio  (manual CSV — not on FRED or Yahoo Finance)
# ---------------------------------------------------------------------------

_CPCE_RECENT_URL  = "https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/equitypc.csv"
_CPCE_ARCHIVE_URL = "https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/equitypcarchive.csv"


def fetch_put_call_ratio(
    cache_dir: Path,
    local_path: Optional[Path] = None,
    timeout: int = 60,
) -> pd.Series:
    """
    CBOE equity-only put/call ratio (CPCE), daily.

    Low put/call (complacency) -> elevated future correction risk.
    Used as an extreme dummy in the correction model (Pan-Poteshman 2006).

    Downloads from CBOE's public CDN (no auth required):
      archive : equitypcarchive.csv  2003-10-17 -> 2012-06-07
      recent  : equitypc.csv         2006-11-01 -> 2019-10-04

    Note: CBOE stopped updating these files after Oct 2019; newer data
    requires a paid DataShop subscription (datashop.cboe.com).
    The 2003-2019 window covers enough correction and bear episodes for
    model training. To extend to the present, supply a local CSV via
    DataConfig(cpce_local_path=...).
    """
    cache_path = cache_dir / "cpce.pkl"
    cached = _load_cache(cache_path)
    if cached is not None:
        return cached

    # 1. User-supplied local file
    if local_path and Path(local_path).exists():
        series = _parse_cpce_csv(Path(local_path))
        _save_cache(cache_path, series)
        return series

    # 2. Download archive + recent from CBOE CDN and merge
    segments: list[pd.Series] = []
    for url in (_CPCE_ARCHIVE_URL, _CPCE_RECENT_URL):
        try:
            resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=timeout)
            resp.raise_for_status()
            raw_path = cache_dir / Path(url).name
            raw_path.write_bytes(resp.content)
            segments.append(_parse_cpce_csv(raw_path))
        except Exception as exc:
            warnings.warn(f"CPCE download from {url} failed: {exc}", stacklevel=3)

    if not segments:
        warnings.warn(
            "CPCE: all download attempts failed. "
            "Supply manually via DataConfig(cpce_local_path=...).",
            stacklevel=2,
        )
        return pd.Series(dtype=float, name="CPCE")

    series = (
        pd.concat(segments)
        .sort_index()
        .loc[lambda s: ~s.index.duplicated(keep="last")]
        .rename("CPCE")
    )
    _save_cache(cache_path, series)
    return series


def _parse_cpce_csv(path: Path) -> pd.Series:
    """
    Parse a CBOE equity put/call CSV.

    CBOE file layout (equitypc.csv and equitypcarchive.csv):
      Row 0: legal disclaimer text
      Row 1: metadata  (, PRODUCT: EQUITY,,EXCHANGE: Cboe,)
      Row 2: header    (DATE,CALL,PUT,TOTAL,P/C Ratio)
      Row 3+: data     (11/1/2006,976510,623929,1600439,0.64)

    Scans for the header row dynamically so it is robust to format changes.
    Tries UTF-8 first, falls back to latin-1 (archive file has non-UTF-8 bytes).
    """
    for encoding in ("utf-8", "latin-1"):
        try:
            raw_text = path.read_text(encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"Cannot decode {path} as UTF-8 or latin-1")

    lines = raw_text.splitlines()
    # Find the first line that looks like a data header (starts with DATE or "Date")
    header_idx = next(
        (i for i, ln in enumerate(lines) if ln.strip().upper().startswith("DATE")),
        None,
    )
    if header_idx is None:
        raise ValueError(f"Cannot find DATE header row in {path}")

    df = pd.read_csv(
        io.StringIO("\n".join(lines[header_idx:])),
        header=0,
    )
    df.columns = [
        c.strip().strip('"').lower().replace(" ", "_").replace("/", "_")
        for c in df.columns
    ]

    date_col  = next((c for c in df.columns if "date" in c), df.columns[0])
    value_col = next(
        (c for c in df.columns if "p_c" in c or "ratio" in c or "put_call" in c),
        next((c for c in df.columns if c in {"cpce", "equity_pc"}), None),
    )
    if value_col is None:
        raise ValueError(
            f"Cannot find put/call ratio column in {path}. "
            f"Columns: {list(df.columns)}"
        )

    dates  = pd.to_datetime(df[date_col], errors="coerce")
    values = pd.to_numeric(df[value_col], errors="coerce")
    series = pd.Series(values.to_numpy(), index=dates, name="CPCE")
    series = series[series.index.notna() & series.notna()]
    series.index = pd.DatetimeIndex(series.index)
    return series.sort_index()


# ---------------------------------------------------------------------------
# Master fetch — both models
# ---------------------------------------------------------------------------

# Each entry: series_id -> (aggregation_method, description)
#
# aggregation_method (FRED convention for daily/weekly → monthly conversion):
#   "eop"  end-of-period  — month-end snapshot; right for rate/price/index series
#   "avg"  average        — mean over the month; right for flow/diffusion series
#
# Series that are already monthly on FRED (UNRATE, SAHMREALTIME, USALOLITOAASTSAM,
# SP500) are still requested with frequency='m' for consistency; FRED ignores
# the aggregation parameter when no frequency conversion is needed.

_BEAR_FRED: dict[str, tuple[str, str]] = {
    "T10Y3M":           ("eop", "10y-3m term spread (Estrella-Mishkin)"),
    "T10Y2Y":           ("eop", "10y-2y term spread"),
    "DGS3MO":           ("eop", "3-month T-bill yield (base for NTFS)"),
    "DGS10":            ("eop", "10-year Treasury yield"),
    "BAMLH0A0HYM2":     ("eop", "HY OAS (FRED free tier ~3yr; BAA10Y used for full history)"),
    "SAHMREALTIME":     ("avg", "Sahm rule real-time vintage"),
    "ICSA":             ("avg", "Initial jobless claims — monthly avg of weekly readings"),
    "USALOLITOAASTSAM": ("avg", "OECD CLI / LEI proxy (US, amplitude adjusted)"),
    "DFF":              ("avg", "Effective federal funds rate — monthly avg"),
    "UNRATE":           ("avg", "Unemployment rate"),
    "BAA10Y":           ("eop", "BAA-10y default spread — long-history HY proxy"),
}

# Correction model series (incremental — not in bear set)
_CORRECTION_FRED: dict[str, tuple[str, str]] = {
    "VIXCLS":  ("avg", "CBOE VIX 30-day — monthly avg"),
    "VXVCLS":  ("avg", "CBOE VIX 3-month — monthly avg (from 2007)"),
    "ANFCI":   ("avg", "Adjusted NFCI — monthly avg of weekly readings"),
    "NFCI":    ("avg", "National Financial Conditions Index — monthly avg"),
    "SP500":   ("eop", "S&P 500 index level (FRED monthly)"),
}


def fetch_all_raw(config: DataConfig) -> dict[str, pd.Series]:
    """
    Fetch all raw series needed by both models at monthly frequency.

    All FRED series are requested at monthly frequency directly from the API
    (daily/weekly series use FRED's built-in aggregation).  Non-FRED series
    (SPX from Yahoo, NTFS from GSW) are fetched at their native frequency
    and resampled to month-end in align_to_monthly().

    Returns
    -------
    dict[str, pd.Series]
        Monthly-indexed (or daily for SPX/NTFS) Series, one per source:
        FRED monthly  : T10Y3M, T10Y2Y, DGS3MO, DGS10, BAMLH0A0HYM2,
                        SAHMREALTIME, ICSA, USALOLITOAASTSAM, DFF, UNRATE,
                        BAA10Y, VIXCLS, VXVCLS, ANFCI, NFCI, SP500
        Yahoo daily   : SPX   (resampled to monthly in align_to_monthly)
        GSW daily     : NTFS  (resampled to monthly in align_to_monthly)
        Fed monthly   : EBP
        CBOE daily    : CPCE  (resampled to monthly in align_to_monthly)
        Shiller monthly: SHILLER_CAPE

    Notes
    -----
    Publication lags are NOT applied here — that is Phase 2 (features.py).
    """
    raw: dict[str, pd.Series] = {}

    # -- FRED series (monthly frequency) --
    all_fred = {**_BEAR_FRED, **_CORRECTION_FRED}
    print(f"\n{'='*60}")
    print(f"  Fetching {len(all_fred)} FRED series at monthly freq  ({config.start} to {config.end})")
    print(f"{'='*60}")
    for sid, (agg, desc) in all_fred.items():
        try:
            s = fetch_fred_series(
                sid, config.start, config.end, config.fred_api_key, config.cache_dir,
                frequency="m",
                aggregation_method=agg,
            )
            raw[sid] = s
            print(f"  OK  {sid:<22}  {len(s):>5} obs  {s.index[0].date()} to {s.index[-1].date()}  {desc}")
        except Exception as exc:
            warnings.warn(f"  FAIL  FRED '{sid}': {exc}", stacklevel=2)
        time.sleep(0.5)

    # -- S&P 500 prices (Yahoo Finance — daily; resampled in align_to_monthly) --
    print(f"\n{'='*60}")
    print("  Fetching SPX prices from Yahoo Finance (daily → resampled monthly)")
    print(f"{'='*60}")
    try:
        raw["SPX"] = fetch_spx(config.start, config.end, config.cache_dir)
        s = raw["SPX"]
        print(f"  OK  SPX (^GSPC)          {len(s):>5} daily obs  "
              f"{s.index[0].date()} to {s.index[-1].date()}")
    except Exception as exc:
        warnings.warn(f"  FAIL  SPX: {exc}", stacklevel=2)

    # -- Shiller CAPE (monthly) --
    print(f"\n{'='*60}")
    print("  Fetching Shiller CAPE (monthly, back to 1871)")
    print(f"{'='*60}")
    try:
        raw["SHILLER_CAPE"] = fetch_shiller_cape(config.cache_dir)
        s = raw["SHILLER_CAPE"]
        print(f"  OK  SHILLER_CAPE         {len(s):>5} monthly obs  "
              f"{s.index[0].date()} to {s.index[-1].date()}")
    except Exception as exc:
        warnings.warn(f"  FAIL  Shiller CAPE: {exc}", stacklevel=2)

    # -- GSW zero-coupon → NTFS (daily; resampled in align_to_monthly) --
    print(f"\n{'='*60}")
    print("  Computing NTFS from GSW zero-coupon curve (daily → resampled monthly)")
    print(f"{'='*60}")
    try:
        gsw_df = fetch_gsw_zero_coupon(config.cache_dir, config.gsw_url, config.request_timeout)
        tbill_daily = raw.get("DGS3MO")
        if tbill_daily is None:
            # DGS3MO was fetched monthly — fetch daily for NTFS computation only
            tbill_daily = fetch_fred_series(
                "DGS3MO", config.start, config.end, config.fred_api_key,
                config.cache_dir / "ntfs_helper",
            )
        raw["NTFS"] = compute_ntfs(gsw_df, tbill_daily)
        s = raw["NTFS"]
        print(f"  OK  NTFS                {len(s):>5} daily obs  "
              f"{s.index[0].date()} to {s.index[-1].date()}")
    except Exception as exc:
        warnings.warn(f"  FAIL  NTFS: {exc}", stacklevel=2)

    # -- CBOE equity put/call ratio (daily; resampled in align_to_monthly) --
    print(f"\n{'='*60}")
    print("  Loading CBOE equity put/call ratio (daily → resampled monthly)")
    print(f"{'='*60}")
    raw["CPCE"] = fetch_put_call_ratio(
        config.cache_dir, config.cpce_local_path, config.request_timeout
    )
    if not raw["CPCE"].empty:
        s = raw["CPCE"]
        print(f"  OK  CPCE                {len(s):>5} daily obs  "
              f"{s.index[0].date()} to {s.index[-1].date()}")
    else:
        print("  --  CPCE: not available — correction model runs without it.")

    # -- Gilchrist-Zakrajšek EBP (monthly) --
    print(f"\n{'='*60}")
    print("  Fetching Gilchrist-Zakrajšek EBP (monthly, back to 1973)")
    print(f"{'='*60}")
    raw["EBP"] = fetch_ebp(
        config.cache_dir,
        config.ebp_url,
        config.ebp_local_path,
        config.request_timeout,
    )
    if not raw["EBP"].empty:
        s = raw["EBP"]
        print(f"  OK  EBP                 {len(s):>5} monthly obs  "
              f"{s.index[0].date()} to {s.index[-1].date()}")
    else:
        print("  --  EBP: not available — bear model will run without it.")
        print("      Supply via DataConfig(ebp_local_path=...)")

    return raw


# ---------------------------------------------------------------------------
# Diagnostic summary
# ---------------------------------------------------------------------------

def summarize_raw(raw: dict[str, pd.Series]) -> pd.DataFrame:
    """
    Return a diagnostic DataFrame with coverage and quality for every series.
    Call after fetch_all_raw().
    """
    rows = []
    for name, s in raw.items():
        lag = publication_lag(name)
        if s is None or s.empty:
            rows.append({
                "Series":      name,
                "Obs":         0,
                "Start":       "-",
                "End":         "-",
                "NaN %":       "-",
                "Pub lag (d)": lag,
                "Status":      "MISSING",
            })
            continue
        rows.append({
            "Series":      name,
            "Obs":         len(s),
            "Start":       str(s.index[0].date()),
            "End":         str(s.index[-1].date()),
            "NaN %":       f"{s.isna().mean() * 100:.1f}%",
            "Pub lag (d)": lag,
            "Status":      "OK",
        })
    return pd.DataFrame(rows).sort_values("Series").reset_index(drop=True)


# ---------------------------------------------------------------------------
# API key resolution
# ---------------------------------------------------------------------------

def _resolve_fred_api_key() -> Optional[str]:
    """
    Resolve FRED API key from (in order):
      1. FRED_API_KEY environment variable
      2. .streamlit/secrets.toml  (FRED_API_KEY = "...")
      3. .streamlit/secrets.toml  (fred.api_key = "...")
    """
    key = os.environ.get("FRED_API_KEY", "").strip()
    if key:
        return key

    search = Path(__file__).resolve().parent
    for _ in range(5):
        candidate = search / ".streamlit" / "secrets.toml"
        if candidate.exists():
            try:
                import tomllib  # Python 3.11+
            except ImportError:
                try:
                    import tomli as tomllib  # type: ignore[no-redef]
                except ImportError:
                    tomllib = None  # type: ignore[assignment]

            if tomllib is not None:
                data = tomllib.loads(candidate.read_text())
            else:
                # Minimal TOML key=value parser (no external dependency)
                data: dict = {}
                for line in candidate.read_text().splitlines():
                    line = line.strip()
                    if "=" in line and not line.startswith("#"):
                        k, _, v = line.partition("=")
                        data[k.strip()] = v.strip().strip('"').strip("'")

            key = data.get("FRED_API_KEY") or (
                data.get("fred", {}).get("api_key")
                if isinstance(data.get("fred"), dict) else None
            )
            if key:
                return str(key).strip()
        search = search.parent

    return None


# ---------------------------------------------------------------------------
# Monthly alignment
# ---------------------------------------------------------------------------

def align_to_monthly(raw: dict[str, pd.Series]) -> pd.DataFrame:
    """
    Resample all raw series to calendar month-end and join into one DataFrame.

    Rules
    -----
    - Daily and weekly series : last available observation within each calendar
      month (month-end snapshot).  Missing trading days at month-end are
      filled by the last available reading during that month.
    - Monthly series           : same resample; months already containing one
      observation are passed through unchanged (date snapped to month-end).

    No publication lags are applied here — this function is for raw-data
    validation only.  Lag-shifting happens in Phase 2 (features.py).

    Returns
    -------
    pd.DataFrame
        Index : monthly period-end dates (e.g. 1995-01-31)
        Columns : one column per series, in the order they appear in `raw`
    """
    segments: dict[str, pd.Series] = {}
    for name, s in raw.items():
        if s is None or s.empty:
            continue
        # Ensure DatetimeIndex before resampling
        s = s.copy()
        s.index = pd.to_datetime(s.index)
        segments[name] = s.resample("ME").last()

    if not segments:
        raise ValueError("No non-empty series available to align.")

    df = pd.concat(segments, axis=1, sort=True)
    df.index.name = "date"
    df.index = pd.to_datetime(df.index)

    # Column order: yield curve → credit → labor → leading → policy →
    # volatility → financial conditions → valuation → sentiment → price
    preferred_order = [
        "NTFS", "T10Y3M", "T10Y2Y", "DGS3MO", "DGS10",
        "BAA10Y", "BAMLH0A0HYM2", "EBP",
        "SAHMREALTIME", "UNRATE", "ICSA",
        "USALOLITOAASTSAM", "DFF",
        "VIXCLS", "VXVCLS",
        "ANFCI", "NFCI",
        "SHILLER_CAPE", "CPCE",
        "SPX", "SP500",
    ]
    ordered = [c for c in preferred_order if c in df.columns]
    remainder = [c for c in df.columns if c not in preferred_order]
    df = df[ordered + remainder]

    return df


# ---------------------------------------------------------------------------
# Smoke-test entry point  (python -m bear.data)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    api_key = _resolve_fred_api_key()
    if not api_key:
        print(
            "FRED API key not found.\n"
            "Provide it via:\n"
            "  * FRED_API_KEY environment variable\n"
            "  * .streamlit/secrets.toml  ->  FRED_API_KEY = \"<key>\""
        )
        sys.exit(1)

    _bear_dir = Path(__file__).resolve().parent
    cfg = DataConfig(
        fred_api_key=api_key,
        start=date(1900, 1, 1),
        end=date.today(),
        cache_dir=_bear_dir / "cache",
    )

    raw = fetch_all_raw(cfg)

    print(f"\n{'='*60}")
    print("  Coverage summary")
    print(f"{'='*60}")
    summary = summarize_raw(raw)
    print(summary.to_string(index=False))
    ok_count = sum(1 for s in raw.values() if s is not None and not s.empty)
    print(f"\n  Total series fetched: {ok_count} / {len(raw)}")

    # -- Monthly alignment & CSV export --
    print(f"\n{'='*60}")
    print("  Aligning to monthly (month-end) and exporting CSV")
    print(f"{'='*60}")
    monthly_df = align_to_monthly(raw)
    # Clip to cfg.start so SHILLER_CAPE (1900) doesn't dominate the index
    monthly_df = monthly_df[monthly_df.index >= pd.Timestamp(cfg.start)]
    out_path = _bear_dir / "raw_monthly.csv"
    monthly_df.to_csv(out_path, date_format="%Y-%m-%d", float_format="%.6f")
    print(f"  Wrote {len(monthly_df)} rows x {len(monthly_df.columns)} columns")
    print(f"  Date range : {monthly_df.index[0].date()} to {monthly_df.index[-1].date()}")
    print(f"  File       : {out_path.resolve()}")
    print(f"\n  Column coverage (non-NaN % per column):")
    coverage = (monthly_df.notna().mean() * 100).rename("non-NaN %").round(1)
    print(coverage.to_string())
