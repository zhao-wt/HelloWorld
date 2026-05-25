from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from typing import Any, Optional
import os


# --- Shared low-level records ---

@dataclass
class OHLCVRecord:
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: int


@dataclass
class SeriesRecord:
    series_id: str
    date: date
    value: float


@dataclass
class ReleaseRecord:
    series_id: str
    release_date: date
    value: float
    vintage_date: date


# --- Abstract base ---

class DataConnector(ABC):

    @abstractmethod
    def connect(self, **credentials: Any) -> None:
        ...

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        ...


# --- Concrete connectors ---

class YahooFinanceConnector(DataConnector):
    """
    Public Yahoo Finance API via yfinance. No credentials required.
    Covers equities, ETFs, indices, FX, crypto, and options.
    """
    _instance: Optional["YahooFinanceConnector"] = None

    def __new__(cls) -> "YahooFinanceConnector":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._connected = False
        return cls._instance

    def connect(self, **credentials: Any) -> None:
        self._connected = True

    @property
    def is_connected(self) -> bool:
        return self._connected

    def fetch_ohlcv(
        self, symbol: str, start: date, end: date, interval: str = "1d"
    ) -> list[OHLCVRecord]:
        """OHLCV bars. interval: 1m 2m 5m 15m 30m 60m 90m 1h 1d 5d 1wk 1mo 3mo"""
        ...

    def fetch_quote(self, symbol: str) -> dict[str, Any]:
        """Real-time bid/ask/last/volume snapshot."""
        ...

    def fetch_info(self, symbol: str) -> dict[str, Any]:
        """Fundamentals, company metadata, and sector info."""
        ...

    def fetch_options_chain(
        self, symbol: str, expiry: Optional[date] = None
    ) -> dict[str, Any]:
        """Calls and puts. Omit expiry to get the nearest expiration."""
        ...


class FREDConnector(DataConnector):
    """
    Federal Reserve Economic Data (FRED) — St. Louis Fed.
    Requires a free API key: https://fred.stlouisfed.org/docs/api/api_key.html
    Set FRED_API_KEY env var or pass api_key to connect().
    """
    _instance: Optional["FREDConnector"] = None

    def __new__(cls) -> "FREDConnector":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._connected = False
            cls._instance._api_key: Optional[str] = None
        return cls._instance

    def connect(self, api_key: Optional[str] = None, **_: Any) -> None:
        self._api_key = api_key or os.getenv("FRED_API_KEY")
        if not self._api_key:
            raise ValueError(
                "FRED API key required — pass api_key= or set FRED_API_KEY env var"
            )
        self._connected = True

    @property
    def is_connected(self) -> bool:
        return self._connected

    def fetch_series(
        self, series_id: str, start: date, end: date
    ) -> list[SeriesRecord]:
        """Historical observations for a series (e.g. 'CPIAUCSL', 'GDP', 'UNRATE')."""
        ...

    def fetch_series_info(self, series_id: str) -> dict[str, Any]:
        """Metadata: title, frequency, units, seasonal adjustment, source."""
        ...

    def fetch_releases(self, start: date, end: date) -> list[ReleaseRecord]:
        """All data releases published within a date range."""
        ...

    def search(self, query: str, limit: int = 100) -> list[dict[str, Any]]:
        """Full-text search across FRED series catalog."""
        ...


class CMEDataMineConnector(DataConnector):
    """
    CME Group DataMine — institutional futures and options data.
    Requires a CME DataMine subscription and login credentials.
    Set CME_DATAMINE_USERNAME and CME_DATAMINE_PASSWORD env vars,
    or pass username= and password= to connect().
    """
    _instance: Optional["CMEDataMineConnector"] = None

    def __new__(cls) -> "CMEDataMineConnector":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._connected = False
            cls._instance._username: Optional[str] = None
            cls._instance._password: Optional[str] = None
        return cls._instance

    def connect(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None,
        **_: Any,
    ) -> None:
        self._username = username or os.getenv("CME_DATAMINE_USERNAME")
        self._password = password or os.getenv("CME_DATAMINE_PASSWORD")
        if not (self._username and self._password):
            raise ValueError(
                "CME DataMine credentials required — pass username=/password= "
                "or set CME_DATAMINE_USERNAME/CME_DATAMINE_PASSWORD env vars"
            )
        self._connected = True

    @property
    def is_connected(self) -> bool:
        return self._connected

    def fetch_ohlcv(
        self, product: str, start: date, end: date
    ) -> list[OHLCVRecord]:
        """Daily OHLCV for a CME product code (e.g. 'ES', 'NQ', 'CL')."""
        ...

    def fetch_tick_data(
        self, product: str, trade_date: date
    ) -> list[dict[str, Any]]:
        """Intraday tick-by-tick data for a single trading day."""
        ...

    def fetch_settlements(
        self, product: str, trade_date: date
    ) -> list[dict[str, Any]]:
        """Official daily settlement prices across all contract months."""
        ...

    def list_products(self) -> list[dict[str, Any]]:
        """Catalog of all products available under the current subscription."""
        ...


class NasdaqDataLinkConnector(DataConnector):
    """
    Nasdaq Data Link (formerly Quandl) — curated financial and alternative data.
    Requires a free or premium API key: https://data.nasdaq.com/
    Set NASDAQ_DATA_LINK_API_KEY env var or pass api_key to connect().
    """
    _instance: Optional["NasdaqDataLinkConnector"] = None

    def __new__(cls) -> "NasdaqDataLinkConnector":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._connected = False
            cls._instance._api_key: Optional[str] = None
        return cls._instance

    def connect(self, api_key: Optional[str] = None, **_: Any) -> None:
        self._api_key = api_key or os.getenv("NASDAQ_DATA_LINK_API_KEY")
        if not self._api_key:
            raise ValueError(
                "Nasdaq Data Link API key required — pass api_key= or set "
                "NASDAQ_DATA_LINK_API_KEY env var"
            )
        self._connected = True

    @property
    def is_connected(self) -> bool:
        return self._connected

    def fetch_dataset(
        self, code: str, start: date, end: date
    ) -> list[dict[str, Any]]:
        """Time-series dataset by database/code slug (e.g. 'WIKI/AAPL', 'FRED/GDP')."""
        ...

    def fetch_table(
        self, datatable: str, **filters: Any
    ) -> list[dict[str, Any]]:
        """Tabular dataset with optional column filters (e.g. datatable='ZACKS/FC')."""
        ...

    def fetch_metadata(self, code: str) -> dict[str, Any]:
        """Dataset description, column definitions, frequency, and provider."""
        ...

    def search(self, query: str) -> list[dict[str, Any]]:
        """Search the Nasdaq Data Link catalog by keyword."""
        ...


# --- Registry ---

class DataConnectors:
    """Entry point for all data source connectors."""

    @staticmethod
    def yahoo_finance() -> YahooFinanceConnector:
        return YahooFinanceConnector()

    @staticmethod
    def fred() -> FREDConnector:
        return FREDConnector()

    @staticmethod
    def cme_datamine() -> CMEDataMineConnector:
        return CMEDataMineConnector()

    @staticmethod
    def nasdaq_data_link() -> NasdaqDataLinkConnector:
        return NasdaqDataLinkConnector()


if __name__ == "__main__":
    yf = DataConnectors.yahoo_finance()
    yf.connect()
    assert yf.is_connected
    assert DataConnectors.yahoo_finance() is yf   # singletons

    fred = DataConnectors.fred()
    cme = DataConnectors.cme_datamine()
    ndl = DataConnectors.nasdaq_data_link()
    assert DataConnectors.fred() is fred
    assert DataConnectors.cme_datamine() is cme
    assert DataConnectors.nasdaq_data_link() is ndl
