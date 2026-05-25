from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from typing import Optional


# --- Data structures ---

@dataclass
class Bar:
    symbol: str
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: int


@dataclass
class Quote:
    symbol: str
    bid: float
    ask: float
    last: float
    volume: int


@dataclass
class Fundamentals:
    symbol: str
    market_cap: float
    pe_ratio: Optional[float]
    eps: Optional[float]
    dividend_yield: Optional[float]


@dataclass
class FutureContract:
    symbol: str
    root: str
    expiry: date
    multiplier: float
    exchange: str


@dataclass
class RollDate:
    root: str
    front: str
    back: str
    roll_date: date


@dataclass
class OptionContract:
    symbol: str
    underlying: str
    expiry: date
    strike: float
    right: str          # 'C' or 'P'
    bid: float
    ask: float
    volume: int
    open_interest: int
    implied_vol: float


@dataclass
class Greeks:
    symbol: str
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float


@dataclass
class EconomicIndicator:
    code: str           # e.g. 'GDP', 'CPI', 'UNRATE'
    name: str
    region: str         # e.g. 'US', 'EU', 'CN'
    date: date
    value: float
    unit: str           # e.g. '%', 'USD Billions'
    frequency: str      # 'monthly', 'quarterly', 'annual'


@dataclass
class EconomicRelease:
    code: str
    name: str
    region: str
    release_date: date
    actual: Optional[float]
    forecast: Optional[float]
    previous: Optional[float]
    unit: str
    importance: str     # 'low', 'medium', 'high'


# --- Abstract base ---

class DataFactory(ABC):

    @abstractmethod
    def get_ohlcv(self, symbol: str, start: date, end: date) -> list[Bar]:
        ...

    @abstractmethod
    def get_quote(self, symbol: str) -> Quote:
        ...


# --- Concrete factories ---

class StockDataFactory(DataFactory):
    _instance: Optional["StockDataFactory"] = None

    def __new__(cls) -> "StockDataFactory":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_ohlcv(self, symbol: str, start: date, end: date) -> list[Bar]:
        ...

    def get_quote(self, symbol: str) -> Quote:
        ...

    def get_fundamentals(self, symbol: str) -> Fundamentals:
        ...


class FutureDataFactory(DataFactory):
    _instance: Optional["FutureDataFactory"] = None

    def __new__(cls) -> "FutureDataFactory":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_ohlcv(self, symbol: str, start: date, end: date) -> list[Bar]:
        ...

    def get_quote(self, symbol: str) -> Quote:
        ...

    def get_contracts(self, root: str) -> list[FutureContract]:
        """All listed contracts for a root symbol (e.g. 'ES')."""
        ...

    def get_front_month(self, root: str) -> str:
        """Symbol of the current front-month contract."""
        ...

    def get_roll_schedule(self, root: str) -> list[RollDate]:
        ...


class OptionDataFactory(DataFactory):
    _instance: Optional["OptionDataFactory"] = None

    def __new__(cls) -> "OptionDataFactory":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_ohlcv(self, symbol: str, start: date, end: date) -> list[Bar]:
        ...

    def get_quote(self, symbol: str) -> Quote:
        ...

    def get_chain(
        self, underlying: str, expiry: Optional[date] = None
    ) -> list[OptionContract]:
        """Full option chain, optionally filtered to a single expiry."""
        ...

    def get_expirations(self, underlying: str) -> list[date]:
        ...

    def get_greeks(self, symbol: str) -> Greeks:
        ...


class EconomicsDataFactory:
    _instance: Optional["EconomicsDataFactory"] = None

    def __new__(cls) -> "EconomicsDataFactory":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_indicator(
        self, code: str, start: date, end: date, region: str = "US"
    ) -> list[EconomicIndicator]:
        """Historical time series for an indicator (e.g. 'CPI', 'GDP')."""
        ...

    def get_calendar(self, start: date, end: date) -> list[EconomicRelease]:
        """Scheduled and past releases within a date range."""
        ...

    def get_release(self, code: str, region: str = "US") -> EconomicRelease:
        """Most recent release for a specific indicator."""
        ...

    def search(self, query: str, region: Optional[str] = None) -> list[EconomicIndicator]:
        """Search available indicators by keyword or partial code."""
        ...


# --- Registry ---

class DataFactories:
    """Entry point for all instrument data factories."""

    @staticmethod
    def stocks() -> StockDataFactory:
        return StockDataFactory()

    @staticmethod
    def futures() -> FutureDataFactory:
        return FutureDataFactory()

    @staticmethod
    def options() -> OptionDataFactory:
        return OptionDataFactory()

    @staticmethod
    def economics() -> EconomicsDataFactory:
        return EconomicsDataFactory()


if __name__ == "__main__":
    stocks = DataFactories.stocks()
    futures = DataFactories.futures()
    options = DataFactories.options()
    print(stocks, futures, options)
    assert DataFactories.stocks() is stocks   # singletons
    assert DataFactories.futures() is futures
    assert DataFactories.options() is options
    economics = DataFactories.economics()
    assert DataFactories.economics() is economics
