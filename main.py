from datetime import date

import pandas as pd

from DataConnectors import DataConnectors

TICKERS = [
    "XLV", "XLP", "XLU", "XLE", "XLB", "XLI", "XLY", "XLF", "XLRE", "XLK",
    "SMH", "SPHQ", "MTUM", "USMV", "VLUE", "IWM", "SCHD", "VOO",
]

START = date(2021, 1, 1)
END = date(2026, 5, 29)


def fetch_monthly_adj_close() -> pd.DataFrame:
    yf = DataConnectors.yahoo_finance()
    yf.connect()

    data: dict[str, dict[date, float]] = {}
    for ticker in TICKERS:
        records = yf.fetch_ohlcv(ticker, start=START, end=END, interval="1mo")
        data[ticker] = {rec.date: rec.adj_close for rec in records}

    df = pd.DataFrame(data)
    df.index.name = "date"
    df.sort_index(inplace=True)
    return df


if __name__ == "__main__":
    df = fetch_monthly_adj_close()
    df.to_csv("etf_mon_data.csv")
    print(df.to_string())
