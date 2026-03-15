from io import StringIO

import pandas as pd
import requests
import structlog
import yfinance as yf

from app.core.exceptions import raise_server_error

log = structlog.get_logger()

_WIKIPEDIA_HEADERS = {"User-Agent": "Mozilla/5.0"}
_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def _fetch_sp500_table() -> pd.DataFrame:
    """Fetch the S&P 500 Wikipedia table, working around the 403 on the default user-agent."""
    resp = requests.get(_SP500_URL, headers=_WIKIPEDIA_HEADERS, timeout=15)
    resp.raise_for_status()
    return pd.read_html(StringIO(resp.text))[0]


class YFinanceProvider:
    def fetch_price_data(self, tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch OHLCV price data for given tickers via yfinance."""
        log.info("fetching_price_data", tickers=tickers, start=start_date, end=end_date)
        data = yf.download(tickers, start=start_date, end=end_date, group_by="ticker", threads=True)

        if not isinstance(data, pd.DataFrame):
            raise raise_server_error(detail="yfinance returned an unexpected data type")

        if data.empty:
            log.warning("no_price_data_returned", tickers=tickers)
            return data

        if len(tickers) > 1:
            for ticker in tickers:
                if ticker in data.columns.get_level_values(0):
                    if data[ticker].isna().all().all():
                        log.warning("no_data_for_ticker", ticker=ticker)

        return data

    def fetch_fundamental_data(self, tickers: list[str]) -> pd.DataFrame:
        """Fetch fundamental data (P/E, P/B, market cap, etc.) for given tickers."""
        log.info("fetching_fundamental_data", tickers=tickers)
        fields = ["trailingPE", "priceToBook", "marketCap", "forwardPE", "dividendYield", "returnOnEquity"]
        records = []

        for ticker in tickers:
            try:
                info = yf.Ticker(ticker).info
                record = {"ticker": ticker}
                for field in fields:
                    record[field] = info.get(field)
                records.append(record)
            except Exception:
                log.warning("failed_to_fetch_fundamentals", ticker=ticker, exc_info=True)

        if not records:
            return pd.DataFrame()

        return pd.DataFrame(records).set_index("ticker")

    def get_sp500_tickers(self) -> list[str]:
        """Fetch current S&P 500 constituent tickers from Wikipedia."""
        log.info("fetching_sp500_tickers")
        try:
            df = _fetch_sp500_table()
            tickers = [t.replace(".", "-") for t in df["Symbol"].tolist()]
            return sorted(tickers)
        except Exception:
            log.warning("failed_to_fetch_sp500_tickers", exc_info=True)
            return []

    def get_sp500_instruments(self) -> list[dict]:
        """Fetch S&P 500 constituents with GICS classification from Wikipedia.

        Returns a list of dicts with keys: ticker, company_name, gics_sector, gics_sub_industry.
        """
        log.info("fetching_sp500_instruments")
        try:
            df = _fetch_sp500_table()[["Symbol", "Security", "GICS Sector", "GICS Sub-Industry"]].copy()
            df = df.rename(
                columns={
                    "Symbol": "ticker",
                    "Security": "company_name",
                    "GICS Sector": "gics_sector",
                    "GICS Sub-Industry": "gics_sub_industry",
                }
            )
            df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)
            return df.to_dict("records")
        except Exception:
            log.warning("failed_to_fetch_sp500_instruments", exc_info=True)
            return []
