from pathlib import Path

import pandas as pd
import structlog
import yfinance as yf

from app.core.exceptions import raise_server_error

log = structlog.get_logger()

DEFAULT_CACHE_DIR = Path(".cache/market_data")


class MarketDataFetcher:
    def __init__(self, cache_dir: Path = DEFAULT_CACHE_DIR):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

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
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        try:
            tables = pd.read_html(url)
            tickers = tables[0]["Symbol"].tolist()
            tickers = [t.replace(".", "-") for t in tickers]
            return sorted(tickers)
        except Exception:
            log.warning("failed_to_fetch_sp500_tickers", exc_info=True)
            return []

    def cache_data(self, data: pd.DataFrame, cache_key: str) -> None:
        """Save DataFrame to disk as parquet."""
        path = self.cache_dir / f"{cache_key}.parquet"
        data.to_parquet(path)
        log.info("data_cached", cache_key=cache_key, path=str(path))

    def load_cached_data(self, cache_key: str) -> pd.DataFrame | None:
        """Load cached DataFrame from disk, or return None if not found."""
        path = self.cache_dir / f"{cache_key}.parquet"
        if not path.exists():
            return None
        return pd.read_parquet(path)
