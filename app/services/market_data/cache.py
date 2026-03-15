import hashlib
from pathlib import Path

import pandas as pd
import structlog

from app.services.market_data.base import MarketDataProvider

log = structlog.get_logger()

DEFAULT_CACHE_DIR = Path(".cache/market_data")


class CachedMarketDataProvider:
    """Wraps any MarketDataProvider with transparent parquet disk caching."""

    def __init__(self, provider: MarketDataProvider, cache_dir: Path = DEFAULT_CACHE_DIR):
        self.provider = provider
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_price_data(self, tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        key = self._key(f"price_{sorted(tickers)}_{start_date}_{end_date}")
        cached = self.load_cached_data(key)
        if cached is not None:
            return cached
        data = self.provider.fetch_price_data(tickers, start_date, end_date)
        if not data.empty:
            self.cache_data(data, key)
        return data

    def fetch_fundamental_data(self, tickers: list[str]) -> pd.DataFrame:
        key = self._key(f"fundamental_{sorted(tickers)}")
        cached = self.load_cached_data(key)
        if cached is not None:
            return cached
        data = self.provider.fetch_fundamental_data(tickers)
        if not data.empty:
            self.cache_data(data, key)
        return data

    def cache_data(self, data: pd.DataFrame, cache_key: str) -> None:
        path = self.cache_dir / f"{cache_key}.parquet"
        data.to_parquet(path)
        log.info("data_cached", cache_key=cache_key, path=str(path))

    def load_cached_data(self, cache_key: str) -> pd.DataFrame | None:
        path = self.cache_dir / f"{cache_key}.parquet"
        if not path.exists():
            return None
        return pd.read_parquet(path)

    @staticmethod
    def _key(raw: str) -> str:
        return hashlib.md5(raw.encode()).hexdigest()
