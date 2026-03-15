from app.services.market_data.base import MarketDataProvider
from app.services.market_data.cache import CachedMarketDataProvider
from app.services.market_data.yfinance import YFinanceProvider

_PROVIDERS = {
    "yfinance": YFinanceProvider,
}


def get_market_data_provider() -> MarketDataProvider:
    from app.core.config import config

    cls = _PROVIDERS.get(config.MARKET_DATA_PROVIDER)
    if cls is None:
        raise ValueError(
            f"Unknown market data provider: {config.MARKET_DATA_PROVIDER!r}. Choose from: {list(_PROVIDERS)}"
        )
    return CachedMarketDataProvider(cls())


__all__ = ["MarketDataProvider", "YFinanceProvider", "CachedMarketDataProvider", "get_market_data_provider"]
