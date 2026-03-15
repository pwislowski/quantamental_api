# Backward-compatible shim. All existing callers import MarketDataFetcher from here.
# Calling MarketDataFetcher() returns the provider configured by MARKET_DATA_PROVIDER env var.
from app.services.market_data import get_market_data_provider

MarketDataFetcher = get_market_data_provider
