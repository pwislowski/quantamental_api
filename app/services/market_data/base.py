from typing import Protocol

import pandas as pd


class MarketDataProvider(Protocol):
    def fetch_price_data(self, tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame: ...
    def fetch_fundamental_data(self, tickers: list[str]) -> pd.DataFrame: ...
