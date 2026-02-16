import numpy as np
import pandas as pd
import structlog
from scipy.stats import zscore

from app.services.data_pipeline import MarketDataFetcher

log = structlog.get_logger()

TRADING_DAYS_PER_YEAR = 252


class FactorEngine:
    """Calculates common equity factors from price and fundamental data.

    All methods are synchronous (CPU-bound). Use run_in_threadpool
    when calling from async routes.
    """

    def __init__(self, fetcher: MarketDataFetcher):
        self.fetcher = fetcher

    def calculate_value_factor(self, price_data: pd.DataFrame, fundamental_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate value factor as earnings-to-price ratio (inverse P/E)."""
        if "trailingPE" not in fundamental_data.columns:
            return pd.DataFrame(columns=["value"])
        pe = fundamental_data["trailingPE"].copy()
        pe = pe[pe > 0]
        ep = 1.0 / pe
        return pd.DataFrame({"value": ep}, index=ep.index)  # type:ignore

    def calculate_momentum_factor(self, price_data: pd.DataFrame, lookback_months: int = 12) -> pd.DataFrame:
        """Calculate momentum factor as trailing 12-1 return."""
        lookback_days = lookback_months * 21
        skip_days = 21

        tickers = self._get_tickers_from_price_data(price_data)
        if not tickers:
            return pd.DataFrame(columns=["momentum"])

        records = {}
        for ticker in tickers:
            close = self._get_close_prices(price_data, ticker, tickers)
            if close is None or len(close) < lookback_days:
                continue
            start_price = close.iloc[-lookback_days]
            end_price = close.iloc[-skip_days] if len(close) > skip_days else close.iloc[-1]
            if start_price > 0:
                records[ticker] = (end_price / start_price) - 1.0

        return pd.DataFrame({"momentum": pd.Series(records)})

    def calculate_quality_factor(self, fundamental_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate quality factor based on return on equity (ROE)."""
        if "returnOnEquity" not in fundamental_data.columns:
            return pd.DataFrame(columns=["quality"])
        roe = fundamental_data["returnOnEquity"].dropna()
        return pd.DataFrame({"quality": roe}, index=roe.index)

    def calculate_low_vol_factor(self, price_data: pd.DataFrame, lookback_months: int = 12) -> pd.DataFrame:
        """Calculate low-volatility factor as inverse annualized volatility."""
        lookback_days = lookback_months * 21
        tickers = self._get_tickers_from_price_data(price_data)
        if not tickers:
            return pd.DataFrame(columns=["low_vol"])

        records = {}
        for ticker in tickers:
            close = self._get_close_prices(price_data, ticker, tickers)
            if close is None or len(close) < lookback_days:
                continue
            returns = close.iloc[-lookback_days:].pct_change().dropna()
            if len(returns) < 2:
                continue
            annual_vol = returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
            if annual_vol > 0:
                records[ticker] = 1.0 / annual_vol

        return pd.DataFrame({"low_vol": pd.Series(records)})

    def calculate_size_factor(self, fundamental_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate size factor as negative log market cap."""
        if "marketCap" not in fundamental_data.columns:
            return pd.DataFrame(columns=["size"])
        mcap = fundamental_data["marketCap"].dropna()
        mcap = mcap[mcap > 0]
        size = -np.log(mcap)
        return pd.DataFrame({"size": size}, index=size.index)  # type:ignore

    def standardize_factor(self, factor_values: pd.Series) -> pd.Series:
        """Standardize factor values using z-score, clipped at +/-3."""
        clean = factor_values.dropna()
        if len(clean) < 2:
            return factor_values
        scores = pd.Series(zscore(clean, nan_policy="omit"), index=clean.index)
        return scores.clip(-3, 3)

    def calculate_all_factors(self, tickers: list[str], start_date: str, end_date: str) -> dict[str, pd.DataFrame]:
        """Fetch data and calculate all factors for the given universe."""
        log.info("calculating_all_factors", tickers_count=len(tickers), start=start_date, end=end_date)

        price_data = self.fetcher.fetch_price_data(tickers, start_date, end_date)
        fundamental_data = self.fetcher.fetch_fundamental_data(tickers)

        factor_methods = {
            "value": lambda: self.calculate_value_factor(price_data, fundamental_data),
            "momentum": lambda: self.calculate_momentum_factor(price_data),
            "quality": lambda: self.calculate_quality_factor(fundamental_data),
            "low_vol": lambda: self.calculate_low_vol_factor(price_data),
            "size": lambda: self.calculate_size_factor(fundamental_data),
        }

        results = {}
        for name, method in factor_methods.items():
            df = method()
            if not df.empty and name in df.columns:
                series = df[name]
                assert isinstance(series, pd.Series)
                df[name] = self.standardize_factor(series)
            results[name] = df

        return results

    def _get_tickers_from_price_data(self, price_data: pd.DataFrame) -> list[str]:
        if price_data.empty:
            return []
        if isinstance(price_data.columns, pd.MultiIndex):
            return list(price_data.columns.get_level_values(0).unique())
        return []

    def _get_close_prices(self, price_data: pd.DataFrame, ticker: str, all_tickers: list[str]) -> pd.Series | None:
        try:
            if isinstance(price_data.columns, pd.MultiIndex):
                close = price_data[(ticker, "Close")].dropna()
            else:
                close = price_data["Close"].dropna()

            assert isinstance(close, pd.Series)

            return close if not close.empty else None
        except KeyError:
            return None
