from datetime import date as Date

import numpy as np
import pandas as pd
import structlog
from sqlalchemy.orm import Session

from app.models.backtest import Backtest, BacktestInstrument, BacktestResult
from app.models.schemas import BacktestRequest
from app.services.factor_engine import FactorEngine
from app.services.market_data.base import MarketDataProvider
from app.services.optimizer import PortfolioOptimizer

log = structlog.get_logger()

TRADING_DAYS_PER_YEAR = 252


class Backtester:
    """Walk-forward backtest engine. Synchronous, CPU-bound."""

    def __init__(self, optimizer: PortfolioOptimizer, fetcher: MarketDataProvider, factor_engine: FactorEngine):
        self.optimizer = optimizer
        self.fetcher = fetcher
        self.factor_engine = factor_engine

    def run(self, config: BacktestRequest, session: Session) -> dict:
        """Run a full backtest: fetch data, simulate, persist results."""
        log.info("backtest_starting", name=config.name, strategy=config.strategy_type)

        # Fetch price data
        price_data = self.fetcher.fetch_price_data(config.tickers, config.start_date, config.end_date)

        # Extract close prices and compute returns
        if isinstance(price_data.columns, pd.MultiIndex):
            close = price_data.xs("Close", axis=1, level=1)
        else:
            close = price_data[["Close"]]
        returns = close.pct_change().dropna()

        # Generate rebalance dates and walk forward
        rebalance_dates = self._generate_rebalance_dates(returns.index, config.rebalance_frequency)
        nav_series, holdings_history = self._walk_forward(returns, rebalance_dates, config)

        # Calculate metrics
        metrics = self.calculate_metrics(nav_series)

        # Serialize time series for JSON storage
        nav_list = [{"date": str(d.date()), "nav": float(v)} for d, v in nav_series.items()]
        daily_returns = nav_series.pct_change().fillna(0.0)
        returns_list = [{"date": str(d.date()), "return": float(v)} for d, v in daily_returns.items()]

        # Persist to DB
        parameters = {
            "constraints": config.constraints,
            "risk_aversion": config.risk_aversion,
            "tilts": config.tilts,
            "tilt_strength": config.tilt_strength,
        }
        backtest = Backtest(
            name=config.name,
            strategy_type=config.strategy_type,
            start_date=Date.fromisoformat(config.start_date),
            end_date=Date.fromisoformat(config.end_date),
            rebalance_frequency=config.rebalance_frequency,
            parameters=parameters,
        )
        session.add(backtest)
        session.flush()

        for ticker in config.tickers:
            session.add(BacktestInstrument(backtest_id=backtest.id, ticker=ticker))

        result = BacktestResult(
            backtest_id=backtest.id,
            total_return=metrics["total_return"],
            annualized_return=metrics["annualized_return"],
            annualized_volatility=metrics["annualized_volatility"],
            sharpe_ratio=metrics["sharpe_ratio"],
            max_drawdown=metrics["max_drawdown"],
            sortino_ratio=metrics["sortino_ratio"],
            calmar_ratio=metrics["calmar_ratio"],
            nav_series=nav_list,
            returns_series=returns_list,
            holdings_history=holdings_history,
        )
        session.add(result)
        session.commit()

        log.info("backtest_complete", name=config.name, total_return=metrics["total_return"])

        return {
            "backtest_id": backtest.id,
            "metrics": metrics,
            "nav_series": nav_list,
            "holdings_history": holdings_history,
        }

    def _walk_forward(
        self, returns: pd.DataFrame, rebalance_dates: list[pd.Timestamp], config: BacktestRequest
    ) -> tuple[pd.Series, list[dict]]:
        """Walk forward through time, rebalancing at specified dates."""
        tickers = list(returns.columns)
        n = len(tickers)

        # Start with equal weights
        current_weights = {t: 1.0 / n for t in tickers}
        nav = 1.0
        nav_values = []
        holdings_history = []
        rebalance_set = set(rebalance_dates)

        for day in returns.index:
            # Rebalance if this is a rebalance date
            if day in rebalance_set:
                lookback_returns = returns.loc[:day]
                current_weights = self._optimize_at_date(lookback_returns, config)
                holdings_history.append({"date": str(day.date()), "weights": dict(current_weights)})

            # Record NAV at start of day (before applying this day's return)
            nav_values.append(nav)

            # Compute portfolio return for this day
            day_returns = returns.loc[day]
            portfolio_return = sum(current_weights.get(t, 0.0) * day_returns.get(t, 0.0) for t in tickers)
            nav *= 1 + portfolio_return

            # Drift weights
            current_weights = self._drift_weights(current_weights, day_returns)

        nav_series = pd.Series(nav_values, index=returns.index)
        return nav_series, holdings_history

    def _optimize_at_date(self, returns: pd.DataFrame, config: BacktestRequest) -> dict[str, float]:
        """Run optimizer with data available up to current date."""
        if config.strategy_type == "mean_variance":
            weights = self.optimizer.mean_variance_optimization(returns, config.risk_aversion)
        elif config.strategy_type == "risk_parity":
            weights = self.optimizer.risk_parity_optimization(returns)
        elif config.strategy_type == "factor_tilt":
            if not config.tilts:
                weights = self.optimizer.mean_variance_optimization(returns, config.risk_aversion)
            else:
                tickers = list(returns.columns)
                factors = self.factor_engine.calculate_all_factors(tickers, config.start_date, config.end_date)
                factor_dfs = {}
                for name, df in factors.items():
                    if not df.empty and name in df.columns:
                        factor_dfs[name] = df[name]
                factor_exposures = pd.DataFrame(factor_dfs) if factor_dfs else pd.DataFrame()

                if factor_exposures.empty:
                    weights = self.optimizer.mean_variance_optimization(returns, config.risk_aversion)
                else:
                    weights = self.optimizer.factor_tilt_optimization(
                        returns, factor_exposures, config.tilts, config.tilt_strength, config.risk_aversion
                    )
        else:
            # Fallback to equal weight
            tickers = list(returns.columns)
            weights = {t: 1.0 / len(tickers) for t in tickers}

        if config.constraints:
            weights = self.optimizer.apply_constraints(weights, config.constraints)

        return weights

    @staticmethod
    def _generate_rebalance_dates(index: pd.DatetimeIndex, frequency: str) -> list[pd.Timestamp]:
        """Generate rebalance dates from a trading calendar index."""
        if len(index) == 0:
            return []

        freq_map = {"monthly": "ME", "quarterly": "QE", "yearly": "YE"}
        freq = freq_map.get(frequency, "ME")

        # Group by period and take the last trading day in each period
        periods = index.to_period(freq[0])  # 'M', 'Q', or 'Y'
        rebalance_dates = []
        for period in periods.unique():
            mask = periods == period
            period_dates = index[mask]
            if len(period_dates) > 0:
                rebalance_dates.append(period_dates[-1])

        return rebalance_dates

    @staticmethod
    def _drift_weights(weights: dict[str, float], daily_returns: pd.Series) -> dict[str, float]:
        """Drift weights based on daily asset returns."""
        new_weights = {}
        for ticker, w in weights.items():
            r = daily_returns.get(ticker, 0.0)
            new_weights[ticker] = w * (1 + r)

        total = sum(new_weights.values())
        if total <= 0:
            return weights
        return {t: w / total for t, w in new_weights.items()}

    @staticmethod
    def calculate_metrics(nav_series: pd.Series) -> dict:
        """Calculate performance metrics from a NAV time series.

        Static method so it can be reused by the Attribution module.
        """
        if len(nav_series) <= 1:
            return {
                "total_return": 0.0,
                "annualized_return": 0.0,
                "annualized_volatility": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "sortino_ratio": 0.0,
                "calmar_ratio": 0.0,
            }

        # Total return
        total_return = nav_series.iloc[-1] / nav_series.iloc[0] - 1

        # Annualized return
        n_days = len(nav_series)
        annualized_return = (1 + total_return) ** (TRADING_DAYS_PER_YEAR / n_days) - 1

        # Daily returns
        daily_returns = nav_series.pct_change().dropna()

        # Annualized volatility
        annualized_volatility = float(daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR))

        # Sharpe ratio (risk-free = 0)
        sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0.0

        # Max drawdown
        cummax = nav_series.cummax()
        drawdown = (nav_series - cummax) / cummax
        max_drawdown = float(drawdown.min())

        # Sortino ratio (downside deviation)
        negative_returns = daily_returns[daily_returns < 0]
        if len(negative_returns) > 0:
            downside_dev = float(negative_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
            sortino_ratio = annualized_return / downside_dev if downside_dev > 0 else 0.0
        else:
            sortino_ratio = 0.0

        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0.0

        return {
            "total_return": float(total_return),
            "annualized_return": float(annualized_return),
            "annualized_volatility": annualized_volatility,
            "sharpe_ratio": float(sharpe_ratio),
            "max_drawdown": max_drawdown,
            "sortino_ratio": float(sortino_ratio),
            "calmar_ratio": float(calmar_ratio),
        }
