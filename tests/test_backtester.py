from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlmodel import SQLModel

from app.models.backtest import Backtest, BacktestResult
from app.models.instrument import Instrument
from app.models.schemas import BacktestRequest
from app.services.backtester import Backtester
from app.services.optimizer import PortfolioOptimizer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def returns_df():
    """252 days of synthetic returns for 4 tickers."""
    np.random.seed(42)
    dates = pd.bdate_range("2023-01-01", periods=252)
    data = {
        "A": np.random.normal(0.001, 0.03, 252),
        "B": np.random.normal(0.0005, 0.01, 252),
        "C": np.random.normal(0.0001, 0.02, 252),
        "D": np.random.normal(0.0004, 0.015, 252),
    }
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def nav_series():
    """A NAV series built from known random returns."""
    np.random.seed(42)
    dates = pd.bdate_range("2023-01-01", periods=252)
    daily_returns = np.random.normal(0.0004, 0.01, 252)
    nav = (1 + pd.Series(daily_returns, index=dates)).cumprod()
    return nav


@pytest.fixture
def optimizer():
    return PortfolioOptimizer()


@pytest.fixture
def backtester(optimizer):
    fetcher = MagicMock()
    factor_engine = MagicMock()
    return Backtester(optimizer=optimizer, fetcher=fetcher, factor_engine=factor_engine)


@pytest.fixture
def backtest_config():
    return BacktestRequest(
        name="Test Backtest",
        tickers=["A", "B", "C", "D"],
        start_date="2023-01-01",
        end_date="2023-12-31",
        strategy_type="mean_variance",
        rebalance_frequency="monthly",
        constraints={},
        risk_aversion=1.0,
    )


def _make_mock_price_data(returns_df: pd.DataFrame) -> pd.DataFrame:
    """Build multi-index price DataFrame from returns, matching MarketDataFetcher output."""
    # Prepend a base row so pct_change().dropna() recovers all N rows of returns
    base_date = returns_df.index[0] - pd.offsets.BDay(1)
    base = pd.DataFrame({col: 100.0 for col in returns_df.columns}, index=[base_date])
    close = pd.concat([base, (1 + returns_df).cumprod() * 100])
    frames = {}
    for ticker in returns_df.columns:
        frames[(ticker, "Close")] = close[ticker]
        frames[(ticker, "Open")] = close[ticker] * 0.999
        frames[(ticker, "High")] = close[ticker] * 1.01
        frames[(ticker, "Low")] = close[ticker] * 0.99
        frames[(ticker, "Volume")] = np.random.randint(1_000_000, 10_000_000, len(close))
    df = pd.DataFrame(frames, index=close.index)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


@pytest.fixture
def db_session():
    """In-memory SQLite session for DB persistence tests."""
    engine = create_engine(
        "sqlite:///file:test_bt.db?mode=memory&cache=shared&uri=true",
        echo=False,
        connect_args={"check_same_thread": False},
    )

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_conn, connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    SQLModel.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, autoflush=False)
    session = SessionLocal()

    # Seed instruments needed by backtest tests
    for ticker in ["A", "B", "C", "D", "ONLY"]:
        session.add(Instrument(ticker=ticker, company_name=ticker))
    session.commit()

    yield session
    session.rollback()
    session.close()
    SQLModel.metadata.drop_all(bind=engine)
    engine.dispose()


# ===========================================================================
# A. calculate_metrics tests
# ===========================================================================


class TestCalculateMetrics:
    def test_total_return(self, nav_series):
        metrics = Backtester.calculate_metrics(nav_series)
        expected = nav_series.iloc[-1] / nav_series.iloc[0] - 1
        assert abs(metrics["total_return"] - expected) < 1e-10

    def test_annualized_return(self, nav_series):
        metrics = Backtester.calculate_metrics(nav_series)
        total = nav_series.iloc[-1] / nav_series.iloc[0] - 1
        n_days = len(nav_series)
        expected = (1 + total) ** (TRADING_DAYS_PER_YEAR / n_days) - 1
        assert abs(metrics["annualized_return"] - expected) < 1e-10

    def test_volatility_positive(self, nav_series):
        metrics = Backtester.calculate_metrics(nav_series)
        assert metrics["annualized_volatility"] > 0

    def test_sharpe_ratio(self, nav_series):
        metrics = Backtester.calculate_metrics(nav_series)
        expected = metrics["annualized_return"] / metrics["annualized_volatility"]
        assert abs(metrics["sharpe_ratio"] - expected) < 1e-10

    def test_max_drawdown_non_positive(self, nav_series):
        metrics = Backtester.calculate_metrics(nav_series)
        assert metrics["max_drawdown"] <= 0

    def test_max_drawdown_known_series(self):
        """Series that goes 1.0 -> 1.2 -> 0.9 -> 1.1 has max DD = (0.9 - 1.2) / 1.2."""
        nav = pd.Series([1.0, 1.2, 0.9, 1.1], index=pd.bdate_range("2023-01-01", periods=4))
        metrics = Backtester.calculate_metrics(nav)
        expected_dd = (0.9 - 1.2) / 1.2  # -0.25
        assert abs(metrics["max_drawdown"] - expected_dd) < 1e-10

    def test_sortino_ratio(self, nav_series):
        metrics = Backtester.calculate_metrics(nav_series)
        # Sortino should be finite for a series with negative returns
        assert np.isfinite(metrics["sortino_ratio"])

    def test_calmar_ratio(self, nav_series):
        metrics = Backtester.calculate_metrics(nav_series)
        if metrics["max_drawdown"] != 0:
            expected = metrics["annualized_return"] / abs(metrics["max_drawdown"])
            assert abs(metrics["calmar_ratio"] - expected) < 1e-10

    def test_flat_nav(self):
        """Flat NAV: all metrics should be 0."""
        nav = pd.Series([1.0] * 100, index=pd.bdate_range("2023-01-01", periods=100))
        metrics = Backtester.calculate_metrics(nav)
        assert metrics["total_return"] == 0.0
        assert metrics["annualized_return"] == 0.0
        assert metrics["annualized_volatility"] == 0.0
        assert metrics["sharpe_ratio"] == 0.0
        assert metrics["max_drawdown"] == 0.0
        assert metrics["sortino_ratio"] == 0.0
        assert metrics["calmar_ratio"] == 0.0

    def test_single_day(self):
        """Single data point — should return zeros."""
        nav = pd.Series([1.0], index=pd.bdate_range("2023-01-01", periods=1))
        metrics = Backtester.calculate_metrics(nav)
        assert metrics["total_return"] == 0.0
        assert metrics["sharpe_ratio"] == 0.0


# ===========================================================================
# B. _generate_rebalance_dates tests
# ===========================================================================

TRADING_DAYS_PER_YEAR = 252


class TestGenerateRebalanceDates:
    def test_monthly_count(self, returns_df):
        dates = Backtester._generate_rebalance_dates(returns_df.index, "monthly")
        # 252 business days ~ 12 months, expect 11-12 rebalance dates
        assert 10 <= len(dates) <= 13

    def test_quarterly_count(self, returns_df):
        dates = Backtester._generate_rebalance_dates(returns_df.index, "quarterly")
        assert 3 <= len(dates) <= 5

    def test_yearly_count(self, returns_df):
        dates = Backtester._generate_rebalance_dates(returns_df.index, "yearly")
        assert 1 <= len(dates) <= 2

    def test_dates_within_index(self, returns_df):
        dates = Backtester._generate_rebalance_dates(returns_df.index, "monthly")
        index_set = set(returns_df.index)
        for d in dates:
            assert d in index_set


# ===========================================================================
# C. _drift_weights tests
# ===========================================================================


class TestDriftWeights:
    def test_sum_to_one(self):
        weights = {"A": 0.4, "B": 0.3, "C": 0.2, "D": 0.1}
        daily_returns = pd.Series({"A": 0.02, "B": -0.01, "C": 0.005, "D": 0.0})
        result = Backtester._drift_weights(weights, daily_returns)
        assert abs(sum(result.values()) - 1.0) < 1e-10

    def test_positive_return_increases_weight(self):
        weights = {"A": 0.5, "B": 0.5}
        daily_returns = pd.Series({"A": 0.10, "B": 0.0})
        result = Backtester._drift_weights(weights, daily_returns)
        assert result["A"] > 0.5

    def test_zero_returns_unchanged(self):
        weights = {"A": 0.6, "B": 0.4}
        daily_returns = pd.Series({"A": 0.0, "B": 0.0})
        result = Backtester._drift_weights(weights, daily_returns)
        assert abs(result["A"] - 0.6) < 1e-10
        assert abs(result["B"] - 0.4) < 1e-10


# ===========================================================================
# D. _walk_forward tests
# ===========================================================================


class TestWalkForward:
    def test_nav_starts_at_one(self, backtester, returns_df, backtest_config):
        rebalance_dates = Backtester._generate_rebalance_dates(returns_df.index, "monthly")
        nav_series, _ = backtester._walk_forward(returns_df, rebalance_dates, backtest_config)
        assert abs(nav_series.iloc[0] - 1.0) < 1e-10

    def test_nav_length_matches_returns(self, backtester, returns_df, backtest_config):
        rebalance_dates = Backtester._generate_rebalance_dates(returns_df.index, "monthly")
        nav_series, _ = backtester._walk_forward(returns_df, rebalance_dates, backtest_config)
        assert len(nav_series) == len(returns_df)

    def test_holdings_recorded_at_rebalance(self, backtester, returns_df, backtest_config):
        rebalance_dates = Backtester._generate_rebalance_dates(returns_df.index, "monthly")
        _, holdings_history = backtester._walk_forward(returns_df, rebalance_dates, backtest_config)
        assert len(holdings_history) == len(rebalance_dates)

    def test_mean_variance_runs(self, backtester, returns_df, backtest_config):
        rebalance_dates = Backtester._generate_rebalance_dates(returns_df.index, "monthly")
        nav_series, holdings = backtester._walk_forward(returns_df, rebalance_dates, backtest_config)
        assert len(nav_series) > 0
        assert len(holdings) > 0

    def test_risk_parity_runs(self, backtester, returns_df, backtest_config):
        config = backtest_config.model_copy(update={"strategy_type": "risk_parity"})
        rebalance_dates = Backtester._generate_rebalance_dates(returns_df.index, "monthly")
        nav_series, holdings = backtester._walk_forward(returns_df, rebalance_dates, config)
        assert len(nav_series) > 0
        assert len(holdings) > 0


# ===========================================================================
# E. run() end-to-end tests (with DB persistence)
# ===========================================================================


class TestRun:
    def test_persists_backtest(self, backtester, returns_df, backtest_config, db_session):
        price_data = _make_mock_price_data(returns_df)
        backtester.fetcher.fetch_price_data.return_value = price_data

        backtester.run(backtest_config, db_session)

        backtests = db_session.query(Backtest).all()
        assert len(backtests) == 1
        assert backtests[0].name == "Test Backtest"
        assert backtests[0].strategy_type == "mean_variance"

    def test_persists_result(self, backtester, returns_df, backtest_config, db_session):
        price_data = _make_mock_price_data(returns_df)
        backtester.fetcher.fetch_price_data.return_value = price_data

        backtester.run(backtest_config, db_session)

        results = db_session.query(BacktestResult).all()
        assert len(results) == 1
        assert results[0].backtest_id is not None

    def test_returns_metrics(self, backtester, returns_df, backtest_config, db_session):
        price_data = _make_mock_price_data(returns_df)
        backtester.fetcher.fetch_price_data.return_value = price_data

        result = backtester.run(backtest_config, db_session)

        expected_keys = {
            "total_return",
            "annualized_return",
            "annualized_volatility",
            "sharpe_ratio",
            "max_drawdown",
            "sortino_ratio",
            "calmar_ratio",
        }
        assert expected_keys.issubset(result["metrics"].keys())

    def test_returns_nav_series(self, backtester, returns_df, backtest_config, db_session):
        price_data = _make_mock_price_data(returns_df)
        backtester.fetcher.fetch_price_data.return_value = price_data

        result = backtester.run(backtest_config, db_session)

        assert isinstance(result["nav_series"], list)
        assert len(result["nav_series"]) > 0
        assert "date" in result["nav_series"][0]
        assert "nav" in result["nav_series"][0]

    def test_returns_holdings_history(self, backtester, returns_df, backtest_config, db_session):
        price_data = _make_mock_price_data(returns_df)
        backtester.fetcher.fetch_price_data.return_value = price_data

        result = backtester.run(backtest_config, db_session)

        assert isinstance(result["holdings_history"], list)
        assert len(result["holdings_history"]) > 0
        assert "date" in result["holdings_history"][0]
        assert "weights" in result["holdings_history"][0]


# ===========================================================================
# F. Edge cases
# ===========================================================================


class TestEdgeCases:
    def test_single_ticker(self, backtester, db_session):
        np.random.seed(42)
        dates = pd.bdate_range("2023-01-01", periods=100)
        returns = pd.DataFrame({"ONLY": np.random.normal(0.001, 0.02, 100)}, index=dates)
        price_data = _make_mock_price_data(returns)
        backtester.fetcher.fetch_price_data.return_value = price_data

        config = BacktestRequest(
            name="Single Ticker",
            tickers=["ONLY"],
            start_date="2023-01-01",
            end_date="2023-05-31",
            strategy_type="mean_variance",
            rebalance_frequency="monthly",
        )
        result = backtester.run(config, db_session)
        assert result["metrics"]["total_return"] != 0.0

    def test_short_period(self, backtester, db_session):
        """Period shorter than rebalance frequency — should still work."""
        np.random.seed(42)
        dates = pd.bdate_range("2023-01-01", periods=10)
        returns = pd.DataFrame(
            {"A": np.random.normal(0.001, 0.02, 10), "B": np.random.normal(0.0005, 0.01, 10)},
            index=dates,
        )
        price_data = _make_mock_price_data(returns)
        backtester.fetcher.fetch_price_data.return_value = price_data

        config = BacktestRequest(
            name="Short Period",
            tickers=["A", "B"],
            start_date="2023-01-01",
            end_date="2023-01-15",
            strategy_type="mean_variance",
            rebalance_frequency="quarterly",
        )
        result = backtester.run(config, db_session)
        assert len(result["nav_series"]) == 10
