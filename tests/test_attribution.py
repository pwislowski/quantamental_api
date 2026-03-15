from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from app.services.attribution import AttributionEngine

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TICKERS = ["A", "B", "C"]
DATES = pd.bdate_range("2023-01-02", periods=60)


@pytest.fixture
def date_index():
    return DATES


@pytest.fixture
def holdings_history():
    """Two rebalance points with known weights."""
    return [
        {"date": str(DATES[0].date()), "weights": {"A": 0.5, "B": 0.3, "C": 0.2}},
        {"date": str(DATES[30].date()), "weights": {"A": 0.4, "B": 0.4, "C": 0.2}},
    ]


@pytest.fixture
def returns_series():
    """Synthetic daily portfolio returns."""
    np.random.seed(42)
    daily = np.random.normal(0.0005, 0.01, len(DATES))
    return [{"date": str(d.date()), "return": float(r)} for d, r in zip(DATES, daily)]


@pytest.fixture
def factor_data():
    """Synthetic factor scores: DataFrame indexed by ticker, one column per factor."""
    np.random.seed(42)
    return {
        "value": pd.DataFrame({"value": [0.8, -0.3, 0.5]}, index=TICKERS),
        "momentum": pd.DataFrame({"momentum": [1.2, 0.1, -0.7]}, index=TICKERS),
    }


@pytest.fixture
def engine():
    fetcher = MagicMock()
    factor_engine = MagicMock()
    return AttributionEngine(factor_engine=factor_engine, fetcher=fetcher)


# ===========================================================================
# A. _build_daily_weights tests
# ===========================================================================


class TestBuildDailyWeights:
    def test_forward_fills_weights(self, engine, holdings_history, date_index):
        result = engine._build_daily_weights(holdings_history, date_index)
        # Before second rebalance, weights should be from first rebalance
        day_15 = date_index[15]
        assert abs(result.loc[day_15, "A"] - 0.5) < 1e-10

    def test_output_length_matches_index(self, engine, holdings_history, date_index):
        result = engine._build_daily_weights(holdings_history, date_index)
        assert len(result) == len(date_index)

    def test_weights_sum_to_one(self, engine, holdings_history, date_index):
        result = engine._build_daily_weights(holdings_history, date_index)
        row_sums = result.sum(axis=1)
        for s in row_sums:
            assert abs(s - 1.0) < 1e-10


# ===========================================================================
# B. _compute_factor_exposures tests
# ===========================================================================


class TestComputeFactorExposures:
    def test_returns_dataframe_with_factor_columns(self, engine, date_index, holdings_history, factor_data):
        daily_weights = engine._build_daily_weights(holdings_history, date_index)
        result = engine._compute_factor_exposures(daily_weights, factor_data)
        assert "value" in result.columns
        assert "momentum" in result.columns

    def test_exposure_is_weighted_sum(self, engine, date_index, holdings_history, factor_data):
        daily_weights = engine._build_daily_weights(holdings_history, date_index)
        result = engine._compute_factor_exposures(daily_weights, factor_data)
        # First day weights: A=0.5, B=0.3, C=0.2
        # value scores: A=0.8, B=-0.3, C=0.5
        # expected = 0.5*0.8 + 0.3*(-0.3) + 0.2*0.5 = 0.4 - 0.09 + 0.1 = 0.41
        expected_value = 0.5 * 0.8 + 0.3 * (-0.3) + 0.2 * 0.5
        assert abs(result.iloc[0]["value"] - expected_value) < 1e-10

    def test_handles_missing_ticker_in_factors(self, engine, date_index):
        """Factor data missing one ticker — should handle gracefully."""
        holdings = [{"date": str(date_index[0].date()), "weights": {"A": 0.5, "B": 0.3, "MISSING": 0.2}}]
        daily_weights = engine._build_daily_weights(holdings, date_index)
        factor_data = {"value": pd.DataFrame({"value": [0.8, -0.3]}, index=["A", "B"])}
        result = engine._compute_factor_exposures(daily_weights, factor_data)
        # Should not raise, exposure based on available tickers
        assert len(result) == len(date_index)


# ===========================================================================
# C. _decompose_returns tests
# ===========================================================================


class TestDecomposeReturns:
    def test_contributions_plus_specific_near_total(self, engine):
        """Factor contributions + specific_return ≈ total return."""
        np.random.seed(42)
        n = 100
        factor_exp = pd.DataFrame({"f1": np.random.normal(0, 1, n), "f2": np.random.normal(0, 1, n)})
        # Construct returns as linear combo of factors + noise
        true_betas = [0.02, -0.01]
        returns = pd.Series(
            factor_exp["f1"] * true_betas[0] + factor_exp["f2"] * true_betas[1] + np.random.normal(0, 0.001, n)
        )
        result = engine._decompose_returns(returns, factor_exp)
        total = float(returns.sum())
        factor_sum = sum(c["contribution"] for c in result["factor_contributions"])
        assert abs(total - (factor_sum + result["specific_return"])) < 0.1

    def test_r_squared_between_zero_and_one(self, engine):
        np.random.seed(42)
        n = 100
        factor_exp = pd.DataFrame({"f1": np.random.normal(0, 1, n)})
        returns = pd.Series(factor_exp["f1"] * 0.01 + np.random.normal(0, 0.005, n))
        result = engine._decompose_returns(returns, factor_exp)
        assert 0.0 <= result["r_squared"] <= 1.0

    def test_perfect_factor_correlation(self, engine):
        """If returns = 0.05 * factor, factor should explain ~100%."""
        np.random.seed(42)
        n = 100
        factor_exp = pd.DataFrame({"f1": np.random.normal(0, 1, n)})
        returns = pd.Series(factor_exp["f1"].values * 0.05)
        result = engine._decompose_returns(returns, factor_exp)
        assert result["r_squared"] > 0.99

    def test_zero_factor_exposures(self, engine):
        """All-zero factors → specific_return ≈ total return."""
        np.random.seed(42)
        n = 50
        factor_exp = pd.DataFrame({"f1": np.zeros(n)})
        returns = pd.Series(np.random.normal(0.001, 0.01, n))
        result = engine._decompose_returns(returns, factor_exp)
        total = float(returns.sum())
        assert abs(result["specific_return"] - total) < 0.01


# ===========================================================================
# D. attribute() end-to-end tests
# ===========================================================================


class TestAttribute:
    def _make_backtest_result(self, holdings_history, returns_series):
        """Create a mock BacktestResult-like object."""
        mock = MagicMock()
        mock.holdings_history = holdings_history
        mock.returns_series = returns_series
        nav = 1.0
        nav_list = []
        for entry in returns_series:
            nav *= 1 + entry["return"]
            nav_list.append({"date": entry["date"], "nav": nav})
        mock.nav_series = nav_list
        mock.total_return = nav / 1.0 - 1
        return mock

    def test_returns_expected_keys(self, engine, holdings_history, returns_series, factor_data):
        engine.factor_engine.calculate_all_factors.return_value = factor_data
        bt_result = self._make_backtest_result(holdings_history, returns_series)
        result = engine.attribute(bt_result, factor_names=["value", "momentum"])

        assert "factor_contributions" in result
        assert "specific_return" in result
        assert "r_squared" in result
        assert "total_return" in result

    def test_factor_contributions_not_nan(self, engine, holdings_history, returns_series, factor_data):
        engine.factor_engine.calculate_all_factors.return_value = factor_data
        bt_result = self._make_backtest_result(holdings_history, returns_series)
        result = engine.attribute(bt_result, factor_names=["value", "momentum"])

        for fc in result["factor_contributions"]:
            assert np.isfinite(fc["contribution"])
            assert np.isfinite(fc["exposure"])

    def test_single_factor(self, engine, holdings_history, returns_series, factor_data):
        engine.factor_engine.calculate_all_factors.return_value = {"value": factor_data["value"]}
        bt_result = self._make_backtest_result(holdings_history, returns_series)
        result = engine.attribute(bt_result, factor_names=["value"])

        assert len(result["factor_contributions"]) == 1
        assert result["factor_contributions"][0]["factor_name"] == "value"


# ===========================================================================
# E. Edge cases
# ===========================================================================


class TestEdgeCases:
    def test_single_day(self, engine):
        """Single-day backtest — should return zeros."""
        dates = pd.bdate_range("2023-01-02", periods=1)
        holdings = [{"date": str(dates[0].date()), "weights": {"A": 1.0}}]
        returns = [{"date": str(dates[0].date()), "return": 0.01}]
        factor_data = {"value": pd.DataFrame({"value": [0.5]}, index=["A"])}
        engine.factor_engine.calculate_all_factors.return_value = factor_data

        bt_result = MagicMock()
        bt_result.holdings_history = holdings
        bt_result.returns_series = returns
        bt_result.nav_series = [{"date": str(dates[0].date()), "nav": 1.01}]
        bt_result.total_return = 0.01

        result = engine.attribute(bt_result, factor_names=["value"])
        assert np.isfinite(result["r_squared"])

    def test_single_ticker(self, engine):
        dates = pd.bdate_range("2023-01-02", periods=30)
        np.random.seed(42)
        holdings = [{"date": str(dates[0].date()), "weights": {"SOLO": 1.0}}]
        daily = np.random.normal(0.001, 0.01, len(dates))
        returns = [{"date": str(d.date()), "return": float(r)} for d, r in zip(dates, daily)]
        factor_data = {"value": pd.DataFrame({"value": [0.8]}, index=["SOLO"])}
        engine.factor_engine.calculate_all_factors.return_value = factor_data

        bt_result = MagicMock()
        bt_result.holdings_history = holdings
        bt_result.returns_series = returns
        nav = 1.0
        nav_list = []
        for entry in returns:
            nav *= 1 + entry["return"]
            nav_list.append({"date": entry["date"], "nav": nav})
        bt_result.nav_series = nav_list
        bt_result.total_return = nav - 1

        result = engine.attribute(bt_result, factor_names=["value"])
        assert "factor_contributions" in result
        assert len(result["factor_contributions"]) == 1
