import numpy as np
import pandas as pd
import pytest

from app.services.optimizer import PortfolioOptimizer


@pytest.fixture
def optimizer():
    return PortfolioOptimizer()


@pytest.fixture
def returns_df():
    """Synthetic daily returns for 4 assets with different risk/return profiles."""
    np.random.seed(42)
    dates = pd.bdate_range("2023-01-01", periods=252)
    # Asset A: high return, high vol
    # Asset B: moderate return, low vol
    # Asset C: low return, moderate vol
    # Asset D: moderate return, moderate vol
    data = {
        "A": np.random.normal(0.001, 0.03, len(dates)),
        "B": np.random.normal(0.0005, 0.01, len(dates)),
        "C": np.random.normal(0.0001, 0.02, len(dates)),
        "D": np.random.normal(0.0004, 0.015, len(dates)),
    }
    return pd.DataFrame(data, index=dates)


# --- mean_variance_optimization ---


def test_mean_variance_weights_sum_to_one(optimizer, returns_df):
    weights = optimizer.mean_variance_optimization(returns_df)
    assert abs(sum(weights.values()) - 1.0) < 1e-6


def test_mean_variance_all_non_negative(optimizer, returns_df):
    weights = optimizer.mean_variance_optimization(returns_df)
    for w in weights.values():
        assert w >= -1e-10


def test_mean_variance_single_asset(optimizer):
    returns = pd.DataFrame({"ONLY": np.random.normal(0, 0.01, 100)})
    weights = optimizer.mean_variance_optimization(returns)
    assert weights == {"ONLY": 1.0}


def test_mean_variance_empty(optimizer):
    returns = pd.DataFrame()
    weights = optimizer.mean_variance_optimization(returns)
    assert weights == {}


def test_mean_variance_risk_aversion_effect(optimizer, returns_df):
    """Higher risk aversion should shift weights toward lower-volatility assets."""
    low_ra = optimizer.mean_variance_optimization(returns_df, risk_aversion=0.1)
    high_ra = optimizer.mean_variance_optimization(returns_df, risk_aversion=10.0)
    # B is lowest vol — should get more weight with higher risk aversion
    assert high_ra["B"] > low_ra["B"] or abs(high_ra["B"] - low_ra["B"]) < 0.01


# --- risk_parity_optimization ---


def test_risk_parity_weights_sum_to_one(optimizer, returns_df):
    weights = optimizer.risk_parity_optimization(returns_df)
    assert abs(sum(weights.values()) - 1.0) < 1e-6


def test_risk_parity_all_non_negative(optimizer, returns_df):
    weights = optimizer.risk_parity_optimization(returns_df)
    for w in weights.values():
        assert w >= -1e-10


def test_risk_parity_lower_vol_gets_more_weight(optimizer, returns_df):
    """Risk parity should give more weight to lower-volatility assets."""
    weights = optimizer.risk_parity_optimization(returns_df)
    # B (vol=0.01) should have at least as much weight as A (vol=0.03)
    assert weights["B"] >= weights["A"] - 0.01


def test_risk_parity_risk_contributions_approximately_equal(optimizer, returns_df):
    """Each asset's risk contribution should be approximately equal."""
    weights = optimizer.risk_parity_optimization(returns_df)
    w = np.array([weights[t] for t in returns_df.columns])
    cov = returns_df.cov().values
    port_vol = np.sqrt(w @ cov @ w)
    marginal = cov @ w
    risk_contrib = w * marginal / port_vol
    # All contributions should be roughly equal
    target = port_vol / len(returns_df.columns)
    for rc in risk_contrib:
        assert abs(rc - target) < 0.005  # within 0.5% tolerance


# --- factor_tilt_optimization ---


def test_factor_tilt_weights_sum_to_one(optimizer, returns_df):
    factor_exposures = pd.DataFrame(
        {"value": [0.5, -0.3, 1.2, -0.8]},
        index=["A", "B", "C", "D"],
    )
    weights = optimizer.factor_tilt_optimization(returns_df, factor_exposures, tilts={"value": 1.0})
    assert abs(sum(weights.values()) - 1.0) < 1e-6


def test_factor_tilt_tilts_toward_target(optimizer, returns_df):
    """Portfolio should tilt toward assets with high factor exposure."""
    factor_exposures = pd.DataFrame(
        {"value": [2.0, -1.0, -1.0, 0.0]},
        index=["A", "B", "C", "D"],
    )
    weights = optimizer.factor_tilt_optimization(returns_df, factor_exposures, tilts={"value": 1.5}, tilt_strength=5.0)
    # A has highest value exposure — should get significant weight
    assert weights["A"] > weights["B"]
    assert weights["A"] > weights["C"]


def test_factor_tilt_no_matching_factors_falls_back(optimizer, returns_df):
    """If no factor columns match tilts, fall back to mean-variance."""
    factor_exposures = pd.DataFrame(
        {"unrelated": [1, 2, 3, 4]},
        index=["A", "B", "C", "D"],
    )
    weights = optimizer.factor_tilt_optimization(returns_df, factor_exposures, tilts={"value": 1.0})
    assert abs(sum(weights.values()) - 1.0) < 1e-6


def test_factor_tilt_empty_tilts_falls_back(optimizer, returns_df):
    factor_exposures = pd.DataFrame(
        {"value": [1, 2, 3, 4]},
        index=["A", "B", "C", "D"],
    )
    weights = optimizer.factor_tilt_optimization(returns_df, factor_exposures, tilts={})
    assert abs(sum(weights.values()) - 1.0) < 1e-6


# --- apply_constraints ---


def test_apply_constraints_max_position(optimizer):
    weights = {"A": 0.6, "B": 0.3, "C": 0.1}
    result = optimizer.apply_constraints(weights, {"max_position": 0.4})
    assert all(v <= 0.4 + 1e-10 for v in result.values())
    assert abs(sum(result.values()) - 1.0) < 1e-6


def test_apply_constraints_min_position(optimizer):
    weights = {"A": 0.5, "B": 0.3, "C": 0.15, "D": 0.04, "E": 0.01}
    result = optimizer.apply_constraints(weights, {"min_position": 0.05})
    assert "D" not in result
    assert "E" not in result
    assert abs(sum(result.values()) - 1.0) < 1e-6


def test_apply_constraints_max_positions(optimizer):
    weights = {"A": 0.4, "B": 0.3, "C": 0.15, "D": 0.1, "E": 0.05}
    result = optimizer.apply_constraints(weights, {"max_positions": 3})
    assert len(result) == 3
    assert "A" in result and "B" in result and "C" in result
    assert abs(sum(result.values()) - 1.0) < 1e-6


def test_apply_constraints_combined(optimizer):
    weights = {"A": 0.5, "B": 0.3, "C": 0.1, "D": 0.07, "E": 0.03}
    result = optimizer.apply_constraints(weights, {"max_positions": 3, "max_position": 0.45})
    assert len(result) <= 3
    assert all(v <= 0.45 + 1e-10 for v in result.values())
    assert abs(sum(result.values()) - 1.0) < 1e-6


def test_apply_constraints_empty(optimizer):
    assert optimizer.apply_constraints({}, {"max_position": 0.5}) == {}


def test_apply_constraints_no_constraints(optimizer):
    weights = {"A": 0.6, "B": 0.4}
    result = optimizer.apply_constraints(weights, {})
    assert result == weights
