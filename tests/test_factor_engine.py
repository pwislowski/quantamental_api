import numpy as np
import pandas as pd
import pytest

from app.services.factor_engine import FactorEngine


@pytest.fixture
def fundamental_data():
    """Sample fundamental data with ticker index."""
    return pd.DataFrame(
        {
            "trailingPE": [25.0, 15.0, 10.0, 50.0],
            "priceToBook": [10.0, 3.0, 1.5, 20.0],
            "marketCap": [3e12, 500e9, 50e9, 1e12],
            "returnOnEquity": [0.30, 0.15, 0.20, 0.05],
            "dividendYield": [0.005, 0.02, 0.03, 0.0],
        },
        index=pd.Index(["AAPL", "MSFT", "XOM", "TSLA"], name="ticker"),
    )


@pytest.fixture
def price_data():
    """Sample multi-ticker price data with MultiIndex columns."""
    dates = pd.bdate_range("2023-01-01", periods=300)
    np.random.seed(42)
    tickers = ["AAPL", "MSFT", "XOM", "TSLA"]
    arrays = []
    for ticker in tickers:
        # Generate random walk prices
        returns = np.random.normal(0.0005, 0.02, len(dates))
        prices = 100 * np.cumprod(1 + returns)
        arrays.append(prices)

    tuples = []
    data = {}
    for i, ticker in enumerate(tickers):
        for field in ["Open", "High", "Low", "Close", "Volume"]:
            tuples.append((ticker, field))
            if field == "Close":
                data[(ticker, field)] = arrays[i]
            elif field == "Volume":
                data[(ticker, field)] = np.random.randint(1e6, 1e7, len(dates)).astype(float)
            else:
                data[(ticker, field)] = arrays[i] * (1 + np.random.normal(0, 0.005, len(dates)))

    columns = pd.MultiIndex.from_tuples(tuples)
    return pd.DataFrame(data, index=dates, columns=columns)


@pytest.fixture
def engine(mocker, tmp_path):
    """FactorEngine with a mocked MarketDataFetcher."""
    mock_fetcher = mocker.MagicMock()
    mock_fetcher.cache_dir = tmp_path
    return FactorEngine(fetcher=mock_fetcher)


# --- calculate_value_factor ---


def test_value_factor(engine, price_data, fundamental_data):
    result = engine.calculate_value_factor(price_data, fundamental_data)
    assert "value" in result.columns
    assert len(result) == 4
    # E/P = 1/PE: XOM (PE=10) should have highest value score
    assert result.loc["XOM", "value"] > result.loc["TSLA", "value"]


def test_value_factor_missing_pe(engine, price_data):
    fund = pd.DataFrame({"marketCap": [1e12]}, index=pd.Index(["AAPL"], name="ticker"))
    result = engine.calculate_value_factor(price_data, fund)
    assert result.empty or "value" in result.columns


def test_value_factor_negative_pe(engine, price_data):
    fund = pd.DataFrame(
        {"trailingPE": [-5.0, 20.0]},
        index=pd.Index(["BAD", "GOOD"], name="ticker"),
    )
    result = engine.calculate_value_factor(price_data, fund)
    # Negative PE should be excluded
    assert "BAD" not in result.index
    assert "GOOD" in result.index


# --- calculate_momentum_factor ---


def test_momentum_factor(engine, price_data):
    result = engine.calculate_momentum_factor(price_data)
    assert "momentum" in result.columns
    assert len(result) > 0
    # All tickers should have momentum values
    for ticker in ["AAPL", "MSFT", "XOM", "TSLA"]:
        assert ticker in result.index


def test_momentum_factor_insufficient_data(engine):
    dates = pd.bdate_range("2024-01-01", periods=10)
    data = pd.DataFrame(
        {("AAPL", "Close"): range(10)},
        index=dates,
        columns=pd.MultiIndex.from_tuples([("AAPL", "Close")]),
    )
    result = engine.calculate_momentum_factor(data)
    # Not enough data for 12-month lookback
    assert result.empty or "AAPL" not in result.index


# --- calculate_quality_factor ---


def test_quality_factor(engine, fundamental_data):
    result = engine.calculate_quality_factor(fundamental_data)
    assert "quality" in result.columns
    assert len(result) == 4
    # AAPL has highest ROE (0.30)
    assert result.loc["AAPL", "quality"] > result.loc["TSLA", "quality"]


def test_quality_factor_missing_roe(engine):
    fund = pd.DataFrame({"marketCap": [1e12]}, index=pd.Index(["AAPL"], name="ticker"))
    result = engine.calculate_quality_factor(fund)
    assert result.empty or "quality" in result.columns


# --- calculate_low_vol_factor ---


def test_low_vol_factor(engine, price_data):
    result = engine.calculate_low_vol_factor(price_data)
    assert "low_vol" in result.columns
    assert len(result) > 0
    # All values should be positive (inverse of volatility)
    assert (result["low_vol"] > 0).all()


# --- calculate_size_factor ---


def test_size_factor(engine, fundamental_data):
    result = engine.calculate_size_factor(fundamental_data)
    assert "size" in result.columns
    assert len(result) == 4
    # XOM (smallest at 50B) should have highest size score (negative log)
    assert result.loc["XOM", "size"] > result.loc["AAPL", "size"]


def test_size_factor_missing_mcap(engine):
    fund = pd.DataFrame({"trailingPE": [20.0]}, index=pd.Index(["AAPL"], name="ticker"))
    result = engine.calculate_size_factor(fund)
    assert result.empty or "size" in result.columns


# --- standardize_factor ---


def test_standardize_factor(engine):
    values = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 100.0], index=list("ABCDEF"))
    result = engine.standardize_factor(values)
    # Should be z-scored
    assert abs(result.mean()) < 0.5  # approximately zero after winsorization
    # Outlier (100.0) should be clipped at 3
    assert result.max() <= 3.0
    assert result.min() >= -3.0


def test_standardize_factor_single_value(engine):
    values = pd.Series([5.0], index=["A"])
    result = engine.standardize_factor(values)
    # Single value — can't compute std, returns as-is
    assert len(result) == 1


def test_standardize_factor_with_nans(engine):
    values = pd.Series([1.0, np.nan, 3.0, 4.0, 5.0], index=list("ABCDE"))
    result = engine.standardize_factor(values)
    # NaN should be dropped, rest standardized
    assert "B" not in result.index
    assert len(result) == 4


# --- calculate_all_factors ---


def test_calculate_all_factors(engine, price_data, fundamental_data):
    engine.fetcher.fetch_price_data.return_value = price_data
    engine.fetcher.fetch_fundamental_data.return_value = fundamental_data

    result = engine.calculate_all_factors(["AAPL", "MSFT", "XOM", "TSLA"], "2023-01-01", "2024-03-01")

    assert isinstance(result, dict)
    assert set(result.keys()) == {"value", "momentum", "quality", "low_vol", "size"}

    # Each factor should have a DataFrame
    for name, df in result.items():
        assert isinstance(df, pd.DataFrame)

    # Standardized values should be clipped to [-3, 3]
    for name, df in result.items():
        if not df.empty and name in df.columns:
            assert df[name].max() <= 3.0
            assert df[name].min() >= -3.0


def test_calculate_all_factors_empty_price_data(engine, fundamental_data):
    engine.fetcher.fetch_price_data.return_value = pd.DataFrame()
    engine.fetcher.fetch_fundamental_data.return_value = fundamental_data

    result = engine.calculate_all_factors(["AAPL"], "2023-01-01", "2024-03-01")
    assert isinstance(result, dict)
    # Price-based factors should be empty
    assert result["momentum"].empty
    assert result["low_vol"].empty
