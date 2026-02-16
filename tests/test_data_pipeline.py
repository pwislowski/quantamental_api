import numpy as np
import pandas as pd
import pytest
from app.services.data_pipeline import MarketDataFetcher


@pytest.fixture
def fetcher(tmp_path):
    return MarketDataFetcher(cache_dir=tmp_path)


# --- fetch_price_data ---


def test_fetch_price_data_success(mocker, fetcher):
    dates = pd.date_range("2024-01-01", periods=3)
    sample_df = pd.DataFrame({"Open": [1, 2, 3], "Close": [1.1, 2.1, 3.1]}, index=dates)
    mocker.patch("app.services.data_pipeline.yf.download", return_value=sample_df)

    result = fetcher.fetch_price_data(["AAPL"], "2024-01-01", "2024-01-03")
    assert not result.empty
    assert len(result) == 3


def test_fetch_price_data_empty(mocker, fetcher):
    mocker.patch("app.services.data_pipeline.yf.download", return_value=pd.DataFrame())

    result = fetcher.fetch_price_data(["INVALID"], "2024-01-01", "2024-01-03")
    assert result.empty


def test_fetch_price_data_partial_failure(mocker, fetcher):
    dates = pd.date_range("2024-01-01", periods=3)
    arrays = [["AAPL", "AAPL", "BAD", "BAD"], ["Close", "Open", "Close", "Open"]]
    tuples = list(zip(*arrays))
    index = pd.MultiIndex.from_tuples(tuples)
    data = pd.DataFrame(
        [[1.0, 2.0, np.nan, np.nan], [3.0, 4.0, np.nan, np.nan], [5.0, 6.0, np.nan, np.nan]],
        index=dates,
        columns=index,
    )
    mocker.patch("app.services.data_pipeline.yf.download", return_value=data)

    result = fetcher.fetch_price_data(["AAPL", "BAD"], "2024-01-01", "2024-01-03")
    assert not result.empty
    # AAPL data should be present
    assert not result["AAPL"].isna().all().all()
    # BAD data should be all NaN
    assert result["BAD"].isna().all().all()


# --- fetch_fundamental_data ---


def test_fetch_fundamental_data_success(mocker, fetcher):
    mock_ticker = mocker.MagicMock()
    mock_ticker.info = {
        "trailingPE": 25.0,
        "priceToBook": 10.0,
        "marketCap": 3_000_000_000_000,
        "forwardPE": 22.0,
        "dividendYield": 0.005,
        "returnOnEquity": 0.15,
    }
    mocker.patch("app.services.data_pipeline.yf.Ticker", return_value=mock_ticker)

    result = fetcher.fetch_fundamental_data(["AAPL"])
    assert not result.empty
    assert result.index.name == "ticker"
    assert "AAPL" in result.index
    assert result.loc["AAPL", "trailingPE"] == 25.0
    assert result.loc["AAPL", "marketCap"] == 3_000_000_000_000


def test_fetch_fundamental_data_ticker_failure(mocker, fetcher):
    def side_effect(ticker):
        if ticker == "BAD":
            raise Exception("not found")
        mock = mocker.MagicMock()
        mock.info = {"trailingPE": 20.0, "priceToBook": 5.0, "marketCap": 1_000_000_000}
        return mock

    mocker.patch("app.services.data_pipeline.yf.Ticker", side_effect=side_effect)

    result = fetcher.fetch_fundamental_data(["AAPL", "BAD"])
    assert len(result) == 1
    assert "AAPL" in result.index


def test_fetch_fundamental_data_all_fail(mocker, fetcher):
    mocker.patch("app.services.data_pipeline.yf.Ticker", side_effect=Exception("fail"))

    result = fetcher.fetch_fundamental_data(["BAD1", "BAD2"])
    assert result.empty


# --- get_sp500_tickers ---


def test_get_sp500_tickers_success(mocker, fetcher):
    sample_table = pd.DataFrame({"Symbol": ["AAPL", "MSFT", "BRK.B", "GOOG"]})
    mocker.patch("app.services.data_pipeline.pd.read_html", return_value=[sample_table])

    result = fetcher.get_sp500_tickers()
    assert isinstance(result, list)
    assert "BRK-B" in result  # dot replaced with dash
    assert "BRK.B" not in result
    assert result == sorted(result)


def test_get_sp500_tickers_failure(mocker, fetcher):
    mocker.patch("app.services.data_pipeline.pd.read_html", side_effect=Exception("network error"))

    result = fetcher.get_sp500_tickers()
    assert result == []


# --- cache_data / load_cached_data ---


def test_cache_roundtrip(fetcher):
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4.0, 5.0, 6.0]})
    fetcher.cache_data(df, "test_key")

    loaded = fetcher.load_cached_data("test_key")
    assert loaded is not None
    pd.testing.assert_frame_equal(df, loaded)


def test_load_cached_data_miss(fetcher):
    result = fetcher.load_cached_data("nonexistent_key")
    assert result is None
