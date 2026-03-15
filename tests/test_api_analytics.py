from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlmodel import SQLModel

from app.core.db import get_db
from app.models.backtest import Backtest, BacktestInstrument, BacktestResult
from app.models.instrument import Instrument


def _make_test_app():
    from fastapi import FastAPI

    from app.api.v1.analytics import router as analytics_router
    from app.core.exceptions import setup_exception_handlers

    test_app = FastAPI()
    setup_exception_handlers(test_app)
    test_app.include_router(analytics_router)
    return test_app


_app = _make_test_app()


@pytest.fixture
def test_session():
    engine = create_engine(
        "sqlite:///file:test_analytics.db?mode=memory&cache=shared&uri=true",
        echo=False,
        connect_args={"check_same_thread": False},
    )
    SQLModel.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
    session = SessionLocal()
    yield session
    session.rollback()
    session.close()
    SQLModel.metadata.drop_all(bind=engine)
    engine.dispose()


@pytest_asyncio.fixture
async def client(test_session):
    def override_get_db():
        yield test_session

    _app.dependency_overrides[get_db] = override_get_db
    async with AsyncClient(transport=ASGITransport(app=_app), base_url="http://test") as ac:
        yield ac
    _app.dependency_overrides.clear()


def _seed_backtest(session, n_days=60) -> tuple[Backtest, BacktestResult]:
    """Insert a Backtest + BacktestResult with realistic time-series data."""
    tickers = ["AAPL", "MSFT", "GOOG"]
    dates = pd.bdate_range("2023-01-01", periods=n_days)

    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.02, n_days)

    returns_series = [{"date": str(d.date()), "return": float(r)} for d, r in zip(dates, returns)]
    nav = np.cumprod(1 + returns)
    nav_series = [{"date": str(d.date()), "nav": float(v)} for d, v in zip(dates, nav)]
    holdings_history = [
        {"date": str(dates[0].date()), "weights": {"AAPL": 0.4, "MSFT": 0.35, "GOOG": 0.25}},
        {"date": str(dates[20].date()), "weights": {"AAPL": 0.35, "MSFT": 0.35, "GOOG": 0.3}},
    ]

    for ticker, name in [("AAPL", "Apple Inc."), ("MSFT", "Microsoft Corp."), ("GOOG", "Alphabet Inc.")]:
        session.merge(Instrument(ticker=ticker, company_name=name))

    backtest = Backtest(
        name="Seeded Backtest",
        strategy_type="mean_variance",
        start_date=date(2023, 1, 1),
        end_date=date(2023, 12, 31),
        rebalance_frequency="monthly",
        parameters={"risk_aversion": 1.0, "constraints": {}},
        created_at=datetime(2024, 6, 1, tzinfo=timezone.utc),
    )
    session.add(backtest)
    session.flush()

    for ticker in tickers:
        session.add(BacktestInstrument(backtest_id=backtest.id, ticker=ticker))

    bt_result = BacktestResult(
        backtest_id=backtest.id,
        total_return=float(nav[-1] - 1),
        annualized_return=0.15,
        annualized_volatility=0.12,
        sharpe_ratio=1.25,
        max_drawdown=-0.08,
        sortino_ratio=1.8,
        calmar_ratio=1.875,
        nav_series=nav_series,
        returns_series=returns_series,
        holdings_history=holdings_history,
    )
    session.add(bt_result)
    session.commit()
    return backtest, bt_result


@pytest.fixture
def seeded_session(test_session):
    _seed_backtest(test_session)
    return test_session


@pytest_asyncio.fixture
async def seeded_client(seeded_session):
    def override_get_db():
        yield seeded_session

    _app.dependency_overrides[get_db] = override_get_db
    async with AsyncClient(transport=ASGITransport(app=_app), base_url="http://test") as ac:
        yield ac
    _app.dependency_overrides.clear()


def _mock_factor_data(tickers):
    """Return synthetic factor DataFrames keyed by factor name."""
    factors = {}
    for name in ["value", "momentum", "quality", "low_vol", "size"]:
        np.random.seed(hash(name) % 2**31)
        scores = {name: pd.Series(np.random.randn(len(tickers)), index=tickers)}
        factors[name] = pd.DataFrame(scores)
    return factors


# ===========================================================================
# POST /api/analytics/attribution
# ===========================================================================


@pytest.mark.asyncio
async def test_attribution_success(seeded_client, mocker):
    mocker.patch("app.api.v1.analytics.MarketDataFetcher")
    mock_fe = mocker.patch("app.api.v1.analytics.FactorEngine")
    mock_fe.return_value.calculate_all_factors.return_value = _mock_factor_data(["AAPL", "MSFT", "GOOG"])

    resp = await seeded_client.post(
        "/api/analytics/attribution",
        json={"backtest_id": 1, "factor_names": ["value", "momentum"]},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["backtest_id"] == 1
    assert "total_return" in data
    assert "factor_contributions" in data
    assert "specific_return" in data
    assert "r_squared" in data
    assert len(data["factor_contributions"]) == 2


@pytest.mark.asyncio
async def test_attribution_backtest_not_found(client):
    resp = await client.post(
        "/api/analytics/attribution",
        json={"backtest_id": 999},
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_attribution_custom_factors(seeded_client, mocker):
    mocker.patch("app.api.v1.analytics.MarketDataFetcher")
    mock_fe = mocker.patch("app.api.v1.analytics.FactorEngine")
    mock_fe.return_value.calculate_all_factors.return_value = _mock_factor_data(["AAPL", "MSFT", "GOOG"])

    resp = await seeded_client.post(
        "/api/analytics/attribution",
        json={"backtest_id": 1, "factor_names": ["quality"]},
    )
    assert resp.status_code == 200
    contributions = resp.json()["factor_contributions"]
    assert len(contributions) == 1
    assert contributions[0]["factor_name"] == "quality"


@pytest.mark.asyncio
async def test_attribution_default_factors(seeded_client, mocker):
    mocker.patch("app.api.v1.analytics.MarketDataFetcher")
    mock_fe = mocker.patch("app.api.v1.analytics.FactorEngine")
    mock_fe.return_value.calculate_all_factors.return_value = _mock_factor_data(["AAPL", "MSFT", "GOOG"])

    resp = await seeded_client.post(
        "/api/analytics/attribution",
        json={"backtest_id": 1},
    )
    assert resp.status_code == 200
    contributions = resp.json()["factor_contributions"]
    assert len(contributions) == 5
    names = {c["factor_name"] for c in contributions}
    assert names == {"value", "momentum", "quality", "low_vol", "size"}


@pytest.mark.asyncio
async def test_attribution_missing_backtest_id(client):
    resp = await client.post("/api/analytics/attribution", json={})
    assert resp.status_code == 422


# ===========================================================================
# GET /api/analytics/{backtest_id}/summary
# ===========================================================================


@pytest.mark.asyncio
async def test_summary_exists(seeded_client):
    resp = await seeded_client.get("/api/analytics/1/summary")
    assert resp.status_code == 200
    data = resp.json()
    assert data["backtest_id"] == 1
    assert data["name"] == "Seeded Backtest"
    assert data["strategy_type"] == "mean_variance"
    assert data["tickers"] == ["AAPL", "GOOG", "MSFT"]
    assert "metrics" in data


@pytest.mark.asyncio
async def test_summary_not_found(client):
    resp = await client.get("/api/analytics/999/summary")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_summary_metrics_shape(seeded_client):
    resp = await seeded_client.get("/api/analytics/1/summary")
    assert resp.status_code == 200
    metrics = resp.json()["metrics"]
    expected_keys = {
        "total_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "max_drawdown",
        "sortino_ratio",
        "calmar_ratio",
    }
    assert set(metrics.keys()) == expected_keys


# ===========================================================================
# Edge cases
# ===========================================================================


@pytest.mark.asyncio
async def test_attribution_empty_returns(test_session, mocker):
    """Backtest with minimal data returns zeros gracefully."""
    test_session.merge(Instrument(ticker="AAPL", company_name="Apple Inc."))
    backtest = Backtest(
        name="Minimal",
        strategy_type="mean_variance",
        start_date=date(2023, 1, 1),
        end_date=date(2023, 1, 2),
        rebalance_frequency="monthly",
        parameters={},
    )
    test_session.add(backtest)
    test_session.flush()
    test_session.add(BacktestInstrument(backtest_id=backtest.id, ticker="AAPL"))

    bt_result = BacktestResult(
        backtest_id=backtest.id,
        total_return=0.0,
        annualized_return=0.0,
        annualized_volatility=0.0,
        sharpe_ratio=0.0,
        max_drawdown=0.0,
        sortino_ratio=0.0,
        calmar_ratio=0.0,
        nav_series=[{"date": "2023-01-02", "nav": 1.0}],
        returns_series=[{"date": "2023-01-02", "return": 0.0}],
        holdings_history=[{"date": "2023-01-02", "weights": {"AAPL": 1.0}}],
    )
    test_session.add(bt_result)
    test_session.commit()

    def override_get_db():
        yield test_session

    _app.dependency_overrides[get_db] = override_get_db

    mocker.patch("app.api.v1.analytics.MarketDataFetcher")
    mock_fe = mocker.patch("app.api.v1.analytics.FactorEngine")
    mock_fe.return_value.calculate_all_factors.return_value = _mock_factor_data(["AAPL"])

    async with AsyncClient(transport=ASGITransport(app=_app), base_url="http://test") as ac:
        resp = await ac.post(
            "/api/analytics/attribution",
            json={"backtest_id": backtest.id, "factor_names": ["value"]},
        )
    _app.dependency_overrides.clear()

    assert resp.status_code == 200
    data = resp.json()
    assert data["r_squared"] >= 0.0


@pytest.mark.asyncio
async def test_attribution_response_fields(seeded_client, mocker):
    """Verify exact keys in attribution response."""
    mocker.patch("app.api.v1.analytics.MarketDataFetcher")
    mock_fe = mocker.patch("app.api.v1.analytics.FactorEngine")
    mock_fe.return_value.calculate_all_factors.return_value = _mock_factor_data(["AAPL", "MSFT", "GOOG"])

    resp = await seeded_client.post(
        "/api/analytics/attribution",
        json={"backtest_id": 1, "factor_names": ["value"]},
    )
    assert resp.status_code == 200
    expected_keys = {"backtest_id", "total_return", "factor_contributions", "specific_return", "r_squared"}
    assert set(resp.json().keys()) == expected_keys

    contribution = resp.json()["factor_contributions"][0]
    assert set(contribution.keys()) == {"factor_name", "contribution", "exposure"}
