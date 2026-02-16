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
from app.models.factor import Factor, FactorData
from app.models.portfolio import Portfolio, PortfolioHolding


# We import app late to avoid conftest.py side effects (pgqueuer mocks etc.)
# Instead, build a minimal FastAPI app that only includes the routers we need.
def _make_test_app():
    from fastapi import FastAPI

    from app.api.v1.factors import router as factors_router
    from app.api.v1.portfolios import router as portfolio_router
    from app.core.exceptions import setup_exception_handlers

    test_app = FastAPI()
    setup_exception_handlers(test_app)
    test_app.include_router(factors_router)
    test_app.include_router(portfolio_router)
    return test_app


_app = _make_test_app()


@pytest.fixture
def test_session():
    """Shared in-memory SQLite session safe for cross-thread use (FastAPI sync routes)."""
    engine = create_engine(
        "sqlite:///file:test_portfolios.db?mode=memory&cache=shared&uri=true",
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
    """AsyncClient with DB dependency overridden to use test session."""

    def override_get_db():
        yield test_session

    _app.dependency_overrides[get_db] = override_get_db
    async with AsyncClient(transport=ASGITransport(app=_app), base_url="http://test") as ac:
        yield ac
    _app.dependency_overrides.clear()


@pytest.fixture
def seeded_session(test_session):
    """Session pre-populated with a portfolio and holdings."""
    portfolio = Portfolio(
        name="Test Portfolio",
        strategy_type="mean_variance",
        created_at=datetime.now(timezone.utc),
        parameters={"risk_aversion": 1.0},
    )
    test_session.add(portfolio)
    test_session.flush()

    for ticker, weight in [("AAPL", 0.4), ("MSFT", 0.35), ("GOOG", 0.25)]:
        test_session.add(
            PortfolioHolding(portfolio_id=portfolio.id, ticker=ticker, weight=weight, date=date(2024, 6, 15))
        )

    test_session.commit()
    return test_session


@pytest_asyncio.fixture
async def seeded_client(seeded_session):
    """AsyncClient with pre-seeded portfolio data."""

    def override_get_db():
        yield seeded_session

    _app.dependency_overrides[get_db] = override_get_db
    async with AsyncClient(transport=ASGITransport(app=_app), base_url="http://test") as ac:
        yield ac
    _app.dependency_overrides.clear()


@pytest.fixture
def seeded_with_factors(seeded_session):
    """Session with portfolio + factor data for exposures tests."""
    value_factor = Factor(name="value", category="equity", created_at=datetime.now(timezone.utc))
    momentum_factor = Factor(name="momentum", category="equity", created_at=datetime.now(timezone.utc))
    seeded_session.add(value_factor)
    seeded_session.add(momentum_factor)
    seeded_session.flush()

    for ticker, val, mom in [("AAPL", 0.5, -0.2), ("MSFT", 0.3, 0.8), ("GOOG", -0.1, 0.4)]:
        seeded_session.add(
            FactorData(factor_id=value_factor.id, ticker=ticker, date=date(2024, 6, 15), value=val, percentile_rank=0.5)
        )
        seeded_session.add(
            FactorData(
                factor_id=momentum_factor.id, ticker=ticker, date=date(2024, 6, 15), value=mom, percentile_rank=0.5
            )
        )

    seeded_session.commit()
    return seeded_session


@pytest_asyncio.fixture
async def factors_client(seeded_with_factors):
    """AsyncClient with portfolio + factor data."""

    def override_get_db():
        yield seeded_with_factors

    _app.dependency_overrides[get_db] = override_get_db
    async with AsyncClient(transport=ASGITransport(app=_app), base_url="http://test") as ac:
        yield ac
    _app.dependency_overrides.clear()


# --- Helper ---


def _mock_price_data(tickers, n_days=300, seed=42):
    """Build a multi-level price DataFrame matching MarketDataFetcher output."""
    np.random.seed(seed)
    dates = pd.bdate_range("2023-01-01", periods=n_days)
    tuples = []
    data = {}
    for ticker in tickers:
        returns = np.random.normal(0.0005, 0.02, n_days)
        prices = 100 * np.cumprod(1 + returns)
        for field in ["Open", "High", "Low", "Close", "Volume"]:
            tuples.append((ticker, field))
            if field == "Close":
                data[(ticker, field)] = prices
            elif field == "Volume":
                data[(ticker, field)] = np.random.randint(1_000_000, 10_000_000, n_days).astype(float)
            else:
                data[(ticker, field)] = prices * (1 + np.random.normal(0, 0.005, n_days))
    return pd.DataFrame(data, index=dates, columns=pd.MultiIndex.from_tuples(tuples))


# --- POST /api/portfolio/optimize ---


@pytest.mark.asyncio
async def test_optimize_mean_variance(client, mocker):
    tickers = ["AAPL", "MSFT", "GOOG"]
    mock_cls = mocker.patch("app.api.v1.portfolios.MarketDataFetcher")
    mock_cls.return_value.fetch_price_data.return_value = _mock_price_data(tickers)

    resp = await client.post(
        "/api/portfolio/optimize",
        json={
            "tickers": tickers,
            "start_date": "2023-01-01",
            "end_date": "2024-03-01",
            "strategy_type": "mean_variance",
            "name": "MV Portfolio",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["portfolio"]["name"] == "MV Portfolio"
    assert data["portfolio"]["strategy_type"] == "mean_variance"
    assert data["portfolio"]["id"] is not None
    assert len(data["holdings"]) == 3
    weights = [h["weight"] for h in data["holdings"]]
    assert abs(sum(weights) - 1.0) < 1e-6


@pytest.mark.asyncio
async def test_optimize_risk_parity(client, mocker):
    tickers = ["AAPL", "MSFT"]
    mock_cls = mocker.patch("app.api.v1.portfolios.MarketDataFetcher")
    mock_cls.return_value.fetch_price_data.return_value = _mock_price_data(tickers)

    resp = await client.post(
        "/api/portfolio/optimize",
        json={
            "tickers": tickers,
            "start_date": "2023-01-01",
            "end_date": "2024-03-01",
            "strategy_type": "risk_parity",
            "name": "RP Portfolio",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["portfolio"]["strategy_type"] == "risk_parity"
    assert len(data["holdings"]) == 2


@pytest.mark.asyncio
async def test_optimize_invalid_strategy(client):
    resp = await client.post(
        "/api/portfolio/optimize",
        json={
            "tickers": ["AAPL"],
            "start_date": "2023-01-01",
            "end_date": "2024-03-01",
            "strategy_type": "invalid_strategy",
            "name": "Bad",
        },
    )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_optimize_with_constraints(client, mocker):
    tickers = ["AAPL", "MSFT", "GOOG", "AMZN"]
    mock_cls = mocker.patch("app.api.v1.portfolios.MarketDataFetcher")
    mock_cls.return_value.fetch_price_data.return_value = _mock_price_data(tickers)

    resp = await client.post(
        "/api/portfolio/optimize",
        json={
            "tickers": tickers,
            "start_date": "2023-01-01",
            "end_date": "2024-03-01",
            "strategy_type": "mean_variance",
            "name": "Constrained",
            "constraints": {"max_position": 0.30},
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    for h in data["holdings"]:
        assert h["weight"] <= 0.30 + 1e-6


# --- GET /api/portfolio/ ---


@pytest.mark.asyncio
async def test_list_portfolios_empty(client):
    resp = await client.get("/api/portfolio/")
    assert resp.status_code == 200
    assert resp.json()["items"] == []


@pytest.mark.asyncio
async def test_list_portfolios_with_data(seeded_client):
    resp = await seeded_client.get("/api/portfolio/")
    assert resp.status_code == 200
    items = resp.json()["items"]
    assert len(items) == 1
    assert items[0]["name"] == "Test Portfolio"


# --- GET /api/portfolio/{portfolio_id} ---


@pytest.mark.asyncio
async def test_get_portfolio_exists(seeded_client):
    resp = await seeded_client.get("/api/portfolio/1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "Test Portfolio"
    assert data["id"] == 1


@pytest.mark.asyncio
async def test_get_portfolio_not_found(client):
    resp = await client.get("/api/portfolio/999")
    assert resp.status_code == 404


# --- GET /api/portfolio/{portfolio_id}/holdings ---


@pytest.mark.asyncio
async def test_get_holdings_exists(seeded_client):
    resp = await seeded_client.get("/api/portfolio/1/holdings")
    assert resp.status_code == 200
    data = resp.json()
    assert data["portfolio_id"] == 1
    assert len(data["holdings"]) == 3
    tickers = {h["ticker"] for h in data["holdings"]}
    assert tickers == {"AAPL", "MSFT", "GOOG"}


@pytest.mark.asyncio
async def test_get_holdings_not_found(client):
    resp = await client.get("/api/portfolio/999/holdings")
    assert resp.status_code == 404


# --- GET /api/portfolio/{portfolio_id}/exposures ---


@pytest.mark.asyncio
async def test_get_exposures_with_factors(factors_client):
    resp = await factors_client.get("/api/portfolio/1/exposures")
    assert resp.status_code == 200
    data = resp.json()
    assert data["portfolio_id"] == 1
    exposures = data["exposures"]
    assert "value" in exposures
    assert "momentum" in exposures
    # Expected: 0.4*0.5 + 0.35*0.3 + 0.25*(-0.1) = 0.28
    assert abs(exposures["value"] - 0.28) < 1e-6
    # Expected: 0.4*(-0.2) + 0.35*0.8 + 0.25*0.4 = 0.30
    assert abs(exposures["momentum"] - 0.30) < 1e-6


@pytest.mark.asyncio
async def test_get_exposures_no_factors(seeded_client):
    resp = await seeded_client.get("/api/portfolio/1/exposures")
    assert resp.status_code == 200
    data = resp.json()
    assert data["exposures"] == {}


# --- DELETE /api/portfolio/{portfolio_id} ---


@pytest.mark.asyncio
async def test_delete_portfolio_exists(seeded_client):
    resp = await seeded_client.delete("/api/portfolio/1")
    assert resp.status_code == 200
    assert resp.json()["status"] == "deleted"

    # Verify it's gone
    resp = await seeded_client.get("/api/portfolio/1")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_delete_portfolio_not_found(client):
    resp = await client.delete("/api/portfolio/999")
    assert resp.status_code == 404
