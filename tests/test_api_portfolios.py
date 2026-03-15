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


# ============================================================
# Additional comprehensive tests
# ============================================================


# --- POST /api/portfolio/optimize – factor_tilt strategy ---


@pytest.mark.asyncio
async def test_optimize_factor_tilt(client, mocker):
    """Factor-tilt strategy full flow: mock both MarketDataFetcher and FactorEngine."""
    tickers = ["AAPL", "MSFT", "GOOG"]
    mock_fetcher_cls = mocker.patch("app.api.v1.portfolios.MarketDataFetcher")
    mock_fetcher_cls.return_value.fetch_price_data.return_value = _mock_price_data(tickers)

    # FactorEngine.calculate_all_factors returns dict[str, DataFrame]
    mock_engine_cls = mocker.patch("app.api.v1.portfolios.FactorEngine")
    mock_engine_cls.return_value.calculate_all_factors.return_value = {
        "value": pd.DataFrame({"value": [0.5, -0.3, 0.8]}, index=tickers),
        "momentum": pd.DataFrame({"momentum": [1.2, -0.1, 0.4]}, index=tickers),
    }

    resp = await client.post(
        "/api/portfolio/optimize",
        json={
            "tickers": tickers,
            "start_date": "2023-01-01",
            "end_date": "2024-03-01",
            "strategy_type": "factor_tilt",
            "name": "FT Portfolio",
            "tilts": {"value": 1.0, "momentum": 0.5},
            "tilt_strength": 0.8,
            "risk_aversion": 2.0,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["portfolio"]["name"] == "FT Portfolio"
    assert data["portfolio"]["strategy_type"] == "factor_tilt"
    assert data["portfolio"]["id"] is not None
    assert len(data["holdings"]) == 3
    weights = [h["weight"] for h in data["holdings"]]
    assert abs(sum(weights) - 1.0) < 1e-6
    # Verify parameters are saved
    params = data["portfolio"]["parameters"]
    assert params["tilts"] == {"value": 1.0, "momentum": 0.5}
    assert params["tilt_strength"] == 0.8
    assert params["risk_aversion"] == 2.0


@pytest.mark.asyncio
async def test_optimize_factor_tilt_missing_tilts(client, mocker):
    """factor_tilt without tilts should return 400."""
    tickers = ["AAPL", "MSFT"]
    mock_cls = mocker.patch("app.api.v1.portfolios.MarketDataFetcher")
    mock_cls.return_value.fetch_price_data.return_value = _mock_price_data(tickers)

    resp = await client.post(
        "/api/portfolio/optimize",
        json={
            "tickers": tickers,
            "start_date": "2023-01-01",
            "end_date": "2024-03-01",
            "strategy_type": "factor_tilt",
            "name": "No Tilts",
        },
    )
    assert resp.status_code == 400
    assert resp.json()["error"] == "missing_tilts"


@pytest.mark.asyncio
async def test_optimize_factor_tilt_empty_factor_data(client, mocker):
    """factor_tilt with empty factor exposures should return 400."""
    tickers = ["AAPL", "MSFT"]
    mock_fetcher_cls = mocker.patch("app.api.v1.portfolios.MarketDataFetcher")
    mock_fetcher_cls.return_value.fetch_price_data.return_value = _mock_price_data(tickers)

    mock_engine_cls = mocker.patch("app.api.v1.portfolios.FactorEngine")
    # Return empty DataFrames for all factors
    mock_engine_cls.return_value.calculate_all_factors.return_value = {
        "value": pd.DataFrame(),
        "momentum": pd.DataFrame(),
    }

    resp = await client.post(
        "/api/portfolio/optimize",
        json={
            "tickers": tickers,
            "start_date": "2023-01-01",
            "end_date": "2024-03-01",
            "strategy_type": "factor_tilt",
            "name": "Empty Factors",
            "tilts": {"value": 1.0},
        },
    )
    assert resp.status_code == 400
    assert resp.json()["error"] == "no_factor_data"


# --- POST /api/portfolio/optimize – error paths ---


@pytest.mark.asyncio
async def test_optimize_empty_price_data(client, mocker):
    """Empty price data from fetcher should return 400."""
    mock_cls = mocker.patch("app.api.v1.portfolios.MarketDataFetcher")
    mock_cls.return_value.fetch_price_data.return_value = pd.DataFrame()

    resp = await client.post(
        "/api/portfolio/optimize",
        json={
            "tickers": ["AAPL"],
            "start_date": "2023-01-01",
            "end_date": "2024-03-01",
            "strategy_type": "mean_variance",
            "name": "Empty Data",
        },
    )
    assert resp.status_code == 400
    assert resp.json()["error"] == "no_price_data"


@pytest.mark.asyncio
async def test_optimize_insufficient_data(client, mocker):
    """Price data with only 1 row (empty after pct_change().dropna()) should return 400."""
    mock_cls = mocker.patch("app.api.v1.portfolios.MarketDataFetcher")
    # Build a MultiIndex DataFrame with only 1 row — pct_change().dropna() will be empty
    dates = pd.bdate_range("2023-01-01", periods=1)
    cols = pd.MultiIndex.from_tuples([("AAPL", "Close")])
    mock_cls.return_value.fetch_price_data.return_value = pd.DataFrame([[100.0]], index=dates, columns=cols)

    resp = await client.post(
        "/api/portfolio/optimize",
        json={
            "tickers": ["AAPL"],
            "start_date": "2023-01-01",
            "end_date": "2023-01-02",
            "strategy_type": "mean_variance",
            "name": "Too Short",
        },
    )
    assert resp.status_code == 400
    assert resp.json()["error"] == "insufficient_data"


# --- POST /api/portfolio/optimize – request validation ---


@pytest.mark.asyncio
async def test_optimize_missing_required_fields(client):
    """Omitting required fields should return 422."""
    resp = await client.post("/api/portfolio/optimize", json={})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_optimize_empty_tickers(client):
    """Empty tickers list should fail validation (min_length=1)."""
    resp = await client.post(
        "/api/portfolio/optimize",
        json={
            "tickers": [],
            "start_date": "2023-01-01",
            "end_date": "2024-03-01",
            "strategy_type": "mean_variance",
            "name": "Bad",
        },
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_optimize_empty_name(client):
    """Empty name should fail validation (min_length=1)."""
    resp = await client.post(
        "/api/portfolio/optimize",
        json={
            "tickers": ["AAPL"],
            "start_date": "2023-01-01",
            "end_date": "2024-03-01",
            "strategy_type": "mean_variance",
            "name": "",
        },
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_optimize_invalid_risk_aversion(client):
    """risk_aversion <= 0 should fail validation (gt=0)."""
    resp = await client.post(
        "/api/portfolio/optimize",
        json={
            "tickers": ["AAPL"],
            "start_date": "2023-01-01",
            "end_date": "2024-03-01",
            "strategy_type": "mean_variance",
            "name": "Bad RA",
            "risk_aversion": 0,
        },
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_optimize_invalid_tilt_strength(client):
    """tilt_strength <= 0 should fail validation (gt=0)."""
    resp = await client.post(
        "/api/portfolio/optimize",
        json={
            "tickers": ["AAPL"],
            "start_date": "2023-01-01",
            "end_date": "2024-03-01",
            "strategy_type": "factor_tilt",
            "name": "Bad TS",
            "tilt_strength": -1.0,
            "tilts": {"value": 1.0},
        },
    )
    assert resp.status_code == 422


# --- POST /api/portfolio/optimize – parameter persistence ---


@pytest.mark.asyncio
async def test_optimize_saves_parameters(client, mocker):
    """Verify the portfolio parameters dict is properly persisted."""
    tickers = ["AAPL", "MSFT"]
    mock_cls = mocker.patch("app.api.v1.portfolios.MarketDataFetcher")
    mock_cls.return_value.fetch_price_data.return_value = _mock_price_data(tickers)

    resp = await client.post(
        "/api/portfolio/optimize",
        json={
            "tickers": tickers,
            "start_date": "2023-01-01",
            "end_date": "2024-03-01",
            "strategy_type": "mean_variance",
            "name": "Params Test",
            "constraints": {"max_position": 0.5},
            "risk_aversion": 3.0,
        },
    )
    assert resp.status_code == 200
    params = resp.json()["portfolio"]["parameters"]
    assert params["constraints"] == {"max_position": 0.5}
    assert params["risk_aversion"] == 3.0
    assert params["tilts"] is None
    assert params["tilt_strength"] == 1.0


@pytest.mark.asyncio
async def test_optimize_custom_risk_aversion(client, mocker):
    """Verify high risk_aversion produces different (more conservative) weights than default."""
    tickers = ["AAPL", "MSFT", "GOOG"]
    mock_cls = mocker.patch("app.api.v1.portfolios.MarketDataFetcher")
    price_data = _mock_price_data(tickers, seed=42)
    mock_cls.return_value.fetch_price_data.return_value = price_data

    # Default risk_aversion=1.0
    resp1 = await client.post(
        "/api/portfolio/optimize",
        json={
            "tickers": tickers,
            "start_date": "2023-01-01",
            "end_date": "2024-03-01",
            "strategy_type": "mean_variance",
            "name": "RA Default",
            "risk_aversion": 1.0,
        },
    )
    # High risk_aversion=100.0
    resp2 = await client.post(
        "/api/portfolio/optimize",
        json={
            "tickers": tickers,
            "start_date": "2023-01-01",
            "end_date": "2024-03-01",
            "strategy_type": "mean_variance",
            "name": "RA High",
            "risk_aversion": 100.0,
        },
    )
    assert resp1.status_code == 200
    assert resp2.status_code == 200
    weights1 = {h["ticker"]: h["weight"] for h in resp1.json()["holdings"]}
    weights2 = {h["ticker"]: h["weight"] for h in resp2.json()["holdings"]}
    # Different risk aversion should produce different weight allocations
    assert weights1 != weights2


# --- GET /api/portfolio/ – ordering ---


@pytest.fixture
def multi_portfolio_session(test_session):
    """Session with multiple portfolios for ordering tests."""
    for i, (name, strategy) in enumerate(
        [("First", "mean_variance"), ("Second", "risk_parity"), ("Third", "mean_variance")]
    ):
        p = Portfolio(
            name=name,
            strategy_type=strategy,
            created_at=datetime(2024, 1, 1 + i, tzinfo=timezone.utc),
            parameters={},
        )
        test_session.add(p)
    test_session.commit()
    return test_session


@pytest_asyncio.fixture
async def multi_client(multi_portfolio_session):
    def override_get_db():
        yield multi_portfolio_session

    _app.dependency_overrides[get_db] = override_get_db
    async with AsyncClient(transport=ASGITransport(app=_app), base_url="http://test") as ac:
        yield ac
    _app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_list_portfolios_ordering(multi_client):
    """Portfolios should be returned newest first (created_at desc)."""
    resp = await multi_client.get("/api/portfolio/")
    assert resp.status_code == 200
    items = resp.json()["items"]
    assert len(items) == 3
    assert items[0]["name"] == "Third"
    assert items[1]["name"] == "Second"
    assert items[2]["name"] == "First"


@pytest.mark.asyncio
async def test_list_portfolios_multiple_count(multi_client):
    """Verify all portfolios are returned."""
    resp = await multi_client.get("/api/portfolio/")
    assert resp.status_code == 200
    items = resp.json()["items"]
    assert len(items) == 3
    strategies = {item["strategy_type"] for item in items}
    assert strategies == {"mean_variance", "risk_parity"}


# --- GET /api/portfolio/{id}/holdings – edge cases ---


@pytest.fixture
def portfolio_no_holdings_session(test_session):
    """Session with a portfolio that has no holdings."""
    portfolio = Portfolio(
        name="Empty Holdings",
        strategy_type="mean_variance",
        created_at=datetime.now(timezone.utc),
        parameters={},
    )
    test_session.add(portfolio)
    test_session.commit()
    return test_session


@pytest_asyncio.fixture
async def no_holdings_client(portfolio_no_holdings_session):
    def override_get_db():
        yield portfolio_no_holdings_session

    _app.dependency_overrides[get_db] = override_get_db
    async with AsyncClient(transport=ASGITransport(app=_app), base_url="http://test") as ac:
        yield ac
    _app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_get_holdings_empty_list(no_holdings_client):
    """Portfolio exists but has no holdings — should return empty list, not 404."""
    resp = await no_holdings_client.get("/api/portfolio/1/holdings")
    assert resp.status_code == 200
    data = resp.json()
    assert data["portfolio_id"] == 1
    assert data["holdings"] == []


# --- GET /api/portfolio/{id}/exposures – partial factor data ---


@pytest.fixture
def partial_factors_session(seeded_session):
    """Portfolio with factor data for only some tickers."""
    value_factor = Factor(name="value", category="equity", created_at=datetime.now(timezone.utc))
    seeded_session.add(value_factor)
    seeded_session.flush()

    # Only add factor data for AAPL (weight=0.4) and MSFT (weight=0.35), not GOOG
    for ticker, val in [("AAPL", 1.0), ("MSFT", -0.5)]:
        seeded_session.add(
            FactorData(factor_id=value_factor.id, ticker=ticker, date=date(2024, 6, 15), value=val, percentile_rank=0.5)
        )
    seeded_session.commit()
    return seeded_session


@pytest_asyncio.fixture
async def partial_factors_client(partial_factors_session):
    def override_get_db():
        yield partial_factors_session

    _app.dependency_overrides[get_db] = override_get_db
    async with AsyncClient(transport=ASGITransport(app=_app), base_url="http://test") as ac:
        yield ac
    _app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_get_exposures_partial_factor_data(partial_factors_client):
    """Exposures computed from partial factor data (only 2 of 3 tickers have data)."""
    resp = await partial_factors_client.get("/api/portfolio/1/exposures")
    assert resp.status_code == 200
    data = resp.json()
    exposures = data["exposures"]
    assert "value" in exposures
    # Expected: 0.4*1.0 + 0.35*(-0.5) = 0.225  (GOOG has no factor data, excluded)
    assert abs(exposures["value"] - 0.225) < 1e-6


# --- GET /api/portfolio/{id}/exposures – portfolio with no holdings ---


@pytest.mark.asyncio
async def test_get_exposures_no_holdings(no_holdings_client):
    """Portfolio with no holdings should return empty exposures."""
    resp = await no_holdings_client.get("/api/portfolio/1/exposures")
    assert resp.status_code == 200
    data = resp.json()
    assert data["exposures"] == {}


# --- DELETE /api/portfolio/{id} – verify holdings cascade ---


@pytest.mark.asyncio
async def test_delete_portfolio_removes_holdings(seeded_client):
    """Deleting a portfolio should also remove its holdings."""
    # Confirm holdings exist first
    resp = await seeded_client.get("/api/portfolio/1/holdings")
    assert resp.status_code == 200
    assert len(resp.json()["holdings"]) == 3

    # Delete
    resp = await seeded_client.delete("/api/portfolio/1")
    assert resp.status_code == 200

    # Portfolio gone
    resp = await seeded_client.get("/api/portfolio/1")
    assert resp.status_code == 404

    # Holdings also gone — portfolio doesn't exist so should be 404
    resp = await seeded_client.get("/api/portfolio/1/holdings")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_delete_response_contains_portfolio_id(seeded_client):
    """Delete response should contain the deleted portfolio's ID."""
    resp = await seeded_client.delete("/api/portfolio/1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "deleted"
    assert data["portfolio_id"] == 1


# --- POST /api/portfolio/optimize – constraints edge cases ---


@pytest.mark.asyncio
async def test_optimize_with_min_position_constraint(client, mocker):
    """min_position constraint enforced on all holdings."""
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
            "name": "Min Position",
            "constraints": {"min_position": 0.10},
        },
    )
    assert resp.status_code == 200
    for h in resp.json()["holdings"]:
        assert h["weight"] >= 0.10 - 1e-6


@pytest.mark.asyncio
async def test_optimize_with_combined_constraints(client, mocker):
    """Combined max_position and min_position constraints."""
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
            "name": "Combined",
            "constraints": {"max_position": 0.40, "min_position": 0.10},
        },
    )
    assert resp.status_code == 200
    for h in resp.json()["holdings"]:
        assert h["weight"] >= 0.10 - 1e-6
        assert h["weight"] <= 0.40 + 1e-6


@pytest.mark.asyncio
async def test_optimize_with_max_positions_constraint(client, mocker):
    """max_positions limits the number of non-zero holdings."""
    tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "META"]
    mock_cls = mocker.patch("app.api.v1.portfolios.MarketDataFetcher")
    mock_cls.return_value.fetch_price_data.return_value = _mock_price_data(tickers)

    resp = await client.post(
        "/api/portfolio/optimize",
        json={
            "tickers": tickers,
            "start_date": "2023-01-01",
            "end_date": "2024-03-01",
            "strategy_type": "mean_variance",
            "name": "Max Positions",
            "constraints": {"max_positions": 3},
        },
    )
    assert resp.status_code == 200
    holdings = resp.json()["holdings"]
    nonzero = [h for h in holdings if h["weight"] > 1e-6]
    assert len(nonzero) <= 3


# --- POST /api/portfolio/optimize – single ticker ---


@pytest.mark.asyncio
async def test_optimize_single_ticker(client, mocker):
    """Single ticker should get 100% weight."""
    tickers = ["AAPL"]
    mock_cls = mocker.patch("app.api.v1.portfolios.MarketDataFetcher")
    mock_cls.return_value.fetch_price_data.return_value = _mock_price_data(tickers)

    resp = await client.post(
        "/api/portfolio/optimize",
        json={
            "tickers": tickers,
            "start_date": "2023-01-01",
            "end_date": "2024-03-01",
            "strategy_type": "mean_variance",
            "name": "Solo",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["holdings"]) == 1
    assert abs(data["holdings"][0]["weight"] - 1.0) < 1e-6


# --- GET /api/portfolio/{id} – verify full response shape ---


@pytest.mark.asyncio
async def test_get_portfolio_response_shape(seeded_client):
    """Verify all expected fields are present in portfolio response."""
    resp = await seeded_client.get("/api/portfolio/1")
    assert resp.status_code == 200
    data = resp.json()
    assert "id" in data
    assert "name" in data
    assert "strategy_type" in data
    assert "created_at" in data
    assert "parameters" in data
