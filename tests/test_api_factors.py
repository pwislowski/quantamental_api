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
from app.main import app
from app.models.factor import Factor, FactorData


@pytest.fixture
def test_session():
    """Shared in-memory SQLite session safe for cross-thread use (FastAPI sync routes)."""
    engine = create_engine(
        "sqlite:///file:test_factors.db?mode=memory&cache=shared&uri=true",
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

    app.dependency_overrides[get_db] = override_get_db
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()


@pytest.fixture
def seeded_session(test_session):
    """Session pre-populated with factor data for query tests."""
    # Create factors
    value_factor = Factor(name="value", category="equity", created_at=datetime.now(timezone.utc))
    momentum_factor = Factor(name="momentum", category="equity", created_at=datetime.now(timezone.utc))
    test_session.add(value_factor)
    test_session.add(momentum_factor)
    test_session.flush()

    # Create factor data
    tickers = ["AAPL", "MSFT", "XOM", "TSLA"]
    for i, ticker in enumerate(tickers):
        test_session.add(
            FactorData(
                factor_id=value_factor.id,
                ticker=ticker,
                date=date(2024, 1, 15),
                value=float(i) * 0.5 - 0.5,
                percentile_rank=float(i) / len(tickers),
            )
        )
        test_session.add(
            FactorData(
                factor_id=momentum_factor.id,
                ticker=ticker,
                date=date(2024, 1, 15),
                value=float(i) * 0.3,
                percentile_rank=float(i) / len(tickers),
            )
        )
    test_session.commit()
    return test_session


@pytest_asyncio.fixture
async def seeded_client(seeded_session):
    """AsyncClient with pre-seeded factor data."""

    def override_get_db():
        yield seeded_session

    app.dependency_overrides[get_db] = override_get_db
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()


# --- GET /api/factors/available ---


@pytest.mark.asyncio
async def test_list_available_factors_empty(client):
    resp = await client.get("/api/factors/available")
    assert resp.status_code == 200
    data = resp.json()
    assert data["items"] == []


@pytest.mark.asyncio
async def test_list_available_factors_with_data(seeded_client):
    resp = await seeded_client.get("/api/factors/available")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["items"]) == 2
    names = {item["name"] for item in data["items"]}
    assert names == {"value", "momentum"}


# --- GET /api/factors/{factor_name}/data ---


@pytest.mark.asyncio
async def test_get_factor_data_success(seeded_client):
    resp = await seeded_client.get("/api/factors/value/data?start_date=2024-01-01&end_date=2024-12-31")
    assert resp.status_code == 200
    data = resp.json()
    assert data["factor_name"] == "value"
    assert "AAPL" in data["data"]


@pytest.mark.asyncio
async def test_get_factor_data_not_found(seeded_client):
    resp = await seeded_client.get("/api/factors/nonexistent/data?start_date=2024-01-01&end_date=2024-12-31")
    assert resp.status_code == 404


# --- GET /api/factors/correlation ---


@pytest.mark.asyncio
async def test_get_factor_correlation(seeded_client):
    resp = await seeded_client.get(
        "/api/factors/correlation?start_date=2024-01-01&end_date=2024-12-31&factors=value,momentum"
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "factors" in data
    assert "matrix" in data
    assert len(data["matrix"]) > 0


@pytest.mark.asyncio
async def test_get_factor_correlation_empty(client):
    resp = await client.get("/api/factors/correlation?start_date=2024-01-01&end_date=2024-12-31&factors=nonexistent")
    assert resp.status_code == 200
    data = resp.json()
    assert data["matrix"] == []


# --- GET /api/factors/{factor_name}/statistics ---


@pytest.mark.asyncio
async def test_get_factor_statistics_success(seeded_client):
    resp = await seeded_client.get("/api/factors/value/statistics?start_date=2024-01-01&end_date=2024-12-31")
    assert resp.status_code == 200
    data = resp.json()
    assert data["factor_name"] == "value"
    stats = data["statistics"]
    assert "mean" in stats
    assert "std" in stats
    assert "count" in stats
    assert stats["count"] == 4


@pytest.mark.asyncio
async def test_get_factor_statistics_not_found(client):
    resp = await client.get("/api/factors/nonexistent/statistics?start_date=2024-01-01&end_date=2024-12-31")
    assert resp.status_code == 404


# --- POST /api/factors/calculate ---


@pytest.mark.asyncio
async def test_calculate_factors(client, mocker):
    # Mock the MarketDataFetcher methods
    dates = pd.bdate_range("2023-01-01", periods=300)
    np.random.seed(42)
    tickers = ["AAPL", "MSFT"]

    # Build multi-level price data
    price_tuples = []
    price_data = {}
    for ticker in tickers:
        returns = np.random.normal(0.0005, 0.02, len(dates))
        prices = 100 * np.cumprod(1 + returns)
        for field in ["Open", "High", "Low", "Close", "Volume"]:
            price_tuples.append((ticker, field))
            if field == "Close":
                price_data[(ticker, field)] = prices
            elif field == "Volume":
                price_data[(ticker, field)] = np.random.randint(1e6, 1e7, len(dates)).astype(float)
            else:
                price_data[(ticker, field)] = prices * (1 + np.random.normal(0, 0.005, len(dates)))

    price_df = pd.DataFrame(price_data, index=dates, columns=pd.MultiIndex.from_tuples(price_tuples))

    fund_df = pd.DataFrame(
        {
            "trailingPE": [25.0, 15.0],
            "marketCap": [3e12, 500e9],
            "returnOnEquity": [0.30, 0.15],
        },
        index=pd.Index(tickers, name="ticker"),
    )

    mock_fetcher_cls = mocker.patch("app.api.v1.factors.MarketDataFetcher")
    mock_fetcher = mock_fetcher_cls.return_value
    mock_fetcher.fetch_price_data.return_value = price_df
    mock_fetcher.fetch_fundamental_data.return_value = fund_df

    resp = await client.post(
        "/api/factors/calculate",
        json={"tickers": tickers, "start_date": "2023-01-01", "end_date": "2024-03-01"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "completed"
    assert set(data["factors"]) == {"value", "momentum", "quality", "low_vol", "size"}
    assert data["tickers_count"] == 2
