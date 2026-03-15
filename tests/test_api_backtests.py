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

    from app.api.v1.backtests import router as backtests_router
    from app.core.exceptions import setup_exception_handlers

    test_app = FastAPI()
    setup_exception_handlers(test_app)
    test_app.include_router(backtests_router)
    return test_app


_app = _make_test_app()


@pytest.fixture
def test_session():
    engine = create_engine(
        "sqlite:///file:test_backtests.db?mode=memory&cache=shared&uri=true",
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


def _seed_backtest(session) -> tuple[Backtest, BacktestResult]:
    """Insert a Backtest + BacktestResult for read/delete tests."""
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

    for ticker in ["AAPL", "MSFT", "GOOG"]:
        session.add(BacktestInstrument(backtest_id=backtest.id, ticker=ticker))

    bt_result = BacktestResult(
        backtest_id=backtest.id,
        total_return=0.15,
        annualized_return=0.15,
        annualized_volatility=0.12,
        sharpe_ratio=1.25,
        max_drawdown=-0.08,
        sortino_ratio=1.8,
        calmar_ratio=1.875,
        nav_series=[{"date": "2023-01-02", "nav": 1.0}, {"date": "2023-12-29", "nav": 1.15}],
        returns_series=[{"date": "2023-01-02", "return": 0.001}, {"date": "2023-12-29", "return": 0.002}],
        holdings_history=[{"date": "2023-01-02", "weights": {"AAPL": 0.4, "MSFT": 0.35, "GOOG": 0.25}}],
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


# --- Helper ---


def _mock_price_data(tickers, n_days=252, seed=42):
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


# ===========================================================================
# POST /api/backtest/run
# ===========================================================================


@pytest.mark.asyncio
async def test_run_mean_variance(client, mocker):
    tickers = ["AAPL", "MSFT", "GOOG"]
    mock_cls = mocker.patch("app.api.v1.backtests.MarketDataFetcher")
    mock_cls.return_value.fetch_price_data.return_value = _mock_price_data(tickers)

    resp = await client.post(
        "/api/backtest/run",
        json={
            "name": "MV Backtest",
            "tickers": tickers,
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "strategy_type": "mean_variance",
            "rebalance_frequency": "monthly",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["backtest_id"] is not None
    assert "metrics" in data
    assert "total_return" in data["metrics"]
    assert "nav_series" in data
    assert "holdings_history" in data
    assert len(data["nav_series"]) > 0


@pytest.mark.asyncio
async def test_run_risk_parity(client, mocker):
    tickers = ["AAPL", "MSFT"]
    mock_cls = mocker.patch("app.api.v1.backtests.MarketDataFetcher")
    mock_cls.return_value.fetch_price_data.return_value = _mock_price_data(tickers)

    resp = await client.post(
        "/api/backtest/run",
        json={
            "name": "RP Backtest",
            "tickers": tickers,
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "strategy_type": "risk_parity",
            "rebalance_frequency": "quarterly",
        },
    )
    assert resp.status_code == 200
    assert resp.json()["metrics"]["total_return"] != 0.0


@pytest.mark.asyncio
async def test_run_invalid_strategy(client):
    resp = await client.post(
        "/api/backtest/run",
        json={
            "name": "Bad",
            "tickers": ["AAPL"],
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "strategy_type": "invalid",
            "rebalance_frequency": "monthly",
        },
    )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_run_missing_fields(client):
    resp = await client.post("/api/backtest/run", json={})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_run_empty_tickers(client):
    resp = await client.post(
        "/api/backtest/run",
        json={
            "name": "Empty",
            "tickers": [],
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "strategy_type": "mean_variance",
        },
    )
    assert resp.status_code == 422


# ===========================================================================
# GET /api/backtest/
# ===========================================================================


@pytest.mark.asyncio
async def test_list_empty(client):
    resp = await client.get("/api/backtest/")
    assert resp.status_code == 200
    assert resp.json()["items"] == []


@pytest.mark.asyncio
async def test_list_with_data(seeded_client):
    resp = await seeded_client.get("/api/backtest/")
    assert resp.status_code == 200
    items = resp.json()["items"]
    assert len(items) == 1
    assert items[0]["name"] == "Seeded Backtest"
    assert items[0]["strategy_type"] == "mean_variance"
    assert items[0]["tickers"] == ["AAPL", "GOOG", "MSFT"]


# ===========================================================================
# GET /api/backtest/{backtest_id}
# ===========================================================================


@pytest.mark.asyncio
async def test_get_exists(seeded_client):
    resp = await seeded_client.get("/api/backtest/1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "Seeded Backtest"
    assert data["id"] == 1
    assert "metrics" in data
    assert data["metrics"]["total_return"] == 0.15
    assert data["metrics"]["sharpe_ratio"] == 1.25


@pytest.mark.asyncio
async def test_get_not_found(client):
    resp = await client.get("/api/backtest/999")
    assert resp.status_code == 404


# ===========================================================================
# GET /api/backtest/{backtest_id}/result
# ===========================================================================


@pytest.mark.asyncio
async def test_get_result_exists(seeded_client):
    resp = await seeded_client.get("/api/backtest/1/result")
    assert resp.status_code == 200
    data = resp.json()
    assert data["backtest_id"] == 1
    assert "metrics" in data
    assert "nav_series" in data
    assert "returns_series" in data
    assert "holdings_history" in data
    assert len(data["nav_series"]) == 2
    assert len(data["holdings_history"]) == 1


@pytest.mark.asyncio
async def test_get_result_not_found(client):
    resp = await client.get("/api/backtest/999/result")
    assert resp.status_code == 404


# ===========================================================================
# DELETE /api/backtest/{backtest_id}
# ===========================================================================


@pytest.mark.asyncio
async def test_delete_exists(seeded_client):
    resp = await seeded_client.delete("/api/backtest/1")
    assert resp.status_code == 200
    assert resp.json()["status"] == "deleted"
    assert resp.json()["backtest_id"] == 1

    # Verify it's gone
    resp = await seeded_client.get("/api/backtest/1")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_delete_not_found(client):
    resp = await client.delete("/api/backtest/999")
    assert resp.status_code == 404


# ===========================================================================
# Edge cases
# ===========================================================================


@pytest.mark.asyncio
async def test_delete_removes_result(seeded_client):
    """Deleting backtest also removes its BacktestResult."""
    # Verify result exists
    resp = await seeded_client.get("/api/backtest/1/result")
    assert resp.status_code == 200

    # Delete
    resp = await seeded_client.delete("/api/backtest/1")
    assert resp.status_code == 200

    # Result also gone
    resp = await seeded_client.get("/api/backtest/1/result")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_run_persists_to_db(client, mocker):
    """After POST /run, GET / should return the backtest."""
    tickers = ["AAPL", "MSFT"]
    mock_cls = mocker.patch("app.api.v1.backtests.MarketDataFetcher")
    mock_cls.return_value.fetch_price_data.return_value = _mock_price_data(tickers)

    resp = await client.post(
        "/api/backtest/run",
        json={
            "name": "Persisted",
            "tickers": tickers,
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "strategy_type": "mean_variance",
            "rebalance_frequency": "monthly",
        },
    )
    assert resp.status_code == 200
    backtest_id = resp.json()["backtest_id"]

    # Verify it shows up in list
    resp = await client.get("/api/backtest/")
    assert resp.status_code == 200
    items = resp.json()["items"]
    assert len(items) == 1
    assert items[0]["name"] == "Persisted"

    # Verify full result is retrievable
    resp = await client.get(f"/api/backtest/{backtest_id}/result")
    assert resp.status_code == 200
    assert len(resp.json()["nav_series"]) > 0
