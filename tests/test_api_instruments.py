from datetime import datetime, timezone

import pytest

from app.models.instrument import Instrument


def _make_instrument(ticker: str, sector: str = "Technology") -> Instrument:
    return Instrument(
        ticker=ticker,
        company_name=f"{ticker} Inc.",
        gics_sector=sector,
        gics_sub_industry="Software",
        is_active=True,
        updated_at=datetime.now(timezone.utc),
    )


# --- GET /api/instruments ---


@pytest.mark.asyncio
async def test_list_instruments_empty(client):
    resp = await client.get("/api/instruments")
    assert resp.status_code == 200
    data = resp.json()
    assert data["items"] == []
    assert data["total"] == 0


@pytest.mark.asyncio
async def test_list_instruments_returns_active_only(client, test_session):
    test_session.add(_make_instrument("AAPL", "Information Technology"))
    test_session.add(_make_instrument("MSFT", "Information Technology"))
    inactive = _make_instrument("REMOVED", "Financials")
    inactive.is_active = False
    test_session.add(inactive)
    test_session.commit()

    resp = await client.get("/api/instruments")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 2
    tickers = [i["ticker"] for i in data["items"]]
    assert "AAPL" in tickers
    assert "MSFT" in tickers
    assert "REMOVED" not in tickers


@pytest.mark.asyncio
async def test_list_instruments_fields(client, test_session):
    test_session.add(_make_instrument("AAPL", "Information Technology"))
    test_session.commit()

    resp = await client.get("/api/instruments")
    item = resp.json()["items"][0]
    assert item["ticker"] == "AAPL"
    assert item["company_name"] == "AAPL Inc."
    assert item["gics_sector"] == "Information Technology"
    assert item["gics_sub_industry"] == "Software"


@pytest.mark.asyncio
async def test_list_instruments_sorted_by_ticker(client, test_session):
    for ticker in ["TSLA", "AAPL", "MSFT"]:
        test_session.add(_make_instrument(ticker))
    test_session.commit()

    resp = await client.get("/api/instruments")
    tickers = [i["ticker"] for i in resp.json()["items"]]
    assert tickers == sorted(tickers)


# --- POST /api/instruments/sync ---


@pytest.mark.asyncio
async def test_sync_instruments(client, mocker):
    mock_provider = mocker.MagicMock()
    mock_provider.get_sp500_instruments.return_value = [
        {"ticker": "AAPL", "company_name": "Apple Inc.", "gics_sector": "Information Technology", "gics_sub_industry": "Tech Hardware"},
        {"ticker": "MSFT", "company_name": "Microsoft Corp.", "gics_sector": "Information Technology", "gics_sub_industry": "Systems Software"},
    ]
    mocker.patch("app.api.v1.instruments.InstrumentSyncService", lambda session: _make_sync_service(session, mock_provider))

    resp = await client.post("/api/instruments/sync")
    assert resp.status_code == 200
    data = resp.json()
    assert data["synced"] == 2
    assert data["deactivated"] == 0


@pytest.mark.asyncio
async def test_sync_instruments_deactivates_removed(client, test_session, mocker):
    # Pre-populate with a ticker that won't be in the sync result
    test_session.add(_make_instrument("OLD", "Financials"))
    test_session.commit()

    mock_provider = mocker.MagicMock()
    mock_provider.get_sp500_instruments.return_value = [
        {"ticker": "AAPL", "company_name": "Apple Inc.", "gics_sector": "Information Technology", "gics_sub_industry": "Tech Hardware"},
    ]
    mocker.patch("app.api.v1.instruments.InstrumentSyncService", lambda session: _make_sync_service(session, mock_provider))

    resp = await client.post("/api/instruments/sync")
    assert resp.status_code == 200
    data = resp.json()
    assert data["synced"] == 1
    assert data["deactivated"] == 1


@pytest.mark.asyncio
async def test_sync_instruments_no_data(client, mocker):
    mock_provider = mocker.MagicMock()
    mock_provider.get_sp500_instruments.return_value = []
    mocker.patch("app.api.v1.instruments.InstrumentSyncService", lambda session: _make_sync_service(session, mock_provider))

    resp = await client.post("/api/instruments/sync")
    assert resp.status_code == 200
    data = resp.json()
    assert data["synced"] == 0


def _make_sync_service(session, provider):
    from app.services.instrument_sync import InstrumentSyncService
    return InstrumentSyncService(session, provider=provider)
