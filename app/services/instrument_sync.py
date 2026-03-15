from datetime import datetime, timezone

import structlog
from sqlalchemy.orm import Session
from sqlmodel import col, select

from app.models.instrument import Instrument
from app.services.market_data.yfinance import YFinanceProvider

log = structlog.get_logger()


class InstrumentSyncService:
    def __init__(self, session: Session, provider: YFinanceProvider | None = None):
        self._session = session
        self._provider = provider or YFinanceProvider()

    def sync(self) -> dict:
        instruments = self._provider.get_sp500_instruments()
        if not instruments:
            log.warning("instrument_sync_no_data")
            return {"synced": 0, "deactivated": 0}

        now = datetime.now(timezone.utc)
        current_tickers = {r["ticker"] for r in instruments}

        # Soft-deactivate constituents no longer in the index
        removed = (
            self._session.execute(
                select(Instrument).where(
                    col(Instrument.is_active) == True,  # noqa: E712
                    col(Instrument.ticker).notin_(current_tickers),
                )
            )
            .scalars()
            .all()
        )
        for inst in removed:
            inst.is_active = False
            inst.updated_at = now

        # Upsert current constituents
        for record in instruments:
            existing = self._session.get(Instrument, record["ticker"])
            if existing:
                existing.company_name = record["company_name"]
                existing.gics_sector = record.get("gics_sector")
                existing.gics_sub_industry = record.get("gics_sub_industry")
                existing.is_active = True
                existing.updated_at = now
            else:
                self._session.add(
                    Instrument(
                        ticker=record["ticker"],
                        company_name=record["company_name"],
                        gics_sector=record.get("gics_sector"),
                        gics_sub_industry=record.get("gics_sub_industry"),
                        is_active=True,
                        updated_at=now,
                    )
                )

        self._session.commit()
        log.info("instrument_sync_complete", synced=len(instruments), deactivated=len(removed))
        return {"synced": len(instruments), "deactivated": len(removed)}
