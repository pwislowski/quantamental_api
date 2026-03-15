from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlmodel import col, select

from app.core.db import get_db
from app.models.instrument import Instrument
from app.services.instrument_sync import InstrumentSyncService

router = APIRouter(prefix="/api/instruments", tags=["instruments"])

SessionDep = Annotated[Session, Depends(get_db)]


@router.get("")
def list_instruments(session: SessionDep) -> dict:
    """Return all active S&P 500 instruments with GICS classification."""
    instruments = (
        session.execute(
            select(Instrument)
            .where(col(Instrument.is_active) == True)  # noqa: E712
            .order_by(col(Instrument.ticker))
        )
        .scalars()
        .all()
    )

    return {
        "items": [
            {
                "ticker": i.ticker,
                "company_name": i.company_name,
                "gics_sector": i.gics_sector,
                "gics_sub_industry": i.gics_sub_industry,
            }
            for i in instruments
        ],
        "total": len(instruments),
    }


@router.post("/sync")
def sync_instruments(session: SessionDep) -> dict:
    """Re-fetch S&P 500 constituents from Wikipedia and update the instruments table."""
    return InstrumentSyncService(session).sync()
