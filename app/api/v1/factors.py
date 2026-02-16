from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlmodel import select

from app.core.db import get_db
from app.core.exceptions import raise_not_found
from app.models.factor import Factor
from app.models.schemas import FactorCalculateRequest
from app.services.data_pipeline import MarketDataFetcher
from app.services.factor_engine import FactorEngine
from app.services.factor_repository import FactorRepository

router = APIRouter(prefix="/api/factors", tags=["factors"])

SessionDep = Annotated[Session, Depends(get_db)]


@router.get("/available")
def list_available_factors(session: SessionDep) -> dict:
    """List all available factors in the database."""
    result = session.execute(select(Factor).order_by(Factor.name))
    factors = result.scalars().all()
    return {
        "items": [
            {
                "id": f.id,
                "name": f.name,
                "description": f.description,
                "category": f.category,
                "created_at": f.created_at.isoformat(),
            }
            for f in factors
        ]
    }


@router.get("/{factor_name}/data")
def get_factor_data(
    factor_name: str,
    session: SessionDep,
    start_date: str = Query(..., description="ISO format YYYY-MM-DD"),
    end_date: str = Query(..., description="ISO format YYYY-MM-DD"),
) -> dict:
    """Get time-series factor data for a specific factor."""
    stmt = select(Factor).where(Factor.name == factor_name)
    result = session.execute(stmt)
    if result.scalar_one_or_none() is None:
        raise_not_found("not_found", resource="factor", name=factor_name)

    repo = FactorRepository(session)
    factors = repo.load_factors([factor_name], start_date, end_date)
    df = factors.get(factor_name)

    if df is None or df.empty:
        return {"factor_name": factor_name, "data": {}}

    data = {}
    for ticker in df.index:
        data[ticker] = {
            str(col): float(df.loc[ticker, col])
            for col in df.columns
            if not df.loc[ticker, col] != df.loc[ticker, col]  # skip NaN
        }

    return {"factor_name": factor_name, "data": data}


@router.get("/correlation")
def get_factor_correlation(
    session: SessionDep,
    start_date: str = Query(..., description="ISO format YYYY-MM-DD"),
    end_date: str = Query(..., description="ISO format YYYY-MM-DD"),
    factors: str = Query(..., description="Comma-separated factor names"),
) -> dict:
    """Get cross-factor correlation matrix."""
    factor_names = [f.strip() for f in factors.split(",") if f.strip()]
    repo = FactorRepository(session)
    corr = repo.get_factor_correlation(factor_names, start_date, end_date)

    if corr.empty:
        return {"factors": factor_names, "matrix": []}

    return {"factors": list(corr.columns), "matrix": corr.values.tolist()}


@router.get("/{factor_name}/statistics")
def get_factor_statistics(
    factor_name: str,
    session: SessionDep,
    start_date: str = Query(..., description="ISO format YYYY-MM-DD"),
    end_date: str = Query(..., description="ISO format YYYY-MM-DD"),
) -> dict:
    """Get summary statistics for a specific factor."""
    repo = FactorRepository(session)
    stats = repo.get_factor_statistics(factor_name, start_date, end_date)
    if not stats:
        raise_not_found("not_found", resource="factor_data", name=factor_name)
    return {"factor_name": factor_name, "statistics": stats}


@router.post("/calculate")
def calculate_factors(body: FactorCalculateRequest, session: SessionDep) -> dict:
    """Calculate all factors for the given tickers and date range."""
    fetcher = MarketDataFetcher()
    engine = FactorEngine(fetcher)

    factors = engine.calculate_all_factors(body.tickers, body.start_date, body.end_date)

    repo = FactorRepository(session)
    repo.save_factors(factors)

    return {
        "status": "completed",
        "factors": list(factors.keys()),
        "tickers_count": len(body.tickers),
    }
