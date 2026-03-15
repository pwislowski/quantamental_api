from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlmodel import select

from app.core.db import get_db
from app.core.exceptions import raise_not_found
from app.models.backtest import Backtest, BacktestInstrument, BacktestResult
from app.models.schemas import AttributionRequest
from app.services.attribution import AttributionEngine
from app.services.data_pipeline import MarketDataFetcher
from app.services.factor_engine import FactorEngine

router = APIRouter(prefix="/api/analytics", tags=["analytics"])

SessionDep = Annotated[Session, Depends(get_db)]


@router.post("/attribution")
def run_attribution(body: AttributionRequest, session: SessionDep) -> dict:
    """Run factor attribution on a completed backtest."""
    stmt = select(BacktestResult).where(BacktestResult.backtest_id == body.backtest_id)
    result = session.execute(stmt)
    bt_result = result.scalar_one_or_none()
    if bt_result is None:
        raise_not_found("not_found", resource="backtest_result", backtest_id=body.backtest_id)

    fetcher = MarketDataFetcher()
    factor_engine = FactorEngine(fetcher)
    engine = AttributionEngine(factor_engine=factor_engine, fetcher=fetcher)

    return engine.attribute(bt_result, body.factor_names)


@router.get("/{backtest_id}/summary")
def get_backtest_summary(backtest_id: int, session: SessionDep) -> dict:
    """Get a backtest summary with key metrics."""
    stmt = select(Backtest).where(Backtest.id == backtest_id)
    result = session.execute(stmt)
    backtest = result.scalar_one_or_none()
    if backtest is None:
        raise_not_found("not_found", resource="backtest", id=backtest_id)

    stmt = select(BacktestResult).where(BacktestResult.backtest_id == backtest_id)
    result = session.execute(stmt)
    bt_result = result.scalar_one_or_none()
    if bt_result is None:
        raise_not_found("not_found", resource="backtest_result", backtest_id=backtest_id)

    tickers_stmt = select(BacktestInstrument).where(BacktestInstrument.backtest_id == backtest_id)
    tickers = sorted(bi.ticker for bi in session.execute(tickers_stmt).scalars().all())

    return {
        "backtest_id": backtest.id,
        "name": backtest.name,
        "strategy_type": backtest.strategy_type,
        "tickers": tickers,
        "start_date": str(backtest.start_date),
        "end_date": str(backtest.end_date),
        "rebalance_frequency": backtest.rebalance_frequency,
        "metrics": {
            "total_return": bt_result.total_return,
            "annualized_return": bt_result.annualized_return,
            "annualized_volatility": bt_result.annualized_volatility,
            "sharpe_ratio": bt_result.sharpe_ratio,
            "max_drawdown": bt_result.max_drawdown,
            "sortino_ratio": bt_result.sortino_ratio,
            "calmar_ratio": bt_result.calmar_ratio,
        },
    }
