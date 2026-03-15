from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlmodel import col, delete, select

from app.core.db import get_db
from app.core.exceptions import raise_bad_request, raise_not_found
from app.models.backtest import Backtest, BacktestInstrument, BacktestResult
from app.models.schemas import BacktestRequest
from app.services.backtester import Backtester
from app.services.data_pipeline import MarketDataFetcher
from app.services.factor_engine import FactorEngine
from app.services.optimizer import PortfolioOptimizer

router = APIRouter(prefix="/api/backtest", tags=["backtest"])

SessionDep = Annotated[Session, Depends(get_db)]

VALID_STRATEGIES = ("mean_variance", "risk_parity", "factor_tilt")
VALID_FREQUENCIES = ("monthly", "quarterly", "yearly")


@router.post("/run")
def run_backtest(body: BacktestRequest, session: SessionDep) -> dict:
    """Run a backtest, persist and return results."""
    if body.strategy_type not in VALID_STRATEGIES:
        raise_bad_request("invalid_strategy", strategy_type=body.strategy_type, valid=list(VALID_STRATEGIES))

    if body.rebalance_frequency not in VALID_FREQUENCIES:
        raise_bad_request(
            "invalid_rebalance_frequency", frequency=body.rebalance_frequency, valid=list(VALID_FREQUENCIES)
        )

    fetcher = MarketDataFetcher()
    optimizer = PortfolioOptimizer()
    factor_engine = FactorEngine(fetcher)
    backtester = Backtester(optimizer=optimizer, fetcher=fetcher, factor_engine=factor_engine)

    return backtester.run(body, session)


@router.get("/")
def list_backtests(session: SessionDep) -> dict:
    """List all backtests ordered by creation date (newest first)."""
    stmt = select(Backtest).order_by(col(Backtest.created_at).desc())
    backtests = session.execute(stmt).scalars().all()

    tickers_by_backtest: dict[int, list[str]] = {b.id: [] for b in backtests}
    if backtests:
        bi_stmt = select(BacktestInstrument).where(col(BacktestInstrument.backtest_id).in_(list(tickers_by_backtest)))
        for bi in session.execute(bi_stmt).scalars().all():
            tickers_by_backtest[bi.backtest_id].append(bi.ticker)
        for tickers in tickers_by_backtest.values():
            tickers.sort()

    return {
        "items": [
            {
                "id": b.id,
                "name": b.name,
                "strategy_type": b.strategy_type,
                "tickers": tickers_by_backtest[b.id],
                "start_date": str(b.start_date),
                "end_date": str(b.end_date),
                "rebalance_frequency": b.rebalance_frequency,
                "created_at": b.created_at.isoformat(),
            }
            for b in backtests
        ]
    }


@router.get("/{backtest_id}")
def get_backtest(backtest_id: int, session: SessionDep) -> dict:
    """Get a single backtest with its result metrics."""
    stmt = select(Backtest).where(Backtest.id == backtest_id)
    result = session.execute(stmt)
    backtest = result.scalar_one_or_none()
    if backtest is None:
        raise_not_found("not_found", resource="backtest", id=backtest_id)

    # Load metrics from result
    stmt = select(BacktestResult).where(BacktestResult.backtest_id == backtest_id)
    result = session.execute(stmt)
    bt_result = result.scalar_one_or_none()

    tickers_stmt = select(BacktestInstrument).where(BacktestInstrument.backtest_id == backtest_id)
    tickers = sorted(bi.ticker for bi in session.execute(tickers_stmt).scalars().all())

    response = {
        "id": backtest.id,
        "name": backtest.name,
        "strategy_type": backtest.strategy_type,
        "tickers": tickers,
        "start_date": str(backtest.start_date),
        "end_date": str(backtest.end_date),
        "rebalance_frequency": backtest.rebalance_frequency,
        "parameters": backtest.parameters,
        "created_at": backtest.created_at.isoformat(),
    }

    if bt_result is not None:
        response["metrics"] = {
            "total_return": bt_result.total_return,
            "annualized_return": bt_result.annualized_return,
            "annualized_volatility": bt_result.annualized_volatility,
            "sharpe_ratio": bt_result.sharpe_ratio,
            "max_drawdown": bt_result.max_drawdown,
            "sortino_ratio": bt_result.sortino_ratio,
            "calmar_ratio": bt_result.calmar_ratio,
        }

    return response


@router.get("/{backtest_id}/result")
def get_backtest_result(backtest_id: int, session: SessionDep) -> dict:
    """Get full backtest result including NAV series and holdings history."""
    stmt = select(Backtest).where(Backtest.id == backtest_id)
    result = session.execute(stmt)
    if result.scalar_one_or_none() is None:
        raise_not_found("not_found", resource="backtest", id=backtest_id)

    stmt = select(BacktestResult).where(BacktestResult.backtest_id == backtest_id)
    result = session.execute(stmt)
    bt_result = result.scalar_one_or_none()
    if bt_result is None:
        raise_not_found("not_found", resource="backtest_result", backtest_id=backtest_id)

    return {
        "backtest_id": backtest_id,
        "metrics": {
            "total_return": bt_result.total_return,
            "annualized_return": bt_result.annualized_return,
            "annualized_volatility": bt_result.annualized_volatility,
            "sharpe_ratio": bt_result.sharpe_ratio,
            "max_drawdown": bt_result.max_drawdown,
            "sortino_ratio": bt_result.sortino_ratio,
            "calmar_ratio": bt_result.calmar_ratio,
        },
        "nav_series": bt_result.nav_series,
        "returns_series": bt_result.returns_series,
        "holdings_history": bt_result.holdings_history,
    }


@router.delete("/{backtest_id}")
def delete_backtest(backtest_id: int, session: SessionDep) -> dict:
    """Delete a backtest and its result."""
    stmt = select(Backtest).where(Backtest.id == backtest_id)
    result = session.execute(stmt)
    backtest = result.scalar_one_or_none()
    if backtest is None:
        raise_not_found("not_found", resource="backtest", id=backtest_id)

    stmt = delete(BacktestResult).where(col(BacktestResult.backtest_id) == backtest_id)
    session.execute(stmt)
    session.delete(backtest)
    session.commit()

    return {"status": "deleted", "backtest_id": backtest_id}
