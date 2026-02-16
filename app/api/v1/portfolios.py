from datetime import date as Date
from typing import Annotated

import pandas as pd
from fastapi import APIRouter, Depends
from pandas.io.parsers.readers import TYPE_CHECKING
from sqlalchemy.orm import Session
from sqlmodel import col, delete, select

from app.core.db import get_db
from app.core.exceptions import raise_bad_request, raise_not_found
from app.models.factor import Factor, FactorData
from app.models.portfolio import Portfolio, PortfolioHolding
from app.models.schemas import OptimizeRequest
from app.services.data_pipeline import MarketDataFetcher
from app.services.factor_engine import FactorEngine
from app.services.optimizer import PortfolioOptimizer

router = APIRouter(prefix="/api/portfolio", tags=["portfolio"])

SessionDep = Annotated[Session, Depends(get_db)]

VALID_STRATEGIES = ("mean_variance", "risk_parity", "factor_tilt")


@router.post("/optimize")
def optimize_portfolio(body: OptimizeRequest, session: SessionDep) -> dict:
    """Run portfolio optimization, save and return results."""
    if body.strategy_type not in VALID_STRATEGIES:
        raise_bad_request("invalid_strategy", strategy_type=body.strategy_type, valid=list(VALID_STRATEGIES))

    # Fetch price data
    fetcher = MarketDataFetcher()
    price_data = fetcher.fetch_price_data(body.tickers, body.start_date, body.end_date)

    if price_data.empty:
        raise_bad_request("no_price_data", tickers=body.tickers)

    # Compute daily returns from close prices
    if isinstance(price_data.columns, pd.MultiIndex):
        close = price_data.xs("Close", axis=1, level=1)
    else:
        close = price_data[["Close"]]
    returns = close.pct_change().dropna()

    assert isinstance(returns, pd.DataFrame)

    if returns.empty:
        raise_bad_request("insufficient_data", message="Not enough data to compute returns")

    # Run optimization (CPU-bound)
    optimizer = PortfolioOptimizer()

    if body.strategy_type == "mean_variance":
        weights = optimizer.mean_variance_optimization(returns, body.risk_aversion)
    elif body.strategy_type == "risk_parity":
        weights = optimizer.risk_parity_optimization(returns)
    elif body.strategy_type == "factor_tilt":
        if not body.tilts:
            raise_bad_request("missing_tilts", message="tilts required for factor_tilt strategy")

        engine = FactorEngine(fetcher)
        factors = engine.calculate_all_factors(body.tickers, body.start_date, body.end_date)

        factor_dfs = {}
        for name, df in factors.items():
            if not df.empty and name in df.columns:
                factor_dfs[name] = df[name]
        factor_exposures = pd.DataFrame(factor_dfs) if factor_dfs else pd.DataFrame()

        if factor_exposures.empty:
            raise_bad_request("no_factor_data", message="Could not compute factor exposures")

        weights = optimizer.factor_tilt_optimization(
            returns,
            factor_exposures,
            body.tilts,
            body.tilt_strength,
            body.risk_aversion,
        )

    # Apply constraints
    if body.constraints:
        weights = optimizer.apply_constraints(weights, body.constraints)

    # Save portfolio
    parameters = {
        "constraints": body.constraints,
        "risk_aversion": body.risk_aversion,
        "tilts": body.tilts,
        "tilt_strength": body.tilt_strength,
    }
    portfolio = Portfolio(name=body.name, strategy_type=body.strategy_type, parameters=parameters)
    session.add(portfolio)
    session.flush()

    # Save holdings
    today = Date.today()
    holdings = []
    for ticker, weight in weights.items():
        holding = PortfolioHolding(portfolio_id=portfolio.id, ticker=ticker, weight=weight, date=today)
        session.add(holding)
        holdings.append({"ticker": ticker, "weight": weight, "date": str(today)})

    session.commit()

    return {
        "portfolio": {
            "id": portfolio.id,
            "name": portfolio.name,
            "strategy_type": portfolio.strategy_type,
            "created_at": portfolio.created_at.isoformat(),
            "parameters": portfolio.parameters,
        },
        "holdings": holdings,
    }


@router.get("/")
def list_portfolios(session: SessionDep) -> dict:
    """List all portfolios ordered by creation date (newest first)."""
    stmt = select(Portfolio).order_by(col(Portfolio.created_at).desc())
    result = session.execute(stmt)
    portfolios = result.scalars().all()
    return {
        "items": [
            {
                "id": p.id,
                "name": p.name,
                "strategy_type": p.strategy_type,
                "created_at": p.created_at.isoformat(),
                "parameters": p.parameters,
            }
            for p in portfolios
        ]
    }


@router.get("/{portfolio_id}")
def get_portfolio(portfolio_id: int, session: SessionDep) -> dict:
    """Get a single portfolio by ID."""
    stmt = select(Portfolio).where(Portfolio.id == portfolio_id)
    result = session.execute(stmt)
    portfolio = result.scalar_one_or_none()
    if portfolio is None:
        raise_not_found("not_found", resource="portfolio", id=portfolio_id)

    if TYPE_CHECKING:
        assert isinstance(portfolio, Portfolio)

    return {
        "id": portfolio.id,
        "name": portfolio.name,
        "strategy_type": portfolio.strategy_type,
        "created_at": portfolio.created_at.isoformat(),
        "parameters": portfolio.parameters,
    }


@router.get("/{portfolio_id}/holdings")
def get_portfolio_holdings(portfolio_id: int, session: SessionDep) -> dict:
    """Get holdings for a portfolio."""
    stmt = select(Portfolio).where(Portfolio.id == portfolio_id)
    result = session.execute(stmt)
    if result.scalar_one_or_none() is None:
        raise_not_found("not_found", resource="portfolio", id=portfolio_id)

    stmt = select(PortfolioHolding).where(PortfolioHolding.portfolio_id == portfolio_id)
    result = session.execute(stmt)
    holdings = result.scalars().all()

    return {
        "portfolio_id": portfolio_id,
        "holdings": [{"ticker": h.ticker, "weight": h.weight, "date": str(h.date)} for h in holdings],
    }


@router.get("/{portfolio_id}/exposures")
def get_portfolio_exposures(portfolio_id: int, session: SessionDep) -> dict:
    """Get portfolio factor exposures (weighted factor scores)."""
    stmt = select(Portfolio).where(Portfolio.id == portfolio_id)
    result = session.execute(stmt)
    if result.scalar_one_or_none() is None:
        raise_not_found("not_found", resource="portfolio", id=portfolio_id)

    stmt = select(PortfolioHolding).where(PortfolioHolding.portfolio_id == portfolio_id)
    result = session.execute(stmt)
    holdings = result.scalars().all()
    if not holdings:
        return {"portfolio_id": portfolio_id, "exposures": {}}

    tickers = [h.ticker for h in holdings]
    weight_map = {h.ticker: h.weight for h in holdings}

    result = session.execute(
        select(Factor.name, FactorData.ticker, FactorData.value)
        .join(FactorData, col(Factor.id) == col(FactorData.factor_id))
        .where(col(FactorData.ticker).in_(tickers))
    )
    rows = result.all()

    if not rows:
        return {"portfolio_id": portfolio_id, "exposures": {}}

    exposures: dict[str, float] = {}
    for factor_name, ticker, value in rows:
        if ticker in weight_map:
            exposures[factor_name] = exposures.get(factor_name, 0.0) + weight_map[ticker] * value

    return {"portfolio_id": portfolio_id, "exposures": exposures}


@router.delete("/{portfolio_id}")
def delete_portfolio(portfolio_id: int, session: SessionDep) -> dict:
    """Delete a portfolio and its holdings."""
    stmt = select(Portfolio).where(Portfolio.id == portfolio_id)
    result = session.execute(stmt)
    portfolio = result.scalar_one_or_none()
    if portfolio is None:
        raise_not_found("not_found", resource="portfolio", id=portfolio_id)

    stmt = delete(PortfolioHolding).where(col(PortfolioHolding.portfolio_id) == portfolio_id)
    session.execute(stmt)
    session.delete(portfolio)
    session.commit()

    return {"status": "deleted", "portfolio_id": portfolio_id}
