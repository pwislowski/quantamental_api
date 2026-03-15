from datetime import date as Date
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import Column
from sqlalchemy.types import JSON
from sqlmodel import Field, SQLModel


class Backtest(SQLModel, table=True):
    __tablename__ = "backtests"  # type:ignore

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    strategy_type: str
    start_date: Date
    end_date: Date
    rebalance_frequency: str  # monthly, quarterly, yearly
    parameters: dict[str, Any] = Field(sa_column=Column(JSON), default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class BacktestInstrument(SQLModel, table=True):
    __tablename__ = "backtest_instruments"  # type:ignore

    backtest_id: int = Field(foreign_key="backtests.id", primary_key=True)
    ticker: str = Field(foreign_key="instruments.ticker", primary_key=True)


class BacktestResult(SQLModel, table=True):
    __tablename__ = "backtest_results"  # type:ignore

    id: Optional[int] = Field(default=None, primary_key=True)
    backtest_id: int = Field(foreign_key="backtests.id")
    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    sortino_ratio: float
    calmar_ratio: float
    nav_series: list[dict] = Field(sa_column=Column(JSON))
    returns_series: list[dict] = Field(sa_column=Column(JSON))
    holdings_history: list[dict] = Field(sa_column=Column(JSON))
