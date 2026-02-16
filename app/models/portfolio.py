from datetime import date as Date
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import Column
from sqlalchemy.types import JSON
from sqlmodel import Field, SQLModel


class Portfolio(SQLModel, table=True):
    __tablename__ = "portfolios"  # type:ignore

    id: int | None = Field(default=None, primary_key=True)
    name: str
    strategy_type: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    parameters: dict[str, Any] = Field(sa_column=Column(JSON), default_factory=dict)


class PortfolioHolding(SQLModel, table=True):
    __tablename__ = "portfolio_holdings"  # type:ignore

    id: int | None = Field(default=None, primary_key=True)
    portfolio_id: int = Field(foreign_key="portfolios.id")
    ticker: str
    weight: float
    date: Date
