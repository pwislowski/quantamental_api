from datetime import date as Date
from datetime import datetime, timezone
from typing import Optional

from sqlmodel import Field, Relationship, SQLModel


class Factor(SQLModel, table=True):
    __tablename__ = "factors"  # type:ignore

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True, nullable=False)
    description: Optional[str] = Field(default=None)
    category: Optional[str] = Field(default=None, index=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), nullable=False)

    data_points: list["FactorData"] = Relationship(back_populates="factor")


class FactorData(SQLModel, table=True):
    __tablename__ = "factor_data"  # type:ignore

    id: Optional[int] = Field(default=None, primary_key=True)
    factor_id: int = Field(foreign_key="factors.id", nullable=False, index=True)
    ticker: str = Field(index=True, nullable=False)
    date: Date = Field(nullable=False, index=True)
    value: float = Field(nullable=False)
    percentile_rank: Optional[float] = Field(default=None)

    factor: Optional[Factor] = Relationship(back_populates="data_points")
