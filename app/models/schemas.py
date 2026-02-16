from datetime import date, datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# -----------------------------
# Factor Schemas
# -----------------------------


class FactorBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(default=None, max_length=1000)
    category: Optional[str] = Field(default=None, max_length=255)


class FactorCreate(FactorBase):
    pass


class FactorRead(FactorBase):
    id: int
    created_at: datetime


class FactorDataBase(BaseModel):
    factor_id: int
    ticker: str = Field(..., min_length=1, max_length=32)
    date: date
    value: float
    percentile_rank: Optional[float] = None


class FactorDataCreate(FactorDataBase):
    pass


class FactorDataRead(FactorDataBase):
    id: int


# -----------------------------
# Portfolio Schemas
# -----------------------------


class PortfolioBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    strategy_type: str = Field(..., min_length=1, max_length=255)
    parameters: Dict[str, Any] = Field(default_factory=dict)


class PortfolioCreate(PortfolioBase):
    pass


class PortfolioRead(PortfolioBase):
    id: int
    created_at: datetime


class PortfolioHoldingBase(BaseModel):
    portfolio_id: int
    ticker: str = Field(..., min_length=1, max_length=32)
    weight: float
    date: date


class PortfolioHoldingCreate(PortfolioHoldingBase):
    pass


class PortfolioHoldingRead(PortfolioHoldingBase):
    id: int


# -----------------------------
# Collections (optional)
# -----------------------------


class FactorList(BaseModel):
    items: List[FactorRead]


class FactorDataList(BaseModel):
    items: List[FactorDataRead]


class PortfolioList(BaseModel):
    items: List[PortfolioRead]


class PortfolioHoldingList(BaseModel):
    items: List[PortfolioHoldingRead]


# -----------------------------
# Factor API Schemas
# -----------------------------


class FactorCalculateRequest(BaseModel):
    tickers: List[str] = Field(..., min_length=1)
    start_date: str = Field(..., description="ISO format YYYY-MM-DD")
    end_date: str = Field(..., description="ISO format YYYY-MM-DD")


# -----------------------------
# Portfolio API Schemas
# -----------------------------


class OptimizeRequest(BaseModel):
    tickers: List[str] = Field(..., min_length=1)
    start_date: str = Field(..., description="ISO format YYYY-MM-DD")
    end_date: str = Field(..., description="ISO format YYYY-MM-DD")
    strategy_type: str = Field(..., description="mean_variance, risk_parity, or factor_tilt")
    name: str = Field(..., min_length=1, max_length=255)
    constraints: Dict[str, Any] = Field(default_factory=dict)
    risk_aversion: float = Field(default=1.0, gt=0)
    tilts: Optional[Dict[str, float]] = None
    tilt_strength: float = Field(default=1.0, gt=0)
