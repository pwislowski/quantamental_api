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


# -----------------------------
# Backtest Schemas
# -----------------------------


class BacktestRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    tickers: List[str] = Field(..., min_length=1)
    start_date: str = Field(..., description="ISO format YYYY-MM-DD")
    end_date: str = Field(..., description="ISO format YYYY-MM-DD")
    strategy_type: str = Field(..., description="mean_variance, risk_parity, or factor_tilt")
    rebalance_frequency: str = Field(default="monthly", description="monthly, quarterly, or yearly")
    constraints: Dict[str, Any] = Field(default_factory=dict)
    risk_aversion: float = Field(default=1.0, gt=0)
    tilts: Optional[Dict[str, float]] = None
    tilt_strength: float = Field(default=1.0, gt=0)


class BacktestMetrics(BaseModel):
    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    sortino_ratio: float
    calmar_ratio: float


class BacktestRead(BaseModel):
    id: int
    name: str
    strategy_type: str
    tickers: List[str]
    start_date: date
    end_date: date
    rebalance_frequency: str
    parameters: Dict[str, Any]
    created_at: datetime
    metrics: Optional[BacktestMetrics] = None


# -----------------------------
# Attribution Schemas
# -----------------------------


class AttributionRequest(BaseModel):
    backtest_id: int
    factor_names: List[str] = Field(default_factory=lambda: ["value", "momentum", "quality", "low_vol", "size"])


class FactorContribution(BaseModel):
    factor_name: str
    contribution: float
    exposure: float


class AttributionResult(BaseModel):
    backtest_id: int
    total_return: float
    factor_contributions: List["FactorContribution"]
    specific_return: float
    r_squared: float
