from datetime import datetime, timezone

from sqlmodel import Field, SQLModel


class Instrument(SQLModel, table=True):
    __tablename__ = "instruments"

    ticker: str = Field(primary_key=True)
    company_name: str
    gics_sector: str | None = None
    gics_sub_industry: str | None = None
    is_active: bool = Field(default=True)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
