import pandas as pd
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlmodel import SQLModel

from app.models.factor import Factor, FactorData
from app.services.factor_repository import FactorRepository


@pytest.fixture
def session():
    """Create an in-memory SQLite sync session for testing."""
    engine = create_engine("sqlite://", echo=False)
    SQLModel.metadata.create_all(bind=engine)

    SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
    with SessionLocal() as s:
        yield s

    engine.dispose()


@pytest.fixture
def sample_factors():
    """Sample factor DataFrames as returned by FactorEngine."""
    return {
        "value": pd.DataFrame(
            {"value": [0.5, -0.3, 1.2, -0.8]},
            index=pd.Index(["AAPL", "MSFT", "XOM", "TSLA"], name="ticker"),
        ),
        "momentum": pd.DataFrame(
            {"momentum": [1.1, 0.2, -0.5, 0.8]},
            index=pd.Index(["AAPL", "MSFT", "XOM", "TSLA"], name="ticker"),
        ),
    }


# --- save_factors ---


def test_save_factors(session, sample_factors):
    repo = FactorRepository(session)
    repo.save_factors(sample_factors)

    factors = session.execute(select(Factor)).scalars().all()
    assert len(factors) == 2
    factor_names = {f.name for f in factors}
    assert factor_names == {"value", "momentum"}

    data = session.execute(select(FactorData)).scalars().all()
    assert len(data) == 8  # 4 tickers * 2 factors


def test_save_factors_upsert(session, sample_factors):
    """Saving the same factors twice should update, not duplicate."""
    repo = FactorRepository(session)
    repo.save_factors(sample_factors)
    repo.save_factors(sample_factors)

    data = session.execute(select(FactorData)).scalars().all()
    assert len(data) == 8  # still 8, not 16


def test_save_factors_empty(session):
    repo = FactorRepository(session)
    repo.save_factors({"empty": pd.DataFrame(columns=["empty"])})

    data = session.execute(select(FactorData)).scalars().all()
    assert len(data) == 0


# --- load_factors ---


def test_load_factors(session, sample_factors):
    repo = FactorRepository(session)
    repo.save_factors(sample_factors)

    loaded = repo.load_factors(["value", "momentum"], "2000-01-01", "2099-12-31")
    assert "value" in loaded
    assert "momentum" in loaded
    assert not loaded["value"].empty
    assert not loaded["momentum"].empty
    assert "AAPL" in loaded["value"].index


def test_load_factors_not_found(session):
    repo = FactorRepository(session)
    loaded = repo.load_factors(["nonexistent"], "2000-01-01", "2099-12-31")
    assert loaded["nonexistent"].empty


# --- get_factor_correlation ---


def test_get_factor_correlation(session, sample_factors):
    repo = FactorRepository(session)
    repo.save_factors(sample_factors)

    corr = repo.get_factor_correlation(["value", "momentum"], "2000-01-01", "2099-12-31")
    assert not corr.empty
    assert "value" in corr.columns
    assert "momentum" in corr.columns
    # Diagonal should be 1.0
    assert abs(corr.loc["value", "value"] - 1.0) < 1e-10


def test_get_factor_correlation_empty(session):
    repo = FactorRepository(session)
    corr = repo.get_factor_correlation(["nonexistent"], "2000-01-01", "2099-12-31")
    assert corr.empty


# --- get_factor_statistics ---


def test_get_factor_statistics(session, sample_factors):
    repo = FactorRepository(session)
    repo.save_factors(sample_factors)

    stats = repo.get_factor_statistics("value", "2000-01-01", "2099-12-31")
    assert "mean" in stats
    assert "std" in stats
    assert "min" in stats
    assert "max" in stats
    assert "median" in stats
    assert "skew" in stats
    assert "kurtosis" in stats
    assert "count" in stats
    assert stats["count"] == 4


def test_get_factor_statistics_empty(session):
    repo = FactorRepository(session)
    stats = repo.get_factor_statistics("nonexistent", "2000-01-01", "2099-12-31")
    assert stats == {}
