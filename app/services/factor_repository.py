from datetime import date

import pandas as pd
import structlog
from sqlalchemy.orm import Session
from sqlmodel import col, delete, select

from app.models.factor import Factor, FactorData

log = structlog.get_logger()


class FactorRepository:
    """Database operations for factor data persistence and retrieval."""

    def __init__(self, session: Session):
        self.session = session

    def save_factors(self, factors: dict[str, pd.DataFrame]) -> None:
        """Persist calculated factors to the database (upsert via delete+insert)."""
        for factor_name, df in factors.items():
            if df.empty:
                continue
            factor = self._get_or_create_factor(factor_name)

            if factor_name not in df.columns:
                continue

            values = df[factor_name].dropna()
            if values.empty:
                continue

            tickers = list(values.index)
            today = date.today()

            stmt = delete(FactorData).where(
                col(FactorData.factor_id) == factor.id,
                col(FactorData.ticker).in_(tickers),
                col(FactorData.date) == today,
            )
            self.session.execute(stmt)

            ranks = values.rank(pct=True)
            rows = [
                FactorData(
                    factor_id=factor.id,
                    ticker=ticker,
                    date=today,
                    value=float(values[ticker]),
                    percentile_rank=float(ranks[ticker]),
                )
                for ticker in tickers
            ]
            self.session.add_all(rows)
            self.session.commit()

    def load_factors(self, factor_names: list[str], start_date: str, end_date: str) -> dict[str, pd.DataFrame]:
        """Load factor data from the database for the given names and date range."""
        start = date.fromisoformat(start_date)
        end = date.fromisoformat(end_date)

        stmt = (
            select(col(Factor.name), col(FactorData.ticker), col(FactorData.date), col(FactorData.value))
            .join(FactorData, col(Factor.id) == FactorData.factor_id)
            .where(col(Factor.name).in_(factor_names), col(FactorData.date) >= start, col(FactorData.date) <= end)
        )
        result = self.session.execute(stmt)
        rows = result.all()

        if not rows:
            return {name: pd.DataFrame() for name in factor_names}

        df = pd.DataFrame(rows, columns=["factor_name", "ticker", "date", "value"])
        factors = {}
        for name in factor_names:
            factor_df = df[df["factor_name"] == name]
            if factor_df.empty:
                factors[name] = pd.DataFrame()
            else:
                pivoted = factor_df.pivot(index="ticker", columns="date", values="value")
                factors[name] = pivoted
        return factors

    def get_factor_correlation(self, factor_names: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Compute cross-factor correlation matrix."""
        factors = self.load_factors(factor_names, start_date, end_date)
        combined = {}
        for name, df in factors.items():
            if df.empty:
                continue
            latest = df.iloc[:, -1] if len(df.columns) > 0 else pd.Series(dtype=float)
            combined[name] = latest

        if not combined:
            return pd.DataFrame()
        return pd.DataFrame(combined).dropna().corr()

    def get_factor_statistics(self, factor_name: str, start_date: str, end_date: str) -> dict:
        """Compute summary statistics for a single factor."""
        factors = self.load_factors([factor_name], start_date, end_date)
        df = factors.get(factor_name, pd.DataFrame())
        if df.empty:
            return {}

        all_values = df.values.flatten()
        all_values = all_values[~pd.isna(all_values)]
        if len(all_values) == 0:
            return {}

        series = pd.Series(all_values)
        std = series.std()
        return {
            "mean": float(series.mean()),
            "std": float(std),  # type:ignore
            "min": float(series.min()),
            "max": float(series.max()),
            "median": float(series.median()),
            "skew": float(series.skew()),  # type:ignore
            "kurtosis": float(series.kurtosis()),  # type:ignore
            "count": int(len(series)),
            "information_ratio": float(series.mean() / std) if std > 0 else None,
        }

    def _get_or_create_factor(self, name: str) -> Factor:
        result = self.session.execute(select(Factor).where(col(Factor.name) == name))
        factor = result.scalar_one_or_none()
        if factor is None:
            factor = Factor(name=name, category="equity")
            self.session.add(factor)
            self.session.flush()
        return factor
