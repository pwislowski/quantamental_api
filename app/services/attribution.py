import numpy as np
import pandas as pd
import structlog

from app.services.factor_engine import FactorEngine
from app.services.market_data.base import MarketDataProvider

log = structlog.get_logger()


class AttributionEngine:
    """Factor-based return attribution. Decomposes portfolio returns into factor contributions via OLS."""

    def __init__(self, factor_engine: FactorEngine, fetcher: MarketDataProvider):
        self.factor_engine = factor_engine
        self.fetcher = fetcher

    def attribute(self, backtest_result, factor_names: list[str]) -> dict:
        """Run factor attribution on a completed backtest.

        Args:
            backtest_result: BacktestResult object (or mock) with holdings_history, returns_series, nav_series.
            factor_names: Which factors to decompose against.

        Returns:
            dict with factor_contributions, specific_return, r_squared, total_return.
        """
        # Reconstruct portfolio returns as a Series
        returns_data = backtest_result.returns_series
        dates = pd.DatetimeIndex([pd.Timestamp(r["date"]) for r in returns_data])
        portfolio_returns = pd.Series([r["return"] for r in returns_data], index=dates)

        if len(portfolio_returns) < 2:
            return self._empty_result(backtest_result, factor_names)

        # Get tickers from holdings
        all_tickers = set()
        for entry in backtest_result.holdings_history:
            all_tickers.update(entry["weights"].keys())
        tickers = sorted(all_tickers)

        # Build daily weights matrix
        daily_weights = self._build_daily_weights(backtest_result.holdings_history, dates)

        # Load factor data
        factor_data = self.factor_engine.calculate_all_factors(
            tickers,
            str(dates[0].date()),
            str(dates[-1].date()),
        )
        # Filter to requested factors
        factor_data = {k: v for k, v in factor_data.items() if k in factor_names}

        if not factor_data:
            total_return = float(portfolio_returns.sum())
            return {
                "backtest_id": getattr(backtest_result, "backtest_id", None),
                "total_return": total_return,
                "factor_contributions": [
                    {"factor_name": f, "contribution": 0.0, "exposure": 0.0} for f in factor_names
                ],
                "specific_return": total_return,
                "r_squared": 0.0,
            }

        # Compute daily factor exposures
        factor_exposures = self._compute_factor_exposures(daily_weights, factor_data)

        # Align indices
        common_idx = portfolio_returns.index.intersection(factor_exposures.index)
        portfolio_returns = portfolio_returns.loc[common_idx]
        factor_exposures = factor_exposures.loc[common_idx]

        # Decompose
        result = self._decompose_returns(portfolio_returns, factor_exposures)
        result["backtest_id"] = getattr(backtest_result, "backtest_id", None)

        return result

    def _build_daily_weights(self, holdings_history: list[dict], date_index: pd.DatetimeIndex) -> pd.DataFrame:
        """Forward-fill rebalance weights to get daily weight matrix (dates × tickers)."""
        # Collect all tickers
        all_tickers = set()
        for entry in holdings_history:
            all_tickers.update(entry["weights"].keys())
        tickers = sorted(all_tickers)

        # Build sparse DataFrame at rebalance dates, then forward-fill
        weight_records = {}
        for entry in holdings_history:
            d = pd.Timestamp(entry["date"])
            weight_records[d] = entry["weights"]

        df = pd.DataFrame(index=date_index, columns=tickers, dtype=float)
        df[:] = np.nan

        for d, weights in weight_records.items():
            if d in df.index:
                for t, w in weights.items():
                    if t in df.columns:
                        df.loc[d, t] = w

        df = df.ffill()
        # Fill any leading NaNs with 0 (before first rebalance)
        df = df.fillna(0.0)

        return df

    def _compute_factor_exposures(
        self, daily_weights: pd.DataFrame, factor_data: dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Compute daily portfolio factor exposures (weighted sum of factor scores)."""
        tickers = daily_weights.columns.tolist()
        exposures = {}

        for factor_name, df in factor_data.items():
            if factor_name not in df.columns:
                continue

            # Factor scores indexed by ticker
            scores = df[factor_name]

            # Align tickers — only use those present in both weights and factor data
            common = [t for t in tickers if t in scores.index]
            if not common:
                exposures[factor_name] = pd.Series(0.0, index=daily_weights.index)
                continue

            aligned_scores = scores.reindex(tickers).fillna(0.0)
            # Daily exposure = dot product of weights and factor scores
            exposures[factor_name] = daily_weights.dot(aligned_scores)

        return pd.DataFrame(exposures, index=daily_weights.index)

    def _decompose_returns(self, portfolio_returns: pd.Series, factor_exposures: pd.DataFrame) -> dict:
        """OLS regression of portfolio returns on factor exposures.

        Returns dict with factor_contributions, specific_return, r_squared, total_return.
        """
        total_return = float(portfolio_returns.sum())

        if len(portfolio_returns) < 2 or factor_exposures.empty:
            return {
                "total_return": total_return,
                "factor_contributions": [
                    {"factor_name": c, "contribution": 0.0, "exposure": 0.0} for c in factor_exposures.columns
                ],
                "specific_return": total_return,
                "r_squared": 0.0,
            }

        # OLS: r = X @ beta + residual (with intercept)
        y = portfolio_returns.values
        X = factor_exposures.values
        n = len(y)

        # Add intercept column
        X_with_intercept = np.column_stack([np.ones(n), X])

        # Solve via least squares
        result, residuals, rank, sv = np.linalg.lstsq(X_with_intercept, y, rcond=None)

        intercept = result[0]
        betas = result[1:]

        # Predicted values
        y_hat = X_with_intercept @ result

        # R²
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        r_squared = max(0.0, min(1.0, r_squared))  # Clamp

        # Factor contributions: beta_f * sum(factor_exposure_f)
        factor_contributions = []
        for i, col in enumerate(factor_exposures.columns):
            contribution = float(betas[i] * factor_exposures[col].sum())
            exposure = float(factor_exposures[col].mean())
            factor_contributions.append(
                {
                    "factor_name": col,
                    "contribution": contribution,
                    "exposure": exposure,
                }
            )

        factor_total = sum(fc["contribution"] for fc in factor_contributions)
        specific_return = total_return - factor_total

        return {
            "total_return": total_return,
            "factor_contributions": factor_contributions,
            "specific_return": float(specific_return),
            "r_squared": r_squared,
        }

    def _empty_result(self, backtest_result, factor_names: list[str]) -> dict:
        """Return zeros for edge cases with insufficient data."""
        return {
            "backtest_id": getattr(backtest_result, "backtest_id", None),
            "total_return": 0.0,
            "factor_contributions": [{"factor_name": f, "contribution": 0.0, "exposure": 0.0} for f in factor_names],
            "specific_return": 0.0,
            "r_squared": 0.0,
        }
