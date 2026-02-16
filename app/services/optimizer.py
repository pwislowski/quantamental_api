import numpy as np
import pandas as pd
import structlog
from scipy.optimize import minimize

log = structlog.get_logger()


class PortfolioOptimizer:
    """Portfolio optimization using scipy. All methods are synchronous (CPU-bound)."""

    def mean_variance_optimization(
        self, returns: pd.DataFrame, risk_aversion: float = 1.0, **kwargs
    ) -> dict[str, float]:
        """Markowitz mean-variance optimization."""
        tickers = list(returns.columns)
        n = len(tickers)
        if n == 0:
            return {}
        if n == 1:
            return {tickers[0]: 1.0}

        mu = returns.mean().values
        cov = returns.cov().values

        def objective(w):
            return -(w @ mu - (risk_aversion / 2) * w @ cov @ w)

        result = self._optimize(objective, n)
        if result is None:
            return self._equal_weight(tickers)
        return dict(zip(tickers, result.x))

    def risk_parity_optimization(self, returns: pd.DataFrame, **kwargs) -> dict[str, float]:
        """Equal risk contribution optimization."""
        tickers = list(returns.columns)
        n = len(tickers)
        if n == 0:
            return {}
        if n == 1:
            return {tickers[0]: 1.0}

        cov = returns.cov().values

        def objective(w):
            port_vol = np.sqrt(w @ cov @ w)
            if port_vol < 1e-12:
                return 0.0
            marginal = cov @ w
            risk_contrib = w * marginal / port_vol
            target = port_vol / n
            return np.sum((risk_contrib - target) ** 2)

        result = self._optimize(objective, n)
        if result is None:
            return self._equal_weight(tickers)
        return dict(zip(tickers, result.x))

    def factor_tilt_optimization(
        self,
        returns: pd.DataFrame,
        factor_exposures: pd.DataFrame,
        tilts: dict[str, float],
        tilt_strength: float = 1.0,
        risk_aversion: float = 1.0,
        **kwargs,
    ) -> dict[str, float]:
        """Mean-variance optimization with factor tilt penalty."""
        tickers = list(returns.columns)
        n = len(tickers)
        if n == 0:
            return {}
        if n == 1:
            return {tickers[0]: 1.0}

        mu = returns.mean().values
        cov = returns.cov().values

        common = [t for t in tickers if t in factor_exposures.index]
        if not common or not tilts:
            return self.mean_variance_optimization(returns, risk_aversion=risk_aversion)

        factor_names = [f for f in tilts if f in factor_exposures.columns]
        if not factor_names:
            return self.mean_variance_optimization(returns, risk_aversion=risk_aversion)

        F = factor_exposures.reindex(index=tickers, columns=factor_names).fillna(0).values
        target = np.array([tilts[f] for f in factor_names])

        def objective(w):
            mv_term = -(w @ mu - (risk_aversion / 2) * w @ cov @ w)
            port_exposure = w @ F
            tilt_penalty = tilt_strength * np.sum((port_exposure - target) ** 2)
            return mv_term + tilt_penalty

        result = self._optimize(objective, n)
        if result is None:
            return self._equal_weight(tickers)
        return dict(zip(tickers, result.x))

    def apply_constraints(self, weights: dict[str, float], constraints: dict) -> dict[str, float]:
        """Apply post-optimization constraints."""
        if not weights:
            return weights

        w = dict(weights)

        max_positions = constraints.get("max_positions")
        if max_positions is not None and len(w) > max_positions:
            sorted_tickers = sorted(w, key=lambda t: w[t], reverse=True)
            w = {t: w[t] for t in sorted_tickers[:max_positions]}

        min_pos = constraints.get("min_position")
        if min_pos is not None:
            w = {t: v for t, v in w.items() if v >= min_pos}

        max_pos = constraints.get("max_position")
        if max_pos is not None:
            w = self._normalize(w)
            locked = {}
            free = dict(w)
            for _ in range(len(w)):
                over = {t for t, v in free.items() if v > max_pos}
                if not over:
                    break
                for t in over:
                    locked[t] = max_pos
                    del free[t]
                remaining = 1.0 - max_pos * len(locked)
                if remaining > 0 and free:
                    free_total = sum(free.values())
                    free = {t: v / free_total * remaining for t, v in free.items()}
            w = {**locked, **free}
            return w

        return self._normalize(w)

    def _optimize(self, objective, n: int):
        w0 = np.ones(n) / n
        bounds = [(0.0, 1.0)] * n
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        try:
            result = minimize(objective, w0, method="SLSQP", bounds=bounds, constraints=constraints)
            if not result.success:
                log.warning("optimization_did_not_converge", message=result.message)
                return None
            result.x = np.maximum(result.x, 0.0)
            result.x /= result.x.sum()
            return result
        except Exception:
            log.warning("optimization_failed", exc_info=True)
            return None

    def _equal_weight(self, tickers: list[str]) -> dict[str, float]:
        log.info("falling_back_to_equal_weight", n=len(tickers))
        w = 1.0 / len(tickers)
        return {t: w for t in tickers}

    def _normalize(self, weights: dict[str, float]) -> dict[str, float]:
        total = sum(weights.values())
        if total <= 0:
            return {}
        return {t: v / total for t, v in weights.items()}
