from __future__ import annotations

"""Portfolio evaluation metrics for simulation outputs.

The evaluator intentionally stays separate from the simulation loop: the
simulator creates the ledger, while this module summarizes returns, risk,
drawdowns, Sharpe ratios, and regime-level behavior.
"""

import pandas as pd


def evaluate_performance(results: pd.DataFrame, starting_capital: float = 10_000.0) -> dict[str, float]:
    """Calculate headline strategy and SPY benchmark performance metrics."""
    if results.empty:
        return {}

    trading_days = 252
    total_return = results["portfolio_value"].iloc[-1] / starting_capital - 1.0
    benchmark_return = results["benchmark_value"].iloc[-1] / starting_capital - 1.0
    strategy_vol = results["daily_return"].std() * trading_days**0.5
    benchmark_vol = results["benchmark_daily_return"].std() * trading_days**0.5

    return {
        "total_return": float(total_return),
        "benchmark_return": float(benchmark_return),
        "excess_return": float(total_return - benchmark_return),
        "max_drawdown": float(results["drawdown"].min()),
        "benchmark_max_drawdown": float(results["benchmark_drawdown"].min()),
        "annualized_volatility": float(strategy_vol),
        "benchmark_annualized_volatility": float(benchmark_vol),
        "sharpe_ratio": float(_annualized_sharpe(results["daily_return"])),
        "benchmark_sharpe_ratio": float(_annualized_sharpe(results["benchmark_daily_return"])),
        "ending_value": float(results["portfolio_value"].iloc[-1]),
        "benchmark_ending_value": float(results["benchmark_value"].iloc[-1]),
    }


def evaluate_regime_performance(results: pd.DataFrame) -> list[dict]:
    """Summarize strategy behavior inside each interpreted macro regime."""
    rows = []
    for regime, group in results.groupby("regime"):
        strategy_growth = (1.0 + group["daily_return"]).prod() - 1.0
        benchmark_growth = (1.0 + group["benchmark_daily_return"]).prod() - 1.0
        rows.append(
            {
                "regime": regime,
                "trading_days": int(len(group)),
                "strategy_return": float(strategy_growth),
                "benchmark_return": float(benchmark_growth),
                "excess_return": float(strategy_growth - benchmark_growth),
                "annualized_volatility": float(group["daily_return"].std() * 252**0.5),
                "sharpe_ratio": float(_annualized_sharpe(group["daily_return"])),
                "max_drawdown_observed": float(group["drawdown"].min()),
            }
        )
    return sorted(rows, key=lambda row: row["trading_days"], reverse=True)


def _annualized_sharpe(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Compute annualized Sharpe ratio from daily returns."""
    if returns.empty:
        return 0.0
    daily_rf = risk_free_rate / 252
    excess = returns - daily_rf
    std = excess.std()
    if pd.isna(std) or std == 0:
        return 0.0
    return excess.mean() / std * 252**0.5
