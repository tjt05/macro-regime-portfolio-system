from __future__ import annotations

"""Portfolio simulation and ledger generation.

This module turns daily regimes into a policy-driven simulated portfolio. It
tracks the ideal model allocation, actionable target allocation, drifted actual
allocation, rebalances, portfolio value, and SPY benchmark value through time.
"""

import pandas as pd

from backend.strategy.allocation import ASSET_ORDER, base_allocation_for_regime
from backend.strategy.decision_engine import generate_decision
from backend.strategy.implementation import simplify_allocation
from backend.strategy.personalization import personalize_allocation


DEFAULT_PORTFOLIO_SETTINGS = {
    "start_date": None,
    "starting_capital": 10_000.0,
    "rebalance_policy": "monthly",
    "min_rebalance_threshold": 0.05,
    "cooldown_days": 5,
    "max_assets": 4,
    "min_position_weight": 0.08,
    "avoid_assets": [],
}


def simulate_portfolio(
    prices: pd.DataFrame,
    regimes: pd.Series,
    raw_regimes: pd.Series | None = None,
    trend_signals: pd.DataFrame | None = None,
    user_profile: dict | None = None,
    portfolio_settings: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Simulate PRAIDS through time under the selected rebalance policy.

    Returns three aligned objects:
    - `results`: full daily portfolio ledger and benchmark series
    - `allocations`: ideal/target/actual allocation columns only
    - `decision_log`: human-readable recommendation records for the UI/API
    """
    settings = {**DEFAULT_PORTFOLIO_SETTINGS, **(portfolio_settings or {})}
    starting_capital = float(settings["starting_capital"])
    start_date = settings.get("start_date")

    simulation_index = regimes.index
    if start_date:
        simulation_index = simulation_index[simulation_index >= pd.Timestamp(start_date)]
    if simulation_index.empty:
        raise ValueError("No regime data available on or after the selected portfolio start date.")

    regimes = regimes.loc[simulation_index]
    raw_regimes = raw_regimes.loc[simulation_index] if raw_regimes is not None else regimes
    aligned_prices = prices.loc[simulation_index, ASSET_ORDER].copy()
    returns = aligned_prices.pct_change().fillna(0.0)
    returns["CASH"] = 0.0

    portfolio_value = starting_capital
    benchmark_value = starting_capital
    actual_allocation: dict[str, float] | None = None
    last_rebalance_date = None
    last_rebalanced_regime = None
    rows: list[dict] = []
    decision_rows: list[dict] = []

    for date, regime in regimes.items():
        ideal_allocation = personalize_allocation(base_allocation_for_regime(regime), user_profile)
        target_allocation, implementation_details = simplify_allocation(ideal_allocation, settings)
        if actual_allocation is None:
            actual_allocation = target_allocation.copy()

        day_return = sum(actual_allocation[asset] * returns.loc[date, asset] for asset in ASSET_ORDER)
        portfolio_value *= 1.0 + day_return
        benchmark_value *= 1.0 + returns.loc[date, "SPY"]
        actual_allocation = _drift_allocation(actual_allocation, returns.loc[date], day_return)

        trend_dict = _trend_dict_for_date(trend_signals, date)
        raw_action = generate_decision(str(regime), trend_dict, user_profile, stable=True)
        should_rebalance = _should_rebalance(
            date=date,
            target_allocation=target_allocation,
            actual_allocation=actual_allocation,
            regime=str(regime),
            last_rebalance_date=last_rebalance_date,
            last_rebalanced_regime=last_rebalanced_regime,
            settings=settings,
        )
        action = "ENTER" if len(rows) == 0 else raw_action if should_rebalance else "HOLD"

        if should_rebalance:
            actual_allocation = target_allocation.copy()
            last_rebalance_date = date
            last_rebalanced_regime = str(regime)

        rows.append(
            {
                "date": date,
                "regime": regime,
                "raw_regime": raw_regimes.loc[date],
                "action": action,
                "rebalance_executed": bool(should_rebalance),
                "portfolio_value": portfolio_value,
                "benchmark_value": benchmark_value,
                "daily_return": day_return,
                "benchmark_daily_return": returns.loc[date, "SPY"],
                **{f"ideal_{asset}": ideal_allocation[asset] for asset in ASSET_ORDER},
                **{f"target_{asset}": target_allocation[asset] for asset in ASSET_ORDER},
                **{f"actual_{asset}": actual_allocation[asset] for asset in ASSET_ORDER},
            }
        )
        decision_rows.append(
            {
                "market_date": date,
                "regime": regime,
                "raw_regime": raw_regimes.loc[date],
                "action": action,
                "rebalance_executed": bool(should_rebalance),
                "ideal_allocation": ideal_allocation.copy(),
                "target_allocation": target_allocation.copy(),
                "actual_allocation": actual_allocation.copy(),
                "implementation_details": implementation_details,
                "portfolio_value": portfolio_value,
            }
        )

    results = pd.DataFrame(rows).set_index("date")
    decision_log = pd.DataFrame(decision_rows).set_index("market_date")
    allocations = results[
        [f"ideal_{asset}" for asset in ASSET_ORDER]
        + [f"target_{asset}" for asset in ASSET_ORDER]
        + [f"actual_{asset}" for asset in ASSET_ORDER]
    ]
    results["cumulative_return"] = results["portfolio_value"] / starting_capital - 1.0
    results["benchmark_cumulative_return"] = results["benchmark_value"] / starting_capital - 1.0
    results["drawdown"] = results["portfolio_value"] / results["portfolio_value"].cummax() - 1.0
    results["benchmark_drawdown"] = results["benchmark_value"] / results["benchmark_value"].cummax() - 1.0
    return results, allocations, decision_log


def _drift_allocation(
    allocation: dict[str, float],
    asset_returns: pd.Series,
    portfolio_return: float,
) -> dict[str, float]:
    """Let weights drift after asset returns before the next rebalance."""
    if 1.0 + portfolio_return <= 0:
        return allocation
    drifted = {
        asset: allocation[asset] * (1.0 + float(asset_returns[asset])) / (1.0 + portfolio_return)
        for asset in ASSET_ORDER
    }
    total = sum(drifted.values())
    return {asset: drifted[asset] / total for asset in ASSET_ORDER}


def _trend_dict_for_date(trend_signals: pd.DataFrame | None, date) -> dict[str, bool]:
    """Extract the decision-engine trend flags for one simulation date."""
    if trend_signals is None or date not in trend_signals.index:
        return {asset: False for asset in ASSET_ORDER if asset != "CASH"}
    row = trend_signals.loc[date]
    return {asset: bool(row.get(asset, False)) for asset in ASSET_ORDER if asset != "CASH"}


def _should_rebalance(
    date,
    target_allocation: dict[str, float],
    actual_allocation: dict[str, float],
    regime: str,
    last_rebalance_date,
    last_rebalanced_regime: str | None,
    settings: dict,
) -> bool:
    """Apply the selected rebalance policy, threshold, and cooldown settings."""
    if last_rebalance_date is None:
        return True

    cooldown_days = int(settings.get("cooldown_days", 0) or 0)
    if cooldown_days > 0 and len(pd.bdate_range(last_rebalance_date, date)) - 1 < cooldown_days:
        return False

    policy = str(settings.get("rebalance_policy", "monthly")).lower()
    allocation_gap = sum(abs(target_allocation[asset] - actual_allocation[asset]) for asset in ASSET_ORDER)
    threshold = float(settings.get("min_rebalance_threshold", 0.05) or 0.0)
    regime_changed = regime != last_rebalanced_regime

    if policy == "daily":
        return allocation_gap >= threshold or regime_changed
    if policy == "weekly":
        return date.isocalendar().week != last_rebalance_date.isocalendar().week and (
            allocation_gap >= threshold or regime_changed
        )
    if policy == "monthly":
        return date.to_period("M") != last_rebalance_date.to_period("M") and (
            allocation_gap >= threshold or regime_changed
        )
    if policy == "regime_change":
        return regime_changed and allocation_gap >= threshold
    if policy == "threshold":
        return allocation_gap >= threshold

    return False
