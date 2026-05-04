from __future__ import annotations

"""User-profile adjustments for PRAIDS allocations.

Personalization happens after the regime template and before the practical
implementation layer. It adjusts exposure based on savings buffer, risk
tolerance, experience level, and investment horizon.
"""

from backend.strategy.allocation import normalize_allocation


DEFAULT_USER_PROFILE = {
    "income": 1500,
    "expenses": 900,
    "savings": 5000,
    "risk_tolerance": "medium",
    "experience_level": "beginner",
    "experience": "beginner",
    "investment_horizon": "long",
}


def personalize_allocation(
    allocation: dict[str, float],
    user_profile: dict | None = None,
) -> dict[str, float]:
    """Adjust an ideal allocation for the user's financial profile."""
    profile = {**DEFAULT_USER_PROFILE, **(user_profile or {})}
    adjusted = allocation.copy()

    expenses = max(float(profile.get("expenses", 0) or 0), 0.0)
    savings = max(float(profile.get("savings", 0) or 0), 0.0)
    months_buffer = savings / expenses if expenses else float("inf")

    high_vol_assets = ["QQQ", "IWM", "BTC", "DBC"]
    growth_assets = ["SPY", "QQQ", "IWM", "BTC"]
    defensive_assets = ["XLP", "XLV", "XLU", "TLT", "IEF", "SHY", "GLD", "CASH"]

    if months_buffer < 3:
        _shift(adjusted, growth_assets + ["DBC"], ["SHY", "CASH", "IEF"], 0.15)

    experience = str(profile.get("experience_level") or profile.get("experience") or "").lower()
    if experience == "beginner":
        _shift(adjusted, high_vol_assets, ["SPY", "SHY", "CASH"], 0.10)

    risk_tolerance = str(profile.get("risk_tolerance", "medium")).lower()
    if risk_tolerance == "high":
        _shift(adjusted, ["SHY", "IEF", "TLT", "CASH", "XLP", "XLU"], ["SPY", "QQQ", "IWM", "BTC"], 0.12)
    elif risk_tolerance == "low":
        _shift(adjusted, growth_assets + ["DBC"], defensive_assets, 0.15)

    horizon = str(profile.get("investment_horizon", "long")).lower()
    if horizon == "short":
        _shift(adjusted, growth_assets + ["DBC"], ["SHY", "CASH"], 0.10)
    elif horizon == "long" and risk_tolerance != "low":
        _shift(adjusted, ["SHY", "CASH"], ["SPY", "QQQ"], 0.04)

    return normalize_allocation(adjusted)


def _shift(
    allocation: dict[str, float],
    sources: list[str],
    destinations: list[str],
    amount: float,
) -> None:
    """Move a fixed amount of weight from source assets into destinations."""
    available = sum(allocation.get(asset, 0.0) for asset in sources)
    moved = min(amount, available)
    if moved <= 0:
        return

    for asset in sources:
        current = allocation.get(asset, 0.0)
        allocation[asset] = current - moved * (current / available)

    each = moved / len(destinations)
    for asset in destinations:
        allocation[asset] = allocation.get(asset, 0.0) + each
