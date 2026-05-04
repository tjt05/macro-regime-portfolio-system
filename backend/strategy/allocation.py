from __future__ import annotations

"""Regime-based model portfolio templates.

The allocation engine starts from simple, interpretable templates instead of a
black-box optimizer. Later layers personalize these weights and simplify them
into a smaller actionable portfolio.
"""

from backend.assets import ASSET_ORDER


BASE_ALLOCATIONS = {
    "growth_expansion": {
        "SPY": 0.28,
        "QQQ": 0.18,
        "IWM": 0.10,
        "XLP": 0.05,
        "XLV": 0.06,
        "XLU": 0.03,
        "TLT": 0.05,
        "IEF": 0.08,
        "SHY": 0.05,
        "GLD": 0.04,
        "DBC": 0.03,
        "BTC": 0.03,
        "CASH": 0.02,
    },
    "liquidity_risk_on": {
        "SPY": 0.22,
        "QQQ": 0.24,
        "IWM": 0.09,
        "XLP": 0.03,
        "XLV": 0.04,
        "XLU": 0.02,
        "TLT": 0.04,
        "IEF": 0.06,
        "SHY": 0.05,
        "GLD": 0.04,
        "DBC": 0.03,
        "BTC": 0.10,
        "CASH": 0.04,
    },
    "recession_risk_off": {
        "SPY": 0.06,
        "QQQ": 0.03,
        "IWM": 0.02,
        "XLP": 0.10,
        "XLV": 0.10,
        "XLU": 0.08,
        "TLT": 0.22,
        "IEF": 0.18,
        "SHY": 0.14,
        "GLD": 0.05,
        "DBC": 0.00,
        "BTC": 0.00,
        "CASH": 0.02,
    },
    "inflation_shock": {
        "SPY": 0.10,
        "QQQ": 0.04,
        "IWM": 0.04,
        "XLP": 0.08,
        "XLV": 0.08,
        "XLU": 0.05,
        "TLT": 0.04,
        "IEF": 0.08,
        "SHY": 0.12,
        "GLD": 0.18,
        "DBC": 0.16,
        "BTC": 0.01,
        "CASH": 0.02,
    },
    "neutral_transition": {
        "SPY": 0.16,
        "QQQ": 0.08,
        "IWM": 0.05,
        "XLP": 0.08,
        "XLV": 0.08,
        "XLU": 0.06,
        "TLT": 0.10,
        "IEF": 0.14,
        "SHY": 0.10,
        "GLD": 0.07,
        "DBC": 0.04,
        "BTC": 0.01,
        "CASH": 0.03,
    },
}


def base_allocation_for_regime(regime: str) -> dict[str, float]:
    """Return the normalized ideal allocation template for one macro regime."""
    return normalize_allocation(BASE_ALLOCATIONS.get(regime, BASE_ALLOCATIONS["neutral_transition"]).copy())


def normalize_allocation(allocation: dict[str, float]) -> dict[str, float]:
    """Clamp negative/missing weights to zero and rescale weights to sum to 1."""
    cleaned = {asset: max(0.0, float(allocation.get(asset, 0.0))) for asset in ASSET_ORDER}
    total = sum(cleaned.values())
    if total <= 0:
        return {asset: 1.0 if asset == "CASH" else 0.0 for asset in ASSET_ORDER}
    return {asset: cleaned[asset] / total for asset in ASSET_ORDER}
