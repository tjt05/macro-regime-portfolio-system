from __future__ import annotations

"""Shared asset universe definitions for PRAIDS.

This file is intentionally small and central: every data, strategy, simulation,
and API module imports asset lists from here so the project has one source of
truth for tickers, internal asset names, and display/order conventions.
"""

TICKER_MAP = {
    "SPY": "SPY",
    "QQQ": "QQQ",
    "IWM": "IWM",
    "XLP": "XLP",
    "XLV": "XLV",
    "XLU": "XLU",
    "TLT": "TLT",
    "IEF": "IEF",
    "SHY": "SHY",
    "GLD": "GLD",
    "DBC": "DBC",
    "BTC": "BTC-USD",
}

RISK_ASSETS = ["SPY", "QQQ", "IWM", "BTC"]
DEFENSIVE_EQUITIES = ["XLP", "XLV", "XLU"]
BONDS = ["TLT", "IEF", "SHY"]
INFLATION_HEDGES = ["GLD", "DBC"]
TRADABLE_ASSETS = RISK_ASSETS + DEFENSIVE_EQUITIES + BONDS + INFLATION_HEDGES
ASSET_ORDER = TRADABLE_ASSETS + ["CASH"]
FEATURE_ASSETS = TRADABLE_ASSETS
