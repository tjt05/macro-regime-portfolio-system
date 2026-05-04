from __future__ import annotations

"""Rule-based action layer for PRAIDS recommendations.

Regime labels describe the market environment; actions translate that environment
into simple guidance terms for the dashboard. The rules deliberately remain
plain Python so users can inspect and adjust the investment logic.
"""


VALID_ACTIONS = ["ENTER", "ADD", "HOLD", "REDUCE", "EXIT"]


def generate_decision(
    regime: str,
    trend_signals: dict[str, bool],
    user_profile: dict | None = None,
    stable: bool = True,
) -> str:
    """Generate ENTER/ADD/HOLD/REDUCE/EXIT from regime and trend context."""
    if not stable:
        return "HOLD"

    broad_equity_trend = trend_signals.get("SPY", False) and (
        trend_signals.get("QQQ", False) or trend_signals.get("IWM", False)
    )
    liquidity_trend = trend_signals.get("QQQ", False) and trend_signals.get("BTC", False)

    if regime == "growth_expansion" and broad_equity_trend:
        return "ADD"
    if regime == "growth_expansion":
        return "ENTER"
    if regime == "liquidity_risk_on" and liquidity_trend:
        return "ADD"
    if regime == "liquidity_risk_on":
        return "ENTER"
    if regime == "recession_risk_off":
        return "EXIT"
    if regime == "inflation_shock":
        return "REDUCE"
    return "HOLD"
