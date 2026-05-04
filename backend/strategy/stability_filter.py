from __future__ import annotations

"""Regime smoothing to avoid reacting to one-day cluster noise."""

import pandas as pd


def is_regime_stable(regimes: pd.Series, min_days: int = 3) -> bool:
    """Return True when the latest raw regime has persisted for `min_days`."""
    if len(regimes) < min_days:
        return False
    latest = regimes.iloc[-1]
    return bool((regimes.iloc[-min_days:] == latest).all())


def apply_stability_filter(regimes: pd.Series, min_days: int = 3, fallback: str = "neutral") -> pd.Series:
    """Hold the previous stable regime until a new regime persists long enough."""
    stable = []
    previous = fallback
    for i in range(len(regimes)):
        window = regimes.iloc[max(0, i - min_days + 1) : i + 1]
        if len(window) == min_days and (window == window.iloc[-1]).all():
            previous = window.iloc[-1]
        stable.append(previous)
    return pd.Series(stable, index=regimes.index, name="stable_regime")
