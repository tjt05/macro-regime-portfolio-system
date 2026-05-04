from __future__ import annotations

"""Practical portfolio implementation constraints.

The model can reason over many macro assets, but real investors often prefer a
compact set of holdings. This module converts ideal diversified weights into an
actionable top-K/min-weight allocation while preserving total portfolio weight.
"""

from backend.strategy.allocation import normalize_allocation


DEFAULT_IMPLEMENTATION_SETTINGS = {
    "max_assets": 4,
    "min_position_weight": 0.08,
    "avoid_assets": [],
}


def simplify_allocation(
    ideal_allocation: dict[str, float],
    implementation_settings: dict | None = None,
) -> tuple[dict[str, float], dict]:
    """
    Convert a diversified model allocation into a practical portfolio.

    The regime model can inspect many assets, but a retail investor may only
    want to hold a few. This layer applies avoid lists, top-K selection, and
    minimum position sizes, then renormalizes the surviving weights.
    """
    settings = {**DEFAULT_IMPLEMENTATION_SETTINGS, **(implementation_settings or {})}
    max_assets = max(1, int(settings.get("max_assets", 4) or 4))
    min_weight = max(0.0, float(settings.get("min_position_weight", 0.08) or 0.0))
    avoided = set(settings.get("avoid_assets") or [])

    candidates = {
        asset: weight
        for asset, weight in ideal_allocation.items()
        if asset not in avoided and weight > 0
    }
    if not candidates:
        candidates = {"CASH": 1.0}

    ranked = sorted(candidates.items(), key=lambda item: item[1], reverse=True)
    selected = dict(ranked[:max_assets])

    above_min = {asset: weight for asset, weight in selected.items() if weight >= min_weight}
    if above_min:
        selected = above_min
    elif selected:
        selected = {ranked[0][0]: ranked[0][1]}

    actionable = normalize_allocation(selected)
    removed_assets = [asset for asset, weight in ideal_allocation.items() if actionable.get(asset, 0.0) == 0 and weight > 0]

    diagnostics = {
        "max_assets": max_assets,
        "min_position_weight": min_weight,
        "avoid_assets": sorted(avoided),
        "selected_assets": [asset for asset, weight in actionable.items() if weight > 0],
        "removed_assets": removed_assets,
    }
    return actionable, diagnostics
