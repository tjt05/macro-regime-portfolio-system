from __future__ import annotations

"""Interpret unsupervised clusters as human-readable macro regimes.

KMeans only produces numeric cluster IDs. This module inspects each cluster's
centroid features and assigns economic labels such as growth expansion,
inflation shock, or recession risk-off. The labels are rule-based and
transparent, which keeps the ML layer interpretable.
"""

import pandas as pd


MACRO_REGIMES = [
    "growth_expansion",
    "inflation_shock",
    "recession_risk_off",
    "liquidity_risk_on",
    "neutral_transition",
]

REGIME_EXPLANATIONS = {
    "growth_expansion": "Broad equity strength with healthier risk appetite across large cap, growth, and small caps.",
    "inflation_shock": "Commodities and gold show relative strength while bonds and growth assets are less dominant.",
    "recession_risk_off": "Weak equity momentum, elevated drawdowns or volatility, and stronger defensive bond behavior.",
    "liquidity_risk_on": "High-beta growth and liquidity-sensitive assets such as QQQ and BTC show strong relative momentum.",
    "neutral_transition": "Mixed macro signals without a clear dominant growth, inflation, or risk-off profile.",
}


def derive_regime_mapping(centroids: pd.DataFrame) -> dict[int, str]:
    """Assign each KMeans cluster to the macro regime with the strongest score."""
    scores = macro_scores(centroids)
    mapping: dict[int, str] = {}
    available = set(centroids.index)

    for regime in ["growth_expansion", "recession_risk_off", "inflation_shock", "liquidity_risk_on"]:
        if not available:
            break
        cluster = scores.loc[list(available), regime].idxmax()
        mapping[int(cluster)] = regime
        available.remove(cluster)

    for cluster in available:
        mapping[int(cluster)] = "neutral_transition"

    return mapping


def macro_scores(centroids: pd.DataFrame) -> pd.DataFrame:
    """Score cluster centroids against each macro-regime interpretation rule."""
    scores = pd.DataFrame(index=centroids.index)
    equity_momentum = _avg(centroids, ["SPY_ret_60d", "QQQ_ret_60d", "IWM_ret_60d"])
    equity_trend = _avg(centroids, ["SPY_price_to_ma200", "QQQ_price_to_ma200", "IWM_price_to_ma200"])
    defensive_momentum = _avg(centroids, ["XLP_ret_60d", "XLV_ret_60d", "XLU_ret_60d"])
    bond_momentum = _avg(centroids, ["TLT_ret_60d", "IEF_ret_60d", "SHY_ret_60d"])
    inflation_momentum = _avg(centroids, ["GLD_ret_60d", "DBC_ret_60d"])
    inflation_trend = _avg(centroids, ["GLD_price_to_ma200", "DBC_price_to_ma200"])
    equity_drawdown = _avg(centroids, ["SPY_max_drawdown", "QQQ_max_drawdown", "IWM_max_drawdown"])
    equity_vol = _avg(centroids, ["SPY_vol_20d", "QQQ_vol_20d", "IWM_vol_20d"])

    scores["growth_expansion"] = (
        equity_momentum
        + equity_trend
        + centroids.get("SPY_TLT_ratio_trend", 0)
        - equity_vol
        - inflation_momentum * 0.25
    )
    scores["recession_risk_off"] = (
        -equity_momentum
        - equity_drawdown
        + bond_momentum
        + defensive_momentum * 0.5
        + equity_vol
    )
    scores["inflation_shock"] = (
        inflation_momentum
        + inflation_trend
        + centroids.get("DBC_SPY_ret_60d_spread", 0)
        - bond_momentum
        - equity_momentum * 0.25
    )
    scores["liquidity_risk_on"] = (
        centroids.get("QQQ_ret_60d", 0)
        + centroids.get("BTC_ret_60d", 0)
        + centroids.get("BTC_SPY_ratio_trend", 0)
        + centroids.get("QQQ_IWM_ratio_trend", 0)
        - bond_momentum * 0.25
    )
    scores["neutral_transition"] = -scores[
        ["growth_expansion", "recession_risk_off", "inflation_shock", "liquidity_risk_on"]
    ].abs().sum(axis=1)
    return scores


def _avg(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Average whichever requested centroid columns exist in the model artifact."""
    existing = [column for column in columns if column in frame.columns]
    if not existing:
        return pd.Series(0.0, index=frame.index)
    return frame[existing].mean(axis=1)
