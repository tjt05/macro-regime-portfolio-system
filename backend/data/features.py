from __future__ import annotations

"""Feature engineering for macro-regime clustering.

The KMeans model is trained on technical and cross-asset signals rather than raw
prices. Each feature describes trend, momentum, risk, or relative asset behavior,
which makes the resulting clusters easier to interpret as macro regimes.
"""

import pandas as pd

from backend.assets import FEATURE_ASSETS

ASSETS = FEATURE_ASSETS


def compute_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Create the feature matrix used for unsupervised market regime learning.

    The output index is the intersection of dates where all rolling-window
    features are available. Rows with incomplete long-window indicators are
    dropped so model training and prediction receive a dense numeric matrix.
    """
    features: dict[str, pd.Series] = {}

    for asset in ASSETS:
        price = prices[asset]
        returns = price.pct_change()

        features[f"{asset}_price_to_ma200"] = price / price.rolling(200).mean()
        features[f"{asset}_ret_20d"] = price.pct_change(20)
        features[f"{asset}_ret_60d"] = price.pct_change(60)
        features[f"{asset}_vol_20d"] = returns.rolling(20).std()
        features[f"{asset}_max_drawdown"] = _rolling_drawdown(price, window=252)

    features["SPY_TLT_ratio_trend"] = (prices["SPY"] / prices["TLT"]).pct_change(60)
    features["SPY_GLD_ratio_trend"] = (prices["SPY"] / prices["GLD"]).pct_change(60)
    features["QQQ_IWM_ratio_trend"] = (prices["QQQ"] / prices["IWM"]).pct_change(60)
    features["BTC_SPY_ratio_trend"] = (prices["BTC"] / prices["SPY"]).pct_change(60)
    features["SPY_TLT_ret_60d_spread"] = prices["SPY"].pct_change(60) - prices["TLT"].pct_change(60)
    features["QQQ_SPY_relative_strength"] = prices["QQQ"].pct_change(60) - prices["SPY"].pct_change(60)
    features["GLD_SPY_corr_60d"] = prices["GLD"].pct_change().rolling(60).corr(prices["SPY"].pct_change())
    features["DBC_SPY_ret_60d_spread"] = prices["DBC"].pct_change(60) - prices["SPY"].pct_change(60)

    matrix = pd.DataFrame(features, index=prices.index)
    return matrix.replace([float("inf"), -float("inf")], pd.NA).dropna()


def latest_trend_signals(prices: pd.DataFrame) -> dict[str, bool]:
    """Return simple positive trend flags for the most recent day."""
    signals = compute_trend_signal_frame(prices)
    latest = signals.iloc[-1]
    return {asset: bool(latest[asset]) for asset in ASSETS}


def compute_trend_signal_frame(prices: pd.DataFrame) -> pd.DataFrame:
    """Return daily positive-trend flags used by the decision engine."""
    ma200 = prices[ASSETS].rolling(200).mean()
    ret20 = prices[ASSETS].pct_change(20)
    return (prices[ASSETS] > ma200) & (ret20 > 0)


def _rolling_drawdown(price: pd.Series, window: int) -> pd.Series:
    """Compute current drawdown versus the rolling peak over `window` days."""
    rolling_peak = price.rolling(window, min_periods=20).max()
    return price / rolling_peak - 1.0
