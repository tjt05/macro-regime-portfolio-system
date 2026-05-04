from __future__ import annotations

"""Market data ingestion utilities.

The backend downloads adjusted daily prices from yfinance, maps external ticker
symbols into PRAIDS' internal asset names, and adds an explicit CASH series with
0% daily return. A CSV cache keeps local runs reproducible and avoids hitting
Yahoo Finance on every dashboard refresh.
"""

from pathlib import Path
from typing import Iterable

import pandas as pd
import yfinance as yf

from backend.assets import ASSET_ORDER, TICKER_MAP, TRADABLE_ASSETS

DEFAULT_TICKERS = list(TICKER_MAP.values())
DEFAULT_START = "2006-01-01"


def fetch_adjusted_close(
    tickers: Iterable[str] = DEFAULT_TICKERS,
    start: str = DEFAULT_START,
    end: str | None = None,
    cache_path: str | Path | None = "artifacts/prices.csv",
    use_cache: bool = True,
) -> pd.DataFrame:
    """Return a clean adjusted-close price matrix for the PRAIDS asset universe.

    Parameters mirror yfinance inputs, while `cache_path` and `use_cache` control
    whether the local `artifacts/prices.csv` file is reused. The returned columns
    always match `ASSET_ORDER`, including a synthetic `CASH` column.
    """
    cache = Path(cache_path) if cache_path else None
    if use_cache and cache and cache.exists():
        prices = pd.read_csv(cache, index_col=0, parse_dates=True)
        if set(TRADABLE_ASSETS).issubset(prices.columns):
            return _finalize_prices(prices)

    data = yf.download(
        list(tickers),
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        group_by="column",
    )

    if data.empty:
        raise RuntimeError("No price data returned by yfinance.")

    if isinstance(data.columns, pd.MultiIndex):
        if "Adj Close" in data.columns.get_level_values(0):
            prices = data["Adj Close"].copy()
        else:
            prices = data["Close"].copy()
    else:
        column = "Adj Close" if "Adj Close" in data.columns else "Close"
        prices = data[[column]].rename(columns={column: list(tickers)[0]})

    prices = prices.rename(columns={external: internal for internal, external in TICKER_MAP.items()})
    prices = _finalize_prices(prices)

    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        prices.to_csv(cache)

    return prices


def _finalize_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Sort, forward-fill, add CASH, and enforce the canonical column order."""
    prices = prices.sort_index()
    prices = prices.ffill().dropna(subset=TRADABLE_ASSETS)
    prices["CASH"] = 1.0
    return prices[ASSET_ORDER]
