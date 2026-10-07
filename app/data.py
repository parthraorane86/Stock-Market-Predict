"""Market-data access (Yahoo Finance) with caching."""
from __future__ import annotations

import logging

import pandas as pd
import yfinance as yf

from .cache import TTLCache
from .config import HISTORY_TTL, MARKET_INDEX, TRAIN_PERIOD

logger = logging.getLogger(__name__)

_history_cache = TTLCache(HISTORY_TTL)

# Periods we can cut out of the 5-year download instead of fetching again.
_SLICE_OFFSETS = {
    "1mo": pd.DateOffset(months=1),
    "3mo": pd.DateOffset(months=3),
    "6mo": pd.DateOffset(months=6),
    "1y": pd.DateOffset(years=1),
    "2y": pd.DateOffset(years=2),
}


def _download(symbol: str, period: str) -> pd.DataFrame | None:
    """Download OHLCV data. Returns None on any failure (never raises)."""
    try:
        df = yf.Ticker(symbol).history(period=period)
    except Exception:
        logger.exception("Download failed for %s (%s)", symbol, period)
        return None
    if df is None or df.empty:
        logger.warning("No data returned for %s (%s)", symbol, period)
        return None

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    # Use plain calendar dates so stock data and index data line up.
    df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
    df = df[~df.index.duplicated(keep="last")].dropna(subset=["Close"])
    return df if not df.empty else None


def get_history(symbol: str, period: str) -> pd.DataFrame | None:
    """Cached daily OHLCV history for `symbol`."""
    return _history_cache.get_or_set(("history", symbol, period), lambda: _download(symbol, period))


def get_chart_data(symbol: str, period: str) -> pd.DataFrame | None:
    """Data for the chart. Short periods are sliced from the cached 5-year download."""
    if period in _SLICE_OFFSETS:
        full = get_history(symbol, TRAIN_PERIOD)
        if full is None:
            return None
        return full[full.index >= full.index[-1] - _SLICE_OFFSETS[period]]
    return get_history(symbol, period)


def get_market_close() -> pd.Series | None:
    """Nifty 50 closing prices (market-context feature). None if unavailable."""
    df = get_history(MARKET_INDEX, TRAIN_PERIOD)
    return df["Close"] if df is not None else None


def clear_cache() -> None:
    _history_cache.clear()
