"""Application-level operations that combine data, model and caching."""
from __future__ import annotations

import logging

import pandas as pd

from .cache import TTLCache
from .config import PREDICTION_TTL, TRAIN_PERIOD
from .data import get_history, get_market_close
from .model import Prediction, train_and_predict

logger = logging.getLogger(__name__)

_prediction_cache = TTLCache(PREDICTION_TTL)

TRADING_DAYS_PER_YEAR = 252


def get_prediction(symbol: str) -> Prediction | None:
    """Cached next-day forecast. The model is retrained at most once per TTL."""

    def build() -> Prediction | None:
        data = get_history(symbol, TRAIN_PERIOD)
        if data is None:
            return None
        try:
            return train_and_predict(data, get_market_close())
        except Exception:
            logger.exception("Prediction failed for %s", symbol)
            return None

    return _prediction_cache.get_or_set(("prediction", symbol), build)


def build_snapshot(history: pd.DataFrame) -> dict:
    """Headline numbers for the stats boxes, computed from price history."""
    last = history.iloc[-1]
    year = history.tail(TRADING_DAYS_PER_YEAR)
    return {
        "current_price": float(last["Close"]),
        "day_high": float(last["High"]),
        "day_low": float(last["Low"]),
        "volume": int(last["Volume"]),
        "week52_high": float(year["High"].max()),
        "week52_low": float(year["Low"].min()),
        "as_of": history.index[-1].date(),
    }


def clear_prediction_cache() -> None:
    _prediction_cache.clear()
