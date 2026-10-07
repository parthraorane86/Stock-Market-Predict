"""Feature engineering and dataset preparation.

Design rules:
  * Every feature is *stationary* (returns, ratios, oscillators) so the model
    isn't thrown off when a stock trades at a new all-time high.
  * Features on row t only use data up to and including day t (no look-ahead).
  * The targets are next-day *returns*; they are converted to prices later.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import MAX_ABS_DAILY_RETURN, MIN_TRAIN_ROWS

BASE_FEATURES = [
    "ret_lag0", "ret_lag1", "ret_lag2", "ret_lag3", "ret_lag4",
    "close_ma5_ratio", "close_ma10_ratio", "close_ma20_ratio",
    "volatility_5", "volatility_20",
    "volume_ratio", "gap", "range_pct",
    "rsi_14", "macd_hist", "bb_pos", "day_of_week",
]
MARKET_FEATURES = ["mkt_ret_1", "mkt_ret_5"]


@dataclass(frozen=True)
class Dataset:
    X: pd.DataFrame            # training features
    y_close: pd.Series         # next-day close return
    y_high: pd.Series          # next-day high, as a return vs today's close
    latest: pd.DataFrame       # features for the most recent day (used for the forecast)
    last_close: float
    last_date: pd.Timestamp


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder smoothing), 0-100."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    return 100 - 100 / (1 + gain / loss)


def trim_to_last_discontinuity(data: pd.DataFrame) -> pd.DataFrame:
    """Drop everything before the last abnormal one-day jump.

    NSE circuit limits make genuine moves above ~20% in a day extremely rare, so
    a bigger jump is almost always a corporate action (demerger, split, bonus)
    that makes older prices not comparable with newer ones.
    """
    jumps = data["Close"].pct_change().abs() > MAX_ABS_DAILY_RETURN
    if jumps.any():
        return data.loc[jumps[jumps].index[-1]:]
    return data


def engineer_features(data: pd.DataFrame, market_close: pd.Series | None = None) -> pd.DataFrame:
    """Return `data` with all feature columns added."""
    df = data.copy()
    close = df["Close"]
    ret = close.pct_change()

    for k in range(5):
        df[f"ret_lag{k}"] = ret.shift(k)          # ret_lag0 = today's return

    for w in (5, 10, 20):
        df[f"close_ma{w}_ratio"] = close / close.rolling(w).mean() - 1

    df["volatility_5"] = ret.rolling(5).std()
    df["volatility_20"] = ret.rolling(20).std()
    df["volume_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()
    df["gap"] = df["Open"] / close.shift(1) - 1
    df["range_pct"] = (df["High"] - df["Low"]) / close
    df["rsi_14"] = rsi(close) / 100

    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    signal = macd.ewm(span=9, adjust=False).mean()
    df["macd_hist"] = (macd - signal) / close

    ma20 = close.rolling(20).mean()
    df["bb_pos"] = (close - ma20) / (2 * close.rolling(20).std())

    df["day_of_week"] = df.index.dayofweek

    if market_close is not None:
        market = market_close.reindex(df.index).ffill()
        df["mkt_ret_1"] = market.pct_change()
        df["mkt_ret_5"] = market.pct_change(5)
        # If the index barely overlaps with this stock, don't let it wipe out the data.
        if df["mkt_ret_1"].notna().mean() < 0.8:
            df = df.drop(columns=MARKET_FEATURES)

    return df.replace([np.inf, -np.inf], np.nan)


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in BASE_FEATURES + MARKET_FEATURES if c in df.columns]


def prepare_dataset(data: pd.DataFrame, market_close: pd.Series | None = None) -> Dataset | None:
    """Build training data plus the feature row to forecast from.

    IMPORTANT: the newest day has no "next day" target yet, so it gets dropped
    from the training set. We therefore grab its features *first* - that row is
    exactly the one we want to predict from.
    """
    data = trim_to_last_discontinuity(data)
    df = engineer_features(data, market_close)
    cols = feature_columns(df)

    latest = df[cols].iloc[[-1]]
    if latest.isna().to_numpy().any():
        return None

    close = df["Close"]
    df["target_close"] = close.shift(-1) / close - 1
    df["target_high"] = df["High"].shift(-1) / close - 1
    train = df.dropna(subset=cols + ["target_close", "target_high"])
    if len(train) < MIN_TRAIN_ROWS:
        return None

    return Dataset(
        X=train[cols],
        y_close=train["target_close"],
        y_high=train["target_high"],
        latest=latest,
        last_close=float(close.iloc[-1]),
        last_date=df.index[-1],
    )
