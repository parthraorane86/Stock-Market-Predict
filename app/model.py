"""Model training, walk-forward backtesting and prediction."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from .config import CV_SPLITS, RIDGE_ALPHA, XGB_PARAMS
from .features import Dataset, prepare_dataset


class EnsembleRegressor:
    """Average of XGBoost (non-linear) and Ridge (linear).

    Averaging two very different models lowers variance, which matters a lot
    on noisy financial data.
    """

    def __init__(self) -> None:
        self.members = [
            XGBRegressor(**XGB_PARAMS),
            make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA)),
        ]

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "EnsembleRegressor":
        for member in self.members:
            member.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.mean([member.predict(X) for member in self.members], axis=0)


@dataclass(frozen=True)
class Backtest:
    """Walk-forward results for the next-day close return (all in percent)."""
    folds: int
    model_mae: float            # mean absolute error of the model
    naive_mae: float            # error of "tomorrow = today" (return of 0%)
    direction_accuracy: float   # % of days the up/down call was right
    error_band: float           # typical size of a miss (1 std of residuals)

    @property
    def beats_naive(self) -> bool:
        return self.model_mae < self.naive_mae


@dataclass(frozen=True)
class Prediction:
    close: float
    high: float
    last_close: float
    band: float                 # +/- price range around `close`
    train_rows: int
    backtest: Backtest


def backtest(dataset: Dataset) -> Backtest:
    """Train on the past, predict the next block, move forward, repeat.

    `gap=1` keeps a one-day buffer between train and test so a training target
    can never overlap with the first test day.
    """
    X, y = dataset.X, dataset.y_close
    splitter = TimeSeriesSplit(n_splits=CV_SPLITS, gap=1)

    actual, predicted = [], []
    for train_idx, test_idx in splitter.split(X):
        model = EnsembleRegressor().fit(X.iloc[train_idx], y.iloc[train_idx])
        predicted.append(model.predict(X.iloc[test_idx]))
        actual.append(y.iloc[test_idx].to_numpy())

    a, p = np.concatenate(actual), np.concatenate(predicted)
    return Backtest(
        folds=CV_SPLITS,
        model_mae=float(np.mean(np.abs(a - p)) * 100),
        naive_mae=float(np.mean(np.abs(a)) * 100),
        direction_accuracy=float(np.mean(np.sign(a) == np.sign(p)) * 100),
        error_band=float(np.std(a - p) * 100),
    )


def train_and_predict(data: pd.DataFrame, market_close: pd.Series | None = None) -> Prediction | None:
    """Forecast the next trading day's close and high. None if there isn't enough data."""
    dataset = prepare_dataset(data, market_close)
    if dataset is None:
        return None

    results = backtest(dataset)
    close_model = EnsembleRegressor().fit(dataset.X, dataset.y_close)
    high_model = EnsembleRegressor().fit(dataset.X, dataset.y_high)

    last = dataset.last_close
    pred_close = last * (1 + float(close_model.predict(dataset.latest)[0]))
    pred_high = last * (1 + float(high_model.predict(dataset.latest)[0]))
    pred_high = max(pred_high, pred_close)      # the day's high can't be below its close

    return Prediction(
        close=round(pred_close, 2),
        high=round(pred_high, 2),
        last_close=round(last, 2),
        band=round(pred_close * results.error_band / 100, 2),
        train_rows=len(dataset.X),
        backtest=results,
    )
