import numpy as np
import pandas as pd
import pytest

from app import create_app
from app import data as data_module
from app import services


def make_prices(n: int = 600, seed: int = 0, end: str = "2026-09-30") -> pd.DataFrame:
    """Synthetic daily OHLCV data (a random walk) for offline tests."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range(end=end, periods=n)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n)))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.004, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.004, n)))
    volume = rng.integers(1_000_000, 5_000_000, n)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low,
                         "Close": close, "Volume": volume}, index=index)


@pytest.fixture
def prices() -> pd.DataFrame:
    return make_prices()


@pytest.fixture
def client(monkeypatch):
    """Flask test client with Yahoo Finance replaced by synthetic data."""
    monkeypatch.setattr(data_module, "_download", lambda symbol, period: make_prices())
    data_module.clear_cache()
    services.clear_prediction_cache()
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()
