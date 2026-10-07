import numpy as np

from app.features import (engineer_features, feature_columns, prepare_dataset,
                          trim_to_last_discontinuity)


def test_latest_row_is_the_last_day(prices):
    """Regression test: the forecast must start from the newest day, not the one before."""
    ds = prepare_dataset(prices)
    assert ds is not None
    assert ds.latest.index[0] == prices.index[-1]
    assert ds.last_date == prices.index[-1]
    assert ds.last_close == float(prices["Close"].iloc[-1])
    # ...and that newest day is correctly NOT part of the training rows.
    assert prices.index[-1] not in ds.X.index


def test_no_nans_in_training_data(prices):
    ds = prepare_dataset(prices)
    assert not ds.X.isna().to_numpy().any()
    assert not ds.y_close.isna().any()


def test_features_do_not_look_into_the_future(prices):
    """Changing tomorrow's price must not change today's features."""
    cut = 400
    base = engineer_features(prices.iloc[:cut + 1])
    altered = prices.copy()
    altered.iloc[cut + 1:, altered.columns.get_loc("Close")] *= 5
    changed = engineer_features(altered)
    cols = feature_columns(base)
    np.testing.assert_allclose(base[cols].iloc[cut].to_numpy(),
                               changed[cols].iloc[cut].to_numpy())


def test_too_little_data_returns_none(prices):
    assert prepare_dataset(prices.iloc[:60]) is None


def test_corporate_action_jump_is_trimmed(prices):
    data = prices.copy()
    data.iloc[300:, :4] *= 0.6                      # -40% overnight (e.g. a demerger)
    trimmed = trim_to_last_discontinuity(data)
    assert trimmed.index[0] == data.index[300]
    assert len(trim_to_last_discontinuity(prices)) == len(prices)   # no jump -> unchanged
