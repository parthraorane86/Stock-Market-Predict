from app.model import train_and_predict


def test_prediction_is_sane(prices):
    pred = train_and_predict(prices)
    assert pred is not None
    last = float(prices["Close"].iloc[-1])
    assert abs(pred.close / last - 1) < 0.15        # a one-day move, not a wild guess
    assert pred.high >= pred.close
    assert pred.band > 0
    assert pred.backtest.folds == 5
    assert 0 <= pred.backtest.direction_accuracy <= 100


def test_not_enough_data_gives_no_prediction(prices):
    assert train_and_predict(prices.iloc[:60]) is None
