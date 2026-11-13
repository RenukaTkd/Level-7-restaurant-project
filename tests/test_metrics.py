import numpy as np

from forecast.baselines import seasonal_naive
from forecast.metrics import bias, mae, wape


def test_mae_and_wape():
    y, yhat = [100, 200, 0], [110, 190, 0]
    assert mae(y, yhat) == 20 / 3
    assert wape(y, yhat) == 20 / 300          # robust to the zero (closed) day


def test_bias_sign():
    assert bias([10, 10], [12, 12]) == 2       # positive = over-forecast


def test_seasonal_naive_repeats_last_week():
    import pandas as pd
    h = pd.Series(np.arange(14, dtype=float))
    assert list(seasonal_naive(h, 9)) == [7, 8, 9, 10, 11, 12, 13, 7, 8]
