"""Baseline forecasters the XGBoost model must beat.

All baselines respect the same information cut-off as the model: they only
see history up to the forecast origin.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def seasonal_naive(history: pd.Series, horizon: int, season: int = 7) -> np.ndarray:
    """Same weekday last week — what most managers do by hand."""
    last = history.values[-season:]
    return np.array([last[i % season] for i in range(horizon)], dtype=float)


def moving_average(history: pd.Series, horizon: int, window: int = 28) -> np.ndarray:
    return np.full(horizon, history.values[-window:].mean(), dtype=float)


def weekday_average(history: pd.Series, dates: pd.Series, horizon: int, weeks: int = 4) -> np.ndarray:
    """Average of the same weekday over the last `weeks` weeks."""
    hist = history.values[-7 * weeks:]
    hist_dow = pd.to_datetime(dates.values[-7 * weeks:]).dayofweek
    future_dow = [(hist_dow[-1] + i + 1) % 7 for i in range(horizon)]
    means = {d: hist[hist_dow == d].mean() for d in range(7)}
    return np.array([means[d] for d in future_dow], dtype=float)


BASELINES = {
    "Seasonal naive (last week)": lambda h, d, H: seasonal_naive(h, H),
    "Moving average (28d)": lambda h, d, H: moving_average(h, H),
    "Weekday average (4 wk)": lambda h, d, H: weekday_average(h, d, H),
}
