"""Forecast accuracy metrics."""
from __future__ import annotations

import numpy as np


def mae(y, yhat) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(np.mean(np.abs(y - yhat)))


def wape(y, yhat) -> float:
    """Weighted Absolute Percentage Error = sum|e| / sum|y|.

    Preferred over MAPE for restaurants: robust to low/zero days (closures)
    and directly interpretable as '% of total volume mis-forecast'.
    """
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    denom = np.sum(np.abs(y))
    return float(np.sum(np.abs(y - yhat)) / denom) if denom else float("nan")


def rmse(y, yhat) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def bias(y, yhat) -> float:
    """Mean (forecast - actual). Positive = over-forecasting."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(np.mean(yhat - y))


def all_metrics(y, yhat) -> dict:
    return {"MAE": mae(y, yhat), "WAPE": wape(y, yhat), "RMSE": rmse(y, yhat), "Bias": bias(y, yhat)}
