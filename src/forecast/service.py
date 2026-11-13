"""Shared service layer used by BOTH the FastAPI API and the Streamlit UI,
so the two front-ends can never disagree."""
from __future__ import annotations

import json
from functools import lru_cache

import pandas as pd

from . import config
from .data import load_sales
from .model import Forecaster
from .operations import plan


@lru_cache(maxsize=1)
def forecaster() -> Forecaster:
    return Forecaster(load_sales())


@lru_cache(maxsize=1)
def metrics() -> dict:
    return json.loads((config.REPORT_DIR / "metrics.json").read_text())


def covers_wape() -> float:
    return metrics()["backtest"]["covers"]["XGBoost"]["WAPE"]


def forecast_and_plan(horizon: int = config.HORIZON, overrides: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    fc = forecaster().forecast(horizon, overrides)
    std = forecaster().meta["residual_quantiles"]["covers"]["std"]
    return fc, plan(fc, std)
