"""Feature engineering.

We use a *direct* multi-step strategy: every lag is >= HORIZON days old, so a
single model can predict any day in the next two weeks using only information
that is genuinely known at forecast time. This removes look-ahead leakage.
"""
from __future__ import annotations

import pandas as pd

from . import config
from .data import calendar_flags

LAGS = [config.MIN_LAG, config.MIN_LAG + 7, 28, 35, 364]


def build_features(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """df must be sorted by date and contain target + exogenous columns."""
    out = pd.DataFrame(index=df.index)
    d = df["date"]
    out["dow"] = d.dt.dayofweek
    out["month"] = d.dt.month
    out["dayofyear"] = d.dt.dayofyear
    out["weekofyear"] = d.dt.isocalendar().week.astype(int).values
    out["is_weekend"] = (out["dow"] >= 5).astype(int)
    out = out.join(calendar_flags(d))
    for col in ["promo", "temp_c", "rain", "closed"]:
        out[col] = df[col].values

    y = df[target]
    for lag in LAGS:
        out[f"lag_{lag}"] = y.shift(lag)
    base = y.shift(config.MIN_LAG)
    out["roll_mean_7"] = base.rolling(7).mean()
    out["roll_mean_28"] = base.rolling(28).mean()
    out["roll_std_28"] = base.rolling(28).std()
    # same-weekday average of the last 4 eligible weeks (MIN_LAG is a multiple of 7)
    out["dow_mean_4w"] = sum(y.shift(config.MIN_LAG + k) for k in (0, 7, 14, 21)) / 4
    return out
