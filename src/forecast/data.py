"""Historical sales data for ONE restaurant.

By default we generate a realistic synthetic history (seeded, so it is
reproducible) because real POS exports contain commercially sensitive data.
To use real data, export a CSV with at least `date, covers, revenue` and
point `load_sales()` at it — the optional columns are filled if missing.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

# England & Wales bank holidays (gov.uk)
UK_BANK_HOLIDAYS = pd.to_datetime([
    "2024-01-01", "2024-03-29", "2024-04-01", "2024-05-06", "2024-05-27", "2024-08-26", "2024-12-25", "2024-12-26",
    "2025-01-01", "2025-04-18", "2025-04-21", "2025-05-05", "2025-05-26", "2025-08-25", "2025-12-25", "2025-12-26",
    "2026-01-01", "2026-04-03", "2026-04-06", "2026-05-04", "2026-05-25", "2026-08-31", "2026-12-25", "2026-12-28",
    "2027-01-01", "2027-03-26", "2027-03-29", "2027-05-03", "2027-05-31", "2027-08-30", "2027-12-27", "2027-12-28",
])
SPECIAL_EVENT_MMDD = {"02-14", "12-24", "12-31", "12-20", "12-19"}   # Valentine's, Xmas Eve, NYE, Xmas parties
CLOSED_MMDD = {"12-25"}                                              # restaurant closed Christmas Day

# Monthly climate for the site (mean max temp °C, chance of rain)
MONTH_TEMP = {1: 7, 2: 8, 3: 11, 4: 14, 5: 17, 6: 20, 7: 22, 8: 22, 9: 19, 10: 15, 11: 10, 12: 8}
MONTH_RAIN_P = {1: .55, 2: .5, 3: .45, 4: .4, 5: .4, 6: .35, 7: .35, 8: .38, 9: .4, 10: .5, 11: .55, 12: .55}

REQUIRED = ["date", "covers", "revenue"]
OPTIONAL_DEFAULTS = {"promo": 0, "temp_c": np.nan, "rain": np.nan, "closed": 0}


def calendar_flags(dates: pd.Series) -> pd.DataFrame:
    d = pd.to_datetime(dates)
    mmdd = d.dt.strftime("%m-%d")
    return pd.DataFrame({
        "is_holiday": d.isin(UK_BANK_HOLIDAYS).astype(int).values,
        "is_special_event": mmdd.isin(SPECIAL_EVENT_MMDD).astype(int).values,
        "is_payday_week": (d.dt.day >= 25).astype(int).values,
    }, index=dates.index)


def climatology(dates: pd.Series) -> pd.DataFrame:
    """Expected weather when no real forecast is supplied."""
    m = pd.to_datetime(dates).dt.month
    return pd.DataFrame({"temp_c": m.map(MONTH_TEMP).astype(float).values,
                         "rain": m.map(MONTH_RAIN_P).values}, index=dates.index)


def generate_synthetic_sales(start: str = "2024-01-01", end: str = "2026-10-31",
                             seed: int = config.SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.Series(pd.date_range(start, end, freq="D"))
    n = len(dates)
    t = np.arange(n)
    dow = dates.dt.dayofweek.values
    month = dates.dt.month.values

    temp = np.array([MONTH_TEMP[m] for m in month]) + rng.normal(0, 3, n)
    rain = (rng.random(n) < np.array([MONTH_RAIN_P[m] for m in month])).astype(int)
    promo = np.zeros(n, dtype=int)
    promo_days = rng.choice(np.where(dow <= 2)[0], size=int(n * 0.06), replace=False)  # Mon–Wed promos
    promo[promo_days] = 1
    cal = calendar_flags(dates)

    dow_effect = np.array([0.72, 0.78, 0.85, 0.95, 1.30, 1.45, 1.10])[dow]
    season = 1 + 0.10 * np.sin(2 * np.pi * (dates.dt.dayofyear.values - 100) / 365.25)
    december = np.where(month == 12, 1.15, 1.0)
    january = np.where(month == 1, 0.88, 1.0)
    trend = 1 + 0.08 * t / 365.25                      # site is growing ~8% a year
    weather = 1 + 0.012 * (temp - 15) - 0.07 * rain
    events = (1 + 0.35 * cal["is_special_event"].values + 0.20 * cal["is_holiday"].values
              + 0.05 * cal["is_payday_week"].values + 0.18 * promo)

    covers = 120 * dow_effect * season * december * january * trend * weather * events
    covers = rng.poisson(np.clip(covers, 1, None) * rng.lognormal(0, 0.06, n)).astype(float)

    spend = config.AVG_SPEND_PER_COVER * (1 + 0.10 * (dow >= 4)) * (1 - 0.12 * promo) * (1 + 0.03 * t / 365.25)
    revenue = covers * spend * rng.normal(1, 0.04, n)

    closed = dates.dt.strftime("%m-%d").isin(CLOSED_MMDD).astype(int).values
    covers[closed == 1] = 0
    revenue[closed == 1] = 0

    return pd.DataFrame({
        "date": dates, "covers": covers.round(0), "revenue": revenue.round(2),
        "promo": promo, "temp_c": temp.round(1), "rain": rain, "closed": closed,
    })


def load_sales(path: Path | str = config.RAW_SALES) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        df = generate_synthetic_sales()
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    df = pd.read_csv(path, parse_dates=["date"])
    return validate(df)


def validate(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Sales data missing required columns: {missing}")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    for col, default in OPTIONAL_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default
    clim = climatology(df["date"])
    df["temp_c"] = df["temp_c"].fillna(clim["temp_c"])
    df["rain"] = df["rain"].fillna(clim["rain"])
    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    full = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    if len(full) != len(df):
        raise ValueError("Sales data has missing dates; fill gaps (closed days = 0 with closed=1).")
    if (df[["covers", "revenue"]] < 0).any().any():
        raise ValueError("Negative covers/revenue found.")
    return df


def future_frame(last_date: date, horizon: int, overrides: dict | None = None) -> pd.DataFrame:
    """Exogenous inputs for the forecast window (promos/weather may be overridden)."""
    dates = pd.Series(pd.date_range(pd.Timestamp(last_date) + pd.Timedelta(days=1), periods=horizon, freq="D"))
    fut = pd.DataFrame({"date": dates, "promo": 0})
    fut = pd.concat([fut, climatology(dates)], axis=1)
    fut["closed"] = dates.dt.strftime("%m-%d").isin(CLOSED_MMDD).astype(int)
    for d, vals in (overrides or {}).items():
        mask = fut["date"] == pd.Timestamp(d)
        for k, v in vals.items():
            if k in fut.columns:
                fut.loc[mask, k] = v
    return fut
