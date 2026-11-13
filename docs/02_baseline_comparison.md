# 2. Baseline comparison

A model only adds value if it beats the simple methods a manager could use instead. Every method was evaluated
on **identical folds** with **identical information** (history up to the forecast origin; known closure days).

## Method — rolling-origin backtesting
- 12 forecast origins, 14 days apart. The last origin is 14 days before the end of the data.
- At each origin, XGBoost is **retrained from scratch** on data up to the origin, then forecasts the next 14 days.
- That gives 168 forecast-days per method per target (`reports/backtest_covers.csv`).

## Metrics
- **MAE**: average absolute error in covers (or £). Easy for a manager to understand ("we're off by about 16 covers a day").
- **WAPE** = Σ|actual − forecast| / Σ actual. The share of total volume that is mis-forecast. Used instead of MAPE
  because MAPE blows up on low or zero days (e.g. Christmas closure).
- RMSE (penalises big misses) and Bias (+ = over-forecast) are also reported.

## Results — covers

| Method | MAE | WAPE | RMSE | Bias | WAPE vs naive |
|---|---|---|---|---|---|
| **XGBoost** | **16.33** | **9.8%** | **20.83** | −3.55 | **+29.2%** |
| Weekday average (4 wk) | 18.54 | 11.2% | 22.99 | −0.06 | +19.6% |
| Seasonal naive (last week) | 23.06 | 13.9% | 28.19 | −1.38 | 0 |
| Moving average (28 d) | 38.30 | 23.1% | 45.80 | −0.06 | −66.1% |

## Results — revenue

| Method | MAE (£) | WAPE | WAPE vs naive |
|---|---|---|---|
| **XGBoost** | **568** | **10.8%** | **+26.1%** |
| Weekday average (4 wk) | 591 | 11.2% | +23.2% |
| Seasonal naive (last week) | 770 | 14.6% | 0 |
| Moving average (28 d) | 1,489 | 28.2% | −93.5% |

## Interpretation
- XGBoost wins on both targets. It gains most on days that baselines cannot anticipate: promotions, bank holidays,
  paydays and weather swings.
- For revenue, the gap over the weekday average is small (10.8% vs 11.2%). Spend per cover is stable, so most of the
  value is in the **covers** forecast that drives staffing and prep.
- XGBoost slightly **under-forecasts** (bias −3.6 covers/day). The operations layer adds a safety buffer, so plans are
  not systematically short.
- The 28-day moving average ignores day-of-week patterns. That is why it does so badly, and why a weekly pattern matters.

Reproduce with `python scripts/train.py`. The numbers come from `reports/metrics.json`.
