# Model report — The Riverside Kitchen (pilot site)

Training data: 2024-01-01 → 2026-10-31 (1035 days), data hash `1b72a17a3c519dc5`, seed 42, 12 rolling-origin folds × 14-day horizon.

## Baseline comparison — covers

| Method | MAE | WAPE | RMSE | Bias | WAPE improvement vs naive |
|---|---|---|---|---|---|
| XGBoost | 16.33 | 9.8% | 20.83 | -3.55 | +29.2% |
| Weekday average (4 wk) | 18.54 | 11.2% | 22.99 | -0.06 | +19.6% |
| Seasonal naive (last week) | 23.06 | 13.9% | 28.19 | -1.38 | +0.0% |
| Moving average (28d) | 38.30 | 23.1% | 45.80 | -0.06 | -66.1% |

## Baseline comparison — revenue

| Method | MAE | WAPE | RMSE | Bias | WAPE improvement vs naive |
|---|---|---|---|---|---|
| XGBoost | 568.47 | 10.8% | 726.19 | -106.62 | +26.1% |
| Weekday average (4 wk) | 591.48 | 11.2% | 759.64 | -9.93 | +23.2% |
| Seasonal naive (last week) | 769.72 | 14.6% | 970.96 | -44.21 | +0.0% |
| Moving average (28d) | 1489.40 | 28.2% | 1732.47 | -9.93 | -93.5% |

## Business impact (annualised, GBP, covers forecast → staffing & prep)

                            overstaff_cost  waste_cost  lost_margin  total_cost  annual_saving_vs_naive_gbp
method                                                                                                     
XGBoost                            13140.0     15082.0       7011.0     35233.0                     20290.0
Weekday average (4 wk)             16425.0     21903.0       7983.0     46312.0                      9211.0
Seasonal naive (last week)         18834.0     25716.0      10973.0     55523.0                         0.0
Moving average (28d)               36135.0     45402.0      26904.0    108441.0                    -52918.0

## Error parity across segments (XGBoost, covers)

| Dimension | Group | Days | MAE | WAPE |
|---|---|---|---|---|
| segment | Friday | 24 | 16.11 | 7.6% |
| segment | Weekday (Mon-Thu) | 96 | 14.30 | 10.7% |
| segment | Weekend (Sat-Sun) | 48 | 20.50 | 9.9% |
| season | Spring/Autumn | 76 | 15.30 | 9.6% |
| season | Summer | 92 | 17.17 | 10.0% |
