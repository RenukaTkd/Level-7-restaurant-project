import pandas as pd

from forecast import config
from forecast.data import generate_synthetic_sales, validate
from forecast.features import build_features
from forecast.operations import plan, staff_needed


def test_synthetic_data_is_reproducible():
    a = generate_synthetic_sales(end="2024-03-31")
    b = generate_synthetic_sales(end="2024-03-31")
    pd.testing.assert_frame_equal(a, b)


def test_no_leakage_lags_respect_horizon():
    df = validate(generate_synthetic_sales(end="2024-06-30"))
    X = build_features(df, "covers")
    lag_cols = [c for c in X.columns if c.startswith("lag_")]
    assert min(int(c.split("_")[1]) for c in lag_cols) >= config.HORIZON


def test_staffing_minimums_and_closed_days():
    assert staff_needed(0) == (0, 0)
    assert staff_needed(10) == (config.MIN_FOH, config.MIN_KITCHEN)
    fc = pd.DataFrame({"date": pd.date_range("2026-11-01", periods=2), "covers": [0.0, 200.0]})
    p = plan(fc, resid_std=10)
    assert p.loc[0, "labour_hours"] == 0 and p.loc[1, "planned_covers"] >= 200


def test_briefing_totals_add_up_and_unsupported_numbers_flagged():
    from forecast.llm.evaluate import numeric_faithfulness
    facts = {"total_covers": 300, "daily": [{"covers": 100, "revenue_gbp": 6622}, {"covers": 200, "revenue_gbp": 6332}]}
    assert numeric_faithfulness("We expect 300 covers; Friday £6,332.", facts)[0] == 1.0
    score, bad = numeric_faithfulness("Friday brings £6,632 revenue.", facts)   # typo near 6,622
    assert score == 0.0 and bad
    assert numeric_faithfulness("About £6.6k on Friday.", facts)[0] == 1.0       # k-rounding allowed
