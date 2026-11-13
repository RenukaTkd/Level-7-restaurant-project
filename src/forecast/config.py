"""Central configuration: paths, seeds, business assumptions.

Every number that drives a business decision lives here so it can be
reviewed by the restaurant manager and changed in one place.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_SALES = DATA_DIR / "sales_history.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

SEED = 42                      # global seed -> reproducible data, model, backtest
RESTAURANT_NAME = "The Riverside Kitchen (pilot site)"

# ---- Forecasting -----------------------------------------------------------
TARGETS = ["covers", "revenue"]   # covers drives staffing/prep, revenue drives P&L
HORIZON = 14                      # days ahead (two-week rota cycle)
MIN_LAG = HORIZON                 # direct strategy: only use lags >= horizon (no leakage)

XGB_PARAMS = {
    "n_estimators": 600,
    "learning_rate": 0.03,
    "max_depth": 5,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "min_child_weight": 3,
    "reg_lambda": 1.0,
    "objective": "reg:squarederror",
    "random_state": SEED,
    "n_jobs": 1,                  # deterministic results across machines
}

# Rolling-origin backtest: train on everything before each cut-off,
# forecast the next HORIZON days, then move forward STEP days.
BACKTEST_FOLDS = 12
BACKTEST_STEP = 14

# ---- Operations (staffing & prep) -----------------------------------------
COVERS_PER_FOH_STAFF = 35         # covers one front-of-house staff member handles per day
COVERS_PER_KITCHEN_STAFF = 50
MIN_FOH, MIN_KITCHEN = 2, 2
SHIFT_HOURS = 8
SERVICE_LEVEL_Z = 0.84            # ~80% chance forecast + buffer covers demand

MENU_MIX = {                      # share of covers ordering each prep item
    "Burger patties": 0.28,
    "Chicken breasts": 0.22,
    "Fish fillets": 0.15,
    "Veg risotto base (portions)": 0.12,
    "Fresh dough (pizza bases)": 0.23,
}

# ---- Business-impact cost assumptions (GBP) — edit to match the site ------
HOURLY_WAGE = 12.60               # UK national living wage-ish
FOOD_COST_PER_COVER = 6.50        # ingredient cost of an over-prepped cover
AVG_SPEND_PER_COVER = 28.0
LOST_MARGIN_RATE = 0.65           # margin lost when an under-staffed cover walks out
UNDERSTAFF_WALKOUT_RATE = 0.30    # share of un-served covers actually lost

# ---- LLM comparison --------------------------------------------------------
# Model IDs are overridable through env vars (see .env.example).
LLM_PRICES_PER_MTOK = {           # (input, output) USD — update from provider pricing pages
    "claude": (4.00, 20.00),
    "openai": (0.25, 2.00),
    "gemini": (0.30, 2.50),
    "template": (0.0, 0.0),
}
