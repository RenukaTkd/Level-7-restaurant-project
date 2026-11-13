"""Turn a covers forecast into staffing and food-prep plans, and cost them."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import config


def staff_needed(covers: float) -> tuple[int, int]:
    if covers <= 0:
        return 0, 0
    foh = max(config.MIN_FOH, math.ceil(covers / config.COVERS_PER_FOH_STAFF))
    kitchen = max(config.MIN_KITCHEN, math.ceil(covers / config.COVERS_PER_KITCHEN_STAFF))
    return foh, kitchen


def plan(forecast: pd.DataFrame, resid_std: float) -> pd.DataFrame:
    """Staffing + prep plan with a safety buffer sized to the model's own error."""
    p = forecast[["date", "covers"]].copy()
    p["planned_covers"] = np.where(p["covers"] > 0, np.ceil(p["covers"] + config.SERVICE_LEVEL_Z * resid_std), 0)
    staff = p["planned_covers"].apply(staff_needed)
    p["foh_staff"] = [s[0] for s in staff]
    p["kitchen_staff"] = [s[1] for s in staff]
    p["labour_hours"] = (p["foh_staff"] + p["kitchen_staff"]) * config.SHIFT_HOURS
    for item, share in config.MENU_MIX.items():
        p[f"prep: {item}"] = np.ceil(p["planned_covers"] * share).astype(int)
    return p


def daily_cost(planned_covers: np.ndarray, actual: np.ndarray) -> pd.DataFrame:
    """Cost of a plan vs what actually happened (no buffer — compare raw forecasts fairly)."""
    rows = []
    for plan_c, act in zip(planned_covers, actual):
        p_foh, p_kit = staff_needed(plan_c)
        a_foh, a_kit = staff_needed(act)
        over_hours = (max(p_foh - a_foh, 0) + max(p_kit - a_kit, 0)) * config.SHIFT_HOURS
        capacity = p_foh * config.COVERS_PER_FOH_STAFF if plan_c > 0 else 0
        unserved = max(act - max(capacity, plan_c), 0)
        rows.append({
            "overstaff_cost": over_hours * config.HOURLY_WAGE,
            "waste_cost": max(plan_c - act, 0) * config.FOOD_COST_PER_COVER,
            "lost_margin": unserved * config.UNDERSTAFF_WALKOUT_RATE * config.AVG_SPEND_PER_COVER
                           * config.LOST_MARGIN_RATE,
        })
    df = pd.DataFrame(rows)
    df["total_cost"] = df.sum(axis=1)
    return df


def business_impact(bt_covers: pd.DataFrame) -> pd.DataFrame:
    """Annualised cost of each method's plans over the backtest period."""
    out = []
    n_days = bt_covers["date"].nunique()
    for method, g in bt_covers.groupby("method"):
        c = daily_cost(g["forecast"].round().values, g["actual"].values).sum()
        out.append({"method": method, **{k: round(v * 365 / n_days, 0) for k, v in c.items()}})
    res = pd.DataFrame(out).set_index("method").sort_values("total_cost")
    base = res.loc["Seasonal naive (last week)", "total_cost"]
    res["annual_saving_vs_naive_gbp"] = (base - res["total_cost"]).round(0)
    return res
