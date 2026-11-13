"""Rolling-origin backtesting: XGBoost vs baselines on identical folds."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .baselines import BASELINES
from .metrics import all_metrics
from .model import fit, predict_window

MODEL_NAME = "XGBoost"


def origins(df: pd.DataFrame, folds: int = config.BACKTEST_FOLDS,
            step: int = config.BACKTEST_STEP, horizon: int = config.HORIZON) -> list[pd.Timestamp]:
    last = df["date"].max()
    return sorted(last - pd.Timedelta(days=horizon + k * step) for k in range(folds))


def run_backtest(df: pd.DataFrame, target: str, folds: int = config.BACKTEST_FOLDS) -> pd.DataFrame:
    """Return one row per (fold, date, method) with actual and forecast."""
    rows = []
    for fold, origin in enumerate(origins(df, folds)):
        hist = df[df["date"] <= origin].reset_index(drop=True)
        fut = df[(df["date"] > origin) & (df["date"] <= origin + pd.Timedelta(days=config.HORIZON))]
        fut = fut.reset_index(drop=True)
        closed = fut["closed"].values == 1

        model, _ = fit(hist, target)
        preds = {MODEL_NAME: predict_window(model, hist, fut.drop(columns=[target]), target)}
        for name, fn in BASELINES.items():
            p = fn(hist[target], hist["date"], len(fut))
            p[closed] = 0                       # managers know closure days too
            preds[name] = p

        for name, p in preds.items():
            rows.append(pd.DataFrame({
                "fold": fold, "origin": origin, "date": fut["date"], "horizon_day": np.arange(1, len(fut) + 1),
                "method": name, "actual": fut[target].values, "forecast": p,
            }))
    return pd.concat(rows, ignore_index=True)


def summarise(bt: pd.DataFrame) -> pd.DataFrame:
    res = pd.DataFrame({m: all_metrics(g["actual"], g["forecast"]) for m, g in bt.groupby("method")}).T
    res = res.sort_values("WAPE")
    res.index.name = "method"
    base = res.loc["Seasonal naive (last week)", "WAPE"]
    res["WAPE_improvement_vs_naive_%"] = (100 * (base - res["WAPE"]) / base).round(1)
    return res


def segment_errors(bt: pd.DataFrame) -> pd.DataFrame:
    """Error parity across operating segments (used in the fairness review)."""
    d = bt[bt["method"] == MODEL_NAME].copy()
    dow = pd.to_datetime(d["date"]).dt.dayofweek
    month = pd.to_datetime(d["date"]).dt.month
    d["segment"] = np.select(
        [dow >= 5, dow == 4], ["Weekend (Sat-Sun)", "Friday"], default="Weekday (Mon-Thu)")
    d["season"] = np.select([month.isin([12, 1, 2]), month.isin([6, 7, 8])], ["Winter", "Summer"], "Spring/Autumn")
    out = []
    for col in ["segment", "season"]:
        for k, g in d.groupby(col):
            m = all_metrics(g["actual"], g["forecast"])
            out.append({"dimension": col, "group": k, "n_days": len(g), **m})
    return pd.DataFrame(out)


def residual_quantiles(bt: pd.DataFrame) -> dict:
    r = bt[(bt["method"] == MODEL_NAME) & (bt["actual"] > 0)]
    e = r["actual"] - r["forecast"]
    return {"q10": float(np.quantile(e, 0.10)), "q90": float(np.quantile(e, 0.90)), "std": float(e.std())}
