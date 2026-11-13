"""XGBoost training, persistence and inference."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import xgboost as xgb

from . import __version__, config
from .data import future_frame
from .features import build_features


def fit(df: pd.DataFrame, target: str, params: dict | None = None) -> tuple[xgb.Booster, list[str]]:
    X = build_features(df, target)
    mask = X.notna().all(axis=1) & (df["closed"] == 0)
    dtrain = xgb.DMatrix(X[mask], label=df.loc[mask, target])
    model = xgb.train(params or config.XGB_PARAMS, dtrain, num_boost_round=config.XGB_ROUNDS)
    return model, list(X.columns)


def predict_window(model: xgb.Booster, history: pd.DataFrame, future: pd.DataFrame,
                   target: str) -> np.ndarray:
    """Predict rows of `future` (len <= HORIZON) given `history` up to the origin."""
    if len(future) > config.HORIZON:
        raise ValueError(f"Horizon {len(future)} exceeds trained maximum {config.HORIZON}.")
    combo = pd.concat([history, future.assign(**{target: np.nan})], ignore_index=True)
    X = build_features(combo, target).iloc[-len(future):]
    pred = np.clip(model.predict(xgb.DMatrix(X)).astype("float64"), 0, None)
    pred[future["closed"].values == 1] = 0
    return pred


# ---- Persistence ------------------------------------------------------------
def save(models: dict, meta: dict) -> None:
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for target, m in models.items():
        m.save_model(config.MODEL_DIR / f"xgb_{target}.json")
    meta = {**meta, "package_version": __version__, "xgboost_version": xgb.__version__,
            "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (config.MODEL_DIR / "model_meta.json").write_text(json.dumps(meta, indent=2, default=str))


def load() -> tuple[dict, dict]:
    meta = json.loads((config.MODEL_DIR / "model_meta.json").read_text())
    models = {}
    for target in config.TARGETS:
        m = xgb.Booster()
        m.load_model(config.MODEL_DIR / f"xgb_{target}.json")
        models[target] = m
    return models, meta


class Forecaster:
    """Loaded once per process (API cold start) and reused."""

    def __init__(self, history: pd.DataFrame):
        self.models, self.meta = load()
        self.history = history

    def forecast(self, horizon: int = config.HORIZON, overrides: dict | None = None) -> pd.DataFrame:
        fut = future_frame(self.history["date"].max(), horizon, overrides)
        out = fut[["date", "promo", "temp_c", "rain", "closed"]].copy()
        for target in config.TARGETS:
            p = predict_window(self.models[target], self.history, fut, target)
            q = self.meta["residual_quantiles"][target]
            out[target] = p.round(1)
            out[f"{target}_low"] = np.where(p > 0, np.clip(p + q["q10"], 0, None), 0).round(1)
            out[f"{target}_high"] = np.where(p > 0, p + q["q90"], 0).round(1)
        return out
