"""FastAPI service — deployed to Vercel as a Python serverless function.

Local:  uvicorn api.index:app --reload      -> http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import logging
import os
import sys
import time
import uuid
from datetime import date
from pathlib import Path
from typing import Dict, List, Literal, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from forecast import __version__, config, service  # noqa: E402
from forecast.llm import build_facts, generate, score  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("forecast-api")

app = FastAPI(
    title="Restaurant Sales Forecasting API",
    version=__version__,
    description="XGBoost demand forecast → staffing & food-prep plan for one pilot restaurant.",
)
app.add_middleware(CORSMiddleware, allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
                   allow_methods=["GET", "POST"], allow_headers=["*"])

class VercelPathFix:
    """Restore the original request path on Vercel.

    vercel.json rewrites every URL to this function as /api/index?__path=<original>,
    so FastAPI would otherwise see "/api/index" for every request. Normal paths
    (local uvicorn, tests) pass through unchanged.
    """

    PREFIX = "/api/index"

    def __init__(self, asgi_app):
        self.app = asgi_app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["path"].startswith(self.PREFIX):
            from urllib.parse import parse_qsl, urlencode
            params = parse_qsl(scope.get("query_string", b"").decode(), keep_blank_values=True)
            original = next((v for k, v in params if k == "__path"), None)
            rest = scope["path"][len(self.PREFIX):]
            path = "/" + (original if original is not None else rest).lstrip("/")
            scope = {**scope, "path": path, "raw_path": path.encode(),
                     "query_string": urlencode([(k, v) for k, v in params if k != "__path"]).encode()}
        await self.app(scope, receive, send)


API_KEY = os.getenv("API_KEY")  # optional: set on Vercel to require an x-api-key header


@app.middleware("http")
async def audit_log(request: Request, call_next):
    """Request log = deployment evidence (visible in Vercel → Logs). No payloads logged (privacy)."""
    if API_KEY and request.url.path not in ("/", "/health", "/docs", "/openapi.json") \
            and request.headers.get("x-api-key") != API_KEY:
        from fastapi.responses import JSONResponse
        return JSONResponse({"detail": "invalid or missing x-api-key"}, status_code=401)
    rid, t0 = uuid.uuid4().hex[:8], time.perf_counter()
    response = await call_next(request)
    log.info("rid=%s %s %s -> %s %.0fms", rid, request.method, request.url.path,
             response.status_code, 1000 * (time.perf_counter() - t0))
    response.headers["x-request-id"] = rid
    return response


# Added last = runs first, so logging and the API-key check see the restored path.
app.add_middleware(VercelPathFix)


class DayOverride(BaseModel):
    promo: Optional[int] = Field(None, ge=0, le=1)
    temp_c: Optional[float] = Field(None, ge=-15, le=40)
    rain: Optional[float] = Field(None, ge=0, le=1)
    closed: Optional[int] = Field(None, ge=0, le=1)


class ForecastRequest(BaseModel):
    horizon: int = Field(config.HORIZON, ge=1, le=config.HORIZON)
    overrides: Dict[date, DayOverride] = Field(default_factory=dict,
                                               description="Planned promos / weather forecast per date")


class ExplainRequest(ForecastRequest):
    provider: Literal["claude", "openai", "gemini", "template"] = "template"


def _overrides(req: ForecastRequest) -> dict:
    return {str(d): o.model_dump(exclude_none=True) for d, o in req.overrides.items()}


def _records(df) -> List[dict]:
    out = df.copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    return out.to_dict(orient="records")


@app.get("/")
def root():
    return {"service": "restaurant-forecast-api", "version": __version__, "docs": "/docs"}


@app.get("/health")
def health():
    meta = service.forecaster().meta
    return {"status": "ok", "model_trained_at": meta["trained_at_utc"], "data_until": meta["train_end"],
            "data_sha256": meta["data_sha256"], "version": __version__}


@app.get("/metrics")
def metrics():
    """Backtest results: XGBoost vs baselines (MAE, WAPE), business impact, segment parity."""
    m = service.metrics()
    return {k: m[k] for k in ("backtest", "business_impact", "segment_errors")}


@app.get("/model-card")
def model_card():
    meta = service.forecaster().meta
    return {"model": "XGBoost regressor (direct multi-horizon)", "targets": meta["targets"],
            "horizon_days": meta["horizon_days"], "features": meta["features"],
            "training_window": [meta["train_start"], meta["train_end"]],
            "intended_use": "Decision support for staffing and prep; a manager approves every rota.",
            "out_of_scope": "Individual employee scheduling decisions, other restaurants without retraining.",
            "personal_data": "None — daily aggregates only."}


@app.post("/forecast")
def forecast(req: ForecastRequest):
    fc, pl = service.forecast_and_plan(req.horizon, _overrides(req))
    return {"forecast": _records(fc), "plan": _records(pl), "model_wape": service.covers_wape()}


@app.post("/explain")
def explain(req: ExplainRequest):
    """Manager briefing written by the chosen LLM from the forecast facts (aggregates only)."""
    fc, pl = service.forecast_and_plan(req.horizon, _overrides(req))
    facts = build_facts(fc, pl, service.covers_wape())
    res = generate(req.provider, facts)
    if res.status.startswith("error"):
        raise HTTPException(502, res.status)
    return {"briefing": res.text, "evaluation": score(res, facts), "facts": facts}
