"""Build briefing facts and score LLM outputs with a transparent, automatic rubric.

Score (0-1) = 0.5 * numeric faithfulness   (every number in the text must exist in FACTS)
            + 0.3 * coverage               (required sections present)
            + 0.2 * length compliance      (120-220 words requested)
Latency and cost are reported separately so the trade-off is visible.
"""
from __future__ import annotations

import re

import pandas as pd

from .. import config
from .providers import LLMResult

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def build_facts(forecast: pd.DataFrame, plan: pd.DataFrame, wape: float) -> dict:
    f = forecast.merge(plan, on=["date", "covers"])
    open_days = f[f["covers"] > 0]
    peak = open_days.loc[open_days["covers"].idxmax()]
    quiet = open_days.loc[open_days["covers"].idxmin()]

    def day(r):
        return {"date": r["date"].strftime("%Y-%m-%d"), "weekday": WEEKDAYS[r["date"].dayofweek],
                "covers": int(round(r["covers"])), "covers_low": int(round(r["covers_low"])),
                "covers_high": int(round(r["covers_high"])), "revenue_gbp": int(round(r["revenue"])),
                "foh_staff": int(r["foh_staff"]), "kitchen_staff": int(r["kitchen_staff"])}

    daily = [day(r) for _, r in f.iterrows()]
    return {
        "restaurant": config.RESTAURANT_NAME,
        "period": f"{f['date'].min():%d %b} to {f['date'].max():%d %b %Y}",
        # totals are sums of the rounded daily figures, so they add up exactly
        "total_covers": sum(d["covers"] for d in daily),
        "total_revenue_gbp": sum(d["revenue_gbp"] for d in daily),
        "total_labour_hours": int(f["labour_hours"].sum()),
        "model_wape_pct": round(100 * wape, 1),
        "peak_day": day(peak),
        "quietest_day": day(quiet),
        "peak_day_prep": {c.replace("prep: ", ""): int(peak[c]) for c in f.columns if c.startswith("prep: ")},
        "promo_days": [d.strftime("%Y-%m-%d") for d in f.loc[f["promo"] == 1, "date"]],
        "closed_days": [d.strftime("%Y-%m-%d") for d in f.loc[f["closed"] == 1, "date"]],
        "daily": daily,
    }


def _fact_numbers(obj) -> set[float]:
    nums: set[float] = set()
    if isinstance(obj, dict):
        for v in obj.values():
            nums |= _fact_numbers(v)
    elif isinstance(obj, list):
        for v in obj:
            nums |= _fact_numbers(v)
    elif isinstance(obj, (int, float)):
        nums.add(float(obj))
    elif isinstance(obj, str):
        nums |= {float(x) for x in re.findall(r"\d+(?:\.\d+)?", obj)}
    return nums


NUM_RE = re.compile(r"(?<![\w.])£?(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)(k)?")


def numeric_faithfulness(text: str, facts: dict) -> tuple[float, list[str]]:
    allowed = _fact_numbers(facts) | {float(i) for i in range(1, 6)}  # section numbering
    found, bad = 0, []
    for m in NUM_RE.finditer(text):
        val = float(m.group(1).replace(",", "")) * (1000 if m.group(2) else 1)
        found += 1
        # exact numbers only (±1 for rounding); "£62.7k" style gets ±0.5% for its own rounding
        tol = (lambda a: max(1.0, 0.005 * abs(a))) if m.group(2) else (lambda a: 1.0)
        if not any(abs(val - a) <= tol(a) for a in allowed):
            bad.append(m.group(0))
    return (1.0 if found == 0 else (found - len(bad)) / found), bad


COVERAGE = {
    "headline_totals": lambda t, f: str(f["total_covers"]) in t.replace(",", ""),
    "peak_day": lambda t, f: f["peak_day"]["weekday"].lower() in t.lower(),
    "quiet_day": lambda t, f: f["quietest_day"]["weekday"].lower() in t.lower(),
    "staffing": lambda t, f: bool(re.search(r"staff|rota|front[- ]of[- ]house|kitchen", t, re.I)),
    "prep": lambda t, f: bool(re.search(r"prep|portion|dough|patt|fillet", t, re.I)),
    "uncertainty": lambda t, f: bool(re.search(r"range|uncertain|risk|error|WAPE|could", t, re.I)),
}


def score(result: LLMResult, facts: dict) -> dict:
    row = {"provider": result.provider, "model": result.model, "status": result.status,
           "latency_s": round(result.latency_s, 2), "input_tokens": result.input_tokens,
           "output_tokens": result.output_tokens, "cost_usd": round(result.cost_usd, 5)}
    if result.status != "ok" or not result.text:
        return {**row, "faithfulness": None, "coverage": None, "length_ok": None, "score": None}
    faith, bad = numeric_faithfulness(result.text, facts)
    cov = sum(fn(result.text, facts) for fn in COVERAGE.values()) / len(COVERAGE)
    words = len(result.text.split())
    length = 1.0 if 120 <= words <= 220 else max(0.0, 1 - min(abs(words - 120), abs(words - 220)) / 120)
    return {**row, "words": words, "faithfulness": round(faith, 3), "unsupported_numbers": " ".join(bad[:10]),
            "coverage": round(cov, 3), "length_ok": round(length, 3),
            "score": round(0.5 * faith + 0.3 * cov + 0.2 * length, 3)}
