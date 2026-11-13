"""Three LLM providers (Claude, OpenAI GPT, Google Gemini) + a deterministic
template baseline, all behind one interface.

The LLM's only job is to turn the forecast/plan *facts* into a short
manager briefing. It never sees raw transactions or staff data — only
aggregated daily numbers (privacy by design).

API keys are read from the environment (.env locally, Vercel/Streamlit
secrets in production). A provider without a key is reported as "skipped".
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

from .. import config

SYSTEM_PROMPT = """You write the weekly operations briefing for a restaurant general manager.
Use ONLY the numbers in the FACTS JSON. Do not invent, re-calculate or estimate any figure.
Write 120-220 words in plain English with these sections:
1. Headline (total expected covers and revenue for the period)
2. Busiest and quietest days
3. Staffing (front-of-house and kitchen) for the peak days
4. Food prep priorities
5. Risks / uncertainty (mention the forecast range and model error)
End with one actionable recommendation."""

DEFAULT_MODELS = {
    "claude": os.getenv("CLAUDE_MODEL", "claude-opus-5-5"),
    "openai": os.getenv("OPENAI_MODEL", "gpt-5-mini"),
    "gemini": os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
    "template": "rule-based-v1",
}


@dataclass
class LLMResult:
    provider: str
    model: str
    text: str = ""
    latency_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    status: str = "ok"
    extra: dict = field(default_factory=dict)

    @property
    def cost_usd(self) -> float:
        pin, pout = config.LLM_PRICES_PER_MTOK.get(self.provider, (0, 0))
        return (self.input_tokens * pin + self.output_tokens * pout) / 1e6


def _user_prompt(facts: dict) -> str:
    return "FACTS:\n" + json.dumps(facts, indent=1, default=str)


# ---- Claude (Anthropic) -------------------------------------------------------
def call_claude(facts: dict) -> LLMResult:
    import anthropic

    model = DEFAULT_MODELS["claude"]
    client = anthropic.Anthropic()          # reads ANTHROPIC_API_KEY
    t0 = time.perf_counter()
    resp = client.beta.messages.create(
        model=model,
        max_tokens=4000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _user_prompt(facts)}],
        output_config={"effort": "low"},     # short summarisation task -> low effort is enough
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",                 # server-side fallback model if a request is refused
    )
    latency = time.perf_counter() - t0
    if resp.stop_reason == "refusal":
        return LLMResult("claude", model, latency_s=latency, status="refused")
    text = "".join(b.text for b in resp.content if b.type == "text")
    return LLMResult("claude", resp.model, text.strip(), latency,
                     resp.usage.input_tokens, resp.usage.output_tokens)


# ---- OpenAI GPT ---------------------------------------------------------------
def call_openai(facts: dict) -> LLMResult:
    from openai import OpenAI

    model = DEFAULT_MODELS["openai"]
    client = OpenAI()                        # reads OPENAI_API_KEY
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": _user_prompt(facts)}],
    )
    latency = time.perf_counter() - t0
    u = resp.usage
    return LLMResult("openai", model, (resp.choices[0].message.content or "").strip(), latency,
                     u.prompt_tokens, u.completion_tokens)


# ---- Google Gemini ------------------------------------------------------------
def call_gemini(facts: dict) -> LLMResult:
    from google import genai
    from google.genai import types

    model = DEFAULT_MODELS["gemini"]
    client = genai.Client()                  # reads GEMINI_API_KEY / GOOGLE_API_KEY
    t0 = time.perf_counter()
    resp = client.models.generate_content(
        model=model, contents=_user_prompt(facts),
        config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
    )
    latency = time.perf_counter() - t0
    um = resp.usage_metadata
    return LLMResult("gemini", model, (resp.text or "").strip(), latency,
                     um.prompt_token_count or 0, um.candidates_token_count or 0)


# ---- Deterministic baseline (no LLM) ------------------------------------------
def call_template(facts: dict) -> LLMResult:
    t0 = time.perf_counter()
    p, q = facts["peak_day"], facts["quietest_day"]
    prep = ", ".join(f"{k} {v}" for k, v in facts["peak_day_prep"].items())
    text = (
        f"Headline: {facts['period']} we expect {facts['total_covers']} covers and "
        f"£{facts['total_revenue_gbp']} revenue.\n"
        f"Busiest day: {p['weekday']} {p['date']} ({p['covers']} covers); quietest: "
        f"{q['weekday']} {q['date']} ({q['covers']} covers).\n"
        f"Staffing: on {p['weekday']} roster {p['foh_staff']} front-of-house and {p['kitchen_staff']} kitchen staff.\n"
        f"Prep for the peak day: {prep}.\n"
        f"Risk: covers could range {p['covers_low']}-{p['covers_high']} on the peak day; "
        f"backtest WAPE is {facts['model_wape_pct']}%.\n"
        f"Recommendation: confirm the {p['weekday']} rota now and review prep daily against bookings."
    )
    return LLMResult("template", DEFAULT_MODELS["template"], text, time.perf_counter() - t0)


PROVIDERS = {"claude": call_claude, "openai": call_openai, "gemini": call_gemini, "template": call_template}
KEY_ENV = {"claude": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY", "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY")}


def available(provider: str) -> bool:
    if provider == "template":
        return True
    keys = KEY_ENV[provider]
    keys = keys if isinstance(keys, tuple) else (keys,)
    return any(os.getenv(k) for k in keys)


def generate(provider: str, facts: dict) -> LLMResult:
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider {provider}; choose from {list(PROVIDERS)}")
    if not available(provider):
        return LLMResult(provider, DEFAULT_MODELS[provider], status="skipped: no API key")
    try:
        return PROVIDERS[provider](facts)
    except ImportError as e:
        return LLMResult(provider, DEFAULT_MODELS[provider], status=f"skipped: SDK not installed ({e.name})")
    except Exception as e:  # network / auth / rate-limit -> recorded, never crashes the app
        return LLMResult(provider, DEFAULT_MODELS[provider], status=f"error: {type(e).__name__}: {e}"[:300])
