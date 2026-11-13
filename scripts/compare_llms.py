"""Compare Claude vs OpenAI GPT vs Google Gemini (+ a no-LLM template baseline)
on the same forecast-briefing task across several what-if scenarios.

    python scripts/compare_llms.py

Writes reports/llm_comparison.csv, reports/llm_summary.csv and reports/llm_outputs.md.
Providers without an API key are recorded as "skipped" (never fabricated).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

import pandas as pd  # noqa: E402

from forecast import config, service  # noqa: E402
from forecast.llm import PROVIDERS, build_facts, generate, score  # noqa: E402


def scenarios() -> dict[str, dict]:
    start = service.forecaster().history["date"].max() + pd.Timedelta(days=1)
    d = lambda k: (start + pd.Timedelta(days=k)).strftime("%Y-%m-%d")  # noqa: E731
    return {
        "baseline_next_14_days": {},
        "tuesday_promo": {d(i): {"promo": 1} for i in range(14) if (start + pd.Timedelta(days=i)).dayofweek == 1},
        "wet_cold_week": {d(i): {"rain": 1, "temp_c": 4} for i in range(7)},
        "warm_spell": {d(i): {"rain": 0, "temp_c": 21} for i in range(14)},
    }


def main() -> None:
    rows, outputs = [], ["# LLM briefing outputs\n"]
    for name, overrides in scenarios().items():
        fc, pl = service.forecast_and_plan(config.HORIZON, overrides)
        facts = build_facts(fc, pl, service.covers_wape())
        for provider in PROVIDERS:
            res = generate(provider, facts)
            r = score(res, facts)
            rows.append({"scenario": name, **r})
            print(f"{name:24s} {provider:9s} {res.status:28s} score={r['score']}")
            outputs += [f"## {name} — {provider} ({res.model})\n", f"_status: {res.status}_\n",
                        res.text or "(no output)", ""]

    df = pd.DataFrame(rows)
    config.REPORT_DIR.mkdir(exist_ok=True)
    df.to_csv(config.REPORT_DIR / "llm_comparison.csv", index=False)
    (config.REPORT_DIR / "llm_outputs.md").write_text("\n".join(outputs))

    ok = df[df["status"] == "ok"]
    summary = (ok.groupby(["provider", "model"])
               .agg(runs=("score", "size"), score=("score", "mean"), faithfulness=("faithfulness", "mean"),
                    coverage=("coverage", "mean"), length_ok=("length_ok", "mean"),
                    latency_s=("latency_s", "mean"), cost_usd_per_brief=("cost_usd", "mean"))
               .round(3).sort_values("score", ascending=False))
    summary.to_csv(config.REPORT_DIR / "llm_summary.csv")
    print("\n", summary.to_string())
    llms = summary.drop(index="template", level="provider", errors="ignore")
    if len(llms):
        print(f"\nBest LLM on this rubric: {llms.index[0][0]} ({llms.index[0][1]})")
    else:
        print("\nNo LLM API keys found — add them to .env to run the comparison (see .env.example).")


if __name__ == "__main__":
    main()
