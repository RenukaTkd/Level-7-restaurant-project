# 8. LLM comparison — Claude vs GPT vs Gemini

## Task
Turn the forecast and plan for the next 14 days into a 120–220-word briefing for the manager. Each model gets the
**same system prompt** and the **same facts JSON** (aggregates only). A rule-based **template** acts as the no-LLM
baseline.

| Provider | Default model (override in `.env`) | SDK |
|---|---|---|
| Anthropic | `claude-opus-5-5` | `anthropic` |
| OpenAI | `gpt-5-mini` | `openai` |
| Google | `gemini-2.5-flash` | `google-genai` |
| Baseline | rule-based template | none |

> Check each provider's model list before running. Model IDs change, and you should compare models in a similar
> price/capability tier, or state clearly that you didn't.

## Scenarios (4)
`baseline_next_14_days`, `tuesday_promo`, `wet_cold_week`, `warm_spell`, each passed through the real forecast API.

## Scoring rubric (automatic, `src/forecast/llm/evaluate.py`)
| Criterion | Weight | Definition |
|---|---|---|
| Numeric faithfulness | 50% | share of numbers in the text that appear in the facts (±1 or ±1%); invented numbers are listed |
| Coverage | 30% | totals, busiest day, quietest day, staffing, prep, uncertainty all mentioned |
| Length compliance | 20% | 120–220 words |
| Latency, cost | reported | seconds per call; USD from token usage × `LLM_PRICES_PER_MTOK` |

Optional: add a human rating (1–5) from the manager for usefulness on each output in `reports/llm_outputs.md`.

## Run
```bash
python scripts/compare_llms.py     # needs the 3 keys in .env
```
Outputs: `reports/llm_comparison.csv` (per run), `reports/llm_summary.csv` (averages), `reports/llm_outputs.md` (texts).

## Results (run 14 Nov 2026, 4 scenarios)

| Provider | Model | Runs | Score | Faithfulness | Coverage | Length | Latency (s) | Cost/brief ($) |
|---|---|---|---|---|---|---|---|---|
| claude | claude-opus-5-5 | 4 | **0.986** | 1.00 | 1.00 | 0.93 | 7.5 | 0.026 |
| openai | gpt-5-mini | — | not run (no API key) | | | | | |
| gemini | gemini-2.5-flash | — | not run (no API key) | | | | | |
| template | rule-based-v1 | 4 | 0.940 | 1.00 | 1.00 | 0.70 | 0.0 | 0 |

Source: `reports/llm_summary.csv`; full texts in `reports/llm_outputs.md`.

### Findings
- **Claude beats the no-AI baseline** (0.986 vs 0.940). Both are fully faithful, but Claude writes readable,
  prioritised advice with a concrete recommendation, and keeps closer to the requested length.
- **Cost and speed are acceptable** for a once-a-day briefing: about $0.03 and 7.5 s per brief.
- **The LLM found a data issue.** In the first run Claude noticed that the daily cover figures did not add up
  to the stated fortnight total. Totals were computed before rounding, so they could differ by a few covers.
  `build_facts` now sums the rounded daily values.
- **The rubric was tightened after review.** One early output wrote "£6,632… correction: £6,332". The original
  ±1% tolerance accepted 6,632 because it is close to another real figure (£6,622). Faithfulness now accepts
  exact numbers only (±1 for rounding), and a unit test covers this case.
- **Limitation:** OpenAI and Gemini are implemented and tested to skip gracefully without a key, but were not
  run for this submission. Adding `OPENAI_API_KEY` / `GEMINI_API_KEY` and re-running `scripts/compare_llms.py`
  completes the three-way comparison with no code changes.

## How to write the conclusion
- Pick the **best** model on score first, then weigh latency and cost. For a briefing produced once a day,
  quality matters more than a few seconds or cents.
- Discuss any **unsupported numbers** an LLM produced. That is the main risk for an operational tool.
- Note that the template gets perfect faithfulness but reads mechanically. The LLMs add value through readable,
  prioritised advice, as long as faithfulness stays at or near 1.0.
- Recommend one provider for the pilot, with a second as fallback. The app already supports switching
  provider per request.
