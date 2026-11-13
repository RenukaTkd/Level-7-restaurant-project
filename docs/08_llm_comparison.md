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

## Results
*Paste `reports/llm_summary.csv` here after running with your keys.*

| Provider | Model | Score | Faithfulness | Coverage | Length | Latency (s) | Cost/brief ($) |
|---|---|---|---|---|---|---|---|
| claude | | | | | | | |
| openai | | | | | | | |
| gemini | | | | | | | |
| template | rule-based-v1 | 0.94 | 1.00 | 1.00 | 0.70 | 0.00 | 0 |

## How to write the conclusion
- Pick the **best** model on score first, then weigh latency and cost. For a briefing produced once a day,
  quality matters more than a few seconds or cents.
- Discuss any **unsupported numbers** an LLM produced. That is the main risk for an operational tool.
- Note that the template gets perfect faithfulness but reads mechanically. The LLMs add value through readable,
  prioritised advice, as long as faithfulness stays at or near 1.0.
- Recommend one provider for the pilot, with a second as fallback. The app already supports switching
  provider per request.
