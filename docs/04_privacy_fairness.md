# 4. Privacy & fairness review

## Privacy (UK GDPR / Data Protection Act 2018)
| Principle | Design decision |
|---|---|
| Data minimisation | The model uses **daily aggregates only** (covers, revenue, promo flag, weather). No customer names, card data, bookings or staff records. |
| Personal data | None is processed, so no DPIA is required for the forecasting component. This was re-checked when the LLM layer was added. |
| Third-party LLMs | Only the aggregated forecast "facts" JSON is sent to Claude, OpenAI and Gemini (see `build_facts`). There are no free-text fields in which personal data could leak. Use API (not consumer) accounts, whose terms exclude training on inputs by default; confirm this in each provider's current data-usage policy. |
| Secrets | API keys are only in `.env` (git-ignored), Vercel environment variables and Streamlit secrets. None are in code. |
| Access control | Optional `API_KEY` header on the API. CORS can be restricted with `CORS_ORIGINS`. |
| Logging | The API logs method, path, status and latency with a request ID. It **never logs request or response bodies**. |
| Commercial sensitivity | Revenue figures are commercially sensitive. That is why the public repo uses synthetic data. |

## Fairness
The model forecasts demand. It does not make decisions about people. Fairness risks come up in two places:

1. **Systematic under-forecasting of particular day types.** This would leave the team on those shifts short-staffed
   and over-worked again and again. It is monitored as *error parity* across segments (`reports/segment_errors.csv`):

   | Segment | Days | MAE | WAPE |
   |---|---|---|---|
   | Weekday (Mon–Thu) | 96 | 14.3 | 10.7% |
   | Friday | 24 | 16.1 | 7.6% |
   | Weekend | 48 | 20.5 | 9.9% |
   | Spring/Autumn | 76 | 15.3 | 9.6% |
   | Summer | 92 | 17.2 | 10.0% |

   WAPE stays in a narrow band (7.6–10.7%), so no segment is badly served. Weekend MAE is higher in absolute terms
   because weekend volumes are higher. *Gap:* the backtest window does not include winter or December. The pilot
   must check December error before the Christmas rota is built.

2. **How the headcount is used.** The output is a *number* of staff. Choosing *who* works stays with the manager,
   under existing HR and working-time policy. The tool does not use or infer protected characteristics, and it
   should not be used to cut contracted hours without consultation.

## Human oversight
- Every number is advisory and can be overridden in the UI. A plan CSV is downloaded for the manager to edit.
- The uncertainty range (P10–P90) is always shown next to the point forecast.
- LLM briefings are scored for numeric faithfulness. Any number not in the source data is flagged.

## Residual risks and mitigations
| Risk | Mitigation |
|---|---|
| Drift (new menu, refurbishment, competitor opens) | Weekly WAPE check; retrain monthly or when WAPE exceeds 15% for 2 weeks |
| LLM hallucination | Faithfulness scoring; template fallback; numbers in UI tables come from the model, not the LLM |
| Over-reliance | Training for the manager; "decision support only" banner |
