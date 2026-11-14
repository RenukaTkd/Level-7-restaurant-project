# 5. Deployment evidence

Deployment route: **GitHub (source + CI) → Vercel (FastAPI) → Streamlit Community Cloud (UI calling the API)**.

## Checklist: capture each item as a screenshot or file and save it in `docs/evidence/`
| # | Evidence | How to capture | Done |
|---|---|---|---|
| 1 | GitHub repo with commit history and green CI run | Actions tab screenshot | ☐ |
| 2 | Vercel deployment "Ready" with the production URL | Vercel → Deployments | ☐ |
| 3 | `/health` response showing model timestamp and data hash | `curl https://<project>.vercel.app/health` | ☐ |
| 4 | Swagger UI at `/docs` | browser screenshot | ☐ |
| 5 | `/forecast` call with a promo override | curl below; save JSON | ☐ |
| 6 | `/explain` call with each of the 3 LLM providers | curl below | ☐ |
| 7 | Vercel runtime logs showing request IDs and latency | Vercel → Logs | ☐ |
| 8 | Streamlit app live URL, in API mode (sidebar shows the API URL) | browser screenshot | ☐ |
| 9 | Plan CSV downloaded by the manager | file | ☐ |
| 10 | Environment variables configured (names only, values hidden) | Vercel → Settings screenshot | ☐ |

## Smoke-test commands
```bash
URL=https://<project>.vercel.app
curl -s $URL/health | jq
curl -s $URL/metrics | jq '.backtest.covers.XGBoost'
curl -s -X POST $URL/forecast -H 'content-type: application/json' \
     -d '{"horizon":7,"overrides":{"2026-11-03":{"promo":1}}}' | jq '.plan[0]'
for p in claude openai gemini; do
  curl -s -X POST $URL/explain -H 'content-type: application/json' -d "{\"provider\":\"$p\"}" | jq '.evaluation'
done
```

## Local verification already completed
- `pytest -q`: **18 passed**. These cover metrics, leakage, staffing rules, `/health`, `/forecast` shape,
  promo uplift, input validation (horizon > 14 → HTTP 422) and briefing faithfulness.
- The pipeline retrains deterministically (identical model hash on repeat runs).

## Operational design
| Concern | Implementation |
|---|---|
| Cold start | Models are loaded once per function instance (`lru_cache`) |
| Function size | Linux uses `xgboost-cpu`; `.vercelignore` excludes the UI, tests and docs |
| Timeouts | `maxDuration: 60` s in `vercel.json` (LLM calls) |
| Failure handling | An LLM error returns HTTP 502 with a reason; a missing key is reported as "skipped"; the UI keeps working |
| Rollback | Vercel → Deployments → "Promote" the previous deployment (instant) |
| Retraining | `python scripts/train.py`, commit the new `models/`, push → automatic redeploy |
