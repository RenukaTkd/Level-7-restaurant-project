# AI Restaurant Sales Forecasting — single-site pilot

Forecasts **daily covers and revenue 14 days ahead** for one restaurant using **XGBoost**, turns the forecast into a
**staffing rota headcount and food-prep plan**, and uses **three LLMs (Claude, OpenAI GPT, Google Gemini)** to write a
manager briefing, with an automatic comparison to show which LLM does that job best.

| Layer | Technology | Where it runs |
|---|---|---|
| Model | XGBoost (direct multi-horizon), pandas | trained locally / CI |
| API | FastAPI (`api/index.py`) | **Vercel** (Python serverless function) |
| UI | Streamlit (`app/streamlit_app.py`) | **Streamlit Community Cloud** (calls the Vercel API) |
| Code & CI | Git + GitHub Actions | **GitHub** |
| LLMs | Anthropic Claude, OpenAI GPT, Google Gemini | provider APIs |

> **Why isn't Streamlit on Vercel?** Vercel runs short-lived serverless functions. Streamlit needs a long-running
> server with WebSockets, so it can't run there. The standard pattern is **API on Vercel + UI on Streamlit Cloud**.
> The UI also has a *local mode* so it works with no API at all.

```
           ┌────────────────────┐   HTTPS/JSON   ┌───────────────────────────┐
 Manager ─▶│ Streamlit UI       │ ─────────────▶ │ FastAPI on Vercel         │
           │ (Streamlit Cloud)  │                │  /forecast  /explain      │
           └────────────────────┘                │  /metrics   /model-card   │
                                                 │  XGBoost models (JSON)    │
                                                 └──────────┬────────────────┘
                                                            │ aggregated facts only
                                              ┌─────────────┼──────────────┐
                                              ▼             ▼              ▼
                                           Claude        OpenAI GPT      Gemini
```

## Results (rolling-origin backtest, 12 folds × 14 days)

| Covers forecast | MAE | WAPE | vs current practice |
|---|---|---|---|
| **XGBoost** | **16.3** | **9.8%** | **−29% error** |
| Weekday average (4 wk) | 18.5 | 11.2% | −20% |
| Seasonal naive (same day last week, *current manual practice*) | 23.1 | 13.9% | — |
| Moving average (28 d) | 38.3 | 23.1% | +66% |

Simulated business impact: about **£20k a year less** in over-staffing, food waste and lost margin compared with current
practice (`reports/business_impact.csv`). Full detail: [`reports/model_report.md`](reports/model_report.md).

> The history is **synthetic but realistic** (seeded, so it can be reproduced), because real POS data is commercially
> sensitive. To use real data, replace `data/sales_history.csv` with columns `date, covers, revenue` (optional:
> `promo, temp_c, rain, closed`) and re-run `scripts/train.py`.

## Quick start (VS Code, macOS/Windows/Linux)

Requires **Python 3.11+** (3.12 recommended, which matches Vercel). On macOS, XGBoost also needs OpenMP:
`brew install libomp`.

```bash
git clone https://github.com/<you>/restaurant-sales-forecasting.git
cd restaurant-sales-forecasting
python3.12 -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
cp .env.example .env                                       # add your 3 LLM API keys

python scripts/train.py          # data → backtest → baselines → model → reports (≈10 s)
pytest -q                        # 18 tests
python scripts/compare_llms.py   # Claude vs GPT vs Gemini → reports/llm_summary.csv

uvicorn api.index:app --reload   # API docs at http://127.0.0.1:8000/docs
streamlit run app/streamlit_app.py
```

In VS Code, choose the interpreter with **Python: Select Interpreter → .venv**. The `Makefile` has shortcuts:
`make train test llm api ui`.

## Deploy

1. **GitHub**: `git push -u origin main`. GitHub Actions retrains, checks the model hash (reproducibility) and runs the tests.
2. **Vercel (API)**: on vercel.com, choose *Add New → Project* and import the repo. Use framework preset **Other** and
   leave the build settings at their defaults. Under *Environment Variables*, add `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`,
   `GEMINI_API_KEY` and optionally `API_KEY`, then deploy. Check `https://<project>.vercel.app/health` and `/docs`.
   (CLI alternative: `npm i -g vercel && vercel --prod`.)
3. **Streamlit Community Cloud (UI)**: on share.streamlit.io, choose *New app*, pick the repo and set the main file to
   `app/streamlit_app.py` and the Python version to 3.12. Under *Secrets*, paste `API_URL = "https://<project>.vercel.app"`
   (plus `API_KEY` if you set one).
4. Collect the evidence listed in [`docs/05_deployment_evidence.md`](docs/05_deployment_evidence.md).

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | status, model timestamp, data hash |
| GET | `/metrics` | backtest MAE/WAPE vs baselines, business impact, segment parity |
| GET | `/model-card` | features, intended use, limitations |
| POST | `/forecast` | `{"horizon":14,"overrides":{"2026-11-03":{"promo":1,"rain":1}}}` → forecast + staffing/prep plan |
| POST | `/explain` | as above + `"provider":"claude"\|"openai"\|"gemini"\|"template"` → LLM briefing + score |

## Repository map

```
api/index.py               FastAPI app (Vercel entry point)
app/streamlit_app.py       Streamlit UI (API mode or local mode)
src/forecast/              data, features, model, baselines, metrics, backtest, operations, service
src/forecast/llm/          3 LLM providers + template baseline, scoring rubric
scripts/train.py           reproducible end-to-end pipeline
scripts/compare_llms.py    LLM comparison across 4 what-if scenarios
data/ models/ reports/     committed artefacts (evidence)
docs/                      pilot enhancement write-up (01–08)
tests/                     unit + API tests
```

## Pilot enhancement documents

1. [Project overview & problem statement](docs/01_project_overview.md)
2. [Baseline comparison](docs/02_baseline_comparison.md)
3. [Reproducibility](docs/03_reproducibility.md)
4. [Privacy & fairness](docs/04_privacy_fairness.md)
5. [Deployment evidence](docs/05_deployment_evidence.md)
6. [Business impact](docs/06_business_impact.md)
7. [KSB mapping](docs/07_ksb_mapping.md)
8. [LLM comparison](docs/08_llm_comparison.md)
