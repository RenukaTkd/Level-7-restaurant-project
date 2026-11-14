# 7. KSB mapping

> **Before submitting:** replace the *KSB code* column with the exact codes from your apprenticeship standard
> (e.g. the Level 7 AI Data Specialist standard, or the one named in your gateway documents). The wording of the
> codes differs between standards, so they are left for you to fill in rather than guessed.

| KSB code | Theme | Evidence in this project |
|---|---|---|
| K__ | Data science / ML lifecycle | `scripts/train.py` end-to-end pipeline; `docs/01` |
| K__ | Statistical & ML methods, choosing the right algorithm | XGBoost vs naive / weekday / moving-average baselines; direct multi-horizon design (`features.py`) |
| K__ | Evaluation metrics & validation | MAE, WAPE, RMSE, Bias (`metrics.py`); rolling-origin backtest (`backtest.py`); `docs/02` |
| K__ | Data governance, legislation & ethics (UK GDPR) | `docs/04` privacy review, data minimisation, LLM data flow |
| K__ | Fairness, bias & responsible AI | segment error parity, human-in-the-loop design (`docs/04`) |
| K__ | Software engineering & deployment architectures | FastAPI service, serverless on Vercel, Streamlit UI, CI (`docs/05`) |
| K__ | Generative AI / LLM capabilities & limitations | 3-provider comparison, faithfulness scoring (`docs/08`) |
| S__ | Acquire, clean, validate data | `data.py::validate` (missing dates, negatives, schema) |
| S__ | Feature engineering | calendar, lag, rolling, weather, event features with no leakage (`features.py`, leakage test) |
| S__ | Build, tune & evaluate models | `model.py`, `config.XGB_PARAMS`, `reports/model_report.md` |
| S__ | Deploy models into production | `api/index.py`, `vercel.json`, live URLs and screenshots |
| S__ | Test & assure AI solutions | 12 pytest tests, GitHub Actions CI, reproducibility hash check |
| S__ | Communicate insight to non-technical stakeholders | Streamlit UI, LLM manager briefing, `docs/06` business case |
| S__ | Evaluate & compare AI technologies | LLM comparison (quality, latency, cost) and recommendation |
| S__ | Quantify business value | cost model in `operations.py`, `docs/06` |
| B__ | Ethical, responsible approach | privacy/fairness review, "decision support only", synthetic public data |
| B__ | Reproducible, well-documented work | `docs/03`, seeds, pins, model lineage |
| B__ | Continuous learning / keeps up with new technology | evaluation of current LLMs from 3 vendors |
| B__ | Stakeholder focus | pilot KPIs and manager trust measure (`docs/06`) |
