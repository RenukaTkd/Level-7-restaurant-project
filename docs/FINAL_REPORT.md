# AI Restaurant Sales Forecasting: Single-Site Pilot

### Final project report

| | |
|---|---|
| **Author** | Renuka |
| **Programme** | Level 7 apprenticeship: *[insert standard name and code]* |
| **Employer / context** | *[insert employer or project sponsor]* |
| **Submission date** | *[insert date]* |
| **Live application** | https://level-7-restaurant-project-ppkk7axcc6vprouutqcwkw.streamlit.app/ |
| **Live API** | https://level-7-restaurant-project-renukas-projects-bea33a15.vercel.app/docs |
| **Source code** | https://github.com/RenukaTkd/Level-7-restaurant-project |

---

## Contents

1. Executive summary
2. Business context and problem statement
3. Objectives, scope and success criteria
4. Data
5. Methodology
6. Evaluation: backtesting and baseline comparison
7. From forecast to decisions: staffing and food preparation
8. Business impact
9. Generative AI layer and LLM comparison
10. System architecture and deployment
11. Testing, quality assurance and reproducibility
12. Privacy, ethics and fairness
13. Pilot enhancement plan
14. Limitations and risks
15. Reflection and lessons learned
16. KSB mapping
17. Conclusion and recommendations
18. References
19. Appendices

---

## 1. Executive summary

Restaurants plan staff rotas about two weeks ahead and prepare food every morning. Both decisions depend on how
many guests ("covers") will arrive. At the pilot site these decisions are currently based on "the same day last
week", adjusted by the manager's judgement. That approach is simple, but it misses predictable drivers such as
promotions, bank holidays, paydays, weather and seasonal trends. The result is avoidable wage cost on quiet days,
stretched teams on busy days, and food waste.

This project delivers an end-to-end AI decision-support tool for one restaurant:

- An **XGBoost forecasting model** predicts daily covers and revenue up to **14 days ahead**, with an uncertainty
  range.
- A **planning layer** turns the forecast into a front-of-house and kitchen **headcount** and a **prep list** per
  dish.
- A **generative AI layer** writes a plain-English manager briefing. Three LLM providers (Anthropic Claude, OpenAI
  GPT and Google Gemini) are supported behind one interface and scored with an automatic faithfulness rubric.
- The system is **deployed**: the API runs on **Vercel** (FastAPI), the manager interface on **Streamlit Community
  Cloud**, and the source code and CI on **GitHub**.

**Key results** (rolling-origin backtest, 12 folds × 14 days, 168 forecast days per method):

| Measure | XGBoost | Current practice (same day last week) | Change |
|---|---|---|---|
| WAPE (covers) | **9.8%** | 13.9% | **29% lower error** |
| MAE (covers per day) | **16.3** | 23.1 | −6.8 covers/day |
| WAPE (revenue) | **10.8%** | 14.6% | 26% lower error |
| Folds where the model wins | **12 of 12** | | |
| Simulated annual cost of forecast errors | **£35,233** | £55,523 | **£20,290 saving (−37%)** |

The Claude briefing scored **0.986** on the rubric against **0.940** for a rule-based template, and every number
it reported traced back to the source data. Through the live API, a briefing costs about **$0.02** and takes about
**7 seconds**.

I recommend an **8-week live pilot** at the site, measuring real labour cost, food waste and manager trust against
a 4-week baseline. Section 13 gives the plan.

---

## 2. Business context and problem statement

### 2.1 The organisation

The pilot site is a mid-sized, independent restaurant (referred to as *The Riverside Kitchen*) serving between
about 50 and 315 covers a day, with an average of 136. It trades seven days a week and closes only on Christmas
Day. Average spend is about £31 per cover.

### 2.2 Current process

- **Rota planning** happens every two weeks. The manager copies the previous fortnight's pattern and adjusts it
  by feel.
- **Food preparation** is decided each morning by the head chef, based on the previous week's sales.
- Nobody records forecasts, so forecast accuracy is never measured or improved.

### 2.3 Problem statement

> *The site cannot reliably anticipate daily demand two weeks ahead. As a result it over-staffs quiet days,
> under-staffs busy days and over-prepares food, raising labour and food costs and harming service quality.*

### 2.4 Stakeholders

| Stakeholder | Interest | How the project addresses it |
|---|---|---|
| General manager | A rota they can trust, less planning time | 14-day forecast, headcount plan, plain-English briefing |
| Head chef | The right prep quantities, less waste | Daily prep list by dish |
| Front-of-house and kitchen staff | Fair, sensible workloads | Fewer under-staffed shifts; no decisions about individuals |
| Owner / finance | Lower labour and waste cost | Quantified business case (Section 8) |
| Customers | Good service, dishes available | Fewer under-staffed services and stock-outs |
| Assessor / IT | Reproducible, secure, maintainable solution | CI, tests, documentation, privacy review |

---

## 3. Objectives, scope and success criteria

### 3.1 Objectives

1. Build a forecasting model for daily covers and revenue with a 14-day horizon.
2. Prove it beats the methods a manager could realistically use, under a fair backtest.
3. Turn forecasts into **actionable** staffing and prep recommendations.
4. Quantify the business value in pounds.
5. Use generative AI to explain the plan, and evaluate it for accuracy, cost and speed.
6. Deploy the solution as a usable product with an API and a user interface.
7. Address reproducibility, privacy, fairness and responsible use.

### 3.2 Scope

**In scope:** one site, daily granularity, covers and revenue, staffing headcount by role, prep quantities for
five key items, deployment, and evaluation.

**Out of scope:** hourly or day-part forecasting; choosing *which* staff work (the manager keeps this decision);
supplier ordering; multi-site roll-out.

### 3.3 Success criteria (agreed before modelling)

| Criterion | Target | Achieved | Status |
|---|---|---|---|
| Relative WAPE improvement on covers vs current practice | ≥ 15% | 29.2% | ✅ |
| MAE on covers | < 20/day | 16.3 | ✅ |
| Beats current practice in every backtest fold | yes | 12/12 | ✅ |
| Simulated annual saving | > £10k | £20,290 | ✅ |
| LLM briefing faithfulness (numbers traceable to the data) | ≥ 0.95 | 1.00 | ✅ |
| Publicly reachable UI and API | yes | yes | ✅ |
| Automated tests and CI | yes | 18 tests, GitHub Actions | ✅ |

---

## 4. Data

### 4.1 Source and rationale

Real point-of-sale data is commercially sensitive and could not be published in a public repository. I therefore
built a **seeded synthetic history** (`src/forecast/data.py`) that reproduces the patterns found in restaurant
trading data: a weekly cycle, seasonality, growth, weather sensitivity, promotions, bank holidays, special events
and closures. Because it is seeded (`SEED = 42`), every run produces exactly the same data.

The pipeline is **source-agnostic**. A real POS export with the columns `date, covers, revenue` (plus optional
`promo, temp_c, rain, closed`) can replace `data/sales_history.csv` without any code changes.

### 4.2 Dataset summary

| Item | Value |
|---|---|
| Period | 1 Jan 2024 – 31 Oct 2026 (1,035 days) |
| Closed days | 2 (Christmas Day) |
| Total covers / revenue | 140,852 covers / £4.33m |
| Mean covers per open day | 136.4 (sd 44.3, min 49, max 315) |
| Average spend per cover | £30.71 |
| Promotion days | 62 (Monday–Wednesday only) |

### 4.3 Schema

| Column | Type | Description |
|---|---|---|
| `date` | date | Trading day (one row per day, no gaps) |
| `covers` | int | Guests served (target 1) |
| `revenue` | float | Net sales in £ (target 2) |
| `promo` | 0/1 | Planned promotion running |
| `temp_c` | float | Maximum temperature (°C) |
| `rain` | 0/1 | Rain on the day |
| `closed` | 0/1 | Restaurant closed |

### 4.4 Exploratory findings

- **Weekly cycle:** mean covers from Monday to Sunday are 99, 104, 115, 126, **173, 193**, 143. Saturday is
  nearly double Monday, so any useful model must capture the day of the week.
- **Seasonality:** monthly means range from 93 (January) to 164 (July), with a smaller December peak (136) from
  festive bookings.
- **Growth:** the mean rises from 126.8 (2024) to 136.4 (2025) and 147.7 (2026), about +8% a year.
- **Promotions:** Monday–Wednesday promotions lift covers by **19%** on those days.
- **Weather:** rainy days average **11.5% fewer** covers.

### 4.5 Data validation

`validate()` rejects or repairs data before training:

- **Missing required columns:** the load fails with a clear message.
- **Missing dates:** the load fails. A gap would silently corrupt the lag features, so closed days must be
  recorded with `closed = 1`.
- **Negative covers or revenue:** the load fails.
- **Duplicate dates:** removed. **Missing weather:** filled from the monthly climate averages.

---

## 5. Methodology

### 5.1 Problem framing

This is a **multi-step time-series regression** problem: on day *t*, predict demand for days *t+1* to *t+14*.
I chose a **direct strategy**: one model, in which every lag feature is at least 14 days old. The alternative,
**recursive** forecasting, feeds predictions back in as inputs. That compounds errors and makes leakage easy to
introduce by accident. With the direct design, every input is genuinely known on the day the forecast is made,
and a unit test enforces this (`test_no_leakage_lags_respect_horizon`).

### 5.2 Features (21)

| Group | Features | Why |
|---|---|---|
| Calendar | day of week, month, day of year, week of year, weekend flag | Weekly and annual cycles |
| Events | bank holiday, special event (Valentine's, Christmas Eve, NYE, Christmas parties), payday week | Known demand spikes |
| Planned / external | promotion, temperature, rain, closed | Drivers known in advance or forecastable |
| Lags (≥ 14 days) | lag 14, 21, 28, 35, 364 | Recent level, same weekday, same day last year |
| Rolling statistics | 7- and 28-day mean, 28-day standard deviation (on the 14-day-lagged series) | Trend and volatility |
| Weekday profile | average of the same weekday over the last 4 eligible weeks | Robust weekday baseline |

### 5.3 Model choice

I chose **XGBoost (gradient-boosted trees)** because:

- it captures non-linear interactions (for example, a promotion on a rainy Tuesday) without manual feature crosses;
- it is strong on small tabular datasets (about 1,000 rows) where deep learning would overfit;
- it trains in seconds, so a 12-fold backtest is cheap;
- it handles mixed feature types without scaling, and is easy to explain through feature importance.

Alternatives I considered: SARIMA/ETS (weak with several external drivers), Prophet (good for seasonality, weaker
on interactions), and LSTMs or other deep models (too data-hungry for one site).

### 5.4 Configuration

The model uses XGBoost's native `Booster` API (`xgb.train`): 600 rounds, learning rate 0.03, depth 5, 90% row
and column subsampling, `min_child_weight` 3, L2 regularisation 1.0, `hist` tree method, seed 42, one thread (for
determinism). I deliberately kept the parameters conservative rather than tuning them heavily. With only 168
backtest days, aggressive tuning would risk overfitting to the evaluation period.

### 5.5 Uncertainty

Prediction intervals come from the **empirical residuals** of the backtest: P10 = forecast − 21.5 covers and
P90 = forecast + 29.1 covers. They are honest by construction because they reflect out-of-sample errors. The
interval is asymmetric because the model slightly under-forecasts busy days.

---

## 6. Evaluation: backtesting and baseline comparison

### 6.1 Rolling-origin backtest

A single train/test split can flatter a model through the luck of one period. I used a **rolling-origin
backtest**:

- 12 forecast origins, 14 days apart, covering 17 May – 31 Oct 2026;
- at each origin the model is **retrained from scratch** on data up to that day only, then forecasts 14 days;
- the baselines are given exactly the same information, including known closure days.

### 6.2 Metrics

- **MAE** (mean absolute error) is in covers per day, which managers can interpret directly.
- **WAPE** = Σ|actual − forecast| / Σ actual is the share of total volume mis-forecast. I preferred WAPE to MAPE
  because MAPE becomes unstable on low-volume or closed days.
- **RMSE** penalises large misses, which hurt service most.
- **Bias** is the mean of (forecast − actual); positive means over-forecasting.

### 6.3 Baselines

| Baseline | Description |
|---|---|
| Seasonal naive | Same day last week: **the manager's current practice** |
| Weekday average (4 wk) | Mean of the same weekday over the last 4 weeks |
| Moving average (28 d) | Flat mean of the last 28 days |

### 6.4 Results: covers

| Method | MAE | WAPE | RMSE | Bias | WAPE vs naive |
|---|---|---|---|---|---|
| **XGBoost** | **16.33** | **9.8%** | **20.83** | −3.55 | **+29.2%** |
| Weekday average (4 wk) | 18.54 | 11.2% | 22.99 | −0.06 | +19.6% |
| Seasonal naive (last week) | 23.06 | 13.9% | 28.19 | −1.38 | 0 |
| Moving average (28 d) | 38.30 | 23.1% | 45.80 | −0.06 | −66.1% |

### 6.5 Results: revenue

| Method | MAE (£) | WAPE | RMSE (£) | WAPE vs naive |
|---|---|---|---|---|
| **XGBoost** | **568** | **10.8%** | **726** | **+26.1%** |
| Weekday average (4 wk) | 591 | 11.2% | 760 | +23.2% |
| Seasonal naive (last week) | 770 | 14.6% | 971 | 0 |
| Moving average (28 d) | 1,489 | 28.2% | 1,732 | −93.5% |

### 6.6 Robustness

- **Every fold:** XGBoost beats current practice in **12 of 12** folds. Its fold-level WAPE ranges from 6.5%
  to 14.9%.
- **Horizon:** XGBoost's error stays flat across the horizon (10.1% for days 1–7, 9.6% for days 8–14). The naive
  method gets worse (12.9% to 14.9%). This matters because rotas are fixed up to two weeks ahead.
- **Day-level accuracy:** 58% of forecast days are within ±10% of the actual covers.

### 6.7 Interpretation

The gains are largest on days whose drivers baselines can't see: promotions, bank holidays, paydays and weather
changes. On revenue, the gap over the weekday average is small (10.8% vs 11.2%) because spend per cover is
stable. Most of the value therefore lies in the **covers** forecast, which drives staffing and prep. The slight
negative bias (−3.6 covers/day) is handled by the safety buffer in the planning layer (Section 7).

---

## 7. From forecast to decisions: staffing and food preparation

A forecast only creates value when it changes a decision. The planning layer (`src/forecast/operations.py`)
turns covers into plans:

1. **Safety buffer:** planned covers = forecast + 0.84 × residual σ (20.6), about an **80% service level**.
   The manager can see the buffer and change it in `config.py`.
2. **Staffing:** front-of-house = ⌈planned covers ÷ 35⌉ and kitchen = ⌈planned covers ÷ 50⌉, with a minimum
   of 2 each. Closed days need 0. Each shift is 8 hours.
3. **Prep:** portions per item = planned covers × menu-mix share (burger 28%, pizza dough 23%, chicken 22%,
   fish 15%, risotto 12%).

For example, the forecast for Friday 6 November 2026 is 200 covers, so the plan is to prepare for 218 covers with
**7 front-of-house and 5 kitchen staff**, 62 burger patties and 48 chicken breasts.

All ratios are **operating assumptions**, kept in one configuration file so the site can calibrate them to its
own service model during the pilot.

---

## 8. Business impact

### 8.1 Cost model

For every backtest day, each method's forecast is turned into a staffing plan and costed against what actually
happened:

| Component | Formula | Assumption |
|---|---|---|
| Over-staffing | surplus staff × 8 h × wage | £12.60/h |
| Food waste | over-forecast covers × ingredient cost | £6.50/cover |
| Lost margin | covers above capacity × walk-out rate × spend × margin | 30% × £28 × 65% |

Costs are annualised (× 365 / 168). No buffer is applied here, so the methods are compared fairly.

### 8.2 Results (annual, £)

| Method | Over-staffing | Waste | Lost margin | **Total** | **Saving vs current** |
|---|---|---|---|---|---|
| **XGBoost** | 13,140 | 15,082 | 7,011 | **35,233** | **20,290** |
| Weekday average | 16,425 | 21,903 | 7,983 | 46,312 | 9,211 |
| Seasonal naive (current) | 18,834 | 25,716 | 10,973 | 55,523 | — |
| Moving average | 36,135 | 45,402 | 26,904 | 108,441 | −52,918 |

### 8.3 What the saving means

- About **£20k a year** saved at a single site, **37%** of the cost of forecast errors under current practice.
- About **450 fewer surplus paid hours** a year, which can be moved to busy shifts.
- About **1,600 fewer over-prepared covers** a year (−41% waste), which supports sustainability reporting.
- **36% less lost margin** from under-staffed services.
- **Running cost:** hosting is free at pilot scale, and LLM briefings cost about $0.02 each (about $7 a year for
  one a day).

These figures come from a **simulation on synthetic data** with assumed cost rates. The pilot exists to confirm
them with real payroll and waste records.

---

## 9. Generative AI layer and LLM comparison

### 9.1 Purpose

Managers asked for "the so-what", not tables. The LLM layer turns the forecast facts into a 120–220-word briefing
with five sections: headline, busiest and quietest days, staffing, prep priorities, and risks, ending with one
recommendation.

### 9.2 Design principles

- **Facts in, prose out.** The model receives a compact JSON of **aggregated** numbers only (`build_facts`). It is
  told to use only those numbers and never to calculate new ones.
- **One interface, several providers.** Claude, GPT and Gemini sit behind `generate(provider, facts)`. A
  deterministic **template** acts as the no-AI baseline. A provider without a key is reported as "skipped" and
  never fabricated.
- **Fail safe.** API errors are caught and reported, and the UI and the forecast tables keep working without
  any LLM.
- **Claude configuration:** model `claude-opus-5-5`, low effort (a short summarisation task), and Anthropic's
  server-side fallback enabled, so a declined request is retried on a fallback model.

### 9.3 Evaluation rubric (automatic)

| Criterion | Weight | Definition |
|---|---|---|
| Numeric faithfulness | 50% | Share of numbers in the text that exist in the facts (exact, ±1 for rounding) |
| Coverage | 30% | Totals, busiest day, quietest day, staffing, prep and uncertainty all present |
| Length compliance | 20% | Within 120–220 words |
| Latency, cost | reported | Seconds per call; USD from token usage |

The same four scenarios were used for every provider: baseline fortnight, Tuesday promotions, a wet and cold
week, and a warm spell.

### 9.4 Results

| Provider | Model | Score | Faithfulness | Coverage | Length | Latency | Cost/brief |
|---|---|---|---|---|---|---|---|
| **Claude** | claude-opus-5-5 | **0.986** | 1.00 | 1.00 | 0.93 | 7.5 s | $0.026 |
| Template (no AI) | rule-based-v1 | 0.940 | 1.00 | 1.00 | 0.70 | 0 s | $0 |
| OpenAI GPT | gpt-5-mini | not run (no key) | | | | | |
| Google Gemini | gemini-2.5-flash | not run (no key) | | | | | |

The live API gave the same result: score **0.985**, faithfulness **1.00**, $0.020, about 7 seconds.

### 9.5 Findings

1. **Claude adds value over the template.** Both are fully faithful, but Claude prioritises, explains trade-offs
   (for example, "Saturday has fewer covers than Friday but the highest revenue") and gives a concrete
   recommendation.
2. **The LLM found a defect in my pipeline.** Claude noticed that the daily covers did not add up to the stated
   fortnight total, because totals were computed before rounding. I fixed `build_facts` so that totals are sums
   of the rounded daily values.
3. **Reviewing outputs improved the evaluation.** One early output contained "£6,632… correction: £6,332". My
   original ±1% tolerance accepted 6,632 because it was close to another real figure (£6,622). I tightened the
   check to exact numbers and added a regression test. This showed me that **an evaluation also needs to be
   evaluated**.
4. **Limitation:** OpenAI and Gemini are implemented and tested to skip cleanly, but were not run because of API
   key access. Running the three-way comparison needs only the keys; no code changes.

---

## 10. System architecture and deployment

### 10.1 Architecture

```
  Manager ──▶ Streamlit UI (Streamlit Community Cloud)
                 │  HTTPS / JSON (API_URL secret)
                 ▼
             FastAPI service (Vercel serverless function, Python 3.12)
               ├── XGBoost models (JSON, loaded once per instance)
               ├── Planning layer (staffing, prep)
               └── /explain ──▶ Claude | GPT | Gemini | template   (aggregates only)
                 ▲
  GitHub ── push ──▶ GitHub Actions (retrain, hash check, 18 tests) + Vercel auto-deploy
```

### 10.2 Why this split

Vercel runs short-lived serverless functions. Streamlit needs a long-running server with WebSockets, so it cannot
run on Vercel. The API therefore runs on Vercel, the UI on Streamlit Community Cloud, and both share **one
service layer** (`src/forecast/service.py`) so they can never disagree. The UI also has a **local mode** that
works without the API.

### 10.3 API

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Status, model timestamp, data hash |
| GET | `/metrics` | Backtest results, business impact, segment parity |
| GET | `/model-card` | Features, intended use, limitations |
| POST | `/forecast` | 1–14-day forecast plus plan; accepts promotion and weather overrides |
| POST | `/explain` | Forecast plus an LLM briefing and its evaluation |

Inputs are validated with Pydantic (for example, a horizon above 14 returns HTTP 422). An optional `API_KEY`
header can protect non-public routes. Every request is logged with an ID and latency, but **request bodies are
never logged**.

### 10.4 Deployment issues and how I solved them

| Issue | Root cause | Fix |
|---|---|---|
| Streamlit: `No module named 'xgboost'` | `numpy==2.0.2` has no wheels for newer Python, so the whole install failed | Version ranges in `app/requirements.txt` |
| CI: `sklearn needs to be installed` | `XGBRegressor` imports scikit-learn behind the scenes | Switched to the native `Booster` API; metrics identical, bundle smaller |
| Two charts rendered blank | `£` is not a valid Vega number-format symbol; a NumPy 2 scalar repr broke a chart expression | `labelExpr` with a £ prefix; cast to `float` |
| Streamlit deprecation | `use_container_width` scheduled for removal | Helper that works on old and new versions |
| Vercel: every route 404 | The rewrite sent all requests to the app as `/api/index` | Rewrite passes `?__path=`; ASGI middleware restores the path |
| Vercel: 302 to login | Deployment protection enabled | Disabled for the public pilot; optional API key instead |
| API key in a tracked file | The key had been pasted into `.env.example` | Caught before commit; file restored; keys live only in `.env` and platform secrets |

### 10.5 Deployment evidence

Screenshots of the green CI run, Vercel deployment and logs, `/health`, `/docs`, every Streamlit tab, and a live
Claude briefing are stored in `docs/evidence/` and indexed in `docs/05_deployment_evidence.md`.

---

## 11. Testing, quality assurance and reproducibility

### 11.1 Automated tests (18, all passing)

| Area | Examples |
|---|---|
| Metrics | MAE and WAPE values, WAPE robust to zero days, bias sign |
| Baselines | Seasonal naive repeats last week correctly |
| Data | Synthetic history is identical across runs |
| Leakage | Every lag ≥ forecast horizon |
| Planning | Staffing minimums; closed days need no staff; buffer applied |
| API | `/health`, forecast shape, promotion uplift, HTTP 422 validation |
| LLM evaluation | Template briefing fully faithful; near-miss numbers flagged |
| Deployment | Vercel-rewritten paths restored; API-key check sees the restored path |

### 11.2 Continuous integration

On every push, GitHub Actions installs the pinned environment, **retrains the model**, compares the model hash
with the committed one, and runs the full test suite.

### 11.3 Reproducibility controls

- **Single seed:** `SEED = 42` controls the data, the model and the backtest.
- **Deterministic training:** one thread and the `hist` method.
- **Pinned versions:** `requirements.txt`, plus `.python-version` (3.12).
- **Data fingerprint:** the SHA-256 of the training data is stored in `model_meta.json` and returned by `/health`.
- **One command:** `python scripts/train.py` regenerates every artefact.
- **Verified:** two consecutive runs produced byte-identical models (`f239423b…`).

---

## 12. Privacy, ethics and fairness

### 12.1 Privacy (UK GDPR and the Data Protection Act 2018)

- **Data minimisation:** only daily aggregates are used. There are no customer, booking, card or employee records.
- **No personal data:** none is processed, so a DPIA is not required for the forecasting component. I re-checked
  this when adding the LLM layer.
- **LLM data flow:** only aggregated numbers are sent to providers, and there are no free-text fields in which
  personal data could leak. API accounts are used, not consumer accounts.
- **Secrets:** keys are kept only in a git-ignored `.env` and in platform secret stores.
- **Commercial sensitivity:** the public repository contains synthetic data only.

### 12.2 Fairness

The model forecasts **demand**, not people. Two fairness risks remain:

1. **Systematic under-forecasting of certain shift types** would repeatedly overload the staff working them. I
   check *error parity* across segments:

   | Segment | Days | MAE | WAPE |
   |---|---|---|---|
   | Weekday (Mon–Thu) | 96 | 14.3 | 10.7% |
   | Friday | 24 | 16.1 | 7.6% |
   | Weekend | 48 | 20.5 | 9.9% |
   | Spring/Autumn | 76 | 15.3 | 9.6% |
   | Summer | 92 | 17.2 | 10.0% |

   WAPE stays within a narrow band of 7.6–10.7%, so no segment is badly served. One gap remains: the backtest
   does not include winter or December, so these must be checked before the Christmas rota.
2. **Misuse of the headcount.** The tool outputs *how many* staff are needed. *Who* works stays with the manager,
   under HR and working-time policy. The tool should not be used to cut contracted hours without consultation.

### 12.3 Human oversight and transparency

Every recommendation is advisory and can be edited. The uncertainty range is always shown. The UI carries a
"decision support only" notice, and a model card is published at `/model-card`.

---

## 13. Pilot enhancement plan

### 13.1 Pilot design

- **Weeks −4 to 0 (baseline):** record the manager's own forecasts, actual covers, labour hours and the food
  waste log.
- **Weeks 1–8 (pilot):** the manager uses the tool for rotas and prep, logs any overrides and the reasons for
  them, and gives weekly feedback.
- **Week 9 (review):** compare the KPIs and decide whether to scale, iterate or stop.

### 13.2 KPIs

| KPI | Source | Target |
|---|---|---|
| WAPE (covers) | Tool vs actual | ≤ 11% |
| Labour as % of revenue | Payroll | −1.5 points |
| Food waste (kg/week) | Waste log | −20% |
| Time spent on the rota | Manager | −1 hour/week |
| Override rate | UI log | < 25% of days |
| Manager trust (1–5) | Survey | ≥ 4 |

### 13.3 Operating model

- **Monitoring:** weekly WAPE. Retrain when WAPE exceeds 15% for two consecutive weeks, or monthly.
- **Retraining:** run `scripts/train.py` and commit the new models. CI checks them and Vercel redeploys
  automatically. Rollback is one click ("Promote" the previous deployment).
- **Enhancements, in priority order:**
  1. Real POS integration.
  2. Live weather API.
  3. Bookings as a feature.
  4. Day-part (lunch/dinner) forecasting.
  5. A manager feedback loop on overrides.
  6. Multi-site roll-out with a site feature.

---

## 14. Limitations and risks

| Limitation / risk | Impact | Mitigation |
|---|---|---|
| Synthetic data | Results may not transfer to the real site | Pipeline accepts real data; the pilot validates it |
| Backtest uses actual weather | Live accuracy beyond about 5 days may be slightly worse | Measured in the pilot; weather API planned |
| No winter in the backtest window | December accuracy unverified | Extended backtest before Christmas |
| Cost assumptions | The £ saving depends on wage, food and walk-out rates | All assumptions in one config; recalibrated in the pilot |
| LLM errors | A wrong number in a briefing | Faithfulness scoring, a template fallback, and tables generated from the model, not the LLM |
| Only one LLM run | Three-way comparison incomplete | Keys only; no code changes needed |
| Concept drift (menu, refurbishment, competition) | Accuracy decays | Monitoring and a retraining trigger |
| Over-reliance on the tool | Less human judgement | Advisory design, visible uncertainty, training |
| Free-tier hosting | Cold starts and rate limits | Acceptable for a pilot; paid tier for production |

---

## 15. Reflection and lessons learned

**Baselines first.** Defining "same day last week" as the benchmark *before* modelling kept the project honest. A
9.8% WAPE means little on its own; "29% better than what you do today, in every fold" is something a manager can
act on.

**Design out leakage.** Choosing the direct multi-horizon design, and enforcing it with a test, removed a whole
class of subtle bugs that recursive forecasting invites.

**Deployment is where the hidden work is.** Most of the debugging happened after the model worked: dependency
wheels, an implicit scikit-learn import, chart format strings, serverless routing, platform authentication. Each
fix taught me to check assumptions in the *target* environment rather than only on my laptop. The diagnostic 404
response I added (showing the path the app actually received) turned a guessing game into a single test.

**Evaluate the evaluator.** The LLM rubric looked sound until real outputs exposed a tolerance that was too loose.
Reading model outputs, not just scores, improved both the pipeline and the metric.

**Security is a habit.** An API key nearly ended up in a tracked file. Keeping secrets only in `.env` and platform
stores, and checking diffs before each commit, is now part of my routine.

**What I would do differently:** start with a small real dataset if one were available, add a live weather
source from day one, and agree the cost assumptions with finance before building the business case.

---

## 16. KSB mapping

*Codes to be completed from my apprenticeship standard; the evidence column points to the artefacts.*

| KSB | Theme | Evidence |
|---|---|---|
| K__ | Data science / ML lifecycle | Sections 4–6; `scripts/train.py` |
| K__ | Statistical and ML methods, algorithm selection | Section 5.3; baselines in Section 6 |
| K__ | Evaluation and validation | Section 6; `backtest.py`, `metrics.py` |
| K__ | Data protection legislation and ethics | Section 12; `docs/04_privacy_fairness.md` |
| K__ | Fairness, bias and responsible AI | Section 12.2; segment parity |
| K__ | Software architecture and deployment | Section 10; `api/index.py`, `vercel.json` |
| K__ | Generative AI capabilities and limitations | Section 9; `src/forecast/llm/` |
| S__ | Acquire, validate and prepare data | Sections 4.1, 4.5; `data.py` |
| S__ | Feature engineering | Section 5.2; `features.py` |
| S__ | Build, tune and evaluate models | Sections 5–6; `model_report.md` |
| S__ | Deploy AI into production | Section 10; live URLs |
| S__ | Test and assure AI solutions | Section 11; 18 tests, CI |
| S__ | Communicate to non-technical stakeholders | Streamlit UI, LLM briefing, Section 8 |
| S__ | Evaluate and compare AI technologies | Section 9 |
| S__ | Quantify business value | Section 8; `operations.py` |
| B__ | Ethical and responsible practice | Section 12 |
| B__ | Reproducible, well-documented work | Section 11.3; `docs/` |
| B__ | Continuous learning and reflection | Section 15 |
| B__ | Stakeholder focus | Sections 2.4, 13 |

---

## 17. Conclusion and recommendations

The project shows that a lightweight, well-evaluated machine-learning model can materially improve demand
planning at a single restaurant. The XGBoost forecast cuts forecast error by **29%** against current practice,
wins in **every** backtest fold, and keeps its accuracy across the whole two-week rota horizon. Translated into
staffing and prep, it is worth about **£20k a year** in simulation. A generative AI layer turns the numbers into a
faithful, readable briefing for about two cents each. The whole system is deployed, tested, reproducible and
designed with privacy and human oversight in mind.

**Recommendations**

1. Approve an **8-week live pilot** using real POS data, with the KPIs in Section 13.2.
2. Calibrate the staffing ratios and cost assumptions with the site manager and finance before the pilot.
3. Add a live weather feed and the bookings data as the first enhancements.
4. Complete the three-provider LLM comparison when API access allows, and select one primary provider and one
   fallback.
5. Review results in week 9 and decide on a multi-site roll-out.

---

## 18. References

1. Chen, T. and Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System.* Proceedings of KDD '16.
2. Hyndman, R.J. and Athanasopoulos, G. (2021). *Forecasting: Principles and Practice*, 3rd ed. OTexts.
3. Taieb, S.B. et al. (2012). *A review and comparison of strategies for multi-step ahead time series
   forecasting.* Expert Systems with Applications 39(8).
4. Information Commissioner's Office. *Guide to the UK GDPR* and *Guidance on AI and data protection*.
5. FastAPI documentation, https://fastapi.tiangolo.com; Vercel FastAPI documentation,
   https://vercel.com/docs/frameworks/backend/fastapi; Streamlit documentation, https://docs.streamlit.io.
6. Anthropic, OpenAI and Google API documentation (model and pricing pages, accessed November 2026).

---

## 19. Appendices

### Appendix A: Repository structure

```
api/index.py               FastAPI app (Vercel entry point)
app/streamlit_app.py       Streamlit UI (API or local mode)
src/forecast/              config, data, features, model, baselines, metrics, backtest, operations, service
src/forecast/llm/          LLM providers and evaluation rubric
scripts/train.py           End-to-end reproducible pipeline
scripts/compare_llms.py    LLM comparison
data/  models/  reports/   Committed artefacts (evidence)
docs/                      Supporting documents 01–08 and this report
tests/                     18 automated tests
.github/workflows/ci.yml   Continuous integration
```

### Appendix B: Reproducing the results

```bash
git clone https://github.com/RenukaTkd/Level-7-restaurant-project.git
cd Level-7-restaurant-project
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
python scripts/train.py        # data → backtest → model → reports
pytest -q                      # 18 tests
python scripts/compare_llms.py # needs API keys in .env
```

### Appendix C: Example API calls

```bash
URL=https://level-7-restaurant-project-renukas-projects-bea33a15.vercel.app
curl -s $URL/health
curl -s -X POST $URL/forecast -H 'content-type: application/json' \
     -d '{"horizon":7,"overrides":{"2026-11-03":{"promo":1}}}'
curl -s -X POST $URL/explain -H 'content-type: application/json' -d '{"provider":"claude"}'
```

### Appendix D: Glossary

| Term | Meaning |
|---|---|
| Cover | One guest served |
| MAE | Mean absolute error, the average size of a miss |
| WAPE | Weighted absolute percentage error, total miss ÷ total actual |
| Backtest | Re-running past forecasts as if made at the time |
| Rolling origin | A backtest that repeatedly moves the forecast start date |
| Data leakage | Using information not available at forecast time |
| P10 / P90 | A 10% / 90% chance the actual value falls below this |
| LLM | Large language model |
| Faithfulness | Whether every number in a briefing exists in the source data |
| Serverless | Code that runs on demand without a managed server |

### Appendix E: Supporting documents

`docs/01_project_overview.md`, `docs/02_baseline_comparison.md`, `docs/03_reproducibility.md`,
`docs/04_privacy_fairness.md`, `docs/05_deployment_evidence.md`, `docs/06_business_impact.md`,
`docs/07_ksb_mapping.md`, `docs/08_llm_comparison.md`, `reports/model_report.md`.
