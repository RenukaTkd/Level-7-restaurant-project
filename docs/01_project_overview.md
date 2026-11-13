# 1. Project overview

## Business problem
The pilot restaurant (≈120–250 covers a day) plans staff rotas two weeks ahead and does food prep each morning.
Today the manager copies **"same day last week"** and adjusts it by gut feel. Because of that:
- **Over-staffing** on quiet days wastes wages.
- **Under-staffing** on busy days (paydays, events, warm evenings) means slow service and walk-outs.
- **Over-prep** produces food waste, and **under-prep** leads to 86'd dishes.

## Objective
Give the manager a **14-day forecast of covers and revenue**, with an uncertainty range, and turn it into a
**headcount and prep plan**. Success criteria agreed for the pilot:

| Criterion | Target | Achieved (backtest) |
|---|---|---|
| WAPE on covers better than current practice | ≥ 15% relative improvement | **29%** (13.9% → 9.8%) |
| MAE on covers | < 20 covers/day | **16.3** |
| Simulated annual cost reduction | > £10k | **≈ £20k** |
| API availability during pilot | ≥ 99% | see deployment evidence |

## Solution
1. **Data**: daily history for one site (covers, revenue, promotions, weather, closures), plus a calendar
   (bank holidays, special events, paydays).
2. **Model**: XGBoost with a *direct* multi-horizon design. Every lag is at least 14 days old, so one model can
   forecast any day in the next two weeks with no look-ahead leakage.
3. **Evaluation**: rolling-origin backtest against three baselines, scored on MAE and WAPE.
4. **Operations layer**: covers plus a safety buffer (z = 0.84 × residual σ ≈ 80% service level) gives the
   front-of-house and kitchen headcount and the prep portions for each item.
5. **Delivery**: FastAPI on Vercel and Streamlit UI on Streamlit Cloud, with code and CI on GitHub.
6. **GenAI layer**: Claude, GPT and Gemini each turn the plan into a plain-English briefing. They are compared on
   faithfulness, coverage, latency and cost.

## Scope and limitations
- One site only. Another site needs its own data and retraining.
- The backtest uses the actual promo and weather values for the forecast window (promos are planned in advance;
  weather forecasts are reasonably accurate up to about 5 days). Live accuracy beyond day 5 may be slightly worse,
  which the pilot will measure.
- The data is synthetic, so the business-impact figures are a simulation and must be confirmed during the pilot.
