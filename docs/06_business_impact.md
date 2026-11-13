# 6. Business impact

## How it is calculated
For every backtest day, each method's covers forecast is turned into a staffing plan. That plan is costed against
what actually happened:

| Cost component | Formula (assumptions in `config.py`) |
|---|---|
| Over-staffing | surplus staff × 8 h × £12.60 |
| Food waste | over-forecast covers × £6.50 ingredient cost |
| Lost margin | covers beyond capacity × 30% walk-out rate × £28 spend × 65% margin |

Costs are annualised (× 365 / backtest days). No safety buffer is applied, so the methods are compared fairly.

## Results (annual, GBP)

| Method | Over-staffing | Waste | Lost margin | **Total** | **Saving vs current** |
|---|---|---|---|---|---|
| **XGBoost** | 13,140 | 15,082 | 7,011 | **35,233** | **£20,290** |
| Weekday average | 16,425 | 21,903 | 7,983 | 46,312 | £9,211 |
| Seasonal naive (current) | 18,834 | 25,716 | 10,973 | 55,523 | — |
| Moving average | 36,135 | 45,402 | 26,904 | 108,441 | −£52,918 |

## Narrative for stakeholders
- About **£20k a year** of avoidable cost is removed at one site. That is **37%** of the cost of forecasting errors
  under current practice.
- Labour: about 450 fewer surplus paid hours a year, which can go to busy shifts instead.
- Waste: about 1,600 fewer over-prepped covers a year (≈ 41% less), which supports sustainability reporting.
- Service: lost margin from under-staffing falls by 36%.
- **Running cost:** Vercel and Streamlit Cloud free/hobby tiers are £0. LLM briefings cost well under $0.10 each
  (see LLM comparison). That is negligible next to the savings.

## Pilot KPIs (to validate the simulation in production)
| KPI | Baseline (4 weeks before) | Target during 8-week pilot |
|---|---|---|
| WAPE (covers) | measure manual forecast | ≤ 11% |
| Labour % of revenue | from payroll | −1.5 pts |
| Food waste (kg/week) | waste log | −20% |
| Manager time on rota (h/week) | self-reported | −1 h |
| Manager trust score (1–5) | — | ≥ 4 |

*Caveat: the figures come from a simulation on synthetic data with assumed cost rates. The pilot exists to confirm
them with real payroll and waste logs.*
