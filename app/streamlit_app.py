"""Streamlit UI for restaurant managers.

Runs in two modes:
  * API mode   — set API_URL (env var or Streamlit secret) to the Vercel FastAPI URL.
  * Local mode — no API_URL: imports the same service layer directly.

    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import os
import sys
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
import requests  # noqa: E402
import streamlit as st  # noqa: E402

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

st.set_page_config(page_title="Restaurant Demand Forecast", page_icon="🍽️", layout="wide")


def secret(name: str, default: str = "") -> str:
    try:
        return st.secrets.get(name, os.getenv(name, default))
    except Exception:
        return os.getenv(name, default)


API_URL = secret("API_URL").rstrip("/")
API_KEY = secret("API_KEY")
for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY"):   # local mode LLM keys
    if secret(k) and not os.getenv(k):
        os.environ[k] = secret(k)


# ---- data access (API or local) ----------------------------------------------
def api(method: str, path: str, payload: dict | None = None) -> dict:
    r = requests.request(method, f"{API_URL}{path}", json=payload, timeout=120,
                         headers={"x-api-key": API_KEY} if API_KEY else {})
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=600)
def get_metrics() -> dict:
    if API_URL:
        return api("GET", "/metrics")
    from forecast import service
    return service.metrics()


def get_forecast(horizon: int, overrides: dict) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    if API_URL:
        res = api("POST", "/forecast", {"horizon": horizon, "overrides": overrides})
        fc, pl = pd.DataFrame(res["forecast"]), pd.DataFrame(res["plan"])
        fc["date"], pl["date"] = pd.to_datetime(fc["date"]), pd.to_datetime(pl["date"])
        return fc, pl, res["model_wape"]
    from forecast import service
    fc, pl = service.forecast_and_plan(horizon, overrides)
    return fc, pl, service.covers_wape()


def get_briefing(provider: str, horizon: int, overrides: dict) -> dict:
    if API_URL:
        return api("POST", "/explain", {"horizon": horizon, "overrides": overrides, "provider": provider})
    from forecast import service
    from forecast.llm import build_facts, generate, score
    fc, pl = service.forecast_and_plan(horizon, overrides)
    facts = build_facts(fc, pl, service.covers_wape())
    res = generate(provider, facts)
    return {"briefing": res.text or res.status, "evaluation": score(res, facts)}


@st.cache_data
def history() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "sales_history.csv", parse_dates=["date"])


# ---- sidebar: what-if inputs ----------------------------------------------------
st.sidebar.title("🍽️ Forecast settings")
st.sidebar.caption(f"Mode: **{'API → ' + API_URL if API_URL else 'local'}**")
horizon = st.sidebar.slider("Days ahead", 1, 14, 14)
hist = history()
start = hist["date"].max() + timedelta(days=1)
dates = [start + timedelta(days=i) for i in range(horizon)]
fmt = lambda d: d.strftime("%a %d %b")  # noqa: E731
promo_days = st.sidebar.multiselect("Planned promotion days", dates, format_func=fmt)
rain_days = st.sidebar.multiselect("Days with rain forecast", dates, format_func=fmt)
use_temp = st.sidebar.checkbox("Override temperature for all days", value=False)
temp = st.sidebar.slider("Expected max temperature (°C)", -5, 35, 15) if use_temp else None

overrides: dict[str, dict] = {}
for d in dates:
    o = {}
    if d in promo_days:
        o["promo"] = 1
    if d in rain_days:
        o["rain"] = 1
    if use_temp and temp is not None:
        o["temp_c"] = temp
    if o:
        overrides[d.strftime("%Y-%m-%d")] = o

st.title("AI Restaurant Sales Forecasting — pilot")
st.caption("XGBoost forecast of covers & revenue → staffing and food-prep plan. "
           "Decision support only: the manager approves every rota.")

tab_fc, tab_bt, tab_imp, tab_llm, tab_gov = st.tabs(
    ["📈 Forecast & plan", "🧪 Backtest vs baselines", "💷 Business impact", "🤖 LLM briefing (3 models)", "🛡️ Governance"])

try:
    fc, pl, wape = get_forecast(horizon, overrides)
    metrics = get_metrics()
except Exception as e:  # pragma: no cover
    st.error(f"Could not load forecast: {e}")
    st.stop()

with tab_fc:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Expected covers", f"{fc['covers'].sum():,.0f}")
    c2.metric("Expected revenue", f"£{fc['revenue'].sum():,.0f}")
    c3.metric("Labour hours planned", f"{pl['labour_hours'].sum():,.0f}")
    c4.metric("Model WAPE (backtest)", f"{wape:.1%}")

    recent = hist[hist["date"] > hist["date"].max() - timedelta(days=42)][["date", "covers"]]
    chart = pd.concat([recent.rename(columns={"covers": "Actual covers"}),
                       fc[["date", "covers", "covers_low", "covers_high"]].rename(columns={
                           "covers": "Forecast", "covers_low": "Low (P10)", "covers_high": "High (P90)"})])
    st.line_chart(chart.set_index("date"), height=320)

    st.subheader("Staffing & prep plan")
    show = pl.copy()
    show.insert(0, "day", show["date"].dt.strftime("%a %d %b"))
    st.dataframe(show.drop(columns=["date"]), hide_index=True, use_container_width=True)
    st.download_button("Download plan (CSV)", show.to_csv(index=False), "staffing_prep_plan.csv")

with tab_bt:
    target = st.radio("Target", ["covers", "revenue"], horizontal=True)
    res = pd.DataFrame(metrics["backtest"][target]).T
    st.dataframe(res.style.format({"MAE": "{:.2f}", "WAPE": "{:.1%}", "RMSE": "{:.2f}", "Bias": "{:+.2f}",
                                   "WAPE_improvement_vs_naive_%": "{:+.1f}%"}), use_container_width=True)
    st.bar_chart(res["WAPE"], horizontal=True, height=220)
    st.caption("Rolling-origin backtest: model retrained at each origin, forecasting the next 14 days. "
               "Lower MAE / WAPE is better. 'Seasonal naive' = same day last week (current manual practice).")
    bt_path = ROOT / "reports" / f"backtest_{target}.csv"
    if bt_path.exists():
        bt = pd.read_csv(bt_path, parse_dates=["date"])
        piv = bt.pivot_table(index="date", columns="method", values="forecast").join(
            bt.drop_duplicates("date").set_index("date")["actual"])
        st.line_chart(piv[["actual", "XGBoost", "Seasonal naive (last week)"]], height=300)

with tab_imp:
    imp = pd.DataFrame(metrics["business_impact"]).T
    st.dataframe(imp.style.format("£{:,.0f}"), use_container_width=True)
    best = imp.loc["XGBoost", "annual_saving_vs_naive_gbp"]
    st.success(f"Simulated annual saving vs current practice: **£{best:,.0f}** "
               "(over-staffing + food waste + lost margin from under-staffing).")
    st.caption("Assumptions are editable in src/forecast/config.py. Figures come from a simulation on "
               "synthetic history and must be validated during the live pilot.")

with tab_llm:
    st.write("Each model turns the same forecast facts into a manager briefing. Scores are automatic: "
             "numeric faithfulness (no invented numbers), coverage of required sections, length, plus latency & cost.")
    providers = st.multiselect("Models to compare", ["claude", "openai", "gemini", "template"],
                               default=["claude", "openai", "gemini", "template"])
    if st.button("Generate briefings", type="primary"):
        cols = st.columns(max(len(providers), 1))
        scores = []
        for col, p in zip(cols, providers):
            with col, st.spinner(p):
                try:
                    out = get_briefing(p, horizon, overrides)
                    ev = out["evaluation"]
                    col.markdown(f"**{p}** · `{ev['model']}`")
                    col.write(out["briefing"])
                    scores.append(ev)
                except Exception as e:
                    col.error(f"{p}: {e}")
        if scores:
            st.dataframe(pd.DataFrame(scores), hide_index=True, use_container_width=True)
    summ = ROOT / "reports" / "llm_summary.csv"
    if summ.exists():
        st.subheader("Offline comparison (scripts/compare_llms.py, 4 scenarios)")
        st.dataframe(pd.read_csv(summ), hide_index=True, use_container_width=True)

with tab_gov:
    st.subheader("Error parity across operating segments")
    st.dataframe(pd.DataFrame(metrics["segment_errors"]).style.format(
        {"MAE": "{:.2f}", "WAPE": "{:.1%}", "RMSE": "{:.2f}", "Bias": "{:+.2f}"}),
        hide_index=True, use_container_width=True)
    st.markdown("""
- **Privacy:** only daily aggregate sales are used — no customer or employee personal data. LLMs receive aggregated
  forecast numbers only.
- **Fairness:** staffing output is a *headcount*, not a decision about individuals; who works is decided by the
  manager under existing HR policy. Error parity is monitored so no day-part is systematically under-staffed.
- **Human in the loop:** recommendations are advisory; the manager can override every number.
- See `docs/` for the full privacy/fairness review, reproducibility steps and KSB mapping.
""")
