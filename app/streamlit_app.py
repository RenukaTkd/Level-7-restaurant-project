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

import altair as alt  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402
import streamlit as st  # noqa: E402

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

st.set_page_config(page_title="Riverside Kitchen · Demand Forecast", page_icon="🍽️", layout="wide")

# ---- Palette (validated categorical slots + neutral) --------------------------------
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GREY = "#8a8984"          # neutral context series (history, baselines)
BAND = "#86b6ef"          # sequential blue step 250 for the P10–P90 band

st.markdown("""
<style>
.block-container {padding-top: 3.2rem; max-width: 1400px;}
.hero {
  background: linear-gradient(120deg, #1c5cab 0%, #2a78d6 55%, #3987e5 100%);
  color: #fff; border-radius: 18px; padding: 26px 30px; margin-bottom: 18px;
  box-shadow: 0 8px 24px rgba(28, 92, 171, .22);
}
.hero h1 {color: #fff; font-size: 1.9rem; margin: 0 0 4px 0; padding: 0; font-weight: 700;}
.hero p {margin: 0; opacity: .92; font-size: 1rem;}
.pill {display: inline-block; background: rgba(255,255,255,.16); border: 1px solid rgba(255,255,255,.28);
  border-radius: 999px; padding: 3px 12px; margin: 12px 8px 0 0; font-size: .82rem; color: #fff;}
.kpi {border: 1px solid rgba(128,128,128,.22); border-radius: 14px; padding: 16px 18px;
  background: rgba(128,128,128,.04); height: 100%;}
.kpi .label {font-size: .8rem; text-transform: uppercase; letter-spacing: .06em; opacity: .7;}
.kpi .value {font-size: 1.75rem; font-weight: 700; margin-top: 4px; line-height: 1.2;}
.kpi .sub {font-size: .85rem; opacity: .7; margin-top: 2px;}
.section {font-size: 1.1rem; font-weight: 650; margin: 18px 0 2px 0;}
.caption {font-size: .85rem; opacity: .7; margin-bottom: 8px;}
.saving {border-radius: 16px; padding: 22px 26px; border: 1px solid rgba(12,163,12,.35);
  background: rgba(12,163,12,.07);}
.saving .big {font-size: 2.6rem; font-weight: 750; line-height: 1.1;}
.llm-card {border: 1px solid rgba(128,128,128,.22); border-radius: 14px; padding: 16px; height: 100%;}
.badge {display:inline-block; border-radius: 999px; padding: 2px 10px; font-size: .78rem; margin-right: 6px;
  border: 1px solid rgba(128,128,128,.35);}
div[data-testid="stTabs"] button p {font-size: 0.98rem; font-weight: 600;}
</style>
""", unsafe_allow_html=True)


def secret(name: str, default: str = "") -> str:
    try:
        return st.secrets.get(name, os.getenv(name, default))
    except Exception:
        return os.getenv(name, default)


API_URL = secret("API_URL").rstrip("/")
API_KEY = secret("API_KEY")
for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY"):   # local-mode LLM keys
    if secret(k) and not os.getenv(k):
        os.environ[k] = secret(k)


# ---- data access (API or local) -------------------------------------------------------
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


@st.cache_data
def backtest(target: str) -> pd.DataFrame | None:
    p = ROOT / "reports" / f"backtest_{target}.csv"
    return pd.read_csv(p, parse_dates=["date"]) if p.exists() else None


def show(chart, **_) -> None:
    """Full-width Altair chart on both older (use_container_width) and newer (width=) Streamlit."""
    try:
        st.altair_chart(chart, width="stretch")
    except TypeError:
        st.altair_chart(chart, use_container_width=True)


def kpi(col, label: str, value: str, sub: str = "") -> None:
    col.markdown(f'<div class="kpi"><div class="label">{label}</div><div class="value">{value}</div>'
                 f'<div class="sub">{sub}</div></div>', unsafe_allow_html=True)


def section(title: str, caption: str = "") -> None:
    st.markdown(f'<div class="section">{title}</div>' + (f'<div class="caption">{caption}</div>' if caption else ""),
                unsafe_allow_html=True)


# ---- sidebar: what-if inputs ---------------------------------------------------------
hist = history()
with st.sidebar:
    st.markdown("### 🍽️ Plan the next fortnight")
    horizon = st.slider("Days ahead", 1, 14, 14)
    start = hist["date"].max() + timedelta(days=1)
    dates = [start + timedelta(days=i) for i in range(horizon)]
    fmt = lambda d: d.strftime("%a %d %b")  # noqa: E731
    st.markdown("**What-if scenario**")
    promo_days = st.multiselect("📣 Promotion days", dates, format_func=fmt)
    rain_days = st.multiselect("🌧️ Rain forecast", dates, format_func=fmt)
    use_temp = st.toggle("🌡️ Override temperature", value=False)
    temp = st.slider("Max temperature (°C)", -5, 35, 15) if use_temp else None
    st.divider()
    st.caption(f"Connection: **{'API · ' + API_URL if API_URL else 'Local model'}**")
    st.caption("Decision support only — the manager approves every rota.")

overrides: dict[str, dict] = {}
for d in dates:
    o = {}
    if d in promo_days:
        o["promo"] = 1
    if d in rain_days:
        o["rain"] = 1
    if temp is not None:
        o["temp_c"] = temp
    if o:
        overrides[d.strftime("%Y-%m-%d")] = o

try:
    fc, pl, wape = get_forecast(horizon, overrides)
    metrics = get_metrics()
except Exception as e:  # pragma: no cover
    st.error(f"Could not load forecast: {e}")
    st.stop()

# ---- hero --------------------------------------------------------------------------------
xgb_cov = metrics["backtest"]["covers"]["XGBoost"]
st.markdown(f"""
<div class="hero">
  <h1>The Riverside Kitchen · Demand Forecast</h1>
  <p>AI forecast of covers &amp; revenue, turned into a staffing rota and a daily prep list.</p>
  <span class="pill">🧠 XGBoost</span>
  <span class="pill">🎯 Backtest WAPE {wape:.1%}</span>
  <span class="pill">📉 {xgb_cov['WAPE_improvement_vs_naive_%']:.0f}% less error than "same day last week"</span>
  <span class="pill">📅 History to {hist['date'].max():%d %b %Y}</span>
  {'<span class="pill">✨ Scenario active</span>' if overrides else ''}
</div>
""", unsafe_allow_html=True)

open_days = fc[fc["covers"] > 0]
peak = open_days.loc[open_days["covers"].idxmax()]
c1, c2, c3, c4 = st.columns(4)
kpi(c1, "Expected covers", f"{fc['covers'].sum():,.0f}", f"next {horizon} days · ~{open_days['covers'].mean():.0f}/day")
kpi(c2, "Expected revenue", f"£{fc['revenue'].sum():,.0f}",
    f"range £{fc['revenue_low'].sum():,.0f} – £{fc['revenue_high'].sum():,.0f}")
kpi(c3, "Labour hours", f"{pl['labour_hours'].sum():,.0f}", f"{(pl['foh_staff'] + pl['kitchen_staff']).sum():,} shifts")
kpi(c4, "Busiest day", f"{peak['date']:%a %d %b}", f"{peak['covers']:.0f} covers expected")
st.write("")

tab_fc, tab_bt, tab_imp, tab_llm, tab_gov = st.tabs(
    ["📈 Forecast & plan", "🧪 Accuracy vs baselines", "💷 Business impact", "🤖 AI briefing · 3 LLMs", "🛡️ Governance"])

# ---- Forecast & plan --------------------------------------------------------------------
with tab_fc:
    section("Covers: last 6 weeks and the forecast",
            "Shaded band = likely range (P10–P90). Hover for daily values.")
    recent = hist[hist["date"] > hist["date"].max() - timedelta(days=42)][["date", "covers"]]
    lines = pd.concat([recent.assign(series="Actual"),
                       fc[["date", "covers"]].assign(series="Forecast")])
    color = alt.Scale(domain=["Actual", "Forecast"], range=[GREY, BLUE])
    x = alt.X("date:T", title=None, axis=alt.Axis(format="%d %b", grid=False))
    band = alt.Chart(fc).mark_area(opacity=0.28, color=BAND).encode(
        x=x, y=alt.Y("covers_low:Q", title="Covers"), y2="covers_high:Q")
    line = alt.Chart(lines).mark_line(strokeWidth=2).encode(
        x=x, y="covers:Q", color=alt.Color("series:N", scale=color, legend=alt.Legend(title=None, orient="top")))
    pts = alt.Chart(fc).mark_point(filled=True, size=70, color=BLUE, stroke="white", strokeWidth=2).encode(
        x=x, y="covers:Q")
    today = alt.Chart(pd.DataFrame({"date": [hist["date"].max()]})).mark_rule(
        strokeDash=[4, 4], color=GREY).encode(x="date:T")
    hover_src = lines.merge(fc[["date", "covers_low", "covers_high"]], on="date", how="left")
    hover = alt.selection_point(fields=["date"], nearest=True, on="pointerover", empty=False)
    hit = alt.Chart(hover_src).mark_rule(color=GREY, strokeWidth=1).encode(
        x=x, opacity=alt.condition(hover, alt.value(0.6), alt.value(0)),
        tooltip=[alt.Tooltip("date:T", format="%a %d %b %Y", title="Date"), alt.Tooltip("series:N", title=""),
                 alt.Tooltip("covers:Q", format=".0f", title="Covers"),
                 alt.Tooltip("covers_low:Q", format=".0f", title="Low (P10)"),
                 alt.Tooltip("covers_high:Q", format=".0f", title="High (P90)")]).add_params(hover)
    show((band + line + pts + today + hit).properties(height=340), width="stretch")

    left, right = st.columns([1.15, 1])
    with left:
        section("Staff per day", "Headcount needed, including a safety buffer for forecast error.")
        staff = pl[["date", "foh_staff", "kitchen_staff"]].melt("date", var_name="role", value_name="staff")
        staff["role"] = staff["role"].map({"foh_staff": "Front of house", "kitchen_staff": "Kitchen"})
        show(alt.Chart(staff).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4,
                                                  stroke="white", strokeWidth=2).encode(
            x=alt.X("yearmonthdate(date):O", title=None, axis=alt.Axis(format="%a %d", labelAngle=0)),
            y=alt.Y("staff:Q", title="Staff", stack="zero"),
            color=alt.Color("role:N", scale=alt.Scale(domain=["Front of house", "Kitchen"], range=[BLUE, ORANGE]),
                            legend=alt.Legend(title=None, orient="top")),
            order=alt.Order("role:N", sort="descending"),
            tooltip=[alt.Tooltip("date:T", format="%a %d %b"), "role:N", "staff:Q"],
        ).properties(height=280), width="stretch")
    with right:
        section("Prep list", "Portions to prepare each day (darker = more).")
        prep_cols = [c for c in pl.columns if c.startswith("prep: ")]
        prep = pl[["date"] + prep_cols].melt("date", var_name="item", value_name="portions")
        prep["item"] = prep["item"].str.replace("prep: ", "", regex=False)
        heat = alt.Chart(prep).encode(
            x=alt.X("yearmonthdate(date):O", title=None, axis=alt.Axis(format="%d", labelAngle=0)),
            y=alt.Y("item:N", title=None, axis=alt.Axis(labelLimit=180)))
        show((heat.mark_rect(cornerRadius=3, stroke="white", strokeWidth=2).encode(
            color=alt.Color("portions:Q", scale=alt.Scale(range=["#cde2fb", "#104281"]), legend=None),
            tooltip=[alt.Tooltip("date:T", format="%a %d %b"), "item:N", "portions:Q"])
            + heat.mark_text(fontSize=10).encode(
                text="portions:Q",
                color=alt.condition(alt.datum.portions > float(prep["portions"].quantile(0.55)),
                                    alt.value("white"), alt.value("#0b0b0b")))
        ).properties(height=280), width="stretch")

    section("Daily plan")
    table = pl.merge(fc[["date", "revenue", "covers_low", "covers_high", "promo", "rain"]], on="date")
    table.insert(0, "Day", table["date"].dt.strftime("%a %d %b"))
    table["Flags"] = (table["promo"].map({1: "📣 promo", 0: ""}) + " "
                      + table["rain"].apply(lambda r: "🌧️ rain" if r >= 1 else "")).str.strip()
    view = table[["Day", "covers", "covers_low", "covers_high", "revenue", "planned_covers",
                  "foh_staff", "kitchen_staff", "labour_hours", "Flags"] + prep_cols]
    st.dataframe(
        view, hide_index=True, width="stretch",
        column_config={
            "covers": st.column_config.ProgressColumn("Covers", format="%.0f", min_value=0,
                                                      max_value=float(fc["covers_high"].max())),
            "covers_low": st.column_config.NumberColumn("Low", format="%.0f"),
            "covers_high": st.column_config.NumberColumn("High", format="%.0f"),
            "revenue": st.column_config.NumberColumn("Revenue", format="£%.0f"),
            "planned_covers": st.column_config.NumberColumn("Plan for", format="%.0f"),
            "foh_staff": st.column_config.NumberColumn("FOH"),
            "kitchen_staff": st.column_config.NumberColumn("Kitchen"),
            "labour_hours": st.column_config.NumberColumn("Hours"),
            **{c: st.column_config.NumberColumn(c.replace("prep: ", "")) for c in prep_cols},
        })
    st.download_button("⬇️ Download plan (CSV)", view.to_csv(index=False), "staffing_prep_plan.csv",
                       type="primary")

# ---- Accuracy vs baselines ------------------------------------------------------------
with tab_bt:
    target = st.segmented_control("Target", ["covers", "revenue"], default="covers") \
        if hasattr(st, "segmented_control") else st.radio("Target", ["covers", "revenue"], horizontal=True)
    target = target or "covers"
    res = pd.DataFrame(metrics["backtest"][target]).T.reset_index(names="method")
    naive = res.set_index("method").loc["Seasonal naive (last week)"]
    best = res.set_index("method").loc["XGBoost"]
    unit = "£" if target == "revenue" else ""
    k1, k2, k3 = st.columns(3)
    kpi(k1, "XGBoost WAPE", f"{best['WAPE']:.1%}", f"vs {naive['WAPE']:.1%} for same-day-last-week")
    kpi(k2, "XGBoost MAE", f"{unit}{best['MAE']:,.1f}", f"vs {unit}{naive['MAE']:,.1f} per day")
    kpi(k3, "Error reduction", f"{best['WAPE_improvement_vs_naive_%']:.0f}%", "12 folds × 14 days, retrained each fold")

    a, b = st.columns([1, 1.25])
    with a:
        section("WAPE by method", "Lower is better.")
        res["is_model"] = res["method"].eq("XGBoost")
        base = alt.Chart(res).encode(y=alt.Y("method:N", sort="x", title=None, axis=alt.Axis(labelLimit=220)),
                                     x=alt.X("WAPE:Q", axis=alt.Axis(format="%"), title=None,
                                             scale=alt.Scale(domain=[0, float(res["WAPE"].max()) * 1.18])))
        show((base.mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4, height=26).encode(
            color=alt.condition(alt.datum.is_model, alt.value(BLUE), alt.value(GREY)),
            tooltip=["method:N", alt.Tooltip("WAPE:Q", format=".1%"), alt.Tooltip("MAE:Q", format=",.1f")])
            + base.mark_text(align="left", dx=6, fontSize=12).encode(text=alt.Text("WAPE:Q", format=".1%"))
        ).properties(height=220), width="stretch")
    bt = backtest(target)
    with b:
        if bt is not None:
            section("Error by days ahead", "WAPE at each forecast horizon day.")
            sub = bt[bt["method"].isin(["XGBoost", "Seasonal naive (last week)"])]
            byh = (sub.assign(ae=(sub["actual"] - sub["forecast"]).abs())
                   .groupby(["method", "horizon_day"]).agg(ae=("ae", "sum"), act=("actual", "sum")).reset_index())
            byh["WAPE"] = byh["ae"] / byh["act"]
            show(alt.Chart(byh).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=60, filled=True)).encode(
                x=alt.X("horizon_day:O", title="Days ahead", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("WAPE:Q", axis=alt.Axis(format="%"), title=None),
                color=alt.Color("method:N", scale=alt.Scale(domain=["XGBoost", "Seasonal naive (last week)"],
                                                            range=[BLUE, ORANGE]),
                                legend=alt.Legend(title=None, orient="top")),
                tooltip=["method:N", "horizon_day:O", alt.Tooltip("WAPE:Q", format=".1%")],
            ).properties(height=220), width="stretch")

    if bt is not None:
        section("Backtest: actual vs forecasts", "Each 14-day block was forecast before it happened.")
        piv = bt[bt["method"].isin(["XGBoost", "Seasonal naive (last week)"])][["date", "method", "forecast"]]
        act = bt.drop_duplicates("date")[["date", "actual"]].rename(columns={"actual": "forecast"}).assign(method="Actual")
        ser = pd.concat([act, piv])
        show(alt.Chart(ser).mark_line(strokeWidth=2).encode(
            x=alt.X("date:T", title=None, axis=alt.Axis(format="%b %Y", grid=False)),
            y=alt.Y("forecast:Q", title=target.capitalize()),
            color=alt.Color("method:N", scale=alt.Scale(domain=["Actual", "XGBoost", "Seasonal naive (last week)"],
                                                        range=[GREY, BLUE, ORANGE]),
                            legend=alt.Legend(title=None, orient="top")),
            strokeDash=alt.condition(alt.datum.method == "Seasonal naive (last week)",
                                     alt.value([5, 4]), alt.value([1, 0])),
            tooltip=[alt.Tooltip("date:T", format="%a %d %b %Y"), "method:N",
                     alt.Tooltip("forecast:Q", format=",.0f", title=target)],
        ).properties(height=300).interactive(bind_y=False), width="stretch")

    with st.expander("Full metrics table"):
        st.dataframe(pd.DataFrame(metrics["backtest"][target]).T.style.format(
            {"MAE": "{:,.2f}", "WAPE": "{:.1%}", "RMSE": "{:,.2f}", "Bias": "{:+.2f}",
             "WAPE_improvement_vs_naive_%": "{:+.1f}%"}), width="stretch")

# ---- Business impact ----------------------------------------------------------------------
with tab_imp:
    imp = pd.DataFrame(metrics["business_impact"]).T
    saving = imp.loc["XGBoost", "annual_saving_vs_naive_gbp"]
    naive_cost = imp.loc["Seasonal naive (last week)", "total_cost"]
    s1, s2 = st.columns([1, 1.6])
    with s1:
        st.markdown(f"""<div class="saving"><div class="label" style="opacity:.75">✅ Simulated annual saving</div>
        <div class="big">£{saving:,.0f}</div>
        <div style="opacity:.8;margin-top:6px">{saving / naive_cost:.0%} less cost from forecast errors than
        planning with "same day last week".</div></div>""", unsafe_allow_html=True)
        st.write("")
        r1, r2 = st.columns(2)
        kpi(r1, "Over-staffing", f"−£{imp.loc['Seasonal naive (last week)', 'overstaff_cost'] - imp.loc['XGBoost', 'overstaff_cost']:,.0f}", "per year")
        kpi(r2, "Food waste", f"−£{imp.loc['Seasonal naive (last week)', 'waste_cost'] - imp.loc['XGBoost', 'waste_cost']:,.0f}", "per year")
    with s2:
        section("Annual cost of forecast errors", "Lower is better. Hover for each component.")
        comp = (imp[["overstaff_cost", "waste_cost", "lost_margin"]].reset_index(names="method")
                .melt("method", var_name="component", value_name="gbp"))
        comp["label"] = comp["gbp"].map(lambda v: f"£{v:,.0f}")
        comp["component"] = comp["component"].map({"overstaff_cost": "Over-staffing", "waste_cost": "Food waste",
                                                   "lost_margin": "Lost margin (under-staffed)"})
        order = list(imp.sort_values("total_cost").index)
        show(alt.Chart(comp).mark_bar(height=28, stroke="white", strokeWidth=2).encode(
            y=alt.Y("method:N", sort=order, title=None, axis=alt.Axis(labelLimit=220)),
            x=alt.X("gbp:Q", title=None, axis=alt.Axis(labelExpr="'£' + format(datum.value, ',.0f')")),
            color=alt.Color("component:N",
                            scale=alt.Scale(domain=["Over-staffing", "Food waste", "Lost margin (under-staffed)"],
                                            range=[BLUE, ORANGE, AQUA]),
                            legend=alt.Legend(title=None, orient="top")),
            tooltip=["method:N", "component:N", alt.Tooltip("label:N", title="Annual cost")],
        ).properties(height=260), width="stretch")
    st.caption("Simulation on synthetic history. Cost assumptions (wage £12.60/h, food £6.50/cover, "
               "30% walk-out when under-staffed) are editable in src/forecast/config.py and will be validated in the pilot.")

# ---- LLM briefing ---------------------------------------------------------------------------
NAMES = {"claude": "Anthropic Claude", "openai": "OpenAI GPT", "gemini": "Google Gemini", "template": "Template (no AI)"}
with tab_llm:
    section("Manager briefing written by AI",
            "Each model gets the same forecast facts. Automatic score = numbers are real (50%) + "
            "required topics covered (30%) + right length (20%).")
    providers = st.multiselect("Models", list(NAMES), default=list(NAMES), format_func=NAMES.get)
    if st.button("✨ Generate briefings", type="primary"):
        cols = st.columns(max(len(providers), 1))
        scores = []
        for col, p in zip(cols, providers):
            with col:
                with st.spinner(f"{NAMES[p]}…"):
                    try:
                        out = get_briefing(p, horizon, overrides)
                        ev = out["evaluation"]
                        scores.append(ev)
                        sc = "—" if ev.get("score") is None else f"{ev['score']:.2f}"
                        st.markdown(f"""<div class="llm-card"><b>{NAMES[p]}</b><br>
                        <span class="badge">score {sc}</span><span class="badge">{ev['latency_s']}s</span>
                        <span class="badge">${ev['cost_usd']:.4f}</span>
                        <div style="font-size:.78rem;opacity:.6;margin-top:6px">{ev['model']}</div></div>""",
                                    unsafe_allow_html=True)
                        if ev["status"] == "ok":
                            st.markdown(out["briefing"])
                        else:
                            st.info(ev["status"])
                    except Exception as e:
                        st.error(f"{NAMES[p]}: {e}")
        ok = [s for s in scores if s.get("score") is not None]
        if ok:
            section("Scores")
            st.dataframe(pd.DataFrame(ok)[["provider", "model", "score", "faithfulness", "coverage", "length_ok",
                                           "latency_s", "cost_usd"]], hide_index=True, width="stretch",
                         column_config={"score": st.column_config.ProgressColumn("Score", min_value=0, max_value=1,
                                                                                 format="%.2f")})
    summ = ROOT / "reports" / "llm_summary.csv"
    if summ.exists():
        with st.expander("Offline comparison across 4 scenarios (scripts/compare_llms.py)"):
            st.dataframe(pd.read_csv(summ), hide_index=True, width="stretch")

# ---- Governance -----------------------------------------------------------------------------
with tab_gov:
    g1, g2 = st.columns([1.2, 1])
    with g1:
        section("Is the forecast equally good for every kind of day?",
                "WAPE by segment — similar bars mean no shift type is systematically under-staffed.")
        seg = pd.DataFrame(metrics["segment_errors"])
        seg["label"] = seg["dimension"].map({"segment": "Day", "season": "Season"}) + " · " + seg["group"]
        sbase = alt.Chart(seg).encode(
            y=alt.Y("label:N", title=None, sort=None, axis=alt.Axis(labelLimit=220)),
            x=alt.X("WAPE:Q", axis=alt.Axis(format="%"), title=None,
                    scale=alt.Scale(domain=[0, float(seg["WAPE"].max()) * 1.2])),
            tooltip=[alt.Tooltip("group:N", title="Segment"), alt.Tooltip("n_days:Q", title="Days"),
                     alt.Tooltip("WAPE:Q", format=".1%"), alt.Tooltip("MAE:Q", format=".1f")])
        show((sbase.mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4, height=22, color=BLUE)
              + sbase.mark_text(align="left", dx=6, fontSize=12).encode(text=alt.Text("WAPE:Q", format=".1%"))
              ).properties(height=230))
    with g2:
        section("Responsible AI")
        st.markdown("""
- 🔒 **Privacy** — daily totals only; no customer or staff personal data. LLMs receive aggregated numbers only.
- ⚖️ **Fairness** — the tool outputs a *headcount*; who works is the manager's decision under HR policy.
- 🧑‍🍳 **Human in the loop** — every number is advisory and can be changed before the rota is published.
- 🔁 **Reproducible** — fixed seed, pinned packages, model hash checked in CI.
- 📉 **Monitoring** — retrain if WAPE exceeds 15% for two weeks.
""")
        st.caption("Full reviews in docs/03 (reproducibility) and docs/04 (privacy & fairness).")
