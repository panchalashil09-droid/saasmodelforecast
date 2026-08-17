"""
SaaS Revenue Intelligence & Forecasting
=========================================
A gamified business intelligence + forecasting dashboard built on top of the
methodology in saas_revenue_forecasting.ipynb.

Run with:  streamlit run app.py
"""

import os
import sys
import json
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Make sure this file's own directory is importable regardless of the working
# directory the hosting platform launches Streamlit from (some deployment
# environments, e.g. Streamlit Community Cloud, don't always run with the repo
# root as the working directory / on sys.path by default).
APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from utils.data_loader import load_data, filter_data, latest_company_snapshot, format_currency, SEGMENTS
from utils.styles import inject_css, kpi_card_html, badge_html
from utils import visualizations as viz
from utils.forecasting import (
    engineer_features, recursive_forecast, forecast_all_segments,
    train_and_evaluate, DRIVER_COLS,
)

warnings.filterwarnings("ignore")

# Resolve all data/artifact paths relative to this file's own directory, not the
# process's current working directory (which varies by hosting platform).
MODEL_PATH = os.path.join(APP_DIR, "saas_revenue_forecasting_model.pkl")
FEATURE_COLS_PATH = os.path.join(APP_DIR, "feature_columns.pkl")
ARTIFACTS_PATH = os.path.join(APP_DIR, "model_artifacts.pkl")
DATA_PATH = os.path.join(APP_DIR, "simulated_saas_subscription_revenue_data.csv")

st.set_page_config(
    page_title="SaaS Revenue Intelligence & Forecasting",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css(st)


# ---------------------------------------------------------------------------
# Model / artifact loading (cached as a resource so it only loads once)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model_artifacts():
    """Load the trained model + supporting artifacts, if they exist."""
    if not (os.path.exists(MODEL_PATH) and os.path.exists(FEATURE_COLS_PATH) and os.path.exists(ARTIFACTS_PATH)):
        return None
    model = joblib.load(MODEL_PATH)
    feature_cols = joblib.load(FEATURE_COLS_PATH)
    artifacts = joblib.load(ARTIFACTS_PATH)
    artifacts["model"] = model
    artifacts["feature_cols"] = feature_cols
    return artifacts


@st.cache_resource(show_spinner=False)
def train_model_now():
    """Fallback: train the model live if no saved artifacts are found."""
    df = load_data(DATA_PATH)
    results = train_and_evaluate(df, tune=True)
    joblib.dump(results["final_model"], MODEL_PATH)
    joblib.dump(results["feature_cols"], FEATURE_COLS_PATH)
    artifacts = {k: v for k, v in results.items() if k not in ("final_model", "feature_cols", "model_df", "feat_df")}
    joblib.dump(artifacts, ARTIFACTS_PATH)
    artifacts["model"] = results["final_model"]
    artifacts["feature_cols"] = results["feature_cols"]
    artifacts["segment_dummy_cols"] = results["segment_dummy_cols"]
    return artifacts


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
try:
    raw_df = load_data(DATA_PATH)
except FileNotFoundError as e:
    st.error(f"⚠️ {e}")
    st.stop()

artifacts = load_model_artifacts()

# model_df (engineered features) is cheap to recompute and always kept in sync with raw_df
feat_df, model_df, computed_feature_cols, segment_dummy_cols = engineer_features(raw_df)

model_ready = artifacts is not None

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎛️ Dashboard Controls")

    segment_choice = st.selectbox("Segment", ["All Segments"] + SEGMENTS, index=0)

    min_date, max_date = raw_df["date"].min().date(), raw_df["date"].max().date()
    date_range = st.date_input("Date Range", value=(min_date, max_date),
                                min_value=min_date, max_value=max_date)
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = min_date, max_date

    horizon_label = st.select_slider("Forecast Horizon", options=["1 Month", "3 Months", "6 Months", "12 Months"],
                                      value="6 Months")
    horizon_months = {"1 Month": 1, "3 Months": 3, "6 Months": 6, "12 Months": 12}[horizon_label]

    available_model_names = []
    if model_ready:
        available_model_names = ["Best Model"] + list(artifacts["final_comparison"].index)
    model_choice = st.selectbox("Model", available_model_names or ["Best Model"], index=0,
                                 disabled=not model_ready,
                                 help="Only models actually trained on this dataset are listed.")

    st.markdown("### 🖥️ Display Controls")
    show_historical = st.checkbox("Show historical revenue", value=True)
    show_forecast = st.checkbox("Show forecast", value=True)
    show_ci = st.checkbox("Show confidence interval", value=True)
    show_target = st.checkbox("Show target values", value=False)
    show_ma = st.checkbox("Show moving average", value=True)

    st.markdown("---")
    st.markdown("### ℹ️ Dataset Info")
    st.markdown(
        f"""
        **Dataset:** SaaS Subscription Revenue
        **Frequency:** Monthly
        **Target:** `target_next_month_revenue_usd`
        **Segments:** {raw_df['segment'].nunique()}
        **Rows:** {len(raw_df)}
        **Range:** {min_date} → {max_date}
        """
    )

filtered_df = filter_data(raw_df, segment_choice, start_date, end_date)
if filtered_df.empty:
    st.warning("⚠️ No data matches the selected filters. Showing full dataset instead.")
    filtered_df = raw_df.copy()

snapshot = latest_company_snapshot(filtered_df if segment_choice != "All Segments" else raw_df)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
status_text = "🟢 Forecasting Engine Online" if model_ready else "🟡 Forecasting Engine Not Trained Yet"
st.markdown(f"""
<div class="app-header">
    <h1>🚀 SaaS Revenue Intelligence & Forecasting</h1>
    <p>AI-powered revenue analytics, forecasting and business insights</p>
    <span class="status-pill">{status_text}</span>
</div>
""", unsafe_allow_html=True)

if not model_ready:
    st.markdown("#### ⚙️ Train Forecasting Model")
    st.info("No saved model artifacts were found. Train the model once (uses the exact "
            "methodology from the forecasting notebook) to unlock predictions.")
    if st.button("⚙️ Train Forecasting Model", type="primary"):
        with st.spinner("Training candidate models (Linear Regression, Random Forest, "
                         "Gradient Boosting, XGBoost, LightGBM) with chronological "
                         "validation... this runs once."):
            artifacts = train_model_now()
            model_ready = True
        st.success("✅ Model trained and saved. Reloading dashboard...")
        st.rerun()


# ---------------------------------------------------------------------------
# Helper: model-aware forecast wrapper
# ---------------------------------------------------------------------------
def get_forecast(n_months, segments=None):
    """Return a forecast dataframe for the requested segments using the saved model."""
    if not model_ready:
        return pd.DataFrame(columns=["segment", "forecast_month", "predicted_revenue", "lower_bound", "upper_bound"])
    model = artifacts["model"]
    feature_cols = artifacts["feature_cols"]
    seg_dummy_cols = artifacts.get("segment_dummy_cols", segment_dummy_cols)
    residual_std = artifacts.get("residual_std", 0.0)
    segs = segments if segments else [c.replace("segment_", "") for c in seg_dummy_cols]
    frames = []
    for seg in segs:
        try:
            frames.append(recursive_forecast(model, model_df, feature_cols, seg_dummy_cols,
                                               seg, n_months, residual_std))
        except Exception:
            continue
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ===========================================================================
# TABS
# ===========================================================================
tab_overview, tab_revenue, tab_customer, tab_forecast, tab_model, tab_gamify, tab_explorer, tab_about = st.tabs(
    ["🏠 Executive Overview", "📈 Revenue Analytics", "👥 Customer Analytics", "🔮 Forecast Center",
     "🤖 Model Intelligence", "🏆 Gamification", "🔍 Data Explorer", "ℹ️ About"]
)

# ---------------------------------------------------------------------------
# TAB 1: EXECUTIVE OVERVIEW
# ---------------------------------------------------------------------------
with tab_overview:
    next_month_forecast_df = get_forecast(1, segments=None if segment_choice == "All Segments" else [segment_choice])
    next_month_total = next_month_forecast_df["predicted_revenue"].sum() if not next_month_forecast_df.empty else None
    forecast_growth = ((next_month_total / snapshot["current_revenue"]) - 1) * 100 if next_month_total and snapshot["current_revenue"] else None

    st.markdown('<div class="section-title">📊 Key Performance Indicators</div>', unsafe_allow_html=True)
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    with k1:
        st.markdown(kpi_card_html("Current Revenue", format_currency(snapshot["current_revenue"]),
                                   f"{snapshot['revenue_growth']:+.1f}% MoM",
                                   "up" if snapshot["revenue_growth"] >= 0 else "down", "💰"), unsafe_allow_html=True)
    with k2:
        val = format_currency(next_month_total) if next_month_total else "—"
        delta = f"{forecast_growth:+.1f}% vs current" if forecast_growth is not None else "Train model to unlock"
        st.markdown(kpi_card_html("Next Month Forecast", val, delta,
                                   "up" if (forecast_growth or 0) >= 0 else "down", "🔮"), unsafe_allow_html=True)
    with k3:
        st.markdown(kpi_card_html("Revenue Growth (MoM)", f"{snapshot['revenue_growth']:.1f}%", None,
                                   "up" if snapshot["revenue_growth"] >= 0 else "down", "📈"), unsafe_allow_html=True)
    with k4:
        st.markdown(kpi_card_html("Active Subscribers", f"{snapshot['active_subscribers']:,}",
                                   f"{snapshot['subscriber_growth']:+.1f}% MoM",
                                   "up" if snapshot["subscriber_growth"] >= 0 else "down", "👥"), unsafe_allow_html=True)
    with k5:
        st.markdown(kpi_card_html("ARPU", f"${snapshot['arpu']:,.0f}", f"{snapshot['arpu_delta']:+.1f}% MoM",
                                   "up" if snapshot["arpu_delta"] >= 0 else "down", "💵"), unsafe_allow_html=True)
    with k6:
        st.markdown(kpi_card_html("NPS Score", f"{snapshot['nps']:.0f}", f"{snapshot['nps_delta']:+.1f} pts MoM",
                                   "up" if snapshot["nps_delta"] >= 0 else "down", "❤️"), unsafe_allow_html=True)

    st.markdown('<hr class="divider"/>', unsafe_allow_html=True)

    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.plotly_chart(viz.revenue_trajectory_chart(filtered_df, moving_avg=show_ma, show_target=show_target),
                         use_container_width=True, key="chart_overview_trajectory")
    with col_b:
        st.plotly_chart(viz.revenue_mix_donut(raw_df), use_container_width=True, key="chart_1")

    # ---- Alerts Center ----
    st.markdown('<div class="section-title">🚨 Revenue Alerts</div>', unsafe_allow_html=True)

    def build_alerts(df, forecast_df, current_revenue):
        alerts = []
        # Revenue decline alert (2+ consecutive months down, company-wide)
        company_rev = df.groupby("date")["monthly_revenue_usd"].sum().sort_index()
        if len(company_rev) >= 3:
            diffs = company_rev.diff().dropna()
            if (diffs.iloc[-2:] < 0).all():
                alerts.append(("critical", "📉 Revenue Alert",
                                "Company-wide revenue has declined for 2 consecutive months."))
        # Churn alert per segment
        for seg in df["segment"].unique():
            seg_df = df[df["segment"] == seg].sort_values("date")
            if len(seg_df) >= 2:
                churn_change = seg_df["churn_rate_pct"].iloc[-1] - seg_df["churn_rate_pct"].iloc[-2]
                if churn_change > 0.5:
                    alerts.append(("warning", "⚠️ Churn Alert",
                                    f"{seg} churn increased by {churn_change:.1f} pts during the latest period."))
        # Subscriber growth slowing
        sub_totals = df.groupby("date")["active_subscribers"].sum().sort_index()
        if len(sub_totals) >= 3:
            growth_rates = sub_totals.pct_change().dropna()
            if len(growth_rates) >= 2 and growth_rates.iloc[-1] < growth_rates.iloc[-2] and growth_rates.iloc[-1] < 0.01:
                alerts.append(("warning", "👥 Subscriber Alert", "Subscriber growth is slowing company-wide."))
        # Forecast-based alerts
        if not forecast_df.empty and current_revenue:
            next_total = forecast_df.groupby("forecast_month")["predicted_revenue"].sum().iloc[0]
            growth_pct = (next_total / current_revenue - 1) * 100
            if growth_pct < 0:
                alerts.append(("critical", "🔮 Forecast Alert",
                                f"Forecasted next-month revenue is {abs(growth_pct):.1f}% below current revenue."))
            elif growth_pct > 5:
                alerts.append(("opportunity", "🚀 Opportunity Alert",
                                f"Forecasted growth of {growth_pct:.1f}% exceeds the 5% threshold."))
        return alerts

    all_seg_forecast = get_forecast(1)
    alerts = build_alerts(raw_df, all_seg_forecast, snapshot["current_revenue"])
    if alerts:
        for severity, title, msg in alerts:
            css_class = {"critical": "alert-critical", "warning": "alert-warning",
                         "opportunity": "alert-opportunity"}[severity]
            st.markdown(f'<div class="alert-card {css_class}"><b>{title}</b><br/>{msg}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="alert-card alert-info"><b>✅ All Clear</b><br/>No significant risk signals detected in the current data.</div>', unsafe_allow_html=True)

    # ---- Business insights ----
    st.markdown('<div class="section-title">💡 AI Business Insights</div>', unsafe_allow_html=True)

    def build_insights(df, forecast_df):
        insights = []
        company_rev = df.groupby("date")["monthly_revenue_usd"].sum().sort_index()
        trend = "increasing 📈" if company_rev.iloc[-1] > company_rev.iloc[0] else "decreasing 📉"
        overall_growth = (company_rev.iloc[-1] / company_rev.iloc[0] - 1) * 100
        insights.append(f"Company-wide revenue is **{trend}** overall — total growth of "
                         f"**{overall_growth:+.1f}%** from {df['date'].min().date()} to {df['date'].max().date()}.")

        seg_growth = {}
        for seg in df["segment"].unique():
            seg_df = df[df["segment"] == seg].sort_values("date")
            g = (seg_df["monthly_revenue_usd"].iloc[-1] / seg_df["monthly_revenue_usd"].iloc[0] - 1) * 100
            seg_growth[seg] = g
        fastest = max(seg_growth, key=seg_growth.get)
        insights.append(f"**{fastest}** is the fastest-growing segment (+{seg_growth[fastest]:.1f}% cumulative).")

        seg_totals = df.groupby("segment")["monthly_revenue_usd"].sum().sort_values(ascending=False)
        insights.append(f"**{seg_totals.index[0]}** contributes the most cumulative revenue "
                         f"({seg_totals.iloc[0] / seg_totals.sum() * 100:.0f}% of the total).")

        churn_by_seg = df.groupby("segment")["churn_rate_pct"].mean().sort_values(ascending=False)
        insights.append(f"**{churn_by_seg.index[0]}** has the highest average churn rate "
                         f"({churn_by_seg.iloc[0]:.1f}%) — a retention priority.")

        arpu_by_seg = df.groupby("segment")["arpu_usd"].mean().sort_values(ascending=False)
        insights.append(f"**{arpu_by_seg.index[0]}** has the highest ARPU (${arpu_by_seg.iloc[0]:,.0f}/user).")

        sub_totals = df.groupby("date")["active_subscribers"].sum().sort_index()
        if len(sub_totals) >= 4:
            recent_growth = sub_totals.pct_change().tail(3).mean()
            older_growth = sub_totals.pct_change().iloc[:-3].mean()
            accel = "accelerating" if recent_growth > older_growth else "decelerating"
            insights.append(f"Subscriber growth is **{accel}** (recent 3-month avg "
                             f"{recent_growth * 100:.1f}% vs. earlier average {older_growth * 100:.1f}%).")

        corr = df[["marketing_spend_usd", "monthly_revenue_usd"]].corr().iloc[0, 1]
        strength = "a strong" if abs(corr) > 0.6 else ("a moderate" if abs(corr) > 0.3 else "a weak")
        insights.append(f"Marketing spend shows **{strength} positive association** with revenue "
                         f"(correlation = {corr:.2f}).")

        if not forecast_df.empty:
            current_total = df[df["date"] == df["date"].max()]["monthly_revenue_usd"].sum()
            next_total = forecast_df.groupby("forecast_month")["predicted_revenue"].sum().iloc[0]
            direction = "above" if next_total > current_total else "below"
            insights.append(f"Forecasted next-month revenue (${next_total:,.0f}) is **{direction}** "
                             f"current revenue (${current_total:,.0f}).")
        return insights

    for line in build_insights(raw_df, all_seg_forecast):
        st.markdown(f"- {line}")

# ---------------------------------------------------------------------------
# TAB 2: REVENUE ANALYTICS
# ---------------------------------------------------------------------------
with tab_revenue:
    st.markdown('<div class="section-title">📈 Revenue Trajectory</div>', unsafe_allow_html=True)
    st.plotly_chart(viz.revenue_trajectory_chart(filtered_df, moving_avg=show_ma, show_target=show_target),
                     use_container_width=True, key="chart_revenue_trajectory")

    st.markdown('<div class="section-title">💼 Revenue by Segment</div>', unsafe_allow_html=True)
    metric_choice = st.radio("Compare segments by:", ["Revenue", "Subscribers", "ARPU", "Churn Rate"],
                              horizontal=True, key="seg_metric_radio")
    metric_map = {
        "Revenue": ("monthly_revenue_usd", "Revenue (USD)"),
        "Subscribers": ("active_subscribers", "Active Subscribers"),
        "ARPU": ("arpu_usd", "ARPU (USD)"),
        "Churn Rate": ("churn_rate_pct", "Churn Rate (%)"),
    }
    col_name, col_label = metric_map[metric_choice]
    st.plotly_chart(viz.segment_bar_chart(raw_df, col_name, col_label), use_container_width=True, key="chart_2")

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(viz.revenue_mix_donut(raw_df), use_container_width=True, key="chart_3")
    with c2:
        st.markdown('<div class="section-title" style="margin-top:0.5rem;">📣 Marketing Spend vs Revenue</div>', unsafe_allow_html=True)
        corr = raw_df[["marketing_spend_usd", "monthly_revenue_usd"]].corr().iloc[0, 1]
        st.plotly_chart(viz.marketing_vs_revenue_scatter(raw_df), use_container_width=True, key="chart_4")
        st.caption(f"Correlation coefficient between marketing spend and revenue: **{corr:.2f}**. "
                   f"{'Higher marketing spend tends to track with higher revenue.' if corr > 0.3 else 'The relationship in this dataset appears weak — spend alone does not explain most of the revenue variation.'}")

    st.markdown('<div class="section-title">🧠 Revenue Driver Intelligence</div>', unsafe_allow_html=True)
    corr_cols = ["monthly_revenue_usd", "target_next_month_revenue_usd"] + DRIVER_COLS
    st.plotly_chart(viz.correlation_heatmap(raw_df, corr_cols), use_container_width=True, key="chart_5")

# ---------------------------------------------------------------------------
# TAB 3: CUSTOMER ANALYTICS
# ---------------------------------------------------------------------------
with tab_customer:
    st.markdown('<div class="section-title">👥 Customer Growth Intelligence</div>', unsafe_allow_html=True)
    st.plotly_chart(viz.subscriber_flow_chart(filtered_df), use_container_width=True, key="chart_6")

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(viz.subscriber_growth_area(raw_df), use_container_width=True, key="chart_7")
    with c2:
        st.plotly_chart(viz.subscribers_by_segment_bar(raw_df), use_container_width=True, key="chart_8")

    st.markdown('<div class="section-title">📉 Churn Analysis</div>', unsafe_allow_html=True)
    churn_by_seg = raw_df.groupby("segment")["churn_rate_pct"].mean()
    latest_churn = raw_df[raw_df["date"] == raw_df["date"].max()].set_index("segment")["churn_rate_pct"]

    cc1, cc2, cc3, cc4 = st.columns(4)
    cc1.metric("Current Avg Churn", f"{latest_churn.mean():.2f}%")
    cc2.metric("Historical Avg Churn", f"{churn_by_seg.mean():.2f}%")
    cc3.metric("Highest Churn Segment", churn_by_seg.idxmax(), f"{churn_by_seg.max():.2f}%")
    cc4.metric("Lowest Churn Segment", churn_by_seg.idxmin(), f"{churn_by_seg.min():.2f}%")

    for seg in raw_df["segment"].unique():
        seg_df = raw_df[raw_df["segment"] == seg].sort_values("date")
        if len(seg_df) >= 2:
            change = seg_df["churn_rate_pct"].iloc[-1] - seg_df["churn_rate_pct"].iloc[-2]
            if change > 0.5:
                st.markdown(f'<div class="alert-card alert-warning"><b>⚠️ Churn Alert</b><br/>'
                             f'{seg} churn increased by {change:.1f} pts during the latest period.</div>',
                             unsafe_allow_html=True)

    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(viz.churn_trend_chart(filtered_df), use_container_width=True, key="chart_9")
    with c4:
        st.plotly_chart(viz.churn_by_segment_bar(raw_df), use_container_width=True, key="chart_10")

    st.markdown('<div class="section-title">💵 Monetization Intelligence</div>', unsafe_allow_html=True)
    c5, c6 = st.columns(2)
    with c5:
        st.plotly_chart(viz.arpu_trend_chart(filtered_df), use_container_width=True, key="chart_11")
    with c6:
        st.plotly_chart(viz.arpu_by_segment_bar(raw_df), use_container_width=True, key="chart_12")
    st.plotly_chart(viz.arpu_vs_revenue_scatter(raw_df), use_container_width=True, key="chart_13")

# ---------------------------------------------------------------------------
# TAB 4: FORECAST CENTER
# ---------------------------------------------------------------------------
with tab_forecast:
    st.markdown('<div class="section-title">🔮 Predict Revenue</div>', unsafe_allow_html=True)
    st.markdown("Select a segment and forecast horizon in the sidebar, then click below "
                "to generate an AI-powered revenue forecast using the trained model.")

    predict_clicked = st.button("🔮 Predict Revenue", type="primary", disabled=not model_ready)

    if not model_ready:
        st.warning("⚠️ Forecasting model not found. Use the **⚙️ Train Forecasting Model** "
                    "button at the top of the page to train it first.")

    if predict_clicked and model_ready:
        target_segments = SEGMENTS if segment_choice == "All Segments" else [segment_choice]
        with st.spinner("Preparing features and generating forecast..."):
            fc_df = get_forecast(horizon_months, segments=target_segments)

        if fc_df.empty:
            st.error("⚠️ Could not generate a forecast for the selected segment(s).")
        else:
            monthly_totals = fc_df.groupby("forecast_month")["predicted_revenue"].sum().sort_index()
            next_month_val = monthly_totals.iloc[0]
            current_val = (raw_df[raw_df["date"] == raw_df["date"].max()]["monthly_revenue_usd"].sum()
                            if segment_choice == "All Segments"
                            else raw_df[(raw_df["date"] == raw_df["date"].max()) & (raw_df["segment"] == segment_choice)]["monthly_revenue_usd"].sum())
            growth_vs_current = (next_month_val / current_val - 1) * 100 if current_val else 0.0

            st.markdown(f"""
            <div class="forecast-hero">
                <div class="label">🔮 Next Month Forecast — {segment_choice}</div>
                <div class="value">{format_currency(next_month_val)}</div>
                <div style="font-size:1.1rem; font-weight:700; color:{'#4ADE80' if growth_vs_current >= 0 else '#F87171'};">
                    {'↑' if growth_vs_current >= 0 else '↓'} {abs(growth_vs_current):.1f}% vs Current Month
                </div>
            </div>
            """, unsafe_allow_html=True)

            if growth_vs_current >= 0:
                st.success("🚀 Revenue momentum is positive!")
            else:
                st.warning("⚠️ Revenue is expected to decline.")

            st.markdown("<br/>", unsafe_allow_html=True)

            if show_ci:
                lower = fc_df[fc_df["forecast_month"] == monthly_totals.index[0]]["lower_bound"].sum()
                upper = fc_df[fc_df["forecast_month"] == monthly_totals.index[0]]["upper_bound"].sum()
                ci1, ci2, ci3 = st.columns(3)
                ci1.metric("Expected Revenue", format_currency(next_month_val))
                ci2.metric("Lower Estimate", format_currency(lower))
                ci3.metric("Upper Estimate", format_currency(upper))
                st.caption("These are model-based **prediction intervals** (derived from historical residual "
                           "error), not formal statistical confidence intervals.")
                pct_width = min(100, max(0, 100 - (upper - lower) / max(next_month_val, 1) * 100))
                st.progress(int(pct_width), text=f"Forecast Confidence Meter — tighter band = higher confidence")

            st.session_state["last_forecast_df"] = fc_df
            st.session_state["last_forecast_segments"] = target_segments

    st.markdown('<hr class="divider"/>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🔮 AI Revenue Forecast Chart</div>', unsafe_allow_html=True)

    chart_segments = SEGMENTS if segment_choice == "All Segments" else [segment_choice]
    chart_forecast = get_forecast(horizon_months, segments=chart_segments)

    if segment_choice == "All Segments":
        hist_plot = raw_df.groupby("date", as_index=False)["monthly_revenue_usd"].sum()
        fc_plot = chart_forecast.groupby("forecast_month", as_index=False).agg(
            predicted_revenue=("predicted_revenue", "sum"),
            lower_bound=("lower_bound", "sum"),
            upper_bound=("upper_bound", "sum"),
        ) if not chart_forecast.empty else pd.DataFrame()
    else:
        hist_plot = raw_df[raw_df["segment"] == segment_choice][["date", "monthly_revenue_usd"]]
        fc_plot = chart_forecast

    if show_forecast:
        st.plotly_chart(viz.forecast_chart(hist_plot, fc_plot, segment_choice), use_container_width=True, key="chart_14")
    elif show_historical:
        st.plotly_chart(viz.revenue_trajectory_chart(filtered_df, moving_avg=show_ma), use_container_width=True, key="chart_15")

    st.markdown('<div class="section-title">🎯 Forecast Challenge</div>', unsafe_allow_html=True)
    st.markdown("**Can we beat the previous month?**")
    if not chart_forecast.empty:
        month1 = chart_forecast.groupby("forecast_month")["predicted_revenue"].sum().sort_index().iloc[0]
        current_total = (raw_df[raw_df["date"] == raw_df["date"].max()]["monthly_revenue_usd"].sum()
                          if segment_choice == "All Segments"
                          else raw_df[(raw_df["date"] == raw_df["date"].max()) & (raw_df["segment"] == segment_choice)]["monthly_revenue_usd"].sum())
        change_pct = (month1 / current_total - 1) * 100 if current_total else 0
        if change_pct > 0:
            momentum = "🔥 Strong" if change_pct > 5 else "🟢 Positive"
            st.markdown(f"""
            <div class="soft-card">
                <h3>🚀 YES!</h3>
                <p><b>Expected Growth:</b> +{change_pct:.1f}%</p>
                <p><b>Revenue Momentum:</b> {momentum}</p>
            </div>""", unsafe_allow_html=True)
        else:
            momentum = "🔴 Critical" if change_pct < -5 else "🟠 Needs Attention"
            st.markdown(f"""
            <div class="soft-card">
                <h3>⚠️ NOT YET</h3>
                <p><b>Expected Change:</b> {change_pct:.1f}%</p>
                <p><b>Revenue Momentum:</b> {momentum}</p>
            </div>""", unsafe_allow_html=True)
    else:
        st.info("Train the model and click **🔮 Predict Revenue** to see the forecast challenge result.")

    st.markdown('<div class="section-title">📋 Forecast Table</div>', unsafe_allow_html=True)
    if not chart_forecast.empty:
        table_df = chart_forecast.copy()
        table_df["forecast_month"] = pd.to_datetime(table_df["forecast_month"]).dt.strftime("%b %Y")
        table_df = table_df.groupby("forecast_month", as_index=False, sort=False).agg(
            predicted_revenue=("predicted_revenue", "sum"),
            lower_bound=("lower_bound", "sum"),
            upper_bound=("upper_bound", "sum"),
        ) if segment_choice == "All Segments" else table_df[["forecast_month", "predicted_revenue", "lower_bound", "upper_bound"]]
        table_df["growth_%"] = table_df["predicted_revenue"].pct_change().fillna(
            (table_df["predicted_revenue"].iloc[0] / current_val - 1) if 'current_val' in dir() and current_val else 0
        ) * 100
        table_df = table_df.rename(columns={
            "forecast_month": "Month", "predicted_revenue": "Forecast",
            "lower_bound": "Lower", "upper_bound": "Upper", "growth_%": "Growth %",
        })
        st.dataframe(table_df.style.format({
            "Forecast": "${:,.0f}", "Lower": "${:,.0f}", "Upper": "${:,.0f}", "Growth %": "{:+.1f}%",
        }), use_container_width=True, hide_index=True)

        csv_bytes = table_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Forecast CSV", data=csv_bytes,
                            file_name=f"revenue_forecast_{segment_choice.replace(' ', '_')}.csv",
                            mime="text/csv")
    else:
        st.info("No forecast available yet — train the model first.")

# ---------------------------------------------------------------------------
# TAB 5: MODEL INTELLIGENCE
# ---------------------------------------------------------------------------
with tab_model:
    if not model_ready:
        st.warning("⚠️ No trained model found yet. Use the **⚙️ Train Forecasting Model** button "
                    "at the top of the page.")
    else:
        st.markdown('<div class="section-title">🤖 Model Performance</div>', unsafe_allow_html=True)
        rec_name = artifacts["recommended_model_name"]
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"""<div class="soft-card"><h4>🏆 Best Model</h4>
                        <h2>{rec_name}</h2></div>""", unsafe_allow_html=True)
        with m2:
            fm = artifacts["final_metrics_best"]
            st.markdown(f"""<div class="soft-card">
                        <p><b>RMSE:</b> ${fm['RMSE']:,.0f}</p>
                        <p><b>MAE:</b> ${fm['MAE']:,.0f}</p>
                        <p><b>MAPE:</b> {fm['MAPE']:.2f}%</p>
                        <p><b>R²:</b> {fm['R2']:.3f}</p></div>""", unsafe_allow_html=True)
        with m3:
            st.markdown(f"""<div class="soft-card"><h4>Residual Std Dev</h4>
                        <h2>${artifacts['residual_std']:,.0f}</h2>
                        <p style="opacity:0.7;">Used to build prediction intervals</p></div>""",
                        unsafe_allow_html=True)

        st.markdown('<div class="section-title">📊 Model Comparison</div>', unsafe_allow_html=True)
        st.plotly_chart(viz.model_comparison_bar(artifacts["final_comparison"], rec_name), use_container_width=True, key="chart_16")
        st.dataframe(artifacts["final_comparison"], use_container_width=True)

        with st.expander("📉 Baseline comparison (naive / moving-average / seasonal-naive)"):
            st.dataframe(artifacts["baseline_results"], use_container_width=True)
            st.caption("Any ML model above should outperform these simple baselines to justify its complexity.")

        with st.expander("🔁 Time-Series Cross-Validation Results (expanding-window folds)"):
            st.dataframe(artifacts["cv_summary"].round(2), use_container_width=True)

        with st.expander("🧩 Segment-level test performance"):
            st.dataframe(artifacts["segment_metrics_df"], use_container_width=True)

        st.markdown('<div class="section-title">🎯 Prediction Accuracy</div>', unsafe_allow_html=True)
        test_pred_df = artifacts["test_predictions_df"]
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(viz.actual_vs_predicted_chart(test_pred_df), use_container_width=True, key="chart_17")
        with c2:
            st.plotly_chart(viz.residual_histogram(test_pred_df), use_container_width=True, key="chart_18")
        st.plotly_chart(viz.prediction_error_over_time(test_pred_df), use_container_width=True, key="chart_19")

        st.markdown('<div class="section-title">🤖 What Drives Revenue?</div>', unsafe_allow_html=True)
        importances = artifacts["importances"]
        if len(importances):
            st.plotly_chart(viz.feature_importance_chart(importances, top_n=15), use_container_width=True, key="chart_20")
        else:
            st.info(f"{rec_name} does not expose feature importances (e.g. it is not tree-based).")

        try:
            import shap  # noqa: F401
            shap_available = True
        except ImportError:
            shap_available = False
        if not shap_available:
            st.caption("ℹ️ SHAP is not installed in this environment — skipping SHAP-based explanations.")

# ---------------------------------------------------------------------------
# TAB 6: GAMIFICATION
# ---------------------------------------------------------------------------
with tab_gamify:
    st.markdown('<div class="section-title">🏆 Revenue Health Score</div>', unsafe_allow_html=True)

    def compute_health_score(df, forecast_df):
        # Revenue growth (25 pts): scaled from -10% (0 pts) to +15% (25 pts)
        company_rev = df.groupby("date")["monthly_revenue_usd"].sum().sort_index()
        rev_growth = (company_rev.iloc[-1] / company_rev.iloc[-2] - 1) * 100 if len(company_rev) > 1 else 0
        rev_score = np.clip((rev_growth + 10) / 25 * 25, 0, 25)

        # Subscriber growth (20 pts): scaled -5% (0) to +10% (20)
        sub_totals = df.groupby("date")["active_subscribers"].sum().sort_index()
        sub_growth = (sub_totals.iloc[-1] / sub_totals.iloc[-2] - 1) * 100 if len(sub_totals) > 1 else 0
        sub_score = np.clip((sub_growth + 5) / 15 * 20, 0, 20)

        # Churn health (20 pts): lower churn = better. 0% churn = 20 pts, 10%+ churn = 0 pts
        avg_churn = df[df["date"] == df["date"].max()]["churn_rate_pct"].mean()
        churn_score = np.clip((10 - avg_churn) / 10 * 20, 0, 20)

        # ARPU (20 pts): scaled relative to historical ARPU range
        arpu_now = df[df["date"] == df["date"].max()]["arpu_usd"].mean()
        arpu_min, arpu_max = df["arpu_usd"].min(), df["arpu_usd"].max()
        arpu_score = np.clip((arpu_now - arpu_min) / max(arpu_max - arpu_min, 1) * 20, 0, 20)

        # NPS (15 pts): scaled -50 (0) to +80 (15)
        nps_now = df[df["date"] == df["date"].max()]["nps_score"].mean()
        nps_score = np.clip((nps_now + 50) / 130 * 15, 0, 15)

        total = rev_score + sub_score + churn_score + arpu_score + nps_score
        return {
            "Revenue Growth": (rev_score, 25), "Subscriber Growth": (sub_score, 20),
            "Churn Health": (churn_score, 20), "ARPU": (arpu_score, 20), "NPS": (nps_score, 15),
            "TOTAL": (total, 100),
        }

    health = compute_health_score(raw_df, all_seg_forecast if 'all_seg_forecast' in dir() else pd.DataFrame())
    total_score, _ = health["TOTAL"]

    if total_score >= 90:
        classification, color = "🚀 Exceptional", "green"
    elif total_score >= 75:
        classification, color = "🟢 Strong", "green"
    elif total_score >= 60:
        classification, color = "🟡 Stable", "yellow"
    elif total_score >= 40:
        classification, color = "🟠 Needs Attention", "orange"
    else:
        classification, color = "🔴 Critical", "red"

    hc1, hc2 = st.columns([1, 2])
    with hc1:
        st.markdown(f"""
        <div class="forecast-hero" style="background:linear-gradient(135deg,#4338CA,#9333EA);">
            <div class="label">Revenue Health Score</div>
            <div class="value">{total_score:.0f}<span style="font-size:1.4rem;">/100</span></div>
            <div>{badge_html(classification, color)}</div>
        </div>
        """, unsafe_allow_html=True)
    with hc2:
        for label, (score, max_score) in health.items():
            if label == "TOTAL":
                continue
            st.progress(min(1.0, score / max_score), text=f"{label}: {score:.0f}/{max_score}")

    st.markdown('<hr class="divider"/>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">🏆 Segment Performance Leaderboard</div>', unsafe_allow_html=True)

    def compute_leaderboard(df):
        rows = []
        for seg in df["segment"].unique():
            seg_df = df[df["segment"] == seg].sort_values("date")
            rev = seg_df["monthly_revenue_usd"].iloc[-1]
            rev_growth = (seg_df["monthly_revenue_usd"].iloc[-1] / seg_df["monthly_revenue_usd"].iloc[0] - 1) * 100
            sub_growth = (seg_df["active_subscribers"].iloc[-1] / seg_df["active_subscribers"].iloc[0] - 1) * 100
            churn = seg_df["churn_rate_pct"].mean()
            arpu = seg_df["arpu_usd"].mean()
            nps = seg_df["nps_score"].mean()
            rows.append({"segment": seg, "revenue": rev, "rev_growth": rev_growth,
                         "sub_growth": sub_growth, "churn": churn, "arpu": arpu, "nps": nps})
        board = pd.DataFrame(rows)
        # Normalize each metric to 0-1 and combine into a transparent composite score
        def norm(s, invert=False):
            r = (s - s.min()) / max(s.max() - s.min(), 1e-9)
            return 1 - r if invert else r
        board["score"] = (
            norm(board["revenue"]) * 25 + norm(board["rev_growth"]) * 25 +
            norm(board["sub_growth"]) * 20 + norm(board["churn"], invert=True) * 15 +
            norm(board["arpu"]) * 10 + norm(board["nps"]) * 5
        )
        return board.sort_values("score", ascending=False).reset_index(drop=True)

    board = compute_leaderboard(raw_df)
    medals = ["🥇", "🥈", "🥉"]
    for i, row in board.iterrows():
        medal = medals[i] if i < 3 else f"#{i+1}"
        st.markdown(f"""
        <div class="rank-row">
            <span>{medal} &nbsp; {row['segment']}</span>
            <span>Score: {row['score']:.1f}/100 &nbsp;|&nbsp; Revenue: {format_currency(row['revenue'])} &nbsp;|&nbsp;
            Growth: {row['rev_growth']:+.1f}% &nbsp;|&nbsp; Churn: {row['churn']:.1f}%</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="divider"/>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🏅 Revenue Milestones</div>', unsafe_allow_html=True)

    company_rev_latest = raw_df[raw_df["date"] == raw_df["date"].max()]["monthly_revenue_usd"].sum()
    company_rev_alltime_max = raw_df.groupby("date")["monthly_revenue_usd"].sum().max()
    milestone_targets = [1_000_000, 2_000_000, 3_000_000, 5_000_000]
    milestone_cols = st.columns(len(milestone_targets))
    for col, target in zip(milestone_cols, milestone_targets):
        achieved = company_rev_alltime_max >= target
        with col:
            label = "🏆" if target < 3_000_000 else "💎"
            status_badge = badge_html("✅ Achieved", "green") if achieved else badge_html("🔒 Upcoming", "blue")
            st.markdown(f"""
            <div class="milestone-card">
                <div style="font-size:1.6rem;">{label}</div>
                <div style="font-weight:700;">${target/1_000_000:.0f}M Revenue Club</div>
                <div style="margin-top:0.4rem;">{status_badge}</div>
            </div>
            """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# TAB 7: DATA EXPLORER
# ---------------------------------------------------------------------------
with tab_explorer:
    st.markdown('<div class="section-title">🔍 Data Explorer</div>', unsafe_allow_html=True)

    exp_col1, exp_col2, exp_col3 = st.columns(3)
    with exp_col1:
        exp_segment = st.multiselect("Filter by segment", SEGMENTS, default=SEGMENTS)
    with exp_col2:
        exp_dates = st.date_input("Filter by date range", value=(min_date, max_date),
                                   min_value=min_date, max_value=max_date, key="explorer_dates")
    with exp_col3:
        search_term = st.text_input("Search (any column, text match)", "")

    exp_df = raw_df[raw_df["segment"].isin(exp_segment)] if exp_segment else raw_df.copy()
    if isinstance(exp_dates, tuple) and len(exp_dates) == 2:
        exp_df = exp_df[(exp_df["date"] >= pd.Timestamp(exp_dates[0])) & (exp_df["date"] <= pd.Timestamp(exp_dates[1]))]
    if search_term:
        mask = exp_df.astype(str).apply(lambda col: col.str.contains(search_term, case=False, na=False)).any(axis=1)
        exp_df = exp_df[mask]

    sort_col = st.selectbox("Sort by column", exp_df.columns.tolist(), index=0)
    sort_asc = st.checkbox("Ascending", value=True)
    exp_df = exp_df.sort_values(sort_col, ascending=sort_asc)

    st.dataframe(exp_df, use_container_width=True, hide_index=True)
    st.caption(f"Showing {len(exp_df)} of {len(raw_df)} rows.")

    st.download_button("⬇️ Download Data", data=exp_df.to_csv(index=False).encode("utf-8"),
                        file_name="saas_revenue_filtered_data.csv", mime="text/csv")

# ---------------------------------------------------------------------------
# TAB 8: ABOUT
# ---------------------------------------------------------------------------
with tab_about:
    st.markdown('<div class="section-title">ℹ️ About This Project</div>', unsafe_allow_html=True)
    st.markdown("""
    **SaaS Revenue Intelligence & Forecasting** is an AI-powered business intelligence and
    forecasting dashboard built on top of a genuine time-series forecasting pipeline.

    **Dataset:** `simulated_saas_subscription_revenue_data.csv` — 141 monthly rows
    (47 months × 3 segments: `AI_Assistant`, `Streaming`, `Telecom_Hosting`),
    January 2021 – November 2024. **This dataset is fully simulated / synthetic**, generated
    for learning and demonstration purposes — it does not represent a real company.

    **Target variable:** `target_next_month_revenue_usd` — next month's revenue for each segment.

    **Methodology (mirrors the source notebook exactly):**
    - Chronological (never random) train/test splits, since this is a forecasting problem.
    - Feature engineering: revenue lags (1/2/3/6/12 months), rolling means & std (3/6/12 months),
      month-over-month and quarter-over-quarter growth rates, and one-hot encoded segments —
      every feature for month *T* only uses information available at or before month *T*.
    - Candidate models: Linear Regression, Random Forest, Gradient Boosting, XGBoost, and
      LightGBM (when available), validated with `TimeSeriesSplit` cross-validation.
    - Model selection: the model that wins on the most metrics among RMSE / MAE / MAPE on a
      held-out chronological test set (not R² alone, which can look artificially strong on a
      small, trending test set).
    - Multi-step forecasts are generated **recursively**: each predicted month feeds the
      lag/rolling features for the next step, with exogenous (non-revenue) drivers held at
      their most recent known values.
    - Prediction intervals are residual-based (test-set residual std, widened by √horizon) —
      labeled as estimates, not formal statistical confidence intervals.
    """)
    if model_ready:
        st.success(f"✅ Currently deployed model: **{artifacts['recommended_model_name']}**")

# ---------------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------------
st.markdown('<hr class="divider"/>', unsafe_allow_html=True)
st.markdown("""
<div class="footer-block">
    🚀 <b>SaaS Revenue Intelligence</b><br/>
    AI-powered revenue forecasting & analytics<br/>
    Built with Python • Streamlit • Scikit-learn • Plotly<br/>
    <i>Dataset is simulated/synthetic, for demonstration purposes only.</i>
</div>
""", unsafe_allow_html=True)
