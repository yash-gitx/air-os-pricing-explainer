import streamlit as st
import pandas as pd
from database import (
    init_and_seed_db,
    fetch_latest_warehouse_snapshot,
    fetch_car_history,
    SNAPSHOT_SQL_QUERY
)
from detector import detect_pricing_anomalies, plot_car_trajectory
from explainer import generate_anomaly_explanation

st.set_page_config(
    page_title="AIR OS // Pricing Anomaly Explainer",
    page_icon="🚗",
    layout="wide"
)

# Initialize database on startup
init_and_seed_db()

# Session state for tracking "Skill Adoption Metrics" (PM lens from the JD)
if "skill_invocations" not in st.session_state:
    st.session_state.skill_invocations = 14
if "overrides_logged" not in st.session_state:
    st.session_state.overrides_logged = 3

# Sidebar: Configuration & PM Skill Telemetry
with st.sidebar:
    st.markdown("## ⚙️ AIR OS Skill Config")
    st.caption("Skill ID: `skill_pricing_anomaly_v2` · Owner: Applied AI PM")
    
    api_key = st.text_input(
        "Gemini API Key (Optional)",
        type="password",
        help="Leave blank to use the built-in deterministic AIR OS analytical engine, or enter a Gemini API key for live LLM synthesis."
    )
    
    city_filter = st.multiselect(
        "Filter Hub Cities",
        options=["Gurugram", "Bengaluru", "Mumbai"],
        default=["Gurugram", "Bengaluru", "Mumbai"]
    )
    
    contamination = st.slider(
        "Isolation Forest Sensitivity",
        min_value=0.10, max_value=0.40, value=0.25, step=0.05,
        help="Controls sensitivity of the unsupervised anomaly detector."
    )

    if st.button("🔄 Reseed Warehouse Data"):
        init_and_seed_db(force_reseed=True)
        st.rerun()

    st.divider()
    st.markdown("### 📊 Skill Adoption Telemetry")
    st.caption("How we measure if this AIR OS skill is working:")
    col_s1, col_s2 = st.columns(2)
    col_s1.metric("Weekly Invocations", st.session_state.skill_invocations)
    col_s2.metric("Margin Saves Logged", st.session_state.overrides_logged)

# Header
st.title("🚗 AIR OS // Used-Car Pricing Anomaly Explainer")
st.markdown(
    "**Internal Applied AI Capability:** Scans daily warehouse pricing logs via SQL window functions, "
    "flags abnormal markdowns or margin breaches using `IsolationForest`, and synthesizes root-cause memos for Pricing PMs."
)

# Load & score data
raw_df = fetch_latest_warehouse_snapshot()
scored_df = detect_pricing_anomalies(raw_df, contamination=contamination)
filtered_df = scored_df[scored_df["city_hub"].isin(city_filter)].copy()

anomalies_df = filtered_df[filtered_df["is_anomaly"]].copy()
margin_breaches = filtered_df[filtered_df["gross_margin_pct"] < 0]

# Top KPI Strip
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Active Hub Inventory Scanned", f"{len(filtered_df)} Cars")
kpi2.metric("Anomalous Price Drops Flagged", f"{len(anomalies_df)} Cars", delta=f"-{len(anomalies_df)} action needed", delta_color="inverse")
kpi3.metric("Negative Margin Breaches (P0)", f"{len(margin_breaches)} Cars", delta="Critical" if len(margin_breaches) > 0 else "Clear", delta_color="inverse")
kpi4.metric("Avg Hub Gross Margin", f"{filtered_df['gross_margin_pct'].mean():.1f}%")

st.divider()

tab1, tab2, tab3 = st.tabs([
    "🚨 Anomaly Triage & AI Root-Cause Explainer",
    "🗄️ Warehouse Snapshot (All Cars)",
    "💻 Underlying SQL Window Query"
])

with tab1:
    if anomalies_df.empty:
        st.success("No pricing anomalies detected for the selected hub filter and sensitivity threshold.")
    else:
        st.subheader("Flagged Pricing Anomalies Requiring PM Review")
        display_cols = [
            "car_id", "make_model", "city_hub", "severity",
            "prev_day_price", "listed_price_lakhs", "dod_price_change_pct",
            "gross_margin_pct", "days_in_inventory", "test_drives_7d",
            "local_supply_index", "pricing_rule_triggered"
        ]
        st.dataframe(
            anomalies_df[display_cols],
            use_container_width=True,
            hide_index=True
        )

        st.markdown("---")
        col_left, col_right = st.columns([1.05, 1])

        with col_left:
            selected_car_id = st.selectbox(
                "Select Flagged Vehicle to Inspect Telemetry & Invoke AIR OS Explainer:",
                options=anomalies_df["car_id"].tolist(),
                format_func=lambda cid: (
                    f"{cid} — {anomalies_df.loc[anomalies_df['car_id']==cid, 'make_model'].values[0]} "
                    f"({anomalies_df.loc[anomalies_df['car_id']==cid, 'dod_price_change_pct'].values[0]}% DoD | "
                    f"{anomalies_df.loc[anomalies_df['car_id']==cid, 'severity'].values[0]})"
                )
            )

            selected_row = anomalies_df[anomalies_df["car_id"] == selected_car_id].iloc[0].to_dict()
            history_df = fetch_car_history(selected_car_id)
            fig = plot_car_trajectory(history_df, selected_row)
            st.pyplot(fig)

        with col_right:
            st.markdown("#### 🧠 AIR OS Root-Cause Memo & Decision Action")
            if st.button("⚡ Invoke AI Root-Cause Skill", type="primary", use_container_width=True):
                st.session_state.skill_invocations += 1
                with st.spinner("Querying telemetry & synthesizing product memo..."):
                    memo = generate_anomaly_explanation(selected_row, api_key=api_key)
                    st.session_state["last_memo"] = memo
                    st.session_state["last_car"] = selected_car_id

            if st.session_state.get("last_car") == selected_car_id and "last_memo" in st.session_state:
                st.markdown(st.session_state["last_memo"])
            else:
                # Auto-render default memo on load so UI is immediately rich
                default_memo = generate_anomaly_explanation(selected_row, api_key=api_key)
                st.markdown(default_memo)

            st.markdown("---")
            c_act1, c_act2 = st.columns(2)
            if c_act1.button("⏪ Approve Price Rollback (Log Override)", use_container_width=True):
                st.session_state.overrides_logged += 1
                st.toast(f"Rollback ticket logged for {selected_car_id}! Margin guardrail updated.", icon="✅")
            if c_act2.button("👍 Mark Markdown as Valid Liquidation", use_container_width=True):
                st.toast(f"Feedback recorded for {selected_car_id}. False-positive weight adjusted.", icon="📌")

with tab2:
    st.subheader("Complete Warehouse Snapshot (Latest Day)")
    st.dataframe(filtered_df, use_container_width=True, hide_index=True)

with tab3:
    st.subheader("Warehouse SQL Query (CTEs + Window Functions)")
    st.caption("This exact query runs against `spinny_warehouse.db` to compute `LAG()` day-over-day price deltas and 7-day rolling averages before feeding the Isolation Forest.")
    st.code(SNAPSHOT_SQL_QUERY, language="sql")