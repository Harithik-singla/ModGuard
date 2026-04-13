# dashboard/app.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh 
import requests

from src.database import (
    get_summary_stats,
    get_recent_logs,
    get_timeline_data,
    get_label_distribution
)

# ── Page config ────────────────────────────────────────
st.set_page_config(
    page_title="ModGuard Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Sidebar ────────────────────────────────────────────
st.sidebar.title("🛡️ ModGuard")
st.sidebar.markdown("Real-time content moderation monitor")
st.sidebar.divider()

auto_refresh = st.sidebar.toggle("Auto refresh", value=True)
refresh_rate = st.sidebar.slider("Refresh every (seconds)", 2, 30, 5)
st.sidebar.divider()

# ── Live test panel in sidebar ─────────────────────────
st.sidebar.subheader("Live test")
test_text = st.sidebar.text_area("Enter text to moderate", height=80)
if st.sidebar.button("Moderate", type="primary"):
    if test_text.strip():
        try:
            r    = requests.post(
                "http://localhost:8000/moderate",
                json={"text": test_text},
                timeout=10
            )
            data = r.json()
            decision = data["decision"]
            color    = {"APPROVED": "green",
                        "FLAGGED":  "orange",
                        "REMOVED":  "red"}.get(decision, "gray")
            st.sidebar.markdown(
                f"**Decision:** :{color}[{decision}]"
            )
            for label, info in data["labels"].items():
                if info["flagged"]:
                    st.sidebar.markdown(
                        f"- `{label}` → {info['score']:.3f}"
                    )
        except Exception as e:
            st.sidebar.error(f"API error: {e}")
    else:
        st.sidebar.warning("Enter some text first")

# ── Main dashboard ─────────────────────────────────────
st.title("🛡️ ModGuard — Content Moderation Dashboard")
st.caption("Live monitoring of all moderation decisions")

placeholder = st.empty()

while True:
    with placeholder.container():

        # ── Row 1: Summary metrics ─────────────────────
        stats = get_summary_stats()

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total requests",  f"{stats['total']:,}")
        col2.metric("Approved",        f"{stats['approved']:,}",
                    delta=f"{stats['approved']/max(stats['total'],1)*100:.1f}%")
        col3.metric("Flagged",         f"{stats['flagged']:,}",
                    delta=f"{stats['flagged']/max(stats['total'],1)*100:.1f}%",
                    delta_color="inverse")
        col4.metric("Removed",         f"{stats['removed']:,}",
                    delta=f"{stats['removed']/max(stats['total'],1)*100:.1f}%",
                    delta_color="inverse")
        col5.metric("Avg latency",     f"{stats['avg_ms']}ms")

        st.divider()

        # ── Row 2: Charts ──────────────────────────────
        col_left, col_right = st.columns(2)

        # Timeline chart
        with col_left:
            st.subheader("Requests over time")
            timeline = get_timeline_data(minutes=60)
            if timeline:
                df_time = pd.DataFrame(timeline)
                fig = px.line(
                    df_time, x="minute", y="count",
                    color="decision",
                    color_discrete_map={
                        "APPROVED": "#22c55e",
                        "FLAGGED":  "#f59e0b",
                        "REMOVED":  "#ef4444"
                    },
                    labels={"minute": "Time", "count": "Requests"}
                )
                fig.update_layout(
                    height=300,
                    margin=dict(l=0, r=0, t=0, b=0),
                    legend=dict(orientation="h", y=1.1)
                )
                st.plotly_chart(fig, use_container_width=True,key="timeline")
            else:
                st.info("No timeline data yet — make some requests first")

        # Label distribution chart
        with col_right:
            st.subheader("Avg score per label")
            dist = get_label_distribution()
            if dist:
                df_dist = pd.DataFrame({
                    "label": list(dist.keys()),
                    "score": list(dist.values())
                }).sort_values("score", ascending=True)

                fig2 = px.bar(
                    df_dist, x="score", y="label",
                    orientation="h",
                    color="score",
                    color_continuous_scale=["#22c55e", "#f59e0b", "#ef4444"],
                    labels={"score": "Avg score", "label": "Label"}
                )
                fig2.update_layout(
                    height=300,
                    margin=dict(l=0, r=0, t=0, b=0),
                    showlegend=False,
                    coloraxis_showscale=False
                )
                st.plotly_chart(fig2, use_container_width=True, key="label_dist")
            else:
                st.info("No label data yet")

        st.divider()

        # ── Row 3: Decision breakdown pie ─────────────
        col_pie, col_table = st.columns([1, 2])

        with col_pie:
            st.subheader("Decision breakdown")
            if stats['total'] > 0:
                fig3 = go.Figure(data=[go.Pie(
                    labels=["Approved", "Flagged", "Removed"],
                    values=[stats['approved'], stats['flagged'], stats['removed']],
                    hole=0.5,
                    marker_colors=["#22c55e", "#f59e0b", "#ef4444"]
                )])
                fig3.update_layout(
                    height=280,
                    margin=dict(l=0, r=0, t=0, b=0),
                    showlegend=True,
                    legend=dict(orientation="h", y=-0.1)
                )
                st.plotly_chart(fig3, use_container_width=True,key="decision_pie")
            else:
                st.info("No data yet")

        # ── Row 4: Recent decisions feed ───────────────
        with col_table:
            st.subheader("Recent decisions")
            logs = get_recent_logs(limit=20)
            if logs:
                df_logs = pd.DataFrame(logs)

                def color_decision(val):
                    colors = {
                        "APPROVED": "color: #22c55e",
                        "FLAGGED":  "color: #f59e0b",
                        "REMOVED":  "color: #ef4444"
                    }
                    return colors.get(val, "")

                styled = df_logs.style.applymap(
                    color_decision, subset=["decision"]
                )
                st.dataframe(styled, use_container_width=True, height=280)
            else:
                st.info("No moderation logs yet — make some requests first")

        # ── Row 5: Queue health ────────────────────────
        st.divider()
        st.subheader("Queue health")
        col_q1, col_q2, col_q3 = st.columns(3)

        try:
            r     = requests.get("http://localhost:8000/queue/stats", timeout=2)
            qdata = r.json()
            col_q1.metric("Queue length",   qdata.get("queue_length", "N/A"))
            col_q2.metric("Total jobs",     qdata.get("total_jobs",   "N/A"))
            redis_ok = qdata.get("redis_connected", False)
            col_q3.metric("Redis",
                          "Connected" if redis_ok else "Disconnected")
        except Exception:
            col_q1.error("API not reachable")

        # ── Footer ─────────────────────────────────────
        st.caption(f"Last updated: {pd.Timestamp.now().strftime('%H:%M:%S')} "
                   f"| Auto-refresh: {'on' if auto_refresh else 'off'}")

    if not auto_refresh:
        break

    time.sleep(refresh_rate)
    placeholder.empty()