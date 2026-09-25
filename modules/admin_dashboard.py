"""
Admin Dashboard — KPI metrics, charts, appointment management & analytics.
Uses Plotly for all visualizations (pure Python, no HTML/JS).
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, timedelta

from data.database import (
    get_appointments, get_patients, get_doctors,
    update_appointment_status
)


# ── KPI Cards ─────────────────────────────────────────────────────────────────

def _render_kpis(appts: pd.DataFrame, patients: pd.DataFrame):
    today = date.today().isoformat()
    total      = len(appts)
    scheduled  = len(appts[appts["status"] == "Scheduled"])
    completed  = len(appts[appts["status"] == "Completed"])
    no_shows   = len(appts[appts["status"] == "No-Show"])
    today_appt = len(appts[appts["appointment_date"] == today])
    high_risk  = len(appts[(appts["no_show_risk"] >= 0.6) & (appts["status"] == "Scheduled")])

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Appointments",  total)
    c2.metric("Scheduled",           scheduled)
    c3.metric("Completed",           completed)
    c4.metric("No-Shows",            no_shows,  delta=f"{round(no_shows/max(total,1)*100,1)}%", delta_color="inverse")
    c5.metric("Today's Appointments", today_appt)
    c6.metric("High-Risk Upcoming",  high_risk,  delta="⚠️ Action Needed" if high_risk > 0 else "✅ Clear", delta_color="inverse")


# ── Charts ────────────────────────────────────────────────────────────────────

def _chart_appointments_by_status(appts: pd.DataFrame):
    counts = appts["status"].value_counts().reset_index()
    counts.columns = ["Status", "Count"]
    fig = px.pie(
        counts, names="Status", values="Count",
        title="Appointments by Status",
        color_discrete_sequence=px.colors.qualitative.Pastel,
        hole=0.4
    )
    fig.update_layout(margin=dict(t=40, b=0, l=0, r=0), height=300)
    return fig


def _chart_appointments_trend(appts: pd.DataFrame):
    appts = appts.copy()
    appts["appointment_date"] = pd.to_datetime(appts["appointment_date"])
    daily = appts.groupby("appointment_date").size().reset_index(name="Count")
    fig = px.line(
        daily, x="appointment_date", y="Count",
        title="Daily Appointment Volume (Last 120 Days)",
        markers=True,
        color_discrete_sequence=["#3b82f6"]
    )
    fig.update_xaxes(title="Date")
    fig.update_yaxes(title="Appointments")
    fig.update_layout(margin=dict(t=40, b=0, l=0, r=0), height=300)
    return fig


def _chart_department_load(appts: pd.DataFrame):
    dept = appts.groupby("department").size().reset_index(name="Appointments")
    fig = px.bar(
        dept.sort_values("Appointments", ascending=True),
        x="Appointments", y="department", orientation="h",
        title="Appointments by Department",
        color="Appointments",
        color_continuous_scale="Blues"
    )
    fig.update_layout(margin=dict(t=40, b=0, l=0, r=0), height=340, yaxis_title="")
    return fig


def _chart_no_show_risk_distribution(appts: pd.DataFrame):
    scheduled = appts[appts["status"] == "Scheduled"].copy()
    scheduled["risk_pct"] = (scheduled["no_show_risk"] * 100).round(0)
    fig = px.histogram(
        scheduled, x="risk_pct", nbins=20,
        title="No-Show Risk Score Distribution (Scheduled)",
        labels={"risk_pct": "Risk Score (%)"},
        color_discrete_sequence=["#f59e0b"]
    )
    fig.add_vline(x=60, line_dash="dash", line_color="red", annotation_text="High Risk Threshold")
    fig.update_layout(margin=dict(t=40, b=0, l=0, r=0), height=300)
    return fig


def _chart_no_show_by_age_group(appts: pd.DataFrame, patients: pd.DataFrame):
    merged = appts.merge(patients[["patient_id", "age"]], on="patient_id", how="left")
    merged["age_group"] = pd.cut(
        merged["age"], bins=[0, 18, 30, 45, 60, 120],
        labels=["<18", "18-30", "31-45", "46-60", "60+"]
    )
    rate = (
        merged.groupby("age_group", observed=True)
        .apply(lambda g: (g["status"] == "No-Show").sum() / max(len(g), 1))
        .reset_index(name="No-Show Rate")
    )
    rate["No-Show Rate"] = (rate["No-Show Rate"] * 100).round(1)
    fig = px.bar(
        rate, x="age_group", y="No-Show Rate",
        title="No-Show Rate by Age Group (%)",
        color="No-Show Rate",
        color_continuous_scale="Reds",
        labels={"age_group": "Age Group"}
    )
    fig.update_layout(margin=dict(t=40, b=0, l=0, r=0), height=300)
    return fig


def _chart_doctor_workload(appts: pd.DataFrame):
    doc_load = appts[appts["status"] == "Scheduled"].groupby("doctor").size().reset_index(name="Scheduled")
    fig = px.bar(
        doc_load.sort_values("Scheduled", ascending=False),
        x="doctor", y="Scheduled",
        title="Upcoming Scheduled Load per Doctor",
        color_discrete_sequence=["#6366f1"]
    )
    fig.update_xaxes(tickangle=-30)
    fig.update_layout(margin=dict(t=40, b=0, l=10, r=0), height=320, xaxis_title="")
    return fig


# ── Appointment Management Table ──────────────────────────────────────────────

def _render_appointment_management(appts: pd.DataFrame):
    st.subheader("📋 Appointment Management")

    col1, col2, col3 = st.columns(3)
    with col1:
        status_filter = st.multiselect(
            "Filter by Status",
            ["Scheduled", "Completed", "No-Show", "Cancelled"],
            default=["Scheduled"]
        )
    with col2:
        dept_filter = st.multiselect(
            "Filter by Department",
            sorted(appts["department"].unique().tolist())
        )
    with col3:
        date_range = st.date_input(
            "Date Range",
            value=(date.today(), date.today() + timedelta(days=14))
        )

    filtered = appts.copy()
    if status_filter:
        filtered = filtered[filtered["status"].isin(status_filter)]
    if dept_filter:
        filtered = filtered[filtered["department"].isin(dept_filter)]
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start, end = date_range
        filtered = filtered[
            (filtered["appointment_date"] >= start.isoformat()) &
            (filtered["appointment_date"] <= end.isoformat())
        ]

    filtered = filtered.sort_values("appointment_date")
    display_cols = [
        "appointment_id", "patient_name", "doctor", "department",
        "appointment_date", "appointment_time", "reason", "status", "no_show_risk"
    ]
    st.dataframe(
        filtered[display_cols].rename(columns={
            "appointment_id": "Appt ID", "patient_name": "Patient",
            "doctor": "Doctor", "department": "Dept",
            "appointment_date": "Date", "appointment_time": "Time",
            "reason": "Reason", "status": "Status", "no_show_risk": "Risk Score"
        }),
        use_container_width=True,
        height=380
    )

    st.markdown("---")
    st.subheader("✏️ Update Appointment Status")
    scheduled = appts[appts["status"] == "Scheduled"]
    if scheduled.empty:
        st.info("No scheduled appointments to update.")
        return

    col_a, col_b, col_c = st.columns([2, 2, 1])
    with col_a:
        aid_options = scheduled.apply(
            lambda r: f"{r['appointment_id']} — {r['patient_name']} ({r['appointment_date']})", axis=1
        ).tolist()
        selected_str = st.selectbox("Select Appointment", aid_options)
        selected_aid = selected_str.split(" — ")[0]
    with col_b:
        new_status = st.selectbox("New Status", ["Completed", "No-Show", "Cancelled"])
    with col_c:
        st.write("")
        st.write("")
        if st.button("Update", type="primary"):
            update_appointment_status(selected_aid, new_status)
            st.success(f"Appointment {selected_aid} marked as **{new_status}**.")
            st.rerun()


# ── High-Risk Patients Section ────────────────────────────────────────────────

def _render_high_risk_section(appts: pd.DataFrame):
    st.subheader("🔴 High No-Show Risk — Upcoming Appointments")
    today = date.today().isoformat()
    high_risk = appts[
        (appts["status"] == "Scheduled") &
        (appts["no_show_risk"] >= 0.6) &
        (appts["appointment_date"] >= today)
    ].sort_values("no_show_risk", ascending=False)

    if high_risk.empty:
        st.success("✅ No high-risk upcoming appointments.")
        return

    for _, row in high_risk.iterrows():
        risk_pct = int(float(row["no_show_risk"]) * 100)
        with st.expander(
            f"🔴 {row['patient_name']} — {row['appointment_date']} {row['appointment_time']} "
            f"— Risk: {risk_pct}%  [{row['appointment_id']}]"
        ):
            c1, c2, c3 = st.columns(3)
            c1.metric("Patient", row["patient_name"])
            c2.metric("Doctor", row["doctor"])
            c3.metric("Department", row["department"])
            st.progress(risk_pct, text=f"No-Show Risk: {risk_pct}%")
            st.caption(f"Reason: {row['reason']} | Booked: {row['booked_on']}")


# ── Entry point ───────────────────────────────────────────────────────────────

def render_admin_dashboard():
    st.title("📊 Admin Dashboard")
    st.markdown("Real-time overview of appointments, risk profiles, and department analytics.")
    st.divider()

    appts    = get_appointments()
    patients = get_patients()
    doctors  = get_doctors()

    _render_kpis(appts, patients)
    st.divider()

    tab_overview, tab_analytics, tab_manage, tab_risk = st.tabs([
        "📈 Overview", "🔬 Analytics", "📋 Manage Appointments", "⚠️ High-Risk Patients"
    ])

    with tab_overview:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(_chart_appointments_by_status(appts), use_container_width=True)
        with col2:
            st.plotly_chart(_chart_appointments_trend(appts), use_container_width=True)
        col3, col4 = st.columns(2)
        with col3:
            st.plotly_chart(_chart_department_load(appts), use_container_width=True)
        with col4:
            st.plotly_chart(_chart_doctor_workload(appts), use_container_width=True)

    with tab_analytics:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(_chart_no_show_risk_distribution(appts), use_container_width=True)
        with col2:
            st.plotly_chart(_chart_no_show_by_age_group(appts, patients), use_container_width=True)

        st.subheader("📊 Department-level No-Show Rate")
        dept_ns = (
            appts.groupby("department")
            .apply(lambda g: pd.Series({
                "Total": len(g),
                "No-Shows": (g["status"] == "No-Show").sum(),
                "No-Show Rate (%)": round((g["status"] == "No-Show").sum() / max(len(g), 1) * 100, 1)
            }))
            .reset_index()
        )
        st.dataframe(dept_ns, use_container_width=True)

    with tab_manage:
        _render_appointment_management(appts)

    with tab_risk:
        _render_high_risk_section(appts)
