"""
No-Show Risk Predictor UI Module.
Interactive prediction tool with feature importance visualization and batch analysis.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import date, timedelta

from data.database import get_patients, get_appointments, get_patient_appointment_history
from ml.predictor import predict_no_show_risk, get_feature_importance


# ── Single Patient Predictor ──────────────────────────────────────────────────

def _single_patient_predictor():
    st.subheader("🔍 Single Patient Risk Assessment")

    patients = get_patients()
    patient_options = {
        f"{row['name']} ({row['patient_id']}) — {row['chronic_conditions']}": row
        for _, row in patients.iterrows()
    }
    selected_key = st.selectbox("Select Patient", list(patient_options.keys()))
    patient_row  = patient_options[selected_key]

    col1, col2, col3 = st.columns(3)
    with col1:
        appt_date = st.date_input(
            "Appointment Date",
            value=date.today() + timedelta(days=7),
            min_value=date.today() + timedelta(days=1)
        )
    with col2:
        appt_time = st.selectbox(
            "Appointment Time",
            ["08:00", "09:00", "10:00", "11:00", "12:00", "13:00", "14:00", "15:00", "16:00"]
        )
    with col3:
        st.write("")
        st.write("")
        predict_btn = st.button("🎯 Predict Risk", type="primary", use_container_width=True)

    if predict_btn:
        history  = get_patient_appointment_history(patient_row["patient_id"])
        appt_hour = int(appt_time.split(":")[0])
        risk = predict_no_show_risk(
            patient_row, history,
            appt_date_str=appt_date.isoformat(),
            appt_hour=appt_hour
        )
        risk_pct = int(risk * 100)

        st.divider()
        col_a, col_b = st.columns([1, 2])
        with col_a:
            if risk < 0.3:
                st.success(f"### 🟢 Low Risk\n## {risk_pct}%")
                recommendation = "Standard scheduling. No special intervention needed."
            elif risk < 0.6:
                st.warning(f"### 🟡 Medium Risk\n## {risk_pct}%")
                recommendation = "Send an SMS/email reminder 24–48 hours before the appointment."
            else:
                st.error(f"### 🔴 High Risk\n## {risk_pct}%")
                recommendation = "Confirm via phone call. Consider double-booking or waitlist management."

            st.progress(risk_pct, text=f"No-Show Probability: {risk_pct}%")

        with col_b:
            st.markdown("#### 📋 Patient Profile")
            info_data = {
                "Field": ["Name", "Age", "Gender", "Chronic Conditions", "Past Appointments"],
                "Value": [
                    patient_row["name"],
                    patient_row["age"],
                    patient_row["gender"],
                    patient_row["chronic_conditions"],
                    len(history)
                ]
            }
            if not history.empty:
                ns_rate     = (history["status"] == "No-Show").sum()
                cancel_rate = (history["status"] == "Cancelled").sum()
                info_data["Field"] += ["Prior No-Shows", "Prior Cancellations"]
                info_data["Value"] += [ns_rate, cancel_rate]
            st.dataframe(pd.DataFrame(info_data), use_container_width=True, hide_index=True)

        st.info(f"💡 **Recommendation:** {recommendation}")

        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=risk_pct,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "No-Show Risk Score", "font": {"size": 18}},
            delta={"reference": 30, "increasing": {"color": "red"}, "decreasing": {"color": "green"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": "#3b82f6"},
                "steps": [
                    {"range": [0, 30],  "color": "#d1fae5"},
                    {"range": [30, 60], "color": "#fef3c7"},
                    {"range": [60, 100],"color": "#fee2e2"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 3},
                    "thickness": 0.75,
                    "value": 60
                }
            }
        ))
        fig.update_layout(height=280, margin=dict(t=40, b=0, l=20, r=20))
        st.plotly_chart(fig, use_container_width=True)


# ── Batch Analysis ────────────────────────────────────────────────────────────

def _batch_risk_analysis():
    st.subheader("📊 Batch Risk Analysis — All Scheduled Appointments")

    appts    = get_appointments()
    patients = get_patients()

    scheduled = appts[appts["status"] == "Scheduled"].merge(
        patients[["patient_id", "age", "gender", "chronic_conditions"]],
        on="patient_id", how="left"
    ).copy()

    if scheduled.empty:
        st.info("No scheduled appointments to analyze.")
        return

    scheduled["risk_pct"]    = (scheduled["no_show_risk"] * 100).round(1)
    scheduled["risk_level"]  = scheduled["no_show_risk"].apply(
        lambda r: "🟢 Low" if r < 0.3 else ("🟡 Medium" if r < 0.6 else "🔴 High")
    )

    col1, col2, col3 = st.columns(3)
    low    = (scheduled["no_show_risk"] < 0.3).sum()
    medium = ((scheduled["no_show_risk"] >= 0.3) & (scheduled["no_show_risk"] < 0.6)).sum()
    high   = (scheduled["no_show_risk"] >= 0.6).sum()
    col1.metric("🟢 Low Risk",    low)
    col2.metric("🟡 Medium Risk", medium)
    col3.metric("🔴 High Risk",   high)

    # Risk distribution bar
    risk_summary = pd.DataFrame({
        "Risk Level": ["Low (<30%)", "Medium (30–60%)", "High (>60%)"],
        "Patients": [low, medium, high]
    })
    fig = px.bar(
        risk_summary, x="Risk Level", y="Patients",
        color="Risk Level",
        color_discrete_map={
            "Low (<30%)": "#6ee7b7",
            "Medium (30–60%)": "#fcd34d",
            "High (>60%)": "#fca5a5"
        },
        title="Scheduled Patients by Risk Level"
    )
    fig.update_layout(height=320, showlegend=False, margin=dict(t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        scheduled[[
            "appointment_id", "patient_name", "appointment_date",
            "appointment_time", "doctor", "department", "risk_pct", "risk_level"
        ]].rename(columns={
            "appointment_id": "ID", "patient_name": "Patient",
            "appointment_date": "Date", "appointment_time": "Time",
            "doctor": "Doctor", "department": "Dept",
            "risk_pct": "Risk %", "risk_level": "Level"
        }).sort_values("Risk %", ascending=False),
        use_container_width=True,
        height=380
    )


# ── Feature Importance ────────────────────────────────────────────────────────

def _feature_importance_chart():
    st.subheader("🧠 Model Feature Importance")

    appts    = get_appointments()
    patients = get_patients()

    importance_df = get_feature_importance(appts, patients)
    fig = px.bar(
        importance_df,
        x="Importance", y="Feature",
        orientation="h",
        title="Random Forest — Feature Importance",
        color="Importance",
        color_continuous_scale="Blues"
    )
    fig.update_layout(height=380, yaxis={"categoryorder": "total ascending"}, margin=dict(t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("ℹ️ Feature Descriptions"):
        st.markdown("""
| Feature | Description |
|---|---|
| `patient_age` | Age of the patient in years |
| `is_chronic_patient` | Whether the patient has any chronic condition (1=Yes, 0=No) |
| `gender_encoded` | Gender encoded numerically (0=Female, 1=Male, 2=Other) |
| `prior_no_show_rate` | Historical ratio of no-shows to total past appointments |
| `prior_total_appointments` | Total appointments the patient has had |
| `prior_cancellation_rate` | Historical cancellation rate |
| `days_until_appointment` | Days between booking and the appointment date |
| `appointment_day_of_week` | Day of the week (0=Monday … 6=Sunday) |
| `appointment_hour` | Hour of the appointment (e.g. 9 for 09:00) |
        """)


# ── Entry point ───────────────────────────────────────────────────────────────

def render_predictor_module():
    st.title("🤖 No-Show Risk Predictor")
    st.markdown(
        "Machine learning–powered risk assessment using Random Forest trained on patient history. "
        "Scores range from 0% (very likely to attend) to 100% (very likely to no-show)."
    )
    st.divider()

    tab1, tab2, tab3 = st.tabs([
        "🔍 Single Patient",
        "📊 Batch Analysis",
        "🧠 Model Insights"
    ])
    with tab1:
        _single_patient_predictor()
    with tab2:
        _batch_risk_analysis()
    with tab3:
        _feature_importance_chart()
