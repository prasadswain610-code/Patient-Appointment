"""
Patient Profiles Module — View detailed patient records with appointment history,
risk timeline, and health summary.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import date

from data.database import get_patients, get_patient_appointment_history


def _risk_timeline_chart(history: pd.DataFrame):
    if history.empty or "no_show_risk" not in history.columns:
        return None
    df = history.copy()
    df["risk_pct"] = (df["no_show_risk"].astype(float) * 100).round(1)
    df["appointment_date"] = pd.to_datetime(df["appointment_date"])
    df = df.sort_values("appointment_date")
    fig = px.line(
        df, x="appointment_date", y="risk_pct",
        markers=True, title="No-Show Risk Score Over Time",
        color_discrete_sequence=["#f59e0b"],
        labels={"appointment_date": "Date", "risk_pct": "Risk %"}
    )
    fig.add_hline(y=60, line_dash="dash", line_color="red", annotation_text="High Risk Threshold")
    fig.update_layout(height=260, margin=dict(t=40, b=0))
    return fig


def render_patient_profiles():
    st.title("👤 Patient Profiles")
    st.markdown("View complete patient records, appointment history, and risk trends.")
    st.divider()

    patients = get_patients()

    col1, col2 = st.columns([1, 3])
    with col1:
        search = st.text_input("🔍 Search patient", placeholder="Name or ID")
        if search:
            mask = (
                patients["name"].str.contains(search, case=False, na=False) |
                patients["patient_id"].str.contains(search, case=False, na=False)
            )
            filtered = patients[mask]
        else:
            filtered = patients

        patient_options = {
            f"{row['name']} ({row['patient_id']})": row["patient_id"]
            for _, row in filtered.iterrows()
        }
        if not patient_options:
            st.warning("No patients found.")
            return

        selected_key = st.radio("Select Patient", list(patient_options.keys()))
        pid = patient_options[selected_key]

    with col2:
        patient = patients[patients["patient_id"] == pid].iloc[0]
        history = get_patient_appointment_history(pid)

        # Profile header
        st.markdown(f"## {patient['name']}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Patient ID", patient["patient_id"])
        c2.metric("Age", patient["age"])
        c3.metric("Gender", patient["gender"])
        c4.metric("Blood Type", patient["blood_type"])

        c5, c6, c7 = st.columns(3)
        c5.metric("Phone", patient["phone"])
        c6.metric("Email", patient["email"] or "—")
        c7.metric("Chronic Conditions", patient["chronic_conditions"])

        st.caption(f"Registered on: {patient['registered_on']}")
        st.divider()

        # Stats
        if not history.empty:
            total     = len(history)
            completed = (history["status"] == "Completed").sum()
            no_shows  = (history["status"] == "No-Show").sum()
            scheduled = (history["status"] == "Scheduled").sum()
            avg_risk  = history["no_show_risk"].astype(float).mean()

            s1, s2, s3, s4, s5 = st.columns(5)
            s1.metric("Total Appts",  total)
            s2.metric("Completed",    completed)
            s3.metric("No-Shows",     no_shows)
            s4.metric("Upcoming",     scheduled)
            s5.metric("Avg Risk",     f"{int(avg_risk*100)}%")

            fig = _risk_timeline_chart(history)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### Appointment History")
            display = history[[
                "appointment_id", "appointment_date", "appointment_time",
                "doctor", "department", "reason", "status", "no_show_risk"
            ]].rename(columns={
                "appointment_id": "ID", "appointment_date": "Date",
                "appointment_time": "Time", "doctor": "Doctor",
                "department": "Dept", "reason": "Reason",
                "status": "Status", "no_show_risk": "Risk"
            })
            st.dataframe(display, use_container_width=True, height=280)
        else:
            st.info("No appointment history for this patient.")
