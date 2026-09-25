"""
Booking Module — Patient appointment booking & management.
Handles new patient registration, appointment booking, and viewing/cancelling bookings.
"""

import streamlit as st
import pandas as pd
from datetime import date, timedelta

from data.database import (
    get_patients, get_doctors, get_appointments,
    add_patient, add_appointment, update_appointment_status,
    get_patient_appointment_history
)
from ml.predictor import predict_no_show_risk


# ── Helpers ───────────────────────────────────────────────────────────────────

DAY_MAP = {
    "Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6
}

def _available_dates(doctor_row, days_ahead: int = 30) -> list[date]:
    """Return the next `days_ahead` available appointment dates for a doctor."""
    allowed_days = [DAY_MAP[d.strip()] for d in doctor_row["available_days"].split(",")]
    available = []
    today = date.today()
    for offset in range(1, days_ahead + 1):
        d = today + timedelta(days=offset)
        if d.weekday() in allowed_days:
            available.append(d)
    return available


def _available_times(doctor_row) -> list[str]:
    return [t.strip() for t in doctor_row["available_times"].split(",")]


def _get_booked_slots(doctor_name: str, appt_date: date) -> set:
    appts = get_appointments()
    mask = (
        (appts["doctor"] == doctor_name) &
        (appts["appointment_date"] == appt_date.isoformat()) &
        (appts["status"].isin(["Scheduled"]))
    )
    return set(appts[mask]["appointment_time"].tolist())


# ── Tabs ──────────────────────────────────────────────────────────────────────

def _tab_register_patient():
    st.subheader("🆕 Register New Patient")
    with st.form("register_patient_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Full Name *")
            age  = st.number_input("Age *", min_value=1, max_value=120, value=30)
            gender = st.selectbox("Gender *", ["Male", "Female", "Other"])
            blood_type = st.selectbox("Blood Type", ["A+","A-","B+","B-","O+","O-","AB+","AB-"])
        with col2:
            phone = st.text_input("Phone Number *")
            email = st.text_input("Email Address")
            chronic_conditions = st.selectbox(
                "Chronic Conditions",
                ["None", "Hypertension", "Diabetes", "Asthma", "Heart Disease",
                 "Obesity", "COPD", "Cancer", "Kidney Disease", "Other"]
            )
        submitted = st.form_submit_button("Register Patient", type="primary")

    if submitted:
        if not name or not phone:
            st.error("Name and Phone are required.")
        else:
            pid = add_patient({
                "name": name, "age": int(age), "gender": gender,
                "phone": phone, "email": email,
                "blood_type": blood_type, "chronic_conditions": chronic_conditions
            })
            st.success(f"✅ Patient **{name}** registered successfully! Patient ID: **{pid}**")
            st.balloons()


def _tab_book_appointment():
    st.subheader("📅 Book an Appointment")

    patients = get_patients()
    doctors  = get_doctors()

    with st.form("book_appointment_form"):
        col1, col2 = st.columns(2)

        with col1:
            patient_options = {
                f"{row['name']} ({row['patient_id']})": row
                for _, row in patients.iterrows()
            }
            selected_patient_key = st.selectbox(
                "Select Patient *",
                list(patient_options.keys())
            )
            patient_row = patient_options[selected_patient_key]

            dept_options = sorted(doctors["department"].unique().tolist())
            department = st.selectbox("Department *", dept_options)

        with col2:
            dept_doctors = doctors[doctors["department"] == department]
            doctor_options = {row["name"]: row for _, row in dept_doctors.iterrows()}
            selected_doctor_key = st.selectbox("Select Doctor *", list(doctor_options.keys()))
            doctor_row = doctor_options[selected_doctor_key]

            reason = st.selectbox(
                "Reason for Visit *",
                ["Routine Checkup", "Follow-up", "New Complaint",
                 "Lab Results Review", "Medication Refill", "Consultation", "Emergency"]
            )

        available_dates = _available_dates(doctor_row)
        if not available_dates:
            st.warning("No available dates in the next 30 days.")
            st.form_submit_button("Book Appointment", disabled=True)
            return

        appt_date = st.selectbox(
            "Appointment Date *",
            available_dates,
            format_func=lambda d: d.strftime("%A, %B %d, %Y")
        )

        booked_slots = _get_booked_slots(selected_doctor_key, appt_date)
        all_times    = _available_times(doctor_row)
        free_times   = [t for t in all_times if t not in booked_slots]

        if not free_times:
            st.warning("All slots are taken for this date. Please pick another date.")
            st.form_submit_button("Book Appointment", disabled=True)
            return

        appt_time = st.selectbox("Appointment Time *", free_times)
        extra_notes = st.text_area("Additional Notes (optional)", height=80)

        submitted = st.form_submit_button("Book Appointment", type="primary")

    if submitted:
        # Predict no-show risk before inserting
        history = get_patient_appointment_history(patient_row["patient_id"])
        risk = predict_no_show_risk(patient_row, history)

        aid = add_appointment({
            "patient_id":       patient_row["patient_id"],
            "patient_name":     patient_row["name"],
            "doctor":           selected_doctor_key,
            "department":       department,
            "appointment_date": appt_date.isoformat(),
            "appointment_time": appt_time,
            "reason":           reason,
            "no_show_risk":     risk,
            "notes":            extra_notes
        })
        risk_pct = int(risk * 100)
        risk_color = "🟢" if risk < 0.3 else "🟡" if risk < 0.6 else "🔴"
        st.success(
            f"✅ Appointment **{aid}** booked for **{patient_row['name']}** "
            f"on **{appt_date.strftime('%B %d, %Y')}** at **{appt_time}** "
            f"with **{selected_doctor_key}**.\n\n"
            f"{risk_color} No-Show Risk Score: **{risk_pct}%**"
        )
        if risk >= 0.6:
            st.warning(
                "⚠️ High no-show risk detected. Consider sending a reminder "
                "or scheduling a confirmation call."
            )


def _tab_my_appointments():
    st.subheader("📋 View & Manage Appointments")

    patients = get_patients()
    patient_options = {
        f"{row['name']} ({row['patient_id']})": row["patient_id"]
        for _, row in patients.iterrows()
    }
    selected_key = st.selectbox("Select Patient", list(patient_options.keys()))
    pid = patient_options[selected_key]

    history = get_patient_appointment_history(pid)

    if history.empty:
        st.info("No appointments found for this patient.")
        return

    tabs = st.tabs(["Upcoming", "All History"])

    with tabs[0]:
        upcoming = history[history["status"] == "Scheduled"].copy()
        if upcoming.empty:
            st.info("No upcoming appointments.")
        else:
            for _, row in upcoming.iterrows():
                with st.expander(
                    f"📅 {row['appointment_date']} {row['appointment_time']} — "
                    f"{row['department']} with {row['doctor']} "
                    f"[{row['appointment_id']}]"
                ):
                    c1, c2 = st.columns(2)
                    c1.metric("Doctor", row["doctor"])
                    c1.metric("Department", row["department"])
                    c2.metric("Date", row["appointment_date"])
                    c2.metric("Time", row["appointment_time"])
                    st.write(f"**Reason:** {row['reason']}")
                    risk_pct = int(float(row["no_show_risk"]) * 100)
                    risk_label = "Low" if risk_pct < 30 else "Medium" if risk_pct < 60 else "High"
                    st.progress(risk_pct, text=f"No-Show Risk: {risk_pct}% ({risk_label})")
                    if st.button(
                        f"❌ Cancel Appointment",
                        key=f"cancel_{row['appointment_id']}"
                    ):
                        update_appointment_status(row["appointment_id"], "Cancelled")
                        st.success("Appointment cancelled.")
                        st.rerun()

    with tabs[1]:
        display_cols = ["appointment_id", "appointment_date", "appointment_time",
                        "doctor", "department", "reason", "status", "no_show_risk"]
        styled = history[display_cols].rename(columns={
            "appointment_id": "ID", "appointment_date": "Date",
            "appointment_time": "Time", "doctor": "Doctor",
            "department": "Dept", "reason": "Reason",
            "status": "Status", "no_show_risk": "Risk Score"
        })
        st.dataframe(styled, use_container_width=True)


# ── Entry point ───────────────────────────────────────────────────────────────

def render_booking_module():
    st.title("🏥 Patient Appointment Booking")
    st.markdown("Book appointments, register new patients, and manage existing bookings.")
    st.divider()

    tab1, tab2, tab3 = st.tabs([
        "📅 Book Appointment",
        "🆕 Register Patient",
        "📋 My Appointments"
    ])
    with tab1:
        _tab_book_appointment()
    with tab2:
        _tab_register_patient()
    with tab3:
        _tab_my_appointments()
