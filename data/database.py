"""
In-memory database using pandas DataFrames with session_state persistence.
Acts as a lightweight data layer for the Streamlit app.
"""

import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, date, timedelta
import random


# ── Schema definitions ────────────────────────────────────────────────────────

PATIENT_COLS = [
    "patient_id", "name", "age", "gender", "phone", "email",
    "blood_type", "chronic_conditions", "registered_on"
]

APPOINTMENT_COLS = [
    "appointment_id", "patient_id", "patient_name", "doctor", "department",
    "appointment_date", "appointment_time", "reason", "status",
    "no_show_risk", "booked_on", "notes"
]

DOCTOR_COLS = ["doctor_id", "name", "department", "available_days", "available_times"]


# ── Seed data helpers ─────────────────────────────────────────────────────────

def _seed_patients() -> pd.DataFrame:
    names = [
        "Alice Johnson", "Bob Martinez", "Carol White", "David Lee",
        "Emma Brown", "Frank Wilson", "Grace Davis", "Henry Miller",
        "Isla Thomas", "Jack Taylor", "Karen Anderson", "Liam Jackson",
        "Mia Harris", "Noah Thompson", "Olivia Garcia", "Peter Robinson",
        "Quinn Lewis", "Rachel Walker", "Sam Hall", "Tina Allen"
    ]
    conditions = [
        "None", "Hypertension", "Diabetes", "Asthma", "None",
        "Heart Disease", "None", "Obesity", "None", "Hypertension",
        "None", "Diabetes", "None", "None", "Asthma",
        "None", "None", "Hypertension", "None", "Diabetes"
    ]
    rows = []
    for i, (name, cond) in enumerate(zip(names, conditions)):
        rows.append({
            "patient_id": f"P{1000 + i}",
            "name": name,
            "age": random.randint(18, 80),
            "gender": random.choice(["Male", "Female"]),
            "phone": f"+1-555-{random.randint(1000,9999)}",
            "email": f"{name.split()[0].lower()}@example.com",
            "blood_type": random.choice(["A+", "A-", "B+", "B-", "O+", "O-", "AB+", "AB-"]),
            "chronic_conditions": cond,
            "registered_on": (date.today() - timedelta(days=random.randint(30, 730))).isoformat()
        })
    return pd.DataFrame(rows, columns=PATIENT_COLS)


def _seed_doctors() -> pd.DataFrame:
    data = [
        ("D001", "Dr. Sarah Kim",    "Cardiology",    "Mon,Wed,Fri", "09:00,10:00,11:00,14:00,15:00"),
        ("D002", "Dr. James Patel",  "General",       "Mon,Tue,Thu", "08:00,09:00,10:00,11:00"),
        ("D003", "Dr. Laura Chen",   "Neurology",     "Tue,Thu,Fri", "10:00,11:00,14:00,15:00,16:00"),
        ("D004", "Dr. Mark Evans",   "Orthopedics",   "Mon,Wed,Fri", "09:00,10:00,13:00,14:00"),
        ("D005", "Dr. Nina Sharma",  "Dermatology",   "Tue,Wed,Thu", "10:00,11:00,12:00,15:00"),
        ("D006", "Dr. Paul Nguyen",  "Pediatrics",    "Mon,Tue,Thu", "08:00,09:00,10:00,11:00,14:00"),
        ("D007", "Dr. Rita Lopez",   "Endocrinology", "Wed,Thu,Fri", "09:00,10:00,11:00,14:00"),
        ("D008", "Dr. Steve Brown",  "Psychiatry",    "Mon,Wed,Fri", "10:00,11:00,14:00,15:00,16:00"),
    ]
    return pd.DataFrame(data, columns=DOCTOR_COLS)


def _seed_appointments(patients: pd.DataFrame, doctors: pd.DataFrame) -> pd.DataFrame:
    statuses = ["Completed", "Scheduled", "No-Show", "Cancelled", "Scheduled"]
    rows = []
    pid_list = patients["patient_id"].tolist()
    pname_map = dict(zip(patients["patient_id"], patients["name"]))

    for i in range(60):
        pid = random.choice(pid_list)
        doc_row = doctors.sample(1).iloc[0]
        days_offset = random.randint(-120, 30)
        appt_date = date.today() + timedelta(days=days_offset)
        status = random.choice(statuses)
        if days_offset > 0:
            status = "Scheduled"
        rows.append({
            "appointment_id": f"A{2000 + i}",
            "patient_id": pid,
            "patient_name": pname_map[pid],
            "doctor": doc_row["name"],
            "department": doc_row["department"],
            "appointment_date": appt_date.isoformat(),
            "appointment_time": random.choice(["09:00", "10:00", "11:00", "14:00", "15:00"]),
            "reason": random.choice([
                "Routine Checkup", "Follow-up", "New Complaint",
                "Lab Results Review", "Medication Refill", "Consultation"
            ]),
            "status": status,
            "no_show_risk": round(random.uniform(0.05, 0.95), 2),
            "booked_on": (appt_date - timedelta(days=random.randint(1, 14))).isoformat(),
            "notes": ""
        })
    return pd.DataFrame(rows, columns=APPOINTMENT_COLS)


# ── Init / accessor ───────────────────────────────────────────────────────────

def init_db() -> None:
    """Initialize all tables in st.session_state if not already present."""
    if "db_patients" not in st.session_state:
        st.session_state["db_patients"] = _seed_patients()
    if "db_doctors" not in st.session_state:
        st.session_state["db_doctors"] = _seed_doctors()
    if "db_appointments" not in st.session_state:
        st.session_state["db_appointments"] = _seed_appointments(
            st.session_state["db_patients"],
            st.session_state["db_doctors"]
        )


def get_patients() -> pd.DataFrame:
    return st.session_state["db_patients"].copy()


def get_doctors() -> pd.DataFrame:
    return st.session_state["db_doctors"].copy()


def get_appointments() -> pd.DataFrame:
    return st.session_state["db_appointments"].copy()


def add_patient(record: dict) -> str:
    df = st.session_state["db_patients"]
    pid = f"P{1000 + len(df)}"
    record["patient_id"] = pid
    record["registered_on"] = date.today().isoformat()
    st.session_state["db_patients"] = pd.concat(
        [df, pd.DataFrame([record])], ignore_index=True
    )
    return pid


def add_appointment(record: dict) -> str:
    df = st.session_state["db_appointments"]
    aid = f"A{2000 + len(df)}"
    record["appointment_id"] = aid
    record["booked_on"] = date.today().isoformat()
    record.setdefault("status", "Scheduled")
    record.setdefault("notes", "")
    st.session_state["db_appointments"] = pd.concat(
        [df, pd.DataFrame([record])], ignore_index=True
    )
    return aid


def update_appointment_status(appointment_id: str, new_status: str) -> None:
    df = st.session_state["db_appointments"]
    df.loc[df["appointment_id"] == appointment_id, "status"] = new_status
    st.session_state["db_appointments"] = df


def update_appointment_risk(appointment_id: str, risk: float) -> None:
    df = st.session_state["db_appointments"]
    df.loc[df["appointment_id"] == appointment_id, "no_show_risk"] = round(risk, 2)
    st.session_state["db_appointments"] = df


def get_patient_appointment_history(patient_id: str) -> pd.DataFrame:
    appts = get_appointments()
    return appts[appts["patient_id"] == patient_id].sort_values("appointment_date", ascending=False)


def get_upcoming_appointments() -> pd.DataFrame:
    appts = get_appointments()
    today = date.today().isoformat()
    return appts[(appts["appointment_date"] >= today) & (appts["status"] == "Scheduled")].sort_values("appointment_date")
