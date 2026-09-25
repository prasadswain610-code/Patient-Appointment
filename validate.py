"""Headless validation script for the MediCare+ application."""
import sys
sys.path.insert(0, ".")

import streamlit as st

class FakeSS(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)
    def __setattr__(self, k, v):
        self[k] = v

st.session_state = FakeSS()

# ── DB layer ──────────────────────────────────────────────────────────────────
from data.database import (
    init_db, get_patients, get_doctors, get_appointments,
    add_patient, add_appointment, update_appointment_status,
    update_appointment_risk, get_patient_appointment_history,
    get_upcoming_appointments
)

init_db()
pts   = get_patients()
docs  = get_doctors()
appts = get_appointments()

assert len(pts)   == 20, f"Expected 20 patients, got {len(pts)}"
assert len(docs)  == 8,  f"Expected 8 doctors, got {len(docs)}"
assert len(appts) == 60, f"Expected 60 appointments, got {len(appts)}"

pid = add_patient({
    "name": "Test Patient", "age": 45, "gender": "Male",
    "phone": "555-0001", "email": "t@t.com",
    "blood_type": "O+", "chronic_conditions": "None"
})
assert pid.startswith("P"), "Patient ID format error"
assert len(get_patients()) == 21, "Patient count after add"

aid = add_appointment({
    "patient_id": pid, "patient_name": "Test Patient",
    "doctor": "Dr. Sarah Kim", "department": "Cardiology",
    "appointment_date": "2025-12-01", "appointment_time": "10:00",
    "reason": "Checkup", "no_show_risk": 0.25
})
assert aid.startswith("A"), "Appointment ID format error"

update_appointment_status(aid, "Completed")
row_status = get_appointments().loc[get_appointments()["appointment_id"] == aid, "status"].iloc[0]
assert row_status == "Completed", f"Status not updated: {row_status}"

update_appointment_risk(aid, 0.75)
row_risk = float(get_appointments().loc[get_appointments()["appointment_id"] == aid, "no_show_risk"].iloc[0])
assert abs(row_risk - 0.75) < 0.01, f"Risk not updated: {row_risk}"

hist = get_patient_appointment_history(pid)
assert len(hist) == 1, f"History length: {len(hist)}"

upcoming = get_upcoming_appointments()
assert upcoming["status"].eq("Scheduled").all(), "Upcoming has non-Scheduled rows"

print("DB layer: ALL TESTS PASSED")

# ── ML layer ──────────────────────────────────────────────────────────────────
from ml.predictor import predict_no_show_risk, get_feature_importance, FEATURES

pts2 = get_patients()
for i in range(5):
    p = pts2.iloc[i]
    h = get_appointments()[get_appointments()["patient_id"] == p["patient_id"]]
    r = predict_no_show_risk(p, h)
    assert 0.0 <= r <= 1.0, f"Risk {r} out of [0,1] for patient {p['name']}"

fi = get_feature_importance(get_appointments(), get_patients())
assert list(fi.columns) == ["Feature", "Importance"], f"Wrong columns: {fi.columns.tolist()}"
assert len(fi) == len(FEATURES), f"Feature count mismatch: {len(fi)} vs {len(FEATURES)}"
importance_sum = fi["Importance"].sum()
assert abs(importance_sum - 1.0) < 0.01, f"Importances sum to {importance_sum}, not 1.0"

print("ML predictor: ALL TESTS PASSED")

# ── google-genai SDK import ───────────────────────────────────────────────────
from google import genai
from google.genai import types
c = genai.Client.__init__  # just verify it's accessible
print("google-genai SDK: IMPORT OK")

# ── plotly import ─────────────────────────────────────────────────────────────
import plotly.express as px
import plotly.graph_objects as go
print("plotly: IMPORT OK")

print()
print("=== FULL VALIDATION PASSED ===")
