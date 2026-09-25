"""
ML No-Show Risk Predictor.

Uses a Random Forest Classifier trained on synthetic patient history features.
The model is trained on-the-fly using the seeded dataset so no pre-trained file
is needed. Predictions are probability scores (0.0 – 1.0).

Feature engineering:
  - patient_age
  - is_chronic_patient         (1 if chronic conditions != "None")
  - gender_encoded             (0=Female, 1=Male, 2=Other)
  - prior_no_show_rate         (historical no-show rate for this patient)
  - prior_total_appointments   (total appointments made so far)
  - prior_cancellation_rate    (historical cancellation rate)
  - days_until_appointment     (how far ahead was the appointment booked)
  - appointment_day_of_week    (0=Mon … 6=Sun)
  - appointment_hour           (int, e.g. 9 for "09:00")
"""

import numpy as np
import pandas as pd
import streamlit as st
from datetime import date, datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


# ── Constants ─────────────────────────────────────────────────────────────────

MODEL_KEY      = "_noshow_model"
FEATURES       = [
    "patient_age", "is_chronic_patient", "gender_encoded",
    "prior_no_show_rate", "prior_total_appointments",
    "prior_cancellation_rate", "days_until_appointment",
    "appointment_day_of_week", "appointment_hour"
]
GENDER_MAP     = {"Male": 1, "Female": 0, "Other": 2}
CHRONIC_NONE   = "None"


# ── Feature extraction ────────────────────────────────────────────────────────

def _extract_features(patient_row: pd.Series, history: pd.DataFrame, appt_date_str: str = None) -> np.ndarray:
    """Build a feature vector for a single patient + optional appointment date."""
    age     = int(patient_row.get("age", 35))
    chronic = 0 if str(patient_row.get("chronic_conditions", "None")) == CHRONIC_NONE else 1
    gender  = GENDER_MAP.get(str(patient_row.get("gender", "Other")), 2)

    if history.empty:
        ns_rate     = 0.15   # prior estimate for new patients
        total       = 0
        cancel_rate = 0.05
    else:
        total       = len(history)
        ns_rate     = (history["status"] == "No-Show").sum() / max(total, 1)
        cancel_rate = (history["status"] == "Cancelled").sum() / max(total, 1)

    # Appointment timing features
    if appt_date_str:
        try:
            appt_date = date.fromisoformat(appt_date_str)
        except Exception:
            appt_date = date.today()
    else:
        appt_date = date.today()

    days_ahead   = (appt_date - date.today()).days
    day_of_week  = appt_date.weekday()
    appt_hour    = 10  # default; overridden by caller if needed

    return np.array([[
        age, chronic, gender,
        round(ns_rate, 3), total, round(cancel_rate, 3),
        max(days_ahead, 0), day_of_week, appt_hour
    ]])


# ── Training ──────────────────────────────────────────────────────────────────

def _build_training_data(appointments: pd.DataFrame, patients: pd.DataFrame) -> tuple:
    """Build X, y matrices from the full appointment history."""
    past = appointments[appointments["status"].isin(["Completed", "No-Show", "Cancelled"])].copy()
    if len(past) < 10:
        return None, None

    past = past.merge(
        patients[["patient_id", "age", "gender", "chronic_conditions"]],
        on="patient_id", how="left"
    )
    past["appointment_date"] = pd.to_datetime(past["appointment_date"], errors="coerce")
    past["booked_on"]        = pd.to_datetime(past["booked_on"], errors="coerce")
    past["days_ahead"]       = (past["appointment_date"] - past["booked_on"]).dt.days.clip(lower=0).fillna(3)
    past["day_of_week"]      = past["appointment_date"].dt.dayofweek.fillna(2)
    past["appt_hour"]        = past["appointment_time"].str.split(":").str[0].astype(int, errors="ignore").fillna(10)
    past["is_chronic"]       = (past["chronic_conditions"] != CHRONIC_NONE).astype(int)
    past["gender_enc"]       = past["gender"].map(GENDER_MAP).fillna(2)
    past["age"]              = past["age"].fillna(35)

    # Per-patient historical stats (computed on records before this one)
    past = past.sort_values("appointment_date")
    past["cum_total"]  = past.groupby("patient_id").cumcount()
    past["cum_ns"]     = past.groupby("patient_id")["status"].transform(
        lambda s: s.eq("No-Show").shift(1).fillna(0).cumsum()
    )
    past["cum_cancel"] = past.groupby("patient_id")["status"].transform(
        lambda s: s.eq("Cancelled").shift(1).fillna(0).cumsum()
    )
    past["prior_ns_rate"]    = (past["cum_ns"]     / past["cum_total"].clip(lower=1)).fillna(0.15)
    past["prior_cancel_rate"]= (past["cum_cancel"] / past["cum_total"].clip(lower=1)).fillna(0.05)

    X = past[[
        "age", "is_chronic", "gender_enc",
        "prior_ns_rate", "cum_total", "prior_cancel_rate",
        "days_ahead", "day_of_week", "appt_hour"
    ]].values

    y = (past["status"] == "No-Show").astype(int).values
    return X, y


def train_model(appointments: pd.DataFrame, patients: pd.DataFrame) -> RandomForestClassifier:
    """Train a Random Forest on available appointment history."""
    X, y = _build_training_data(appointments, patients)

    clf = RandomForestClassifier(
        n_estimators=150,
        max_depth=6,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42
    )
    if X is None or len(X) < 10:
        # Minimal data — fit on synthetic fallback
        rng = np.random.default_rng(42)
        X_fake = rng.random((200, len(FEATURES)))
        y_fake = (X_fake[:, 3] > 0.4).astype(int)   # ns_rate feature drives label
        clf.fit(X_fake, y_fake)
    else:
        clf.fit(X, y)

    return clf


def _get_or_train_model(appointments: pd.DataFrame, patients: pd.DataFrame) -> RandomForestClassifier:
    """Retrieve cached model from session_state or train a new one."""
    appt_hash = len(appointments)  # simple invalidation key
    cached = st.session_state.get(MODEL_KEY)

    if cached is None or st.session_state.get("_model_hash") != appt_hash:
        clf = train_model(appointments, patients)
        st.session_state[MODEL_KEY]     = clf
        st.session_state["_model_hash"] = appt_hash
        return clf

    return cached


# ── Public API ────────────────────────────────────────────────────────────────

def predict_no_show_risk(
    patient_row: pd.Series,
    history: pd.DataFrame,
    appt_date_str: str = None,
    appt_hour: int = 10
) -> float:
    """
    Returns a float in [0.0, 1.0] representing no-show probability.

    Parameters
    ----------
    patient_row    : a single row from the patients DataFrame
    history        : all past appointments for this patient
    appt_date_str  : ISO date string of the new appointment (optional)
    appt_hour      : integer hour of the appointment (optional)
    """
    from data.database import get_appointments, get_patients  # lazy to avoid circular import

    appointments = get_appointments()
    patients     = get_patients()

    clf  = _get_or_train_model(appointments, patients)
    feat = _extract_features(patient_row, history, appt_date_str)
    feat[0, 8] = appt_hour  # override hour

    prob = clf.predict_proba(feat)[0]
    # index 1 = "No-Show" class
    if clf.classes_.tolist() == [0, 1]:
        return float(prob[1])
    return float(prob[0])


def get_feature_importance(appointments: pd.DataFrame, patients: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame of feature importances from the trained model."""
    clf = _get_or_train_model(appointments, patients)
    return pd.DataFrame({
        "Feature": FEATURES,
        "Importance": clf.feature_importances_
    }).sort_values("Importance", ascending=False).reset_index(drop=True)
