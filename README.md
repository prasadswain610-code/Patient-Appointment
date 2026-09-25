# 🏥 MediCare+ — AI-Powered Patient Appointment Management System

A fully featured **Streamlit** healthcare web application with:
- 📅 **Patient Appointment Booking** module
- 📊 **Admin Dashboard** with real-time analytics
- 🤖 **ML No-Show Risk Predictor** (Random Forest, trained on patient history)
- 🤖 **MediAssist AI** — conversational assistant powered by **Gemini 2.5 Flash**
- 👤 **Patient Profiles** with appointment history & risk timelines

> No HTML, CSS, or JavaScript — pure Python + Streamlit.

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Gemini API Key

**Option A — Environment variable (recommended):**
```bash
cp .env.example .env
# Edit .env and set: GEMINI_API_KEY=your_key_here
```

**Option B — In-app:** Enter the key in the MediAssist AI page when prompted.

Get your free API key at: https://aistudio.google.com/app/apikey

### 3. Run the app
```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
├── app.py                        # Main Streamlit entry point
├── requirements.txt
├── .env.example
│
├── data/
│   └── database.py               # In-memory pandas DataFrames (session_state)
│
├── modules/
│   ├── booking.py                # Appointment booking + patient registration
│   ├── admin_dashboard.py        # KPIs, Plotly charts, appointment management
│   ├── predictor_ui.py           # ML risk predictor UI
│   ├── ai_assistant.py           # Gemini 2.5 Flash chat assistant
│   └── patient_profiles.py       # Patient profile viewer
│
└── ml/
    └── predictor.py              # Random Forest no-show risk model
```

---

## 🧠 ML No-Show Risk Predictor

**Algorithm:** Random Forest Classifier (scikit-learn)  
**Training:** Automatically trained at runtime on session data (no pre-trained file needed)

**Features used:**
| Feature | Description |
|---|---|
| `patient_age` | Age of the patient |
| `is_chronic_patient` | Has chronic conditions |
| `gender_encoded` | Gender (0=Female, 1=Male, 2=Other) |
| `prior_no_show_rate` | Historical no-show ratio |
| `prior_total_appointments` | Total past appointments |
| `prior_cancellation_rate` | Historical cancellation ratio |
| `days_until_appointment` | Days between booking and appointment |
| `appointment_day_of_week` | 0=Monday … 6=Sunday |
| `appointment_hour` | Hour of appointment |

**Risk bands:**
- 🟢 **< 30%** — Low risk (standard scheduling)
- 🟡 **30–60%** — Medium risk (send reminder)
- 🔴 **> 60%** — High risk (phone confirmation, double-booking consideration)

---

## 🤖 MediAssist AI (Gemini 2.5 Flash)

The AI assistant receives live clinic context with every message:
- Current appointment counts by status
- Today's schedule
- High-risk upcoming appointments
- Department workload breakdown
- Patient demographics

It can help with scheduling decisions, no-show reduction strategies, workload analysis, and more.

---

## 📊 Admin Dashboard Features

- KPI cards: Total, Scheduled, Completed, No-Shows, Today's count, High-risk count
- Appointment volume trend (line chart)
- Department load (horizontal bar chart)
- Doctor workload (upcoming scheduled)
- No-show risk distribution histogram
- No-show rate by age group
- Appointment management table with filters
- One-click status updates (Completed / No-Show / Cancelled)
- High-risk patient alert section

---

## 🔧 Tech Stack

| Layer | Technology |
|---|---|
| UI Framework | Streamlit |
| AI Model | Google Gemini 2.5 Flash (google-genai SDK) |
| ML | scikit-learn RandomForestClassifier |
| Data | pandas + numpy |
| Charts | Plotly Express + Graph Objects |
| Config | python-dotenv |
