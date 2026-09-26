"""
Gemini AI Assistant Module.
Provides a conversational AI interface powered by Google Gemini 2.5 Flash.
Uses the official google-genai SDK (v2+).
The assistant has full context of appointment data and can answer healthcare queries,
provide recommendations, and help with patient management decisions.
"""

import streamlit as st
from google import genai
from google.genai import types
import pandas as pd
from datetime import date
import os
from dotenv import load_dotenv

from data.database import get_appointments, get_patients, get_doctors

load_dotenv()


# ── Configuration ─────────────────────────────────────────────────────────────

MODEL_NAME      = "gemini-flash-latest"   # primary
MODEL_FALLBACK  = "gemini-3.8-flash"      # fallback if primary is unavailable
CHAT_HISTORY    = "gemini_chat_history"
CHAT_SESSION    = "gemini_chat_session"

SYSTEM_PROMPT = """You are MediAssist AI, a specialized healthcare assistant integrated into a 
Patient Appointment Management System. You help clinic staff, administrators, and healthcare 
providers with:

1. Understanding patient no-show patterns and risk factors
2. Appointment scheduling recommendations  
3. Analyzing department workload and capacity
4. Answering questions about specific patients or appointments
5. Providing evidence-based suggestions to reduce no-show rates
6. Interpreting analytics and KPIs from the dashboard

You have access to real-time clinic data that is injected into your context.
Always be concise, professional, and prioritize patient privacy.
When discussing risk scores, explain them clearly and suggest actionable interventions.
Never fabricate patient data — only reference what is provided in the context.
If asked something outside healthcare/clinic management, politely redirect."""


def _get_api_key() -> str | None:
    """Retrieve Gemini API key from session state or environment."""
    if "gemini_api_key" in st.session_state and st.session_state["gemini_api_key"]:
        return st.session_state["gemini_api_key"]
    env_key = os.getenv("GEMINI_API_KEY", "")
    if env_key and env_key != "your_gemini_api_key_here":
        return env_key
    return None


def _build_context_summary() -> str:
    """Build a text summary of current clinic data to inject as context."""
    appts    = get_appointments()
    patients = get_patients()
    doctors  = get_doctors()
    today    = date.today().isoformat()

    total_appts     = len(appts)
    scheduled       = len(appts[appts["status"] == "Scheduled"])
    completed       = len(appts[appts["status"] == "Completed"])
    no_shows        = len(appts[appts["status"] == "No-Show"])
    today_count     = len(appts[appts["appointment_date"] == today])
    high_risk_count = len(appts[(appts["no_show_risk"] >= 0.6) & (appts["status"] == "Scheduled")])

    dept_load = appts.groupby("department").size().to_dict()
    top_depts = sorted(dept_load.items(), key=lambda x: x[1], reverse=True)[:4]

    context = f"""
=== CURRENT CLINIC DATA (as of {today}) ===

📊 Appointment Summary:
- Total appointments in system: {total_appts}
- Scheduled (upcoming): {scheduled}
- Completed: {completed}
- No-Shows: {no_shows} ({round(no_shows/max(total_appts,1)*100,1)}% no-show rate)
- Today's appointments: {today_count}
- High-risk upcoming (risk ≥ 60%): {high_risk_count}

🏥 Departments by volume: {', '.join(f'{d}: {c}' for d, c in top_depts)}

👨‍⚕️ Available doctors: {len(doctors)} ({', '.join(doctors['name'].tolist()[:4])} ...)

👥 Registered patients: {len(patients)}
- Patients with chronic conditions: {(patients['chronic_conditions'] != 'None').sum()}

⚠️ High-Risk Upcoming Appointments:
"""
    high_risk = appts[
        (appts["status"] == "Scheduled") & (appts["no_show_risk"] >= 0.6)
    ][["patient_name", "doctor", "appointment_date", "no_show_risk"]].head(5)

    if high_risk.empty:
        context += "  None at this time.\n"
    else:
        for _, row in high_risk.iterrows():
            context += (
                f"  - {row['patient_name']} with {row['doctor']} on {row['appointment_date']} "
                f"(risk: {int(float(row['no_show_risk'])*100)}%)\n"
            )

    context += "\n=== END OF CLINIC DATA ==="
    return context


def _initialize_history():
    if CHAT_HISTORY not in st.session_state:
        st.session_state[CHAT_HISTORY] = []


def _call_model(client, model: str, contents: list) -> str:
    """Call a specific model and return response text."""
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.4,
            max_output_tokens=2048
        )
    )
    return response.text


def _send_message(api_key: str, user_message: str) -> str:
    """Send a message with automatic retry and model fallback for 503 errors."""
    import time
    client = genai.Client(api_key=api_key)

    # Build conversation history
    history = st.session_state.get(CHAT_HISTORY, [])
    contents = []
    for msg in history:
        role = "user" if msg["role"] == "user" else "model"
        contents.append(types.Content(
            role=role,
            parts=[types.Part(text=msg["content"])]
        ))

    context = _build_context_summary()
    full_user_text = f"{context}\n\nUser question: {user_message}"
    contents.append(types.Content(
        role="user",
        parts=[types.Part(text=full_user_text)]
    ))

    # Try primary model with 2 retries, then fall back
    for attempt in range(3):
        try:
            return _call_model(client, MODEL_NAME, contents)
        except Exception as e:
            err = str(e)
            if "503" in err or "UNAVAILABLE" in err:
                if attempt < 2:
                    time.sleep(2 ** attempt)   # 1s, 2s backoff
                    continue
                # Primary exhausted — try fallback
                try:
                    return _call_model(client, MODEL_FALLBACK, contents)
                except Exception as e2:
                    raise Exception(str(e2)) from None
            raise   # non-503 errors bubble up immediately


# ── Suggested Prompts ─────────────────────────────────────────────────────────

SUGGESTED_PROMPTS = [
    "Which patients are at highest risk of not showing up this week?",
    "What are the most common reasons for no-shows in our clinic?",
    "Suggest strategies to reduce our no-show rate.",
    "Which department has the most cancellations?",
    "What interventions work best for high-risk patients?",
    "Summarize today's appointment load.",
    "How should we handle double-booking for high-risk slots?",
    "What factors most influence no-show probability?",
]


# ── Render ────────────────────────────────────────────────────────────────────

def render_ai_assistant():
    st.title("🤖 MediAssist AI — Gemini 3.8 Flash")
    st.markdown(
        "Your intelligent healthcare assistant. Ask anything about appointments, "
        "patient risk, scheduling strategy, or clinic analytics."
    )
    st.divider()

    _initialize_history()

    # ── API Key Management ────────────────────────────────────────────────────
    api_key = _get_api_key()
    if not api_key:
        st.warning("⚠️ Gemini API key not configured. Enter your key below to activate the AI assistant.")
        with st.expander("🔑 Configure Gemini API Key", expanded=True):
            key_input = st.text_input(
                "Gemini API Key",
                type="password",
                placeholder="AIza...",
                help="Get your API key from https://aistudio.google.com/app/apikey"
            )
            if st.button("Save API Key", type="primary"):
                if key_input.strip():
                    st.session_state["gemini_api_key"] = key_input.strip()
                    st.session_state.pop(CHAT_SESSION, None)
                    st.success("API key saved! You can now chat with MediAssist AI.")
                    st.rerun()
                else:
                    st.error("Please enter a valid API key.")
        st.stop()

    # ── Sidebar Controls ──────────────────────────────────────────────────────
    with st.sidebar:
        st.divider()
        st.markdown("### 🤖 AI Assistant")
        if st.button("🗑️ Clear Chat History"):
            st.session_state[CHAT_HISTORY] = []
            st.session_state.pop(CHAT_SESSION, None)
            st.rerun()
        st.caption(f"Model: {MODEL_NAME}")

    # ── Suggested Prompts ─────────────────────────────────────────────────────
    if not st.session_state[CHAT_HISTORY]:
        st.markdown("#### 💡 Try asking:")
        cols = st.columns(2)
        for i, prompt in enumerate(SUGGESTED_PROMPTS):
            with cols[i % 2]:
                if st.button(f"_{prompt}_", key=f"sug_{i}", use_container_width=True):
                    st.session_state["_pending_prompt"] = prompt
                    st.rerun()

    # ── Chat Display ──────────────────────────────────────────────────────────
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state[CHAT_HISTORY]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # ── Input ─────────────────────────────────────────────────────────────────
    pending    = st.session_state.pop("_pending_prompt", None)
    user_input = st.chat_input("Ask MediAssist AI about appointments, risk, scheduling...") or pending

    if user_input:
        with chat_container:
            with st.chat_message("user"):
                st.markdown(user_input)
        st.session_state[CHAT_HISTORY].append({"role": "user", "content": user_input})

        try:
            with st.spinner("MediAssist AI is thinking..."):
                answer = _send_message(api_key, user_input)

            with chat_container:
                with st.chat_message("assistant"):
                    st.markdown(answer)

            st.session_state[CHAT_HISTORY].append({"role": "assistant", "content": answer})

        except Exception as e:
            err_msg = str(e)
            if "API_KEY_INVALID" in err_msg or "api key" in err_msg.lower():
                st.error("❌ Invalid API key. Please reconfigure your Gemini API key.")
                st.session_state.pop("gemini_api_key", None)
            elif "quota" in err_msg.lower() or "429" in err_msg:
                st.error("❌ API quota exceeded. Please check your Gemini API usage limits.")
            elif "model" in err_msg.lower() and "not found" in err_msg.lower():
                st.error(f"❌ Model `{MODEL_NAME}` not available. Check your API access tier.")
            else:
                st.error(f"❌ Gemini API error: {err_msg}")
