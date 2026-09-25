"""
MediCare+ — AI-Powered Patient Appointment Management System
Main Streamlit application entry point.

Run with:  streamlit run app.py
"""

import sys
import os

# Ensure the project root is on sys.path so subpackages (data/, modules/, ml/)
# are always importable — required for Streamlit Cloud deployment.
_root = os.path.dirname(os.path.abspath(__file__))
if _root not in sys.path:
    sys.path.insert(0, _root)

import streamlit as st
from data.database import init_db

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MediCare+ | Patient Appointment System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "MediCare+ — AI-Powered Patient Appointment Management System\nPowered by Gemini 2.5 Flash & scikit-learn"
    }
)

# ── Initialize Database ───────────────────────────────────────────────────────
init_db()

# ── Sidebar Navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style='text-align:center; padding: 10px 0 6px 0;'>
            <span style='font-size:2.4rem;'>🏥</span><br>
            <span style='font-size:1.3rem; font-weight:700; color:#1f2328;'>MediCare+</span><br>
            <span style='font-size:0.78rem; color:#57606a;'>AI Patient Appointment System</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.divider()

    pages = {
        "📅 Book Appointment":     "booking",
        "📊 Admin Dashboard":      "dashboard",
        "🤖 No-Show Risk Predictor": "predictor",
        "🤖 MediAssist AI":         "ai_assistant",
        "👤 Patient Profiles":      "profiles",
    }

    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "booking"

    for label, key in pages.items():
        if st.sidebar.button(
            label,
            use_container_width=True,
            type="primary" if st.session_state["current_page"] == key else "secondary"
        ):
            st.session_state["current_page"] = key
            st.rerun()

    st.divider()
    st.markdown(
        "<div style='font-size:0.72rem; color:#57606a; text-align:center;'>"
        "Powered by<br><b>Gemini 2.5 Flash</b> + <b>scikit-learn</b><br>"
        "Built with Streamlit"
        "</div>",
        unsafe_allow_html=True
    )

# ── Page Routing ──────────────────────────────────────────────────────────────
page = st.session_state.get("current_page", "booking")

if page == "booking":
    from modules.booking import render_booking_module
    render_booking_module()

elif page == "dashboard":
    from modules.admin_dashboard import render_admin_dashboard
    render_admin_dashboard()

elif page == "predictor":
    from modules.predictor_ui import render_predictor_module
    render_predictor_module()

elif page == "ai_assistant":
    from modules.ai_assistant import render_ai_assistant
    render_ai_assistant()

elif page == "profiles":
    from modules.patient_profiles import render_patient_profiles
    render_patient_profiles()
