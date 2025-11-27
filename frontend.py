import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import datetime
import base64
import json
import os
import time
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff

# == Teachable Machine Integration ==
# =====================  T E A C H A B L E   M A C H I N E  =====================

import tensorflow as tf

TM_MODEL_PATH = "tm_model/keras_model.h5"
TM_LABELS_PATH = "tm_model/labels.txt"

# Global cached model + labels
_tm_model = None
_tm_labels = None

def load_tm_model():
    """Loads Teachable Machine model once globally."""
    global _tm_model, _tm_labels

    if _tm_model is not None:
        return _tm_model, _tm_labels

    try:
        # Load model
        _tm_model = tf.keras.models.load_model(TM_MODEL_PATH, compile=False)

        # Load labels
        with open(TM_LABELS_PATH, "r") as f:
            _tm_labels = [line.strip() for line in f.readlines() if line.strip()]

        if not _tm_labels:
            _tm_labels = ["Class 1", "Class 2", "Class 3"]

        return _tm_model, _tm_labels

    except Exception as e:
        st.error(f"Failed to load Teachable Machine model: {e}")
        return None, None



def predict_tm_posture(image_bgr):
    """
    Run Teachable Machine image model on a BGR image.
    Returns: (label, confidence_percent) or (None, None) on error.
    """
    model, labels = load_tm_model()
    if model is None or not labels:
        return None, None

    # Teachable Machine default input size (usually 224x224)
    IMG_SIZE = (224, 224)

    try:
        # BGR -> RGB
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, IMG_SIZE)
        img_normalized = img_resized.astype("float32") / 255.0
        input_tensor = np.expand_dims(img_normalized, axis=0)

        preds = model.predict(input_tensor)
        idx = int(np.argmax(preds[0]))
        conf = float(preds[0][idx] * 100.0)
        label = labels[idx] if idx < len(labels) else f"Class {idx}"
        return label, conf
    except Exception as e:
        st.error(f"❌ TM prediction error: {e}")
        return None, None

# 🔗 IMPORT FROM LINKED BACKEND
try:
    from backend import (
        IntelliFitPatient,
        save_patient_data,
        load_patient_data,
        calculate_angle,
        get_exercise_landmarks_both,
        assess_medical_progress,
        pelvic_analyzer,
        report_generator,
        AdvancedPelvicAnalyzer,
        ProfessionalReportGenerator,
        MEDICAL_ROM_STANDARDS
    )
    print("✅ Successfully linked with backend.py")
except ImportError as e:
    st.error(f"❌ Backend linking error: {str(e)}")
    st.error("📁 Make sure 'backend.py' exists in the same folder")
    st.stop()

# 🎨 PROFESSIONAL MEDICAL UI CONFIGURATION
st.set_page_config(
    page_title="🏥 IntelliFit Pro - Medical Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.intellifit.com/help',
        'Report a bug': 'https://www.intellifit.com/bug',
        'About': "IntelliFit Pro - Professional Healthcare Analytics Platform v2.0"
    }
)

# 🎨 PROFESSIONAL MEDICAL STYLING
st.markdown("""
<style>
    /* Global App Styling */
    .stApp {
    background: linear-gradient(135deg, #d3d3d3 0%, #bcbcbc 100%);
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}
    
    /* Professional Medical Header */
    .medical-header {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 50%, #1e40af 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(37, 99, 235, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .medical-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    .medical-header p {
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    /* Professional Sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
        border-right: 2px solid #334155;
    }
    
    /* Dashboard Cards */
    .dashboard-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #475569;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 4px 16px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    
    .dashboard-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.3);
        border-color: #2563eb;
    }
    
    .dashboard-card h3 {
        color: #60a5fa;
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .dashboard-card p {
        color: #cbd5e1;
        font-size: 0.9rem;
        line-height: 1.5;
        margin: 0;
    }
    
    /* Status Indicators */
    .status-card {
        background: linear-gradient(135deg, #059669 0%, #047857 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .status-card.warning {
        background: linear-gradient(135deg, #d97706 0%, #b45309 100%);
    }
    
    .status-card.danger {
        background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
    }
    
    /* Professional Metrics */
    .metric-container {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #475569;
        text-align: center;
        margin: 0.5rem 0;
        color: white;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #60a5fa;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #cbd5e1;
        margin: 0.5rem 0 0 0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Camera Feed Professional Styling */
    .camera-feed {
        border: 2px solid #2563eb;
        border-radius: 12px;
        background: #1e293b;
        padding: 1rem;
        box-shadow: 0 8px 24px rgba(37, 99, 235, 0.2);
    }
    
    .camera-feed img {
        border-radius: 8px;
        width: 100%;
        max-width: 640px;
        height: auto;
    }
    
    /* Professional Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.7rem 1.5rem;
        font-weight: 600;
        font-size: 0.9rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(37, 99, 235, 0.4);
        background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%);
    }
    
    /* Professional Alerts */
    .alert-success {
        background: linear-gradient(135deg, #059669 0%, #047857 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #10b981;
    }
    
    .alert-warning {
        background: linear-gradient(135deg, #d97706 0%, #b45309 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #f59e0b;
    }
    
    .alert-danger {
        background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #ef4444;
    }
    
    /* Professional Forms */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div {
        background: #334155 !important;
        border: 1px solid #475569 !important;
        border-radius: 8px !important;
        color: white !important;
    }
    
    /* Live Indicator */
    .live-indicator {
        background: #ef4444;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        animation: pulse 2s infinite;
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    /* Professional Progress Bars */
    .progress-container {
        background: #334155;
        border-radius: 10px;
        overflow: hidden;
        margin: 1rem 0;
    }
    
    .progress-bar {
        height: 8px;
        background: linear-gradient(90deg, #2563eb 0%, #60a5fa 100%);
        transition: width 0.3s ease;
    }
</style>
""", unsafe_allow_html=True)

# 🔧 SESSION STATE INITIALIZATION
def init_session_state():
    """Initialize all session state variables"""
    session_vars = {
        "patient": None,
        "analysis_running": False,
        "current_analysis": None,
        "rep_count": 0,
        "exercise_stage": "down",
        "session_logs": [],
        "dashboard_view": "overview",
        "user_preferences": {"theme": "dark", "notifications": True},
        "system_status": {"camera": "ready", "ai": "ready", "storage": "ready"},
        "selected_page": "dashboard"
    }
    
    for var, default_value in session_vars.items():
        if var not in st.session_state:
            st.session_state[var] = default_value

init_session_state()

# 🏥 PROFESSIONAL SIDEBAR NAVIGATION
def render_professional_sidebar():
    """Render professional medical sidebar with proper navigation"""
    with st.sidebar:
        # Logo and Title
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0; border-bottom: 1px solid #334155; margin-bottom: 1rem;">
            <h2 style="color: #60a5fa; margin: 0; font-weight: 700;">🏥 IntelliFit Pro</h2>
            <p style="color: #94a3b8; font-size: 0.8rem; margin: 0.5rem 0 0 0;">Medical Analytics Platform v2.0</p>
        </div>
        """, unsafe_allow_html=True)
        
        # System Status
        st.markdown("### 📊 System Status")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="status-card">
                🟢 AI Engine<br>
                <small>Online</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="status-card">
                📹 Camera<br>
                <small>Ready</small>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Navigation Menu
        st.markdown("### 🧭 Navigation")
        
        # Navigation buttons
        if st.button("🏠 Dashboard", use_container_width=True, key="nav_dashboard"):
            st.session_state.selected_page = "dashboard"
            st.rerun()
            
        if st.button("👤 Patient Management", use_container_width=True, key="nav_patients"):
            st.session_state.selected_page = "patients"
            st.rerun()
            
        if st.button("📐 Pelvic Analysis", use_container_width=True, key="nav_analysis"):
            st.session_state.selected_page = "analysis"
            st.rerun()
            
        if st.button("🏋️ Exercise Therapy", use_container_width=True, key="nav_exercises"):
            st.session_state.selected_page = "exercises"
            st.rerun()
            
        if st.button("📊 Analytics", use_container_width=True, key="nav_analytics"):
            st.session_state.selected_page = "analytics"
            st.rerun()
            
        if st.button("📋 Reports", use_container_width=True, key="nav_reports"):
            st.session_state.selected_page = "reports"
            st.rerun()
        
        # Patient Quick Info
        if st.session_state.patient:
            st.markdown("---")
            st.markdown("### 👤 Current Patient")
            st.markdown(f"""
            <div class="dashboard-card">
                <strong>{st.session_state.patient.name}</strong><br>
                <small>ID: {st.session_state.patient.patient_id}</small><br>
                <small>Age: {st.session_state.patient.age}</small><br>
                <small>Condition: {st.session_state.patient.condition}</small>
            </div>
            """, unsafe_allow_html=True)
        
        # Quick Actions
        st.markdown("---")
        st.markdown("### ⚡ Quick Actions")
        if st.button("🚨 Emergency Analysis", type="primary", key="emergency_btn"):
            st.session_state.selected_page = "analysis"
            st.rerun()
        if st.button("📷 Quick Capture", key="quick_capture_btn"):
            st.session_state.selected_page = "analysis"
            st.rerun()
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #64748b; font-size: 0.7rem;">
            <strong>IntelliFit Pro</strong><br>
            Medical Grade Analytics<br>
            <small>v2.0 | Licensed for Clinical Use</small>
        </div>
        """, unsafe_allow_html=True)
    
    return st.session_state.selected_page

# 📊 PROFESSIONAL DASHBOARD COMPONENTS
def render_dashboard_metrics():
    """Render professional dashboard metrics"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_patients = 1 if st.session_state.patient else 0
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{total_patients}</div>
            <div class="metric-label">Active Patients</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        analyses_count = len(st.session_state.patient.pelvic_history) if st.session_state.patient else 0
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{analyses_count}</div>
            <div class="metric-label">Total Analyses</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        sessions_count = len(st.session_state.patient.sessions) if st.session_state.patient else 0
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{sessions_count}</div>
            <div class="metric-label">Exercise Sessions</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_score = "N/A"
        if st.session_state.patient and st.session_state.patient.pelvic_history:
            scores = [h['posture_score'] for h in st.session_state.patient.pelvic_history]
            avg_score = f"{np.mean(scores):.1f}"
        
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{avg_score}</div>
            <div class="metric-label">Avg Posture Score</div>
        </div>
        """, unsafe_allow_html=True)

def render_professional_alerts():
    """Render professional medical alerts"""
    st.markdown("### 🚨 Clinical Alerts")
    
    if st.session_state.patient and st.session_state.patient.pelvic_history:
        latest_analysis = st.session_state.patient.pelvic_history[-1]
        score = latest_analysis['posture_score']
        
        if score >= 8:
            st.markdown("""
            <div class="alert-success">
                ✅ <strong>Excellent Clinical Status</strong><br>
                Patient demonstrates excellent postural alignment. Continue current therapy protocol.
            </div>
            """, unsafe_allow_html=True)
        elif score >= 6:
            st.markdown("""
            <div class="alert-warning">
                ⚠️ <strong>Moderate Clinical Attention</strong><br>
                Patient shows moderate postural deviation. Consider therapy plan adjustments.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="alert-danger">
                🚨 <strong>Immediate Clinical Intervention Required</strong><br>
                Patient shows significant postural dysfunction. Schedule urgent consultation.
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="alert-warning">
            📋 <strong>No Clinical Data Available</strong><br>
            Register a patient and perform analysis to view clinical status alerts.
        </div>
        """, unsafe_allow_html=True)

# 🏥 MAIN APPLICATION PAGES
def render_dashboard_page():
    """Render main professional dashboard"""
    st.markdown("""
    <div class="medical-header">
        <h1>🏥 IntelliFit Pro Dashboard</h1>
        <p>Professional Healthcare Analytics & Advanced Posture Assessment Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Dashboard Metrics
    render_dashboard_metrics()
    
    # Two column layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Recent Clinical Activity
        st.markdown("### 📈 Recent Clinical Activity")
        if st.session_state.patient and st.session_state.patient.pelvic_history:
            df_data = []
            for entry in st.session_state.patient.pelvic_history[-5:]:
                df_data.append({
                    'Date': entry['date'].strftime('%Y-%m-%d %H:%M'),
                    'Pelvic Tilt': f"{entry['left_pelvic_angle']:.1f}°",
                    'Posture Score': f"{entry['posture_score']:.1f}/10",
                    'Clinical Status': entry['interpretation'][:50] + "..." if len(entry['interpretation']) > 50 else entry['interpretation']
                })
            
            if df_data:
                df = pd.DataFrame(df_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("📊 No recent clinical activity. Start by registering a patient and performing analysis.")
        
        # Progress Visualization
        if st.session_state.patient and len(st.session_state.patient.pelvic_history) > 1:
            st.markdown("### 📊 Clinical Progress Visualization")
            df_chart = pd.DataFrame([
                {
                    'Date': entry['date'],
                    'Posture Score': entry['posture_score'],
                    'Pelvic Tilt': abs(entry['left_pelvic_angle'])
                }
                for entry in st.session_state.patient.pelvic_history
            ])
            
            fig = px.line(df_chart, x='Date', y=['Posture Score', 'Pelvic Tilt'],
                         title="📈 Patient Clinical Progress Tracking",
                         color_discrete_sequence=['#60a5fa', '#f59e0b'])
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                height=400,
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Clinical Status Alerts
        render_professional_alerts()
        
        # Patient Summary Card
        if st.session_state.patient:
            st.markdown("### 👤 Patient Summary")
            progress = st.session_state.patient.get_progress_summary()
            
            # Clinical status color coding
            clinical_status = progress.get('clinical_status', 'No data')
            if 'Excellent' in clinical_status:
                status_class = 'status-card'
            elif 'Good' in clinical_status:
                status_class = 'status-card warning'
            else:
                status_class = 'status-card danger'
            
            st.markdown(f"""
            <div class="dashboard-card">
                <h3>📊 Clinical Overview</h3>
                <p><strong>Patient:</strong> {st.session_state.patient.name}</p>
                <p><strong>Therapy Duration:</strong> {progress.get('therapy_duration_days', 0)} days</p>
                <p><strong>Total Assessments:</strong> {progress.get('total_analyses', 0)}</p>
                <p><strong>Exercise Sessions:</strong> {progress.get('total_sessions', 0)}</p>
            </div>
            
            <div class="{status_class}">
                <strong>Clinical Status</strong><br>
                <small>{clinical_status}</small>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="dashboard-card">
                <h3>👤 No Active Patient</h3>
                <p>Please register a patient to begin clinical assessment and monitoring.</p>
            </div>
            """, unsafe_allow_html=True)

# (Patient management + other pages stay exactly as in your version – I’m not changing those)
def render_dashboard_page():
    """Render main professional dashboard"""
    st.markdown("""
    <div class="medical-header">
        <h1>🏥 IntelliFit Pro Dashboard</h1>
        <p>Professional Healthcare Analytics & Advanced Posture Assessment Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Dashboard Metrics
    render_dashboard_metrics()
    
    # Two column layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Recent Clinical Activity
        st.markdown("### 📈 Recent Clinical Activity")
        if st.session_state.patient and st.session_state.patient.pelvic_history:
            df_data = []
            for entry in st.session_state.patient.pelvic_history[-5:]:
                df_data.append({
                    'Date': entry['date'].strftime('%Y-%m-%d %H:%M'),
                    'Pelvic Tilt': f"{entry['left_pelvic_angle']:.1f}°",
                    'Posture Score': f"{entry['posture_score']:.1f}/10",
                    'Clinical Status': entry['interpretation'][:50] + "..." if len(entry['interpretation']) > 50 else entry['interpretation']
                })
            
            if df_data:
                df = pd.DataFrame(df_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("📊 No recent clinical activity. Start by registering a patient and performing analysis.")
        
        # Progress Visualization
        if st.session_state.patient and len(st.session_state.patient.pelvic_history) > 1:
            st.markdown("### 📊 Clinical Progress Visualization")
            df_chart = pd.DataFrame([
                {
                    'Date': entry['date'],
                    'Posture Score': entry['posture_score'],
                    'Pelvic Tilt': abs(entry['left_pelvic_angle'])
                }
                for entry in st.session_state.patient.pelvic_history
            ])
            
            fig = px.line(df_chart, x='Date', y=['Posture Score', 'Pelvic Tilt'],
                         title="📈 Patient Clinical Progress Tracking",
                         color_discrete_sequence=['#60a5fa', '#f59e0b'])
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                height=400,
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Clinical Status Alerts
        render_professional_alerts()
        
        # Patient Summary Card
        if st.session_state.patient:
            st.markdown("### 👤 Patient Summary")
            progress = st.session_state.patient.get_progress_summary()
            
            # Clinical status color coding
            clinical_status = progress.get('clinical_status', 'No data')
            if 'Excellent' in clinical_status:
                status_class = 'status-card'
            elif 'Good' in clinical_status:
                status_class = 'status-card warning'
            else:
                status_class = 'status-card danger'
            
            st.markdown(f"""
            <div class="dashboard-card">
                <h3>📊 Clinical Overview</h3>
                <p><strong>Patient:</strong> {st.session_state.patient.name}</p>
                <p><strong>Therapy Duration:</strong> {progress.get('therapy_duration_days', 0)} days</p>
                <p><strong>Total Assessments:</strong> {progress.get('total_analyses', 0)}</p>
                <p><strong>Exercise Sessions:</strong> {progress.get('total_sessions', 0)}</p>
            </div>
            
            <div class="{status_class}">
                <strong>Clinical Status</strong><br>
                <small>{clinical_status}</small>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="dashboard-card">
                <h3>👤 No Active Patient</h3>
                <p>Please register a patient to begin clinical assessment and monitoring.</p>
            </div>
            """, unsafe_allow_html=True)

def render_patient_management_page():
    """Enhanced patient management with professional medical forms"""
    st.markdown("""
    <div class="medical-header">
        <h1>👤 Patient Management System</h1>
        <p>Professional Patient Registration & Comprehensive Medical History Management</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Professional Patient Registration
    st.markdown("### 📝 Professional Patient Registration")
    
    with st.form("enhanced_patient_registration", clear_on_submit=False):
        st.markdown("#### 📋 Patient Demographics & Contact Information")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            patient_id = st.text_input("🆔 Patient ID*", placeholder="Unique patient identifier")
            full_name = st.text_input("👤 Full Name*", placeholder="Last, First Middle")
            
        with col2:
            age = st.number_input("📅 Age*", min_value=0, max_value=120, value=25)
            gender = st.selectbox("⚧ Gender", ["Male", "Female"])
            
        with col3:
            email = st.text_input("📧 Email Address", placeholder="patient@email.com")
        
        st.markdown("#### 🏥 Medical Information")
        col1, col2 = st.columns(2)
        
        with col1:
            condition = st.selectbox("🩺 Primary Diagnosis*", [
                "Pelvic Floor Dysfunction",
                "Lumbar Spine Disorder",
                "Post-Surgical Rehabilitation",
                "Spinal Alignment Disorder", 
                "Sports-Related Injury",
                "Chronic Pain Syndrome",
                "Postural Correction Needed",
                "Musculoskeletal Disorder",
                "Neurological Rehabilitation",
                "Work-Related Injury",
                "Motor Vehicle Accident",
                "General Physiotherapy Assessment"
            ])
            
            referring_physician = st.text_input("👨‍⚕️ Referring Physician", placeholder="Dr. Last Name")
            primary_complaint = st.text_area("🩺 Chief Complaint", placeholder="Patient's primary concern or symptoms")
            
       
        st.markdown("#### 📋 Clinical Assessment & History")
        col1, col2 = st.columns(2)
        
        with col1:
            pain_level = st.slider("🩹 Current Pain Level (0-10)", 0, 10, 0)
            previous_therapy = st.checkbox("Previous Physiotherapy Treatment")
            surgical_history = st.checkbox("Relevant Surgical History")
            
        with col2:
            medications = st.text_area("💊 Current Medications", placeholder="List current medications")
            medical_history = st.text_area("🏥 Relevant Medical History", placeholder="Relevant medical conditions")
        
        clinical_notes = st.text_area("📝 Initial Clinical Notes", 
                                     placeholder="Initial assessment notes, observations, and treatment goals")
        
        # Submit button
        submitted = st.form_submit_button("✅ Register Patient", type="primary", use_container_width=True)
        
        if submitted:
            if patient_id.strip() and full_name.strip() and age > 0 and condition:
                # Create enhanced patient profile
                new_patient = IntelliFitPatient(
                    patient_id.strip(),
                    full_name.strip(),
                    age,
                    condition,
                    datetime.datetime.now()
                )
                
                # Add enhanced professional attributes
                new_patient.gender = gender
                new_patient.email = email
                new_patient.referring_physician = referring_physician
                new_patient.primary_complaint = primary_complaint
                new_patient.clinical_notes = clinical_notes
                new_patient.current_pain_level = pain_level
                new_patient.medications = medications
                new_patient.medical_history = medical_history
                new_patient.previous_therapy = previous_therapy
                new_patient.surgical_history = surgical_history
                
                # Registration timestamp
                new_patient.registration_date = datetime.datetime.now()
                
                st.session_state.patient = new_patient
                
                if save_patient_data(new_patient):
                    st.markdown("""
                    <div class="alert-success">
                        ✅ <strong>Patient Successfully Registered!</strong><br>
                        Professional patient profile has been created with complete medical information.<br>
                        Patient is now ready for clinical assessment and treatment planning.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="alert-danger">
                        ❌ <strong>Registration Failed</strong><br>
                        Unable to save patient data to system. Please verify all information and try again.
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="alert-warning">
                    ⚠️ <strong>Required Information Missing</strong><br>
                    Please complete all required fields marked with (*) before submitting registration.
                </div>
                """, unsafe_allow_html=True)
    
    # Current Patient Profile Display
    if st.session_state.patient:
        st.markdown("---")
        st.markdown("### 👤 Current Patient Profile")
        
        # Professional patient info display
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="dashboard-card">
                <h3>📋 Demographics</h3>
                <p><strong>Name:</strong> {st.session_state.patient.name}</p>
                <p><strong>ID:</strong> {st.session_state.patient.patient_id}</p>
                <p><strong>Age:</strong> {st.session_state.patient.age}</p>
                <p><strong>Gender:</strong> {getattr(st.session_state.patient, 'gender', 'Not specified')}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="dashboard-card">
                <h3>🏥 Medical Information</h3>
                <p><strong>Diagnosis:</strong> {st.session_state.patient.condition}</p>
                <p><strong>Physician:</strong> {getattr(st.session_state.patient, 'referring_physician', 'Not specified')}</p>
                <p><strong>Start Date:</strong> {st.session_state.patient.therapy_start_date.strftime('%Y-%m-%d')}</p>
                <p><strong>Pain Level:</strong> {getattr(st.session_state.patient, 'current_pain_level', 'Not assessed')}/10</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            progress = st.session_state.patient.get_progress_summary()
            st.markdown(f"""
            <div class="dashboard-card">
                <h3>📈 Clinical Progress</h3>
                <p><strong>Duration:</strong> {progress.get('therapy_duration_days', 0)} days</p>
                <p><strong>Assessments:</strong> {progress.get('total_analyses', 0)}</p>
                <p><strong>Sessions:</strong> {progress.get('total_sessions', 0)}</p>
                <p><strong>Status:</strong> {progress.get('clinical_status', 'No data')[:30]}...</p>
                <p><strong>Last Visit:</strong> {datetime.datetime.now().strftime('%Y-%m-%d')}</p>
            </div>
            """, unsafe_allow_html=True)


# ---------------------- PELVIC ANALYSIS PAGES -------------------------

def render_pelvic_analysis_page():
    """Professional pelvic analysis with advanced clinical interface"""
    st.markdown("""
    <div class="medical-header">
        <h1>📐 Professional Pelvic Analysis</h1>
        <p>Advanced AI-Powered Clinical Posture Assessment & Diagnostic Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.patient:
        st.markdown("""
        <div class="alert-warning">
            ⚠️ <strong>No Patient Selected</strong><br>
            Please register a patient in the Patient Management section before performing clinical analysis.
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Analysis method selection
    st.markdown("### 🎯 Clinical Analysis Method")
    analysis_method = st.radio(
        "Select Professional Analysis Method:",
        ["📷 Static Image Analysis", "🎥 Live Camera Assessment", "📊 Historical Analysis Review"],
        horizontal=True
    )
    
    if analysis_method == "📷 Static Image Analysis":
        render_static_image_analysis()
    elif analysis_method == "🎥 Live Camera Assessment":
        render_live_camera_analysis()
    else:
        render_historical_analysis_review()

def render_static_image_analysis():
    """Professional static image analysis interface with TM"""
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### 📤 Upload Clinical Image")
        
        uploaded_file = st.file_uploader(
            "Select patient assessment image",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a clear, full-body frontal or sagittal view image for optimal clinical analysis"
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.markdown('<div class="camera-feed">', unsafe_allow_html=True)
            st.image(image, caption="📷 Patient Assessment Image", width=350)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Image quality check
            img_array = np.array(image)
            height, width = img_array.shape[:2]
            
            if width < 400 or height < 600:
                st.warning("⚠️ Image resolution may be too low for optimal analysis")
            else:
                st.success("✅ Image quality suitable for clinical analysis")
    
    with col2:
        if uploaded_file and st.button("🔍 Perform Clinical Analysis", type="primary", use_container_width=True):
            with st.spinner("🔄 Processing advanced clinical analysis..."):
                # Convert and process image
                image_array = np.array(image)
                if len(image_array.shape) == 3:
                    image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                else:
                    image_bgr = image_array
                
                # MediaPipe processing
                mp_pose = mp.solutions.pose
                pose = mp_pose.Pose(
                    static_image_mode=True,
                    model_complexity=2,
                    min_detection_confidence=0.7,
                    min_tracking_confidence=0.5
                )
                
                results = pose.process(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
                
                if results.pose_landmarks:
                    # Extract landmark coordinates
                    landmark_array = []
                    for lm in results.pose_landmarks.landmark:
                        landmark_array.append([lm.x, lm.y, lm.z])
                    
                    # Perform advanced pelvic analysis
                    analysis_result = pelvic_analyzer.calculate_pelvic_tilt(landmark_array)

                    # ---------- NEW: Teachable Machine prediction ----------
                    tm_label, tm_conf = predict_tm_posture(image_bgr)
                    if tm_label is not None:
                        analysis_result["tm_label"] = tm_label
                        analysis_result["tm_confidence"] = tm_conf
                    # ------------------------------------------------------
                    
                    if 'error' not in analysis_result:
                        
                        # Display comprehensive results
                        st.markdown("### 📊 Clinical Analysis Results")
                        
                        # Primary metrics display
                        col_a, col_b, col_c, col_d, col_e = st.columns(5)
                        
                        with col_a:
                            st.markdown(f"""
                            <div class="metric-container">
                                <div class="metric-value">{analysis_result.get('pelvic_tilt_angle', 0):.1f}°</div>
                                <div class="metric-label">Pelvic Tilt</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col_b:
                            st.markdown(f"""
                            <div class="metric-container">
                                <div class="metric-value">{analysis_result.get('lateral_tilt_angle', 0):.1f}°</div>
                                <div class="metric-label">Lateral Tilt</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col_c:
                            score = analysis_result.get('posture_score', 0)
                            score_color = "#10b981" if score >= 8 else "#f59e0b" if score >= 6 else "#ef4444"
                            st.markdown(f"""
                            <div class="metric-container">
                                <div class="metric-value" style="color: {score_color}">{score:.1f}</div>
                                <div class="metric-label">Posture Score</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col_d:
                            confidence = analysis_result.get('measurement_confidence', 0)
                            st.markdown(f"""
                            <div class="metric-container">
                                <div class="metric-value">{confidence:.0f}%</div>
                                <div class="metric-label">MP Confidence</div>
                            </div>
                            """, unsafe_allow_html=True)

                        with col_e:
                            tm_label = analysis_result.get("tm_label", "N/A")
                            tm_conf = analysis_result.get("tm_confidence", 0.0)
                            st.markdown(f"""
                            <div class="metric-container">
                                <div class="metric-value">{tm_conf:.0f}%</div>
                                <div class="metric-label">TM: {tm_label}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Professional clinical interpretation
                        st.markdown("### 🏥 Clinical Interpretation")
                        
                        col_i, col_ii = st.columns(2)
                        
                        with col_i:
                            st.markdown(f"""
                            <div class="dashboard-card">
                                <h3>🔬 Clinical Assessment</h3>
                                <p><strong>Functional Impact:</strong><br>{analysis_result.get('functional_impact', 'Unable to assess')}</p>
                                <p><strong>Analysis Quality:</strong> {analysis_result.get('analysis_quality', 'Unknown').title()}</p>
                                <p><strong>Weight Distribution:</strong> {analysis_result.get('weight_distribution', {}).get('distribution', 'Not assessed')}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col_ii:
                            st.markdown(f"""
                            <div class="dashboard-card">
                                <h3>⚠️ Risk Factors</h3>
                            """, unsafe_allow_html=True)
                            
                            risk_factors = analysis_result.get('risk_factors', [])
                            for risk in risk_factors[:4]:
                                st.markdown(f"<p>• {risk}</p>", unsafe_allow_html=True)
                            
                            st.markdown("</div>", unsafe_allow_html=True)
                        
                        # Clinical recommendations
                        st.markdown("### 📋 Evidence-Based Clinical Recommendations")
                        recommendations = analysis_result.get('clinical_recommendations', [])
                        
                        for i, rec in enumerate(recommendations[:6], 1):
                            priority = "High" if i <= 2 else "Medium" if i <= 4 else "Low"
                            priority_color = "#ef4444" if priority == "High" else "#f59e0b" if priority == "Medium" else "#10b981"
                            
                            st.markdown(f"""
                            <div class="dashboard-card">
                                <h3 style="color: {priority_color};">#{i} - {priority} Priority</h3>
                                <p>{rec}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # ---------- SAVE TO PATIENT RECORD (WITH TM DATA) ----------
                        analysis_entry = st.session_state.patient.add_pelvic_analysis(
                            analysis_result.get('pelvic_tilt_angle', 0),
                            analysis_result.get('lateral_tilt_angle', 0),
                            analysis_result.get('posture_score', 0),
                            analysis_result     # contains tm_label + tm_confidence
                        )
                        
                        if save_patient_data(st.session_state.patient):
                            st.markdown("""
                            <div class="alert-success">
                                ✅ <strong>Clinical Analysis Complete</strong><br>
                                Assessment has been saved to patient's medical record with comprehensive clinical documentation, including Teachable Machine classification.
                            </div>
                            """, unsafe_allow_html=True)
                        # ---------------------------------------------------------
                        
                    else:
                        st.markdown("""
                        <div class="alert-danger">
                            ❌ <strong>Analysis Processing Error</strong><br>
                            Unable to complete pelvic analysis. Please ensure the image shows a clear full-body view.
                        </div>
                        """, unsafe_allow_html=True)
                
                else:
                    st.markdown("""
                    <div class="alert-warning">
                        ⚠️ <strong>Pose Detection Failed</strong><br>
                        No human pose detected in the image. Please ensure:
                        • Clear full-body view (head to feet visible)
                        • Good lighting and contrast
                        • Patient standing upright
                        • Minimal background distractions
                    </div>
                    """, unsafe_allow_html=True)
                
                pose.close()

def render_live_camera_analysis():
    """Professional live camera analysis interface with TM"""
    st.markdown("#### 🎥 Live Clinical Assessment")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.markdown("""
        <div class="dashboard-card">
            <h3>📋 Assessment Protocol</h3>
            <p>• Position patient 6-8 feet from camera</p>
            <p>• Ensure complete body visibility</p>
            <p>• Optimize lighting conditions</p>
            <p>• Patient in relaxed standing position</p>
            <p>• Minimal clothing for landmark visibility</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Camera controls
        if st.button("📹 Start Live Assessment", type="primary", use_container_width=True):
            st.session_state.analysis_running = True
            st.rerun()
        
        if st.button("⏹️ Stop Assessment", type="secondary", use_container_width=True):
            st.session_state.analysis_running = False
            st.rerun()
        
        if st.session_state.analysis_running:
            st.markdown('<div class="live-indicator">🔴 LIVE ANALYSIS</div>', unsafe_allow_html=True)
    
    with col1:
        if st.session_state.analysis_running:
            # Live camera analysis implementation
            camera_placeholder = st.empty()
            metrics_placeholder = st.empty()
            
            try:
                cap = cv2.VideoCapture(0)
                mp_pose = mp.solutions.pose
                pose = mp_pose.Pose(
                    static_image_mode=False,
                    model_complexity=1,
                    min_detection_confidence=0.7,
                    min_tracking_confidence=0.5
                )
                
                frame_count = 0
                while cap.isOpened() and st.session_state.analysis_running:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_count += 1
                    if frame_count % 3 != 0:  # Process every 3rd frame for performance
                        continue
                    
                    # Resize for consistent processing
                    frame_resized = cv2.resize(frame, (640, 480))
                    
                    # Process with MediaPipe
                    image_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                    results = pose.process(image_rgb)
                    
                    tm_label = None
                    tm_conf = None

                    if results.pose_landmarks:
                        # Draw landmarks
                        mp.solutions.drawing_utils.draw_landmarks(
                            frame_resized, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                            mp.solutions.drawing_utils.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                            mp.solutions.drawing_utils.DrawingSpec(color=(0,0,255), thickness=2)
                        )
                        
                        # Real-time analysis
                        landmark_array = []
                        for lm in results.pose_landmarks.landmark:
                            landmark_array.append([lm.x, lm.y, lm.z])
                        
                        analysis_result = pelvic_analyzer.calculate_pelvic_tilt(landmark_array)

                        # ---------- NEW: TM prediction on live frame ----------
                        tm_label, tm_conf = predict_tm_posture(frame_resized)
                        if tm_label is not None:
                            analysis_result["tm_label"] = tm_label
                            analysis_result["tm_confidence"] = tm_conf
                        # ------------------------------------------------------
                        
                        if 'error' not in analysis_result:
                            # Update metrics display
                            with metrics_placeholder.container():
                                col_a, col_b, col_c, col_d, col_e = st.columns(5)
                                
                                with col_a:
                                    st.metric("🎯 Pelvic Tilt", f"{analysis_result.get('pelvic_tilt_angle', 0):.1f}°")
                                
                                with col_b:
                                    st.metric("📐 Lateral Tilt", f"{analysis_result.get('lateral_tilt_angle', 0):.1f}°")
                                
                                with col_c:
                                    score = analysis_result.get('posture_score', 0)
                                    color = "🟢" if score >= 8 else "🟡" if score >= 6 else "🔴"
                                    st.metric(f"{color} Score", f"{score:.1f}/10")
                                
                                with col_d:
                                    st.metric("📊 MP Quality", analysis_result.get('analysis_quality', 'Unknown').title())

                                with col_e:
                                    label_display = analysis_result.get("tm_label", "N/A")
                                    conf_display = analysis_result.get("tm_confidence", 0.0)
                                    st.metric("🤖 TM Class", f"{label_display} ({conf_display:.0f}%)")
                            
                            # Store for potential saving (WITH TM)
                            st.session_state.current_analysis = analysis_result
                    
                    # Display video feed
                    camera_placeholder.image(
                        cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB),
                        caption="🎥 Live Clinical Assessment Feed",
                        width=640
                    )
                    
                    # Break after reasonable time
                    if frame_count > 1800:  # ~1 minute at 30fps
                        break
                
                cap.release()
                pose.close()
                
            except Exception as e:
                st.error(f"❌ Camera error: {str(e)}")
                st.session_state.analysis_running = False
        
        else:
            st.info("📹 Click 'Start Live Assessment' to begin real-time clinical analysis")
    
    # Save live analysis option
    if hasattr(st.session_state, 'current_analysis') and st.session_state.current_analysis:
        st.markdown("---")
        if st.button("💾 Save Live Analysis to Patient Record", type="primary"):
            analysis_entry = st.session_state.patient.add_pelvic_analysis(
                st.session_state.current_analysis.get('pelvic_tilt_angle', 0),
                st.session_state.current_analysis.get('lateral_tilt_angle', 0),
                st.session_state.current_analysis.get('posture_score', 0),
                st.session_state.current_analysis   # includes TM fields
            )
            
            if save_patient_data(st.session_state.patient):
                st.success("✅ Live analysis saved to patient medical record (with TM classification)!")
                st.session_state.current_analysis = None

# -------------- historical review + reports etc: same as your file --------------
def render_historical_analysis_review():
    """Professional historical analysis review"""
    if not st.session_state.patient.pelvic_history:
        st.info("📊 No historical analysis data available. Perform some assessments first.")
        return
    
    st.markdown("#### 📈 Historical Clinical Analysis Review")
    
    # Create comprehensive historical data table
    df_data = []
    for i, entry in enumerate(st.session_state.patient.pelvic_history):
        df_data.append({
            'Assessment #': i + 1,
            'Date': entry['date'].strftime('%Y-%m-%d %H:%M'),
            'Pelvic Tilt (°)': f"{entry['left_pelvic_angle']:.1f}",
            'Lateral Tilt (°)': f"{entry.get('right_pelvic_angle', entry['analysis_data'].get('lateral_tilt_angle', 0)):.1f}",
            'Posture Score': f"{entry['posture_score']:.1f}/10",
            'Clinical Status': entry['interpretation'][:40] + "..." if len(entry['interpretation']) > 40 else entry['interpretation'],
            'Quality': entry['analysis_data'].get('analysis_quality', 'Good').title(),
            'Confidence': f"{entry['analysis_data'].get('measurement_confidence', 0):.0f}%"
        })
    
    df = pd.DataFrame(df_data)
    
    # Display professional data table
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Professional trend analysis
    st.markdown("#### 📈 Clinical Trend Analysis")
    
    # Create trend visualization
    fig = go.Figure()
    
    dates = [entry['date'] for entry in st.session_state.patient.pelvic_history]
    posture_scores = [entry['posture_score'] for entry in st.session_state.patient.pelvic_history]
    pelvic_angles = [abs(entry['left_pelvic_angle']) for entry in st.session_state.patient.pelvic_history]
    
    # Add posture score trend
    fig.add_trace(go.Scatter(
        x=dates,
        y=posture_scores,
        mode='lines+markers',
        name='Posture Score',
        line=dict(color='#60a5fa', width=3),
        marker=dict(size=8)
    ))
    
    # Add pelvic angle trend
    fig.add_trace(go.Scatter(
        x=dates,
        y=pelvic_angles,
        mode='lines+markers',
        name='Pelvic Deviation (°)',
        line=dict(color='#f59e0b', width=3),
        marker=dict(size=8),
        yaxis='y2'
    ))
    
    # Update layout for professional appearance
    fig.update_layout(
        title="📊 Comprehensive Clinical Progress Tracking",
        xaxis_title="Assessment Date",
        yaxis_title="Posture Score (0-10)",
        yaxis2=dict(
            title="Pelvic Deviation (degrees)",
            overlaying='y',
            side='right'
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        height=500,
        showlegend=True,
        legend=dict(
            bgcolor='rgba(30, 41, 59, 0.8)',
            bordercolor='#475569',
            borderwidth=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Clinical summary statistics
    st.markdown("#### 📋 Clinical Summary Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_score = np.mean(posture_scores)
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{avg_score:.1f}</div>
            <div class="metric-label">Average Score</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        score_trend = posture_scores[-1] - posture_scores[0] if len(posture_scores) > 1 else 0
        trend_color = "#10b981" if score_trend > 0 else "#ef4444" if score_trend < 0 else "#64748b"
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value" style="color: {trend_color}">{score_trend:+.1f}</div>
            <div class="metric-label">Score Change</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        best_score = max(posture_scores)
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{best_score:.1f}</div>
            <div class="metric-label">Best Score</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        consistency = 100 - (np.std(posture_scores) / np.mean(posture_scores) * 100)
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{consistency:.0f}%</div>
            <div class="metric-label">Consistency</div>
        </div>
        """, unsafe_allow_html=True)

def render_reports_page():
    """Professional clinical reports interface"""
    st.markdown("""
    <div class="medical-header">
        <h1>📋 Clinical Documentation & Reports</h1>
        <p>Professional Medical Reports & Comprehensive Clinical Documentation</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.patient:
        st.markdown("""
        <div class="alert-warning">
            ⚠️ <strong>No Patient Selected</strong><br>
            Please register a patient and perform clinical assessments before generating reports.
        </div>
        """, unsafe_allow_html=True)
        return
    
    if not st.session_state.patient.pelvic_history:
        st.markdown("""
        <div class="alert-warning">
            📊 <strong>No Clinical Data Available</strong><br>
            No assessment data found for this patient. Please perform clinical analysis before generating reports.
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Professional report options
    st.markdown("### 📋 Professional Report Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="dashboard-card">
            <h3>👤 Patient Summary</h3>
            <p><strong>Name:</strong> {st.session_state.patient.name}</p>
            <p><strong>ID:</strong> {st.session_state.patient.patient_id}</p>
            <p><strong>Assessments:</strong> {len(st.session_state.patient.pelvic_history)}</p>
            <p><strong>Latest Score:</strong> {st.session_state.patient.pelvic_history[-1]['posture_score']:.1f}/10</p>
            <p><strong>Report Date:</strong> {datetime.datetime.now().strftime('%Y-%m-%d')}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        latest_analysis = st.session_state.patient.pelvic_history[-1]
        progress = st.session_state.patient.get_progress_summary()
        
        st.markdown(f"""
        <div class="dashboard-card">
            <h3>📊 Clinical Status</h3>
            <p><strong>Pelvic Tilt:</strong> {latest_analysis['left_pelvic_angle']:.1f}°</p>
            <p><strong>Lateral Tilt:</strong> {latest_analysis['analysis_data'].get('lateral_tilt_angle', 0):.1f}°</p>
            <p><strong>Quality:</strong> {latest_analysis['analysis_data'].get('analysis_quality', 'Good').title()}</p>
            <p><strong>Status:</strong> {progress.get('clinical_status', 'No data')[:30]}...</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Report generation options
    st.markdown("### 📄 Generate Professional Report")
    
    report_options = st.multiselect(
        "Select report components:",
        [
            "Patient Demographics & Medical History",
            "Complete Assessment History",
            "Latest Clinical Analysis",
            "Progress Trend Analysis",
            "Clinical Recommendations",
            "Risk Factor Assessment",
            "Treatment Planning Suggestions"
        ],
        default=[
            "Patient Demographics & Medical History",
            "Latest Clinical Analysis", 
            "Clinical Recommendations"
        ]
    )
    
    report_format = st.selectbox(
        "Report Format:",
        ["Comprehensive Text Report", "Clinical Summary Report", "Progress Report"]
    )
    
    # Generate report button
    if st.button("📄 Generate Professional Clinical Report", type="primary", use_container_width=True):
        with st.spinner("🔄 Generating comprehensive clinical documentation..."):
            try:
                # Get latest analysis for report
                latest_analysis_data = st.session_state.patient.pelvic_history[-1]['analysis_data']
                
                # Prepare patient data for report
                patient_data_dict = {
                    'name': st.session_state.patient.name,
                    'patient_id': st.session_state.patient.patient_id,
                    'age': st.session_state.patient.age,
                    'gender': getattr(st.session_state.patient, 'gender', 'Not specified'),
                    'condition': st.session_state.patient.condition,
                    'referring_physician': getattr(st.session_state.patient, 'referring_physician', 'Not specified'),
                    'email': getattr(st.session_state.patient, 'email', 'Not provided')
                }
                
                # Generate professional report
                report_path = report_generator.generate_posture_report(patient_data_dict, latest_analysis_data)
                
                if report_path and os.path.exists(report_path):
                    # Read the generated report
                    with open(report_path, 'r', encoding='utf-8') as f:
                        report_content = f.read()
                    
                    # Success message
                    st.markdown("""
                    <div class="alert-success">
                        ✅ <strong>Professional Clinical Report Generated Successfully!</strong><br>
                        Comprehensive medical documentation has been created with all clinical findings and recommendations.
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Download button
                    st.download_button(
                        label="📥 Download Professional Clinical Report",
                        data=report_content.encode('utf-8'),
                        file_name=f"IntelliFit_Clinical_Report_{st.session_state.patient.patient_id}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        help="Download comprehensive clinical documentation for patient medical record",
                        use_container_width=True
                    )
                    
                    # Report preview
                    with st.expander("📄 Clinical Report Preview", expanded=False):
                        st.text(report_content[:1500] + "\n\n[... Report continues with complete clinical documentation ...]")
                    
                    # Report statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.info(f"📄 Report Type: {report_format}")
                    with col2:
                        st.info(f"👤 Patient: {st.session_state.patient.name}")
                    with col3:
                        st.info(f"📅 Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
                    
                    # Cleanup temporary file
                    try:
                        os.remove(report_path)
                    except:
                        pass
                
                else:
                    st.markdown("""
                    <div class="alert-danger">
                        ❌ <strong>Report Generation Failed</strong><br>
                        Unable to generate clinical report. Please try again or contact system administrator.
                    </div>
                    """, unsafe_allow_html=True)
                
            except Exception as e:
                st.markdown(f"""
                <div class="alert-danger">
                    ❌ <strong>Report Generation Error</strong><br>
                    Technical error occurred: {str(e)}<br>
                    Please verify patient data and try again.
                </div>
                """, unsafe_allow_html=True)

# (I’m not rewriting the rest because it’s unchanged; just keep your existing
# render_historical_analysis_review(), render_reports_page(), etc.)

# 🚀 MAIN APPLICATION CONTROLLER
def main():
    """Main application controller with proper routing"""
    try:
        # Render sidebar and get selected page
        selected_page = render_professional_sidebar()
        
        # Route to appropriate page based on selection
        if selected_page == "dashboard":
            render_dashboard_page()
        elif selected_page == "patients":
            # use your existing render_patient_management_page()
            render_patient_management_page()
        elif selected_page == "analysis":
            render_pelvic_analysis_page()
        elif selected_page == "reports":
            render_reports_page()
        else:
            # Default to dashboard
            render_dashboard_page()
        
        # Professional footer
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #64748b; padding: 2rem 0; border-top: 1px solid #334155;'>
            <strong>🏥 IntelliFit Pro</strong> - Professional Healthcare Analytics Platform<br>
            <small>Medical-Grade AI Analysis | Clinical Documentation | Evidence-Based Recommendations</small><br>
            <small>© 2025 IntelliFit Technologies. Licensed for Professional Medical Use.</small><br>
            <small><em>This software is designed for use by qualified healthcare professionals as a clinical assessment tool.</em></small>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"❌ Application Error: {str(e)}")
        st.info("🔧 Please restart the application or contact system administrator.")

# 🚀 APPLICATION ENTRY POINT
if __name__ == "__main__":
    main()
