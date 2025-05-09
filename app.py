import streamlit as st
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

# Load model and features
model = joblib.load("relapse_risk_model.pkl")
features = joblib.load("relapse_risk_features.pkl")

# Fake auth password
APP_PASSWORD = "Morce1428!$@*"

relapse_day_contrib = {
    1: 0.021, 2: 0.022, 3: 0.025, 4: 0.032, 5: 0.038, 6: 0.045, 7: 0.021,
    8: 0.026, 9: 0.030, 10: 0.034, 11: 0.035, 12: 0.033, 13: 0.028, 14: 0.027,
    15: 0.021, 16: 0.022, 17: 0.026, 18: 0.031, 19: 0.032, 20: 0.034, 21: 0.029,
    22: 0.035, 23: 0.037, 24: 0.036, 25: 0.025, 26: 0.039, 27: 0.040, 28: 0.041,
    29: 0.042, 30: 0.044, 31: 0.015
}

def streak_weight(s):
    return 0.6 + 0.4 * (np.cos(np.pi * s / 10) ** 2)

def predict_relapse_risk(date_str, streak_age):
    today = pd.to_datetime(date_str)
    day = today.day
    weekday = today.weekday()
    month = today.month
    is_holiday_month = 1 if month == 12 else 0
    relapse_day_contribution = relapse_day_contrib.get(day, 0)

    features_dict = {
        'IsHolidayMonth': is_holiday_month,
        'Day_sin': np.sin(2 * np.pi * day / 31),
        'Day_cos': np.cos(2 * np.pi * day / 31),
        'Month_sin': np.sin(2 * np.pi * month / 12),
        'Month_cos': np.cos(2 * np.pi * month / 12),
        'Weekday_sin': np.sin(2 * np.pi * weekday / 7),
        'Weekday_cos': np.cos(2 * np.pi * weekday / 7),
        'RelapseDayContribution': relapse_day_contribution
    }

    df = pd.DataFrame([features_dict])[features]
    base_prob = model.predict_proba(df)[0][0]
    adjusted = base_prob * streak_weight(streak_age)
    return round(adjusted * 100, 2), round(base_prob * 100, 2)

# --- Custom CSS ---
st.markdown("""
<style>
body {
    background-color: #0d0d0d;
}
.big-title {
    font-size: 38px;
    font-weight: bold;
    color: #00ff99;
    animation: glow 2s ease-in-out infinite alternate;
    text-align: center;
}
@keyframes glow {
    from { text-shadow: 0 0 10px #00ff99; }
    to { text-shadow: 0 0 25px #00ff99; }
}
.fade-in {
    animation: fadeIn 2s ease-in;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}
.lock-icon {
    font-size: 40px;
    animation: pulse 1.5s infinite;
}
@keyframes pulse {
    0% { transform: scale(1); color: #ccc; }
    50% { transform: scale(1.2); color: #00ffcc; }
    100% { transform: scale(1); color: #ccc; }
}
</style>
""", unsafe_allow_html=True)

# --- Authentication ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.markdown('<div class="lock-icon">🔐</div>', unsafe_allow_html=True)
    st.markdown('<h1 class="big-title">Welcome, Operator</h1>', unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: grey;'>This space is private. Authentication required...</p>", unsafe_allow_html=True)

    pwd = st.text_input("Enter your secret key:", type="password")
    if st.button("Unlock"):
        if pwd == APP_PASSWORD:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("🔴 Intrusion detected. Access denied, traitor.")
    st.stop()

# --- Main App ---
st.markdown('<h1 class="fade-in">🧠 Relapse Risk Predictor</h1>', unsafe_allow_html=True)
st.success("🟢 Access Granted. Welcome back, Boss.")

date = st.date_input("Select Date", value=datetime.today())
streak = st.number_input("Enter Current Streak Age", min_value=0, max_value=100, value=0)

if st.button("Predict"):
    risk, base = predict_relapse_risk(str(date), streak)
    st.success(f"📅 {date.strftime('%A, %d %B %Y')}")
    st.markdown(f"🔥 **Adjusted Relapse Risk:** `{risk}%`")
    st.caption(f"(Base model risk was {base}%)")
