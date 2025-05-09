import streamlit as st
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

# --- Secure login logic ---
def check_password():
    def password_entered():
        if st.session_state["password"] == "your-secret-password":  # <-- CHANGE THIS
            st.session_state["authenticated"] = True
        else:
            st.session_state["authenticated"] = False
            st.warning("😕 Incorrect password")

    if "authenticated" not in st.session_state:
        st.text_input("🔐 Enter password", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["authenticated"]:
        st.text_input("🔐 Enter password", type="password", on_change=password_entered, key="password")
        return False
    else:
        return True

# --- Only show the app if password is correct ---
if check_password():
    # Load model and features
    model = joblib.load("relapse_risk_model.pkl")
    features = joblib.load("relapse_risk_features.pkl")

    # Hardcoded relapse contribution from each day of month
    relapse_day_contrib = {
        1: 0.021, 2: 0.022, 3: 0.025, 4: 0.032, 5: 0.038, 6: 0.045, 7: 0.021,
        8: 0.026, 9: 0.030, 10: 0.034, 11: 0.035, 12: 0.033, 13: 0.028, 14: 0.027,
        15: 0.021, 16: 0.022, 17: 0.026, 18: 0.031, 19: 0.032, 20: 0.034, 21: 0.029,
        22: 0.035, 23: 0.037, 24: 0.036, 25: 0.025, 26: 0.039, 27: 0.040, 28: 0.041,
        29: 0.042, 30: 0.044, 31: 0.015
    }

    # Risk curve based on streak age
    def streak_weight(s):
        return 0.6 + 0.4 * (np.cos(np.pi * s / 10) ** 2)

    # Core prediction function
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

    # --- UI ---
    st.title("🧠 Relapse Risk Predictor")

    date = st.date_input("Select Date", value=datetime.today())
    streak = st.number_input("Enter Current Streak Age", min_value=0, max_value=100, value=0)

    if st.button("Predict"):
        risk, base = predict_relapse_risk(str(date), streak)
        st.success(f"📅 {date.strftime('%A, %d %B %Y')}")
        st.markdown(f"🔥 **Adjusted Relapse Risk:** `{risk}%`")
        st.caption(f"(Base model risk was {base}%)")
