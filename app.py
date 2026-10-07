import os
import joblib
import numpy as np
import streamlit as st

MODEL_FILE = "rover_risk_model.joblib"

st.set_page_config(page_title="Mars Rover Risk Predictor", page_icon="🚀")
st.title("🚀 Mars Rover Predictive Immobilization System")
st.caption("MVP: Predicting terrain and traction risk before rover immobilization")

if not os.path.exists(MODEL_FILE):
    st.error("Model not found. Run generate_data.py and train_model.py first.")
    st.stop()

bundle = joblib.load(MODEL_FILE)
model = bundle["model"]

st.sidebar.header("Rover Sensor Inputs")

motor_current = st.sidebar.slider("Motor current (A)", 0.5, 4.5, 1.8, 0.1)
left = st.sidebar.slider("Left wheel speed (RPM)", 20, 180, 120, 1)
right = st.sidebar.slider("Right wheel speed (RPM)", 20, 180, 120, 1)
tilt = st.sidebar.slider("Rover tilt (degrees)", 0.0, 25.0, 4.0, 0.5)
distance = st.sidebar.slider("Obstacle distance (cm)", 8, 150, 55, 1)
moisture = st.sidebar.slider("Terrain moisture (%)", 0, 100, 35, 1)

avg_speed = (left + right) / 2
slip = abs(left - right) / max(avg_speed, 1) * 100

X = np.array([[motor_current, left, right, tilt, distance, moisture, slip]])
prediction = model.predict(X)[0]
probs = model.predict_proba(X)[0]

st.metric("Wheel slip", f"{slip:.1f}%")

if prediction == "SAFE":
    st.success("🟢 SAFE — Rover can continue.")
    action = "Continue normal movement."
elif prediction == "HIGH_RISK":
    st.warning("🟡 HIGH RISK — Conditions are degrading.")
    action = "Reduce speed and evaluate an alternate path."
else:
    st.error("🔴 IMMOBILIZATION RISK — Rover may become stuck.")
    action = "Stop or reroute before entering the high-risk region."

st.subheader(f"Prediction: {prediction.replace('_', ' ')}")
st.write(f"**Recommended action:** {action}")

st.subheader("Model confidence")
for label, probability in zip(model.classes_, probs):
    st.write(f"{label}: {probability:.1%}")
    st.progress(float(probability))

st.info(
    "This MVP uses simulated sensor data. At HACKTIVATE, the input layer can "
    "be replaced by live ESP32/rover sensor data."
)
