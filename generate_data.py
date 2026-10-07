import numpy as np
import pandas as pd

np.random.seed(42)
N = 1200

motor_current = np.random.normal(1.8, 0.55, N).clip(0.5, 4.5)
left = np.random.normal(120, 25, N).clip(20, 180)
right = np.random.normal(120, 25, N).clip(20, 180)
tilt = np.random.normal(4, 5, N).clip(0, 25)
distance = np.random.normal(55, 25, N).clip(8, 150)
moisture = np.random.normal(35, 18, N).clip(0, 100)

avg_speed = (left + right) / 2
slip = (np.abs(left - right) / np.maximum(avg_speed, 1)) * 100

score = (
    0.30 * motor_current
    + 0.25 * slip
    + 0.18 * tilt
    + 0.15 * (100 - distance / 1.5)
    + 0.12 * moisture
    + np.random.normal(0, 2.5, N)
)

risk = np.where(score < 18, "SAFE",
         np.where(score < 32, "HIGH_RISK", "IMMOBILIZATION_RISK"))

df = pd.DataFrame({
    "motor_current_a": motor_current,
    "wheel_speed_left_rpm": left,
    "wheel_speed_right_rpm": right,
    "tilt_deg": tilt,
    "obstacle_distance_cm": distance,
    "terrain_moisture_pct": moisture,
    "slip_ratio_pct": slip,
    "risk": risk
})

df.to_csv("rover_data.csv", index=False)
print(f"Created rover_data.csv with {len(df)} samples.")
