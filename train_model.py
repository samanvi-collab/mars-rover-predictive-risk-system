import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

FEATURES = [
    "motor_current_a",
    "wheel_speed_left_rpm",
    "wheel_speed_right_rpm",
    "tilt_deg",
    "obstacle_distance_cm",
    "terrain_moisture_pct",
    "slip_ratio_pct",
]

df = pd.read_csv("rover_data.csv")
X, y = df[FEATURES], df["risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=150, max_depth=8, random_state=42
)
model.fit(X_train, y_train)

pred = model.predict(X_test)
print(f"Test accuracy: {accuracy_score(y_test, pred):.2%}")
print(classification_report(y_test, pred))

joblib.dump({"model": model, "features": FEATURES}, "rover_risk_model.joblib")
print("Saved rover_risk_model.joblib")
