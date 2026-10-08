import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# Load rover dataset
data = pd.read_csv("rover_data.csv")

# Use the same four features as the quantum model
features = [
    "motor_current_a",
    "slip_ratio_pct",
    "tilt_deg",
    "obstacle_distance_cm"
]

X = data[features].astype(float)
y = data["risk"]

# Convert risk labels to numbers
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Same train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# Classical machine-learning model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("==============================")
print("CLASSICAL BASELINE")
print("==============================")

print(f"\nAccuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=encoder.classes_
    )
)