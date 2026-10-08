import numpy as np
import pandas as pd

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# -----------------------------
# 1. Load rover dataset
# -----------------------------
data = pd.read_csv("rover_data.csv")

features = [
    "motor_current_a",
    "slip_ratio_pct",
    "tilt_deg",
    "obstacle_distance_cm"
]

X = data[features].astype(float)
y = data["risk"]

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)


# -----------------------------
# 2. Normalize features
# -----------------------------
minimum = X.min()
maximum = X.max()

X_normalized = (X - minimum) / (maximum - minimum + 1e-9)


# -----------------------------
# 3. Quantum feature generator
# -----------------------------
simulator = AerSimulator()


def quantum_features(values):

    qc = QuantumCircuit(4, 4)

    # Encode rover features
    for i, value in enumerate(values):
        qc.ry(float(value) * np.pi, i)

    # Feature interactions
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(2, 3)

    # Measure
    qc.measure([0, 1, 2, 3], [0, 1, 2, 3])

    # Execute circuit
    result = simulator.run(
        qc,
        shots=256
    ).result()

    counts = result.get_counts()

    probabilities = np.zeros(16)

    for state, count in counts.items():
        probabilities[int(state, 2)] = count / 256

    return probabilities


# -----------------------------
# 4. Generate quantum features
# -----------------------------
print("Generating quantum features...")

quantum_X = []

for _, row in X_normalized.iterrows():
    quantum_X.append(
        quantum_features(row.values)
    )

quantum_X = np.array(quantum_X)

print("Quantum features created:", quantum_X.shape)


# -----------------------------
# 5. Combine classical + quantum
# -----------------------------
classical_features = X_normalized.values

hybrid_X = np.concatenate(
    [classical_features, quantum_X],
    axis=1
)

print(
    "Combined hybrid feature matrix:",
    hybrid_X.shape
)


# -----------------------------
# 6. Train/test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    hybrid_X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)


# -----------------------------
# 7. Hybrid classifier
# -----------------------------
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)


# -----------------------------
# 8. Results
# -----------------------------
accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n==============================")
print("IMPROVED HYBRID MODEL")
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