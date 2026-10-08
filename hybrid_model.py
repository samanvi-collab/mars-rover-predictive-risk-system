import numpy as np
import pandas as pd

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
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

# Target: rover risk
y = data["risk"]

# Convert risk labels to numbers
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Normalize features
X_min = X.min()
X_max = X.max()

X_normalized = (X - X_min) / (X_max - X_min + 1e-9)


# -----------------------------
# 2. Quantum feature generator
# -----------------------------
simulator = AerSimulator()


def quantum_features(values):
    """
    Convert four rover features into
    quantum measurement probabilities.
    """

    qc = QuantumCircuit(4, 4)

    # Encode rover features
    for i, value in enumerate(values):
        qc.ry(float(value) * np.pi, i)

    # Interactions between rover features
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(2, 3)

    # Measure
    qc.measure([0, 1, 2, 3], [0, 1, 2, 3])

    # Run circuit
    result = simulator.run(qc, shots=256).result()
    counts = result.get_counts()

    # Convert measurement counts into probabilities
    probabilities = np.zeros(16)

    for state, count in counts.items():
        index = int(state, 2)
        probabilities[index] = count / 256

    return probabilities


# -----------------------------
# 3. Generate quantum features
# -----------------------------
print("Generating quantum features...")

quantum_X = []

for _, row in X_normalized.iterrows():
    quantum_X.append(
        quantum_features(row.values)
    )

quantum_X = np.array(quantum_X)

print("Quantum feature matrix shape:", quantum_X.shape)


# -----------------------------
# 4. Train hybrid classifier
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    quantum_X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

model = LogisticRegression(
    max_iter=1000
)

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("HYBRID QUANTUM-CLASSICAL MODEL")
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


# -----------------------------
# 5. Test one rover condition
# -----------------------------
sample = X_normalized.iloc[0].values

sample_quantum = quantum_features(sample)

prediction = model.predict(
    sample_quantum.reshape(1, -1)
)[0]

risk = encoder.inverse_transform([prediction])[0]

print("\nExample rover prediction:")
print("Predicted risk:", risk)