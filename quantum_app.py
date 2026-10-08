import streamlit as st
import numpy as np
import pandas as pd


from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


st.set_page_config(
    page_title="Quantum Rover Risk Predictor",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 Quantum-Enhanced Mars Rover Risk Predictor")
st.write(
    "Hybrid quantum-classical prediction of rover immobilization risk."
)


# -----------------------------
# Load training data
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

# Normalize training data
X_min = X.min()
X_max = X.max()

X_normalized = (
    (X - X_min) /
    (X_max - X_min + 1e-9)
)


# -----------------------------
# Quantum feature generator
# -----------------------------
simulator = AerSimulator()


def quantum_features(values):

    qc = QuantumCircuit(4, 4)

    for i, value in enumerate(values):
        qc.ry(float(value) * np.pi, i)

    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(2, 3)

    qc.measure(
        [0, 1, 2, 3],
        [0, 1, 2, 3]
    )

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
# Build hybrid training data
# -----------------------------
@st.cache_resource
def train_hybrid_model():

    quantum_X = []

    for _, row in X_normalized.iterrows():
        quantum_X.append(
            quantum_features(row.values)
        )

    quantum_X = np.array(quantum_X)

    hybrid_X = np.concatenate(
        [
            X_normalized.values,
            quantum_X
        ],
        axis=1
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(
        hybrid_X,
        y_encoded
    )

    return model


model = train_hybrid_model()


# -----------------------------
# Rover inputs
# -----------------------------
st.sidebar.header("Rover Conditions")

motor_current = st.sidebar.slider(
    "Motor Current (A)",
    float(X["motor_current_a"].min()),
    float(X["motor_current_a"].max()),
    float(X["motor_current_a"].median())
)

slip_ratio = st.sidebar.slider(
    "Wheel Slip (%)",
    float(X["slip_ratio_pct"].min()),
    float(X["slip_ratio_pct"].max()),
    float(X["slip_ratio_pct"].median())
)

tilt = st.sidebar.slider(
    "Rover Tilt (degrees)",
    float(X["tilt_deg"].min()),
    float(X["tilt_deg"].max()),
    float(X["tilt_deg"].median())
)

obstacle_distance = st.sidebar.slider(
    "Obstacle Distance (cm)",
    float(X["obstacle_distance_cm"].min()),
    float(X["obstacle_distance_cm"].max()),
    float(X["obstacle_distance_cm"].median())
)


# -----------------------------
# Normalize current input
# -----------------------------
current = np.array([
    motor_current,
    slip_ratio,
    tilt,
    obstacle_distance
])

normalized = (
    current - X_min.values
) / (
    X_max.values - X_min.values + 1e-9
)

normalized = np.clip(
    normalized,
    0,
    1
)


# -----------------------------
# Quantum processing
# -----------------------------
quantum_output = quantum_features(
    normalized
)
# -----------------------------
# Display Quantum Circuit
# -----------------------------
st.subheader("⚛️ 4-Qubit Quantum Circuit")

display_qc = QuantumCircuit(4, 4)

for i, value in enumerate(normalized):
    display_qc.ry(float(value) * np.pi, i)

display_qc.cx(0, 1)
display_qc.cx(1, 2)
display_qc.cx(2, 3)

display_qc.measure(
    [0, 1, 2, 3],
    [0, 1, 2, 3]
)

st.code(
    display_qc.draw(output="text"),
    language="text"
)

# Combine classical + quantum features
hybrid_input = np.concatenate(
    [
        normalized,
        quantum_output
    ]
).reshape(1, -1)


# -----------------------------
# Prediction
# -----------------------------
prediction = model.predict(
    hybrid_input
)[0]

prediction_label = encoder.inverse_transform(
    [prediction]
)[0]

probabilities = model.predict_proba(
    hybrid_input
)[0]

confidence = max(probabilities) * 100


# -----------------------------
# Display
# -----------------------------
st.subheader("Prediction")

if prediction_label == "SAFE":
    st.success("🟢 SAFE")

elif prediction_label == "HIGH_RISK":
    st.warning("🟡 HIGH RISK")

else:
    st.error("🔴 IMMOBILIZATION RISK")


st.metric(
    "Prediction Confidence",
    f"{confidence:.1f}%"
)


st.subheader("Quantum Processing")

st.write(
    "The four rover features are encoded into a "
    "4-qubit circuit and combined with the original "
    "classical features."
)

st.code(
    str(quantum_output),
    language="text"
)


st.subheader("Current Rover Inputs")

input_table = pd.DataFrame({
    "Feature": features,
    "Value": current.tolist()
})


st.dataframe(
    input_table,
    use_container_width=True
)