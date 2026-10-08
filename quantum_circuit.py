import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

# Load the actual rover dataset
data = pd.read_csv("rover_data.csv")

# Select 4 important rover features
features = [
    "motor_current_a",
    "slip_ratio_pct",
    "tilt_deg",
    "obstacle_distance_cm"
]

# Take the first real sample from the dataset
row = data.iloc[0]

values = row[features].astype(float).values

# Normalize each feature using the complete dataset
minimum = data[features].min().values
maximum = data[features].max().values

normalized = (values - minimum) / (maximum - minimum + 1e-9)

print("Actual rover features:")
for name, value in zip(features, values):
    print(f"{name}: {value}")

print("\nNormalized values:")
print(normalized)

# Create a 4-qubit quantum circuit
qc = QuantumCircuit(4, 4)

# Encode the four rover features into four qubits
for i, value in enumerate(normalized):
    qc.ry(float(value) * np.pi, i)

# Create interactions between the qubits
qc.cx(0, 1)
qc.cx(1, 2)
qc.cx(2, 3)

# Measure the qubits
qc.measure([0, 1, 2, 3], [0, 1, 2, 3])

# Run the quantum circuit
simulator = AerSimulator()
job = simulator.run(qc, shots=1024)

result = job.result()
counts = result.get_counts()

print("\nMARS ROVER QUANTUM CIRCUIT")
print(qc)

print("\nQuantum Measurement Results:")
print(counts)