#  Mars Rover Predictive Immobilization & Terrain Risk System

## Problem
A planetary rover can lose traction, experience abnormal motor load, or
approach difficult terrain before becoming immobilized.

## Solution
The system predicts rover risk using motor current, wheel speeds, wheel slip,
tilt, obstacle distance and terrain moisture.

Risk states:
- SAFE
- HIGH_RISK
- IMMOBILIZATION_RISK

## Architecture
Sensors / simulated data
→ Feature calculation
→ Machine-learning model
→ Risk classification
→ Early-warning decision

## Run
```bash
pip install -r requirements.txt
python generate_data.py
python train_model.py
streamlit run app.py
```

## MVP note
The current dataset is synthetic. It demonstrates the predictive pipeline.
For the final prototype, replace the simulated inputs with measurements from
the actual rover.

## Team roles
- AI/ML: prediction model and risk logic
- AIML: preprocessing and evaluation
- ECE: sensors, controller and electronics
- CSE: dashboard and system integration
