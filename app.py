import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model and scaler
model_path = "D:/IBA/NEST/model.pkl"
scaler_path = "D:/IBA/NEST/scaler.pkl"

# Load the model and scaler
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# List of all 29 features (used during training)
feature_names = [
    'Enrollment', 'Trial Duration (Days)', 'Time Since Last Update', 'Enrollment Rate Per Day',
    'Sponsor Experience Score', 'Number of Interventions', 'Primary Outcome Complexity', 'Randomization Factor',
    'Number of Locations', 'Multinational Trial Flag', 'Encoded Study Status', 'Encoded Study Phase',
    'Relative Competition Intensity', 'Study Status', 'Study Results', 'Conditions', 'Interventions',
    'Primary Outcome Measures', 'Secondary Outcome Measures', 'Sponsor', 'Sex', 'Age', 'Phases', 'Funder Type',
    'Study Type', 'Study Design', 'Primary Completion Date', 'Results First Posted', 'Locations'
]

# Streamlit App Title
st.title("Clinical Trial Recruitment Rate Prediction")
st.write("Predict the recruitment rate for clinical trials based on input features.")

# Sidebar for user inputs
st.sidebar.header("Input Features")

# Input fields for numerical features
enrollment = st.sidebar.number_input("Enrollment", min_value=0, value=100)
trial_duration = st.sidebar.number_input("Trial Duration (Days)", min_value=0, value=30)
time_since_last_update = st.sidebar.number_input("Time Since Last Update (Days)", min_value=0, value=10)
enrollment_rate_per_day = st.sidebar.number_input("Enrollment Rate Per Day", min_value=0.0, value=1.0)
sponsor_experience_score = st.sidebar.number_input("Sponsor Experience Score", min_value=0, value=5)
number_of_interventions = st.sidebar.number_input("Number of Interventions", min_value=0, value=1)
primary_outcome_complexity = st.sidebar.number_input("Primary Outcome Complexity", min_value=0, value=10)
randomization_factor = st.sidebar.selectbox("Randomization Factor", [0, 1], index=0)
number_of_locations = st.sidebar.number_input("Number of Locations", min_value=0, value=1)
multinational_trial_flag = st.sidebar.selectbox("Multinational Trial Flag", [0, 1], index=0)

# Input fields for categorical features
encoded_study_status = st.sidebar.selectbox("Encoded Study Status", [0, 1, 2])  # Example: 0 = Completed, 1 = Recruiting, 2 = Terminated
encoded_study_phase = st.sidebar.selectbox("Encoded Study Phase", [1, 2, 3, 4])  # Example: Phase 1, 2, 3, 4
relative_competition_intensity = st.sidebar.number_input("Relative Competition Intensity", min_value=0.0, value=1.0)

# Collect all user inputs into a dictionary
user_inputs = {
    "Enrollment": enrollment,
    "Trial Duration (Days)": trial_duration,
    "Time Since Last Update": time_since_last_update,
    "Enrollment Rate Per Day": enrollment_rate_per_day,
    "Sponsor Experience Score": sponsor_experience_score,
    "Number of Interventions": number_of_interventions,
    "Primary Outcome Complexity": primary_outcome_complexity,
    "Randomization Factor": randomization_factor,
    "Number of Locations": number_of_locations,
    "Multinational Trial Flag": multinational_trial_flag,
    "Encoded Study Status": encoded_study_status,
    "Encoded Study Phase": encoded_study_phase,
    "Relative Competition Intensity": relative_competition_intensity
}

# Fill missing features with default values (e.g., 0)
for feature in feature_names:
    if feature not in user_inputs:
        user_inputs[feature] = 0  # Default value for missing features

# Create a DataFrame with all features in the correct order
input_data = pd.DataFrame([user_inputs])[feature_names]

# Scale the input data
scaled_data = scaler.transform(input_data)

# Predict recruitment rate
if st.button("Predict Recruitment Rate"):
    prediction = model.predict(scaled_data)
    # Apply inverse log transformation (if log1p was used during training)
    predicted_rate = np.expm1(prediction[0])
    st.success(f"Predicted Recruitment Rate: {predicted_rate:.2f} patients/site/month")