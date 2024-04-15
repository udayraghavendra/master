
import streamlit as st
import numpy as np
import pickle

# Load the trained CatBoost model
with open("catboost_model.pkl", "rb") as f:
    catboost_model = pickle.load(f)

# Load the trained TabNet model
with open("tabnet_model.pkl", "rb") as f:
    tabnet_model = pickle.load(f)

# Load the scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Function to make prediction
def make_prediction(user_data):
    # Preprocess user entry data
    user_data_scaled = scaler.transform(user_data)

    # Obtain predictions from CatBoost model
    catboost_user_probs = catboost_model.predict_proba(user_data_scaled)[:, 1]

    # Combine original features with predicted probabilities from CatBoost
    user_data_meta = np.column_stack((user_data_scaled, catboost_user_probs))

    # Use the trained TabNet model to predict the outcome
    prediction = tabnet_model.predict(user_data_meta)

    return prediction

# Streamlit UI
st.title("Diabetes Prediction")

# User input section
st.header("Enter Patient Information")
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, step=1)
plasma_glucose = st.number_input("Plasma Glucose", min_value=0, max_value=300, step=1)
diastolic_blood_pressure = st.number_input("Diastolic Blood Pressure", min_value=0, max_value=200, step=1)
triceps_thickness = st.number_input("Triceps Thickness", min_value=0, max_value=100, step=1)
serum_insulin = st.number_input("Serum Insulin", min_value=0, max_value=1000, step=1)
bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, step=0.1)
diabetes_pedigree = st.number_input("Diabetes Pedigree", min_value=0.0, max_value=2.0, step=0.01)
age = st.number_input("Age", min_value=0, max_value=120, step=1)

# Make prediction on user input
user_data = np.array([[pregnancies, plasma_glucose, diastolic_blood_pressure, triceps_thickness, serum_insulin, bmi, diabetes_pedigree, age]])
if st.button("Predict"):
    prediction = make_prediction(user_data)
    if prediction == 1:
        st.error("Based on the input data, the individual is predicted to be diabetic.")
    else:
        st.success("Based on the input data, the individual is predicted not to be diabetic.")

