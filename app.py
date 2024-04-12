'''
import streamlit as st
import numpy as np
import pickle
import joblib

# Define model names and their corresponding file paths
model_paths = {
    "TabNet": "tabnet_model.pickle",
    "CatBoost": "catboost_model.pickle",
    "LightGBM": "lightgbm_model.pickle",
    "XGBoost": "xgboost_model.pickle",
}

# Function to preprocess input data
def preprocess_input(input_data):
    columns = ["Pregnancies", "Glucose Level", "Blood Pressure", "Skin Thickness", "Insulin Level", "BMI", "Diabetes Pedigree Function", "Age"]
    for i in range(len(input_data)):
        for j in range(len(input_data[i])):
            if input_data[i][j] is None:
                input_data[i][j] = st.number_input(f"Enter {columns[j]}", min_value=0, step=0.1)
    return input_data

# Function to make predictions using all models
def predict_all_models(input_data, model_paths):
    predictions = []
    for _, path in model_paths.items():
        if path.endswith('.pickle'):
            with open(path, "rb") as f:
                model = pickle.load(f)
        elif path.endswith('.joblib'):
            model = joblib.load(path)
        
        preprocessed_data = preprocess_input(input_data.copy())
        # Check if the model supports probability prediction
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(preprocessed_data)
            prediction = probabilities[:, 1]  # Probability of positive class (diabetic)
        else:
            # For models that don't support probability prediction, fallback to regular prediction
            prediction = model.predict(preprocessed_data)
        predictions.append(prediction[0])
    return predictions

# Function to render the main page
def main_page():
    st.title("Welcome to the Diabetes Prediction App")
    st.write("Please enter your name:")
    name = st.text_input("Name")
    if name:
        st.write(f"Hi, {name}!")
        st.write("Please select your gender:")
        gender = st.selectbox("Gender", ["Male", "Female"])
        if gender == "Male":
            render_male_details()
        else:
            render_female_details()

# Function to render details for male users
def render_male_details():
    st.write("Please enter the following details:")
    glucose = st.number_input("Glucose Level", min_value=0, max_value=200, value=None, help="Glucose concentration in plasma (mg/dL)")
    blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=122, value=None, help="Blood pressure (mm Hg)")
    skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=99, value=None, help="Triceps skin fold thickness (mm)")
    insulin = st.number_input("Insulin Level", min_value=0, max_value=846, value=None, help="Insulin concentration in serum (mu U/ml)")
    bmi = st.number_input("BMI", min_value=0.0, max_value=67.1, value=None, help="Body mass index (weight in kg/(height in m)^2)")
    diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.078, max_value=2.42, step=0.01, value=None, help="Diabetes pedigree function")
    age = st.number_input("Age", min_value=21, max_value=81, value=None, help="Age in years")

    # Predict button with tooltip
    predict_button = st.button("Predict", help="Click to make a prediction")

    # Perform prediction upon button click
    if predict_button:
        with st.spinner("Predicting..."):
            input_data = np.array([[0, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])  # Assuming 0 pregnancies for males
            predictions = predict_all_models(input_data, model_paths)
            diabetic_count = sum(pred >= 0.5 for pred in predictions)
            non_diabetic_count = len(predictions) - diabetic_count
            st.subheader("Predictions by Different Models:")
            for name, prediction in zip(model_paths.keys(), predictions):
                if prediction < 0.5:
                    st.error(f"{name} predicts that the patient is not diabetic. Probability: {1-prediction:.2f}")
                else:
                    st.success(f"{name} predicts that the patient is diabetic. Probability: {prediction:.2f}")
            if diabetic_count > non_diabetic_count:
                st.warning("It is recommended to see a doctor immediately!")
                

# Function to render details for female users
def render_female_details():
    st.write("Please enter the following details:")
    pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=17, value=None, help="Total number of pregnancies")
    glucose = st.number_input("Glucose Level", min_value=0, max_value=200, value=None, help="Glucose concentration in plasma (mg/dL)")
    blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=122, value=None, help="Blood pressure (mm Hg)")
    skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=99, value=None, help="Triceps skin fold thickness (mm)")
    insulin = st.number_input("Insulin Level", min_value=0, max_value=846, value=None, help="Insulin concentration in serum (mu U/ml)")
    bmi = st.number_input("BMI", min_value=0.0, max_value=67.1, value=None, help="Body mass index (weight in kg/(height in m)^2)")
    diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.078, max_value=2.42, step=0.01, value=None, help="Diabetes pedigree function")
    age = st.number_input("Age", min_value=21, max_value=81, value=None, help="Age in years")

    # Predict button with tooltip
    predict_button = st.button("Predict", help="Click to make a prediction")

    # Perform prediction upon button click
    if predict_button:
        with st.spinner("Predicting..."):
            input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])  # Assuming 0 smoking for females
            predictions = predict_all_models(input_data, model_paths)
            diabetic_count = sum(pred >= 0.5 for pred in predictions)
            non_diabetic_count = len(predictions) - diabetic_count
            st.subheader("Predictions by Different Models:")
            for name, prediction in zip(model_paths.keys(), predictions):
                if prediction < 0.5:
                    st.error(f"{name} predicts that the patient is not diabetic. Probability: {1-prediction:.2f}")
                else:
                    st.success(f"{name} predicts that the patient is diabetic. Probability: {prediction:.2f}")
            if diabetic_count > non_diabetic_count:
                st.warning("It is recommended to see a doctor immediately!")

# Main function
def main():
    main_page()

if __name__ == "__main__":
    main()
'''
import streamlit as st
import numpy as np
import pickle
import joblib

# Define model names and their corresponding file paths
model_paths = {
    "TabNet": "tabnet_model.pickle",
    "CatBoost": "catboost_model.pickle",
    "LightGBM": "lightgbm_model.pickle",
    "XGBoost": "xgboost_model.pickle",
}

# Function to preprocess input data
def preprocess_input(input_data):
    columns = ["Pregnancies", "Glucose Level", "Blood Pressure", "Skin Thickness", "Insulin Level", "BMI", "Diabetes Pedigree Function", "Age"]
    for i in range(len(input_data)):
        for j in range(len(input_data[i])):
            if input_data[i][j] is None:
                input_data[i][j] = st.number_input(f"Enter {columns[j]}", min_value=0, step=0.1)
    return input_data

# Function to make predictions using all models
def predict_all_models(input_data, model_paths):
    predictions = []
    for _, path in model_paths.items():
        if path.endswith('.pickle'):
            with open(path, "rb") as f:
                model = pickle.load(f)
        elif path.endswith('.joblib'):
            model = joblib.load(path)
        
        preprocessed_data = preprocess_input(input_data.copy())
        # Check if the model supports probability prediction
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(preprocessed_data)
            prediction = probabilities[:, 1]  # Probability of positive class (diabetic)
        else:
            # For models that don't support probability prediction, fallback to regular prediction
            prediction = model.predict(preprocessed_data)
        predictions.append(prediction[0])
    return predictions

# Function to render the main page
def main_page():
    st.title(" Diabetes Prediction")
    st.write("Please enter the following details:")
    pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=17, value=None, help="Total number of pregnancies")
    glucose = st.number_input("Glucose Level", min_value=0, max_value=200, value=None, help="Glucose concentration in plasma (mg/dL)")
    blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=122, value=None, help="Blood pressure (mm Hg)")
    skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=99, value=None, help="Triceps skin fold thickness (mm)")
    insulin = st.number_input("Insulin Level", min_value=0, max_value=846, value=None, help="Insulin concentration in serum (mu U/ml)")
    bmi = st.number_input("BMI", min_value=0.0, max_value=67.1, value=None, help="Body mass index (weight in kg/(height in m)^2)")
    diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.078, max_value=2.42, step=0.01, value=None, help="Diabetes pedigree function")
    age = st.number_input("Age", min_value=21, max_value=81, value=None, help="Age in years")

    # Predict button with tooltip
    predict_button = st.button("Predict", help="Click to make a prediction")

    # Perform prediction upon button click
    if predict_button:
        with st.spinner("Predicting..."):
            input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])
            predictions = predict_all_models(input_data, model_paths)
            diabetic_count = sum(pred >= 0.5 for pred in predictions)
            non_diabetic_count = len(predictions) - diabetic_count
            st.subheader("Predictions by Different Models:")
            for name, prediction in zip(model_paths.keys(), predictions):
                if prediction < 0.5:
                    st.error(f"{name} predicts that the patient is not diabetic. Probability: {1-prediction:.2f}")
                else:
                    st.success(f"{name} predicts that the patient is diabetic. Probability: {prediction:.2f}")
            if diabetic_count > non_diabetic_count:
                st.warning("It is recommended to see a doctor immediately!")

# Main function
def main():
    main_page()

if __name__ == "__main__":
    main()
