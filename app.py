'''import numpy as np
import pickle
import streamlit as st

# Define model names and their corresponding pickle file paths
model_paths = {
    "Logistic Regression": "logistic_regression_model.pickle",
    "TabNet": "tabnet_model.pickle",
    "MLP Classifier": "mlp_classifier_model.pickle",
    "CatBoost": "catboost_model.pickle",
    "LightGBM": "lightgbm_model.pickle",
    "XGBoost": "xgboost_model.pickle",
    "Support Vector Machine": "support_vector_machine_model.pickle",
    "Gaussian Naive Bayes": "gaussian_naive_bayes_model.pickle"
}

# Function to preprocess input data
def preprocess_input(input_data):
    # Implement your preprocessing steps here if needed
    return input_data

# Function to make predictions using all models
def predict_all_models(input_data, model_paths):
    predictions = {}
    for name, path in model_paths.items():
        with open(path, "rb") as f:
            model = pickle.load(f)
            preprocessed_data = preprocess_input(input_data)
            prediction = model.predict(preprocessed_data)
            predictions[name] = prediction[0]
    return predictions

# Streamlit UI
def main():
    st.title("Diabetes Prediction App")
    st.write("Enter patient information:")
    
    # Input fields
    pregnancies = st.number_input("Number of Pregnancies", value=1, min_value=0, max_value=17)
    glucose = st.number_input("Glucose Level", value=100, min_value=0, max_value=200)
    blood_pressure = st.number_input("Blood Pressure", value=69, min_value=0, max_value=122)
    skin_thickness = st.number_input("Skin Thickness", value=20, min_value=0, max_value=99)
    insulin = st.number_input("Insulin Level", value=79, min_value=0, max_value=846)
    bmi = st.number_input("BMI", value=32.0, min_value=0.0, max_value=67.1)
    diabetes_pedigree = st.number_input("Diabetes Pedigree Function", value=0.3725, min_value=0.078, max_value=2.42, step=0.01)
    age = st.number_input("Age", value=35, min_value=21, max_value=81)


    # Make prediction
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])
    if st.button("Predict", key="predict_button", help="Click to make a prediction"):
        predictions = predict_all_models(input_data, model_paths)
        st.write("<h2>Predictions by Different Models:</h2>", unsafe_allow_html=True)
        for name, prediction in predictions.items():
            if prediction == 0:
                st.markdown(f"<h3>{name} predicts that the patient is not diabetic.</h3>", unsafe_allow_html=True)
            else:
                st.markdown(f"<h3>{name} predicts that the patient is diabetic.</h3>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()

import numpy as np
import pickle
import streamlit as st

# Define model names and their corresponding pickle file paths
model_paths = {
    "Logistic Regression": "logistic_regression_model.pickle",
    "TabNet": "tabnet_model.pickle",
    "MLP Classifier": "mlp_classifier_model.pickle",
    "CatBoost": "catboost_model.pickle",
    "LightGBM": "lightgbm_model.pickle",
    "XGBoost": "xgboost_model.pickle",
    "Support Vector Machine": "support_vector_machine_model.pickle",
    "Gaussian Naive Bayes": "gaussian_naive_bayes_model.pickle"
}

# Function to preprocess input data
def preprocess_input(input_data):
    # Implement your preprocessing steps here if needed
    return input_data

# Function to make predictions using all models
def predict_all_models(input_data, model_paths):
    predictions = {}
    for name, path in model_paths.items():
        with open(path, "rb") as f:
            model = pickle.load(f)
            preprocessed_data = preprocess_input(input_data)
            # Check if the model supports probability prediction
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(preprocessed_data)
                prediction = probabilities[:, 1]  # Probability of positive class (diabetic)
            else:
                # For models that don't support probability prediction, fallback to regular prediction
                prediction = model.predict(preprocessed_data)
            predictions[name] = prediction[0]
    return predictions


# Streamlit UI
def main():
    st.title("Diabetes Prediction App")
    
    # Input fields
    st.subheader("Patient Information")
    pregnancies = st.number_input("Number of Pregnancies", value=1, min_value=0, max_value=17, help="Total number of pregnancies")
    glucose = st.number_input("Glucose Level", value=100, min_value=0, max_value=200, help="Glucose concentration in plasma")
    blood_pressure = st.number_input("Blood Pressure", value=69, min_value=0, max_value=122, help="Blood pressure (mm Hg)")
    skin_thickness = st.number_input("Skin Thickness", value=20, min_value=0, max_value=99, help="Triceps skin fold thickness (mm)")
    insulin = st.number_input("Insulin Level", value=79, min_value=0, max_value=846, help="Insulin concentration in serum (mu U/ml)")
    bmi = st.number_input("BMI", value=32.0, min_value=0.0, max_value=67.1, help="Body mass index (weight in kg/(height in m)^2)")
    diabetes_pedigree = st.number_input("Diabetes Pedigree Function", value=0.3725, min_value=0.078, max_value=2.42, step=0.01, help="Diabetes pedigree function")
    age = st.number_input("Age", value=35, min_value=21, max_value=81, help="Age in years")


    # Make prediction button
    st.subheader("Make Prediction")
    if st.button("Predict", key="predict_button", help="Click to make a prediction"):
        input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])
        predictions = predict_all_models(input_data, model_paths)
        st.subheader("Predictions by Different Models:")
        for name, prediction in predictions.items():
            if prediction < 0.5:
                st.error(f"{name} predicts that the patient is not diabetic. Probability: {1-prediction:.2f}")
            else:
                st.success(f"{name} predicts that the patient is diabetic. Probability: {prediction:.2f}")

if __name__ == "__main__":
    main() 


# Display LinkedIn icon
st.markdown('<a href="https://www.linkedin.com/in/adabala-uday-raghavendra-kumar-ab35bb1a3/" target="_blank"><img src="https://image.flaticon.com/icons/png/512/174/174857.png" width="30"></a>', unsafe_allow_html=True)
'''
import numpy as np
import pickle
import streamlit as st
import time

# Define model names and their corresponding pickle file paths
model_paths = {
    "Logistic Regression": "logistic_regression_model.pickle",
    "TabNet": "tabnet_model.pickle",
    "MLP Classifier": "mlp_classifier_model.pickle",
    "CatBoost": "catboost_model.pickle",
    "LightGBM": "lightgbm_model.pickle",
    "XGBoost": "xgboost_model.pickle",
    "Support Vector Machine": "support_vector_machine_model.pickle",
    "Gaussian Naive Bayes": "gaussian_naive_bayes_model.pickle"
}

# Function to preprocess input data
def preprocess_input(input_data):
    # Implement your preprocessing steps here if needed
    return input_data

# Function to make predictions using all models
def predict_all_models(input_data, model_paths):
    predictions = {}
    for name, path in model_paths.items():
        with open(path, "rb") as f:
            model = pickle.load(f)
            preprocessed_data = preprocess_input(input_data)
            # Check if the model supports probability prediction
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(preprocessed_data)
                prediction = probabilities[:, 1]  # Probability of positive class (diabetic)
            else:
                # For models that don't support probability prediction, fallback to regular prediction
                prediction = model.predict(preprocessed_data)
            predictions[name] = prediction[0]
    return predictions

# Streamlit UI for sign-in page
def signin(registered_users):
    st.title("Sign In")

    # Generate unique keys for username and password input fields
    timestamp = str(int(time.time()))
    signin_username_key = "signin_username_" + timestamp
    signin_password_key = "signin_password_" + timestamp

    # Input fields for username and password
    username = st.text_input("Username", key=signin_username_key)
    password = st.text_input("Password", type="password", key=signin_password_key)

    # Sign-in button
    if st.button("Sign In"):
        # Check if username exists in registered users
        if username in registered_users:
            # Verify password
            if registered_users[username] == password:
                st.success("Logged in successfully!")
                return True  # Return True upon successful sign-in
            else:
                st.error("Invalid password. Please try again.")
        else:
            st.warning("Username not found. Please sign up first.")
            if st.button("Sign Up"):
                signup(registered_users)  # Provide option to sign up
    return False  # Return False if sign-in fails

# Streamlit UI for user registration page
def signup(registered_users):
    st.title("Sign Up")

    # Generate unique keys for new username and password input fields
    timestamp = str(int(time.time()))
    signup_username_key = "signup_username_" + timestamp
    signup_password_key = "signup_password_" + timestamp

    # Input fields for new username and password
    new_username = st.text_input("New Username", key=signup_username_key)
    new_password = st.text_input("New Password", type="password", key=signup_password_key)

    # Sign-up button
    if st.button("Sign Up"):
        # Check if username already exists
        if new_username in registered_users:
            st.error("Username already exists. Please choose a different one.")
        else:
            # Add new user to registered users
            registered_users[new_username] = new_password
            st.success(f"User registered successfully! Username: {new_username}")

# Streamlit UI for the main page
def main_page():
    st.title("Diabetes Prediction App")
    
    # Input fields
    st.subheader("Patient Information")
    pregnancies = st.number_input("Number of Pregnancies", value=1, min_value=0, max_value=17, help="Total number of pregnancies")
    glucose = st.number_input("Glucose Level", value=100, min_value=0, max_value=200, help="Glucose concentration in plasma")
    blood_pressure = st.number_input("Blood Pressure", value=69, min_value=0, max_value=122, help="Blood pressure (mm Hg)")
    skin_thickness = st.number_input("Skin Thickness", value=20, min_value=0, max_value=99, help="Triceps skin fold thickness (mm)")
    insulin = st.number_input("Insulin Level", value=79, min_value=0, max_value=846, help="Insulin concentration in serum (mu U/ml)")
    bmi = st.number_input("BMI", value=32.0, min_value=0.0, max_value=67.1, help="Body mass index (weight in kg/(height in m)^2)")
    diabetes_pedigree = st.number_input("Diabetes Pedigree Function", value=0.3725, min_value=0.078, max_value=2.42, step=0.01, help="Diabetes pedigree function")
    age = st.number_input("Age", value=35, min_value=21, max_value=81, help="Age in years")

    # Make prediction button
    st.subheader("Make Prediction")
    if st.button("Predict", key="predict_button", help="Click to make a prediction"):
        input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])
        predictions = predict_all_models(input_data, model_paths)
        st.subheader("Predictions by Different Models:")
        for name, prediction in predictions.items():
            if prediction < 0.5:
                st.error(f"{name} predicts that the patient is not diabetic. Probability: {1-prediction:.2f}")
            else:
                st.success(f"{name} predicts that the patient is diabetic. Probability: {prediction:.2f}")

# Main function to run the app
def main():
    registered_users = {}  # Dictionary to store registered users and their passwords
    while not signin(registered_users):  # Continue sign-in loop until successful sign-in
        pass  # Continue looping until successful sign-in
    main_page()  # Once signed in, proceed to the
