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
    st.set_page_config(layout="wide")  # Set page layout to wide

    st.title("Diabetes Prediction App")

    # Sidebar with input fields and user documentation
    st.sidebar.title("Patient Information")
    st.sidebar.write("Please enter the following patient information:")

    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    
    # If gender is male, skip the pregnancies input
    if gender == "Female":
        pregnancies = st.sidebar.number_input("Number of Pregnancies", min_value=0, max_value=17, value=1, help="Total number of pregnancies")
    else:
        pregnancies = 0  # Skip pregnancy for male
    
    glucose = st.sidebar.number_input("Glucose Level", min_value=0, max_value=200, value=100, help="Glucose concentration in plasma (mg/dL)")
    blood_pressure = st.sidebar.number_input("Blood Pressure", min_value=0, max_value=122, value=69, help="Blood pressure (mm Hg)")
    skin_thickness = st.sidebar.number_input("Skin Thickness", min_value=0, max_value=99, value=20, help="Triceps skin fold thickness (mm)")
    insulin = st.sidebar.number_input("Insulin Level", min_value=0, max_value=846, value=79, help="Insulin concentration in serum (mu U/ml)")
    bmi = st.sidebar.number_input("BMI", min_value=0.0, max_value=67.1, value=32.0, help="Body mass index (weight in kg/(height in m)^2)")
    diabetes_pedigree = st.sidebar.number_input("Diabetes Pedigree Function", min_value=0.078, max_value=2.42, step=0.01, value=0.3725, help="Diabetes pedigree function")
    age = st.sidebar.number_input("Age", min_value=21, max_value=81, value=35, help="Age in years")

    # Predict button with tooltip
    predict_button = st.sidebar.button("Predict", help="Click to make a prediction")

    # Main content area
    col1, col2 = st.columns([1, 3])  # Divide page into two columns, 1:3 ratio

    # Perform prediction upon button click
    if predict_button:
        with st.spinner("Predicting..."):
            input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])
            predictions = predict_all_models(input_data, model_paths)
            col2.subheader("Predictions by Different Models:")
            for name, prediction in predictions.items():
                if prediction < 0.5:
                    col2.error(f"{name} predicts that the patient is not diabetic. Probability: {1-prediction:.2f}")
                else:
                    col2.success(f"{name} predicts that the patient is diabetic. Probability: {prediction:.2f}")

if __name__ == "__main__":
    main()

# Display LinkedIn icon with tooltip
st.sidebar.write("Connect with me on LinkedIn:")
st.sidebar.markdown('<a href="https://www.linkedin.com/in/adabala-uday-raghavendra-kumar-ab35bb1a3/" target="_blank" title="Connect with me on LinkedIn"><img src="https://image.flaticon.com/icons/png/512/174/174857.png" width="30"></a>', unsafe_allow_html=True)
'''
import streamlit as st
import numpy as np
import pickle
from auth0.v3.authentication import GetToken
from auth0.v3.authentication import Users
from auth0.v3.management import Auth0
import json


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

# Function to perform Google login using Auth0
def login_with_google():
    # Initialize Auth0 client
    auth0 = Auth0(domain='dev-v2r66cqhzhl2mheg.us.auth0.com', client_id='gGvyg53rPAhzMpBwPnS0IIlN22F5axJZ', client_secret='qnvr19IAI5jEQDmYB8CUs1QANPRgeqfj4HrwqwYFuSAgQJqpZS9KiYp11yvd_rFz')
    token = GetToken(auth0)
    
    # Generate Google login URL
    google_auth_url = token.login(
        client_id='ID',
        client_secret='seret',
        connection='google-oauth2',
        redirect_uri='https://diapredicttl.streamlit.app/',  # Replace with your app's redirect URI
    )
    
    # Display login button
    st.markdown(f'<a href="{google_auth_url}">Login with Google</a>', unsafe_allow_html=True)

# Streamlit UI
def main():
    st.set_page_config(layout="wide")  # Set page layout to wide

    st.title("Diabetes Prediction App")

    # Sidebar with input fields and user documentation
    st.sidebar.title("Patient Information")
    st.sidebar.write("Please enter the following patient information:")

    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    
    # If gender is male, skip the pregnancies input
    if gender == "Female":
        pregnancies = st.sidebar.number_input("Number of Pregnancies", min_value=0, max_value=17, value=1, help="Total number of pregnancies")
    else:
        pregnancies = 0  # Skip pregnancy for male
    
    glucose = st.sidebar.number_input("Glucose Level", min_value=0, max_value=200, value=100, help="Glucose concentration in plasma (mg/dL)")
    blood_pressure = st.sidebar.number_input("Blood Pressure", min_value=0, max_value=122, value=69, help="Blood pressure (mm Hg)")
    skin_thickness = st.sidebar.number_input("Skin Thickness", min_value=0, max_value=99, value=20, help="Triceps skin fold thickness (mm)")
    insulin = st.sidebar.number_input("Insulin Level", min_value=0, max_value=846, value=79, help="Insulin concentration in serum (mu U/ml)")
    bmi = st.sidebar.number_input("BMI", min_value=0.0, max_value=67.1, value=32.0, help="Body mass index (weight in kg/(height in m)^2)")
    diabetes_pedigree = st.sidebar.number_input("Diabetes Pedigree Function", min_value=0.078, max_value=2.42, step=0.01, value=0.3725, help="Diabetes pedigree function")
    age = st.sidebar.number_input("Age", min_value=21, max_value=81, value=35, help="Age in years")

    # Predict button with tooltip
    predict_button = st.sidebar.button("Predict", help="Click to make a prediction")

    # Main content area
    col1, col2 = st.columns([1, 3])  # Divide page into two columns, 1:3 ratio

    # Perform prediction upon button click
    if predict_button:
        with st.spinner("Predicting..."):
            input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])
            predictions = predict_all_models(input_data, model_paths)
            col2.subheader("Predictions by Different Models:")
            for name, prediction in predictions.items():
                if prediction < 0.5:
                    col2.error(f"{name} predicts that the patient is not diabetic. Probability: {1-prediction:.2f}")
                else:
                    col2.success(f"{name} predicts that the patient is diabetic. Probability: {prediction:.2f}")

if __name__ == "__main__":
    main()

# Display LinkedIn icon with tooltip
st.sidebar.write("Connect with me on LinkedIn:")
st.sidebar.markdown('<a href="https://www.linkedin.com/in/adabala-uday-raghavendra-kumar-ab35bb1a3/" target="_blank" title="Connect with me on LinkedIn"><img src="https://image.flaticon.com/icons/png/512/174/174857.png" width="30"></a>', unsafe_allow_html=True)
