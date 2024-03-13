#!/usr/bin/env python
# coding: utf-8

# In[2]:


pip install streamlit


# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
'''from sklearn.preprocessing import StandardScaler'''
import pickle
from pytorch_tabnet.tab_model import TabNetClassifier
import numpy as np

def standard_scaler(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    scaled_X = (X - mean) / std
    return scaled_X


# Load the pickled model
with open("tabnet_model.pickle", "rb") as f:
    model = pickle.load(f)

# Function to preprocess input data
def preprocess_input(input_data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(input_data)
    return scaled_data

# Function to make predictions
def predict(data):
    preprocessed_data = preprocess_input(data)
    prediction = model.predict(preprocessed_data)
    return prediction

# Streamlit UI
def main():
    st.title("Diabetes Prediction App")

    # Input fields
    st.header("Input Data")
    glucose = st.slider("Glucose Level", 0, 200, 100)
    blood_pressure = st.slider("Blood Pressure", 0, 150, 70)
    insulin = st.slider("Insulin Level", 0, 500, 100)
    bmi = st.slider("BMI", 0.0, 70.0, 25.0)

    # Make prediction
    input_data = np.array([[glucose, blood_pressure, insulin, bmi]])
    if st.button("Predict"):
        prediction = predict(input_data)
        if prediction[0] == 0:
            st.error("The model predicts that the patient is not diabetic.")
        else:
            st.success("The model predicts that the patient is diabetic.")

if __name__ == "__main__":
    main()



# In[ ]:




