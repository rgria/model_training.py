import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the model and scaler
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Define the prediction function
def predict_readmission(data):
    data_scaled = scaler.transform(data)
    prediction = model.predict(data_scaled)
    return prediction

# Streamlit app interface
st.title("Hospital Readmission Predictor")
st.write("Predict if a patient will be readmitted within 30 days, after 30 days, or not at all.")

# Collect user input
age = st.slider("Age (years)", 0, 120, 50)
admission_type = st.selectbox("Admission Type", ["Emergency", "Urgent", "Elective", "Newborn"])
discharge_disposition = st.selectbox("Discharge Disposition", ["Discharged to home", "Transferred", "Expired"])
diagnosis = st.text_input("Primary Diagnosis Code", "250.00 (Diabetes)")
num_lab_procedures = st.number_input("Number of Lab Procedures", 0, 100, 20)
num_medications = st.number_input("Number of Medications", 0, 50, 10)

# Prepare input data for prediction
input_data = pd.DataFrame({
    'age': [age],
    'admission_type': [admission_type],
    'discharge_disposition': [discharge_disposition],
    'diagnosis': [diagnosis],
    'num_lab_procedures': [num_lab_procedures],
    'num_medications': [num_medications]
})

# Process input on submit
if st.button("Predict"):
    input_data_encoded = pd.get_dummies(input_data).reindex(columns=model.feature_importances_, fill_value=0)
    prediction = predict_readmission(input_data_encoded)
    st.write("Prediction:", "No Readmission" if prediction == 0 else ("Readmitted < 30 days" if prediction == 1 else "Readmitted > 30 days"))
