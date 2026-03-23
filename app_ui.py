import streamlit as st
import pickle
import numpy as np
import pandas as pd

model = pickle.load(open("churn_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title(" Customer Churn Prediction App")
st.write("Enter customer details to predict churn probability")

credit = st.number_input("Credit Score", 300, 900, 600)

geo = st.selectbox("Geography", ["France", "Germany", "Spain"])
geo_germany = 1 if geo == "Germany" else 0
geo_spain = 1 if geo == "Spain" else 0

gender = st.selectbox("Gender", ["Female", "Male"])
gender_val = 1 if gender == "Male" else 0

age = st.number_input("Age", 18, 100, 40)
tenure = st.number_input("Tenure", 0, 10, 3)
balance = st.number_input("Balance", value=60000.0)
products = st.number_input("Number of Products", 1, 4, 2)

card = st.selectbox("Has Credit Card?", ["No", "Yes"])
card_val = 1 if card == "Yes" else 0

active = st.selectbox("Is Active Member?", ["No", "Yes"])
active_val = 1 if active == "Yes" else 0

salary = st.number_input("Estimated Salary", value=50000.0)

if st.button("Predict"):

    input_dict = {
        "CreditScore": credit,
        "Geography_Germany": geo_germany,
        "Geography_Spain": geo_spain,
        "Gender": gender_val,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": products,
        "HasCrCard": card_val,
        "IsActiveMember": active_val,
        "EstimatedSalary": salary
    }

    data_df = pd.DataFrame([input_dict])

    data_df = data_df[scaler.feature_names_in_]

    data_scaled = scaler.transform(data_df)

    probability = model.predict_proba(data_scaled)[0][1]
    threshold = 0.3
    prediction = 1 if probability > threshold else 0

    st.subheader("Result")

    if prediction == 1:
        st.error(f" Customer likely to churn\n\nProbability: {probability:.2f}")
    else:
        st.success(f"  Customer not likely to churn\n\nProbability: {probability:.2f}")