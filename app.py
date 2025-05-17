import streamlit as st
import numpy as np
import pickle
import pandas as pd
from tensorflow.keras.models import load_model
from datetime import datetime
import os

# --- Load model and encoders ---
model = load_model('mdoel.h5')

with open('label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

with open('onehot.pkl', 'rb') as file:
    onehot = pickle.load(file)

with open('scalar.pkl', 'rb') as file:
    scalar = pickle.load(file)

# --- Set page config ---
st.set_page_config(page_title="💼 Churn Prediction", layout="centered")

# --- Custom CSS styling ---
st.markdown("""
    <style>
        body {
            background-color: #f4f6f9;
            font-family: 'Segoe UI', sans-serif;
        }
        .main {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #1f77b4;
        }
    </style>
""", unsafe_allow_html=True)

# --- App Header ---
st.markdown("<div class='main'>", unsafe_allow_html=True)
st.title("💼 Bank Customer Churn Prediction")
st.markdown("Use this interactive tool to predict whether a customer will leave the bank based on input data.")
st.markdown("---")

# --- Form UI ---
with st.form("churn_form"):
    col1, col2 = st.columns(2)
    with col1:
        credit_score = st.slider("Credit Score", 300, 900, 600)
        geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
        age = st.number_input("Age", 18, 100, 40)
        balance = st.number_input("Account Balance", 0.0, 250000.0, 50000.0)
        has_cr_card = st.selectbox("Has Credit Card", [0, 1])
    with col2:
        gender = st.selectbox("Gender", ["Male", "Female"])
        tenure = st.number_input("Tenure (years with bank)", 0, 10, 3)
        num_of_products = st.selectbox("Number of Products", [1, 2, 3, 4])
        is_active = st.selectbox("Is Active Member", [0, 1])
        est_salary = st.number_input("Estimated Salary", 0.0, 200000.0, 50000.0)

    submitted = st.form_submit_button("🔍 Predict")

# --- Prediction Logic ---
if submitted:
    gender_encoded = label_encoder.transform([gender])[0]
    geo_encoded = onehot.transform([[geography]]).toarray()[0]

    input_array = np.array([[credit_score, gender_encoded, age, tenure,
                             balance, num_of_products, has_cr_card,
                             is_active, est_salary]])

    input_full = np.concatenate((input_array, geo_encoded.reshape(1, -1)), axis=1)
    input_scaled = scalar.transform(input_full)

    prediction = model.predict(input_scaled)[0][0]
    result = "🔥 Likely to Churn" if prediction >= 0.5 else "✅ Likely to Stay"
    confidence = prediction if prediction >= 0.5 else 1 - prediction

    # --- Output Results ---
    st.markdown("## 🧾 Prediction Result")
    st.success(f"**{result}**")
    st.info(f"🔢 Confidence Score: **{confidence:.2%}**")
    st.progress(int(confidence * 100))

    # --- Save log to CSV ---
    log_data = {
        'Timestamp': [datetime.now()],
        'CreditScore': [credit_score],
        'Geography': [geography],
        'Gender': [gender],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active],
        'EstimatedSalary': [est_salary],
        'Prediction': [result],
        'Confidence': [f"{confidence:.4f}"]
    }

    log_df = pd.DataFrame(log_data)
    log_file = 'logs.csv'

    if os.path.exists(log_file):
        log_df.to_csv(log_file, mode='a', index=False, header=False)
    else:
        log_df.to_csv(log_file, index=False)

st.markdown("</div>", unsafe_allow_html=True)
