import streamlit as st
import requests

# Streamlit App Configuration
st.title("Customer Retention Prediction Bot")
st.markdown("Provide customer details below to predict if they will churn or be retained.")

# User Inputs
age = st.number_input("Enter Age:", min_value=10, max_value=100, step=1)
gender = st.selectbox("Select Gender:", ["Male", "Female"])
purchase_amount = st.number_input("Enter Purchase Amount:", min_value=0, step=1)

# Backend URL
backend_url = "http://127.0.0.1:5000/chatbot"

# Predict Button
if st.button("Predict"):
    # Prepare the payload
    payload = {
        "age": age,
        "gender": gender,
        "purchase_amount": purchase_amount
    }
    try:
        # Send the request to the Flask API
        response = requests.post(backend_url, json=payload)
        if response.status_code == 200:
            result = response.json().get("response", "No response received.")
            st.success(result)
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
    except requests.exceptions.RequestException as e:
        st.error(f"Connection Error: {e}")
