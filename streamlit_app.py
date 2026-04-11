import streamlit as st
import requests
import pandas as pd

st.title("📊 Customer Churn Prediction")
st.write("Enter customer details to predict churn")

st.subheader("Customer Details")
st.divider()

# Inputs
tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.number_input("Monthly Charges", value=50.0)
total_charges = st.number_input("Total Charges", value=500.0)

# Button
if st.button("Predict", key="predict_btn"):

    data = {
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }

    response = requests.post("http://127.0.0.1:8000/predict", json=data)
    result = response.json()

    churn = result["churn"]
    prob = result["probability"]

    # Show result
    if churn:
        st.error("Customer is likely to churn.")
    else:
        st.success("Customer is unlikely to churn.")

    st.metric("Churn Probability", f"{prob:.2%}")

    chart_data = pd.DataFrame({
        "Probability": [prob, 1 - prob]
    }, index=["Churn", "No Churn"])

    st.bar_chart(chart_data)

# Always visible info
st.info("This model predicts customer churn based on usage and billing patterns.")