import streamlit as st
import pickle
import numpy as np
import pandas as pd

st.set_page_config(page_title="💳 EMI Eligibility Predictor", layout="centered")
st.title(" EMI Eligibility Prediction App")

try:
    model = pickle.load(open("best_model.pkl", "rb"))
    le = pickle.load(open("label_encoder.pkl", "rb"))
    st.success("✅ Model and encoder loaded successfully!")
except Exception as e:
    st.error(f" Error loading model: {e}")
    st.stop()

st.header("Enter Applicant Details")

age = st.number_input("Age", 18, 70, 30)
salary = st.number_input("Monthly Salary (₹)", 5000, 2000000, 50000)
credit_score = st.number_input("Credit Score", 300, 900, 700)
bank_balance = st.number_input("Bank Balance (₹)", 0, 5000000, 100000)
requested_amount = st.number_input("Requested Loan Amount (₹)", 10000, 5000000, 250000)
requested_tenure = st.number_input("Requested Tenure (months)", 3, 120, 12)
monthly_rent = st.number_input("Monthly Rent (₹)", 0, 100000, 10000)
current_emi = st.number_input("Current EMI (₹)", 0, 100000, 5000)
groceries = st.number_input("Groceries & Utilities (₹)", 0, 100000, 8000)
travel = st.number_input("Travel Expenses (₹)", 0, 100000, 4000)
other_expenses = st.number_input("Other Monthly Expenses (₹)", 0, 100000, 3000)
emergency_fund = st.number_input("Emergency Fund (₹)", 0, 1000000, 20000)

employment_type = st.selectbox("Employment Type", ["Private", "Government", "Self-employed"])
existing_loans = st.selectbox("Existing Loans", ["Yes", "No"])
house_type = st.selectbox("House Type", ["Own", "Rented", "Family"])


if st.button("Predict EMI Eligibility"):

    try:
        df = pd.DataFrame({
            "age": [age],
            "monthly_salary": [salary],
            "credit_score": [credit_score],
            "bank_balance": [bank_balance],
            "requested_amount": [requested_amount],
            "requested_tenure": [requested_tenure],
            "monthly_rent": [monthly_rent],
            "current_emi_amount": [current_emi],
            "groceries_utilities": [groceries],
            "travel_expenses": [travel],
            "other_monthly_expenses": [other_expenses],
            "emergency_fund": [emergency_fund],
            "employment_type": [employment_type],
            "existing_loans": [existing_loans],
            "house_type": [house_type],
        })

        df["affordability_ratio"] = (
            (df["bank_balance"] + df["emergency_fund"]) / (df["requested_amount"] + 1)
        )
        df["expense_to_income"] = (
            (df["groceries_utilities"] + df["travel_expenses"] + df["other_monthly_expenses"])
            / (df["monthly_salary"] + 1)
        )

        df["max_monthly_emi"] = 0  

        df.fillna(0, inplace=True)

        prediction = model.predict(df)[0]
        pred_label = le.inverse_transform([prediction])[0] if hasattr(le, "inverse_transform") else prediction

        st.subheader(" Prediction Result:")
        if str(pred_label).lower() == "eligible":
            st.success(" The applicant is Eligible for EMI!")
        elif str(pred_label).lower() == "partially_eligible":
            st.warning(" The applicant is Partially Eligible for EMI.")
        else:
            st.error(" The applicant is Not Eligible for EMI.")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.markdown("---")
st.caption("Developed as part of EMI Prediction AI Project | © Vibha Chauhan 2025")
