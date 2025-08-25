import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained XGBoost model
model_path = "notebook/credit_card_fraud_xgb.pkl"
xgb_model = joblib.load(model_path)

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")
st.title("Credit Card Fraud Detection (XGBoost)")

st.write("""
This demo predicts the probability of a credit card transaction being fraudulent.
""")

# Sidebar for user input
st.sidebar.header("Transaction Features Input")

# Key features for user input
key_features = ["V3", "V7", "V14", "V17", "Amount_log", "Amount_x_Hour"]
input_data = {}

# User inputs for key features
for f in key_features:
    input_data[f] = st.sidebar.number_input(f, value=0.0)

# Full feature list in the same order as training
features = ['V1','V2','V3','V4','V5','V6','V7','V8','V9','V10',
            'V11','V12','V13','V14','V15','V16','V17','V18','V19','V20',
            'V21','V22','V23','V24','V25','V26','V27','V28','Amount_log','Amount_x_Hour']

# Initialize DataFrame with all zeros
input_df = pd.DataFrame([{f: 0.0 for f in features}])

# Overwrite user inputs for key features
for f in key_features:
    input_df[f] = input_data[f]

# Predict button
if st.button("Predict Fraud Probability"):
    prob = xgb_model.predict_proba(input_df)[:, 1][0]
    st.subheader("Prediction Result")
    st.write(f"Fraud Probability: **{prob:.4f}**")
    
    if prob > 0.5:
        st.error("⚠️ This transaction is likely fraudulent!")
    else:
        st.success("✅ This transaction is likely legitimate.")
