import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the model (update path if needed)
model = joblib.load(r"C:\Users\gargm\Downloads\fraud_detection_model (1).joblib")

# Feature names in correct order (30 total from training)
all_features = [
    'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
    'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
    'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Hours', 'AvgRatio_of_Amount'
]

# Default values for hidden features (based on training dataset stats or zeros)
default_values = {
    'V1': -0.541529,
    'V2': 0.350027,
    'V3': -0.769281,
    'V5': -0.267808,
    'V6': -0.181686,
    'V7': -0.591665,
    'V9': -0.250592,
    'V10': -0.55679,
    'V11': 0.385734,
    'V13': 0.005275,
    'V15': -0.013339,
    'V16': -0.420481,
    'V18': -0.257139,
    'V19': 0.092033,
    'V20': 0.033299,
    'V21': 0.099229,
    'V22': -0.014931,
    'V23': -0.012127,
    'V24': -0.00647,
    'V25': 0.003167,
    'V26': 0.003133,
    'V27': 0.019565,
    'V28': 0.02235
}

# Streamlit UI
st.title("Credit Card Fraud Detection")

st.markdown("Enter values for selected transaction features:")

# User input for selected features
v4 = st.number_input("V4", value=0.0, step=0.1)
v8 = st.number_input("V8", value=0.0, step=0.1)
v12 = st.number_input("V12", value=0.0, step=0.1)
v14 = st.number_input("V14", value=0.0, step=0.1)
v17 = st.number_input("V17", value=0.0, step=0.1)
hour = st.slider("Hours", 0, 23, 12)
avg_ratio = st.number_input("AvgRatio_of_Amount", value=0.0, step=0.01)

# Build full feature vector
input_data = {col: default_values.get(col, 0) for col in all_features}  # Fill all with defaults

# Update with user inputs
input_data.update({
    'V4': v4,
    'V8': v8,
    'V12': v12,
    'V14': v14,
    'V17': v17,
    'Hours': hour,
    'AvgRatio_of_Amount': avg_ratio
})

# Create DataFrame for prediction
input_df = pd.DataFrame([input_data])

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    if prediction == 1:
        st.error("🚨 Fraudulent Transaction")
    else:
        st.success("✅ Transaction is Legitimate.")
