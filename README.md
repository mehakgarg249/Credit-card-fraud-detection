💳 Credit Card Fraud Detection

A machine learning project that detects fraudulent credit card transactions using transaction feature analysis and a trained classification model.
The app provides a Streamlit-based web interface where users can input transaction data and instantly get fraud predictions.

🧠 Project Overview

This project aims to identify potentially fraudulent credit card transactions by analyzing anonymized transaction features (V1–V28) and additional engineered features such as Hours and AvgRatio_of_Amount.

It leverages a trained machine learning pipeline (saved in .joblib format) that has been preprocessed and optimized for fraud detection.
🚀 Features

Interactive Streamlit App for easy prediction

Pretrained ML model (via Joblib) for instant inference

Custom feature input fields for real transaction simulation

Clear output with fraud/legit classification message

🧩 Technologies Used

Python 3.10+

Streamlit

Pandas

NumPy

Joblib

Scikit-learn

Jupyter Notebook

🧮 How It Works

The user inputs selected transaction features such as:

V4, V8, V12, V14, V17

Hours

AvgRatio_of_Amount

The system fills in default values for all other hidden features (V1–V28) using precomputed averages.

The feature vector is passed into the trained model.

The model predicts:

✅ Legitimate Transaction

🚨 Fraudulent Transaction

📊 Model Training

The model was developed and trained in pipeline_model.ipynb, which includes:

Data preprocessing and cleaning

Feature scaling and transformation

Model selection and evaluation

Saving the final model using joblib

🧠 Future Enhancements

Integrate real-time API for live transaction data

Add visual analytics dashboard for transaction trends

Deploy app on Streamlit Cloud / AWS / Render
