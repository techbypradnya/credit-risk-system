import os
import joblib
import numpy as np
import pandas as pd

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load models
model = joblib.load(os.path.join(BASE_DIR, "models", "model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "models", "scaler.pkl"))
pca = joblib.load(os.path.join(BASE_DIR, "models", "pca.pkl"))
imputer = joblib.load(os.path.join(BASE_DIR, "models", "imputer.pkl"))
columns = joblib.load(os.path.join(BASE_DIR, "models", "columns.pkl"))


def predict_loan(data_dict):
    
    df = pd.DataFrame([data_dict])

    # Apply encoding
    df = pd.get_dummies(df)

    # Match training columns
    df = df.reindex(columns=columns, fill_value=0)

    # Apply pipeline
    df = imputer.transform(df)
    df = scaler.transform(df)
    df = pca.transform(df)

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return prediction, probability


def risk_category(prob):
   def risk_category(prob):
    if prob < 0.33:
        return "Low Risk"
    elif prob < 0.66:
        return "Medium Risk"
    else:
        return "High Risk"