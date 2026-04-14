import os
import joblib
import pandas as pd

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load saved objects
model = joblib.load(os.path.join(BASE_DIR, "models", "model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "models", "scaler.pkl"))
pca = joblib.load(os.path.join(BASE_DIR, "models", "pca.pkl"))
imputer = joblib.load(os.path.join(BASE_DIR, "models", "imputer.pkl"))
columns = joblib.load(os.path.join(BASE_DIR, "models", "columns.pkl"))

# Prediction function
def predict_loan(data_dict):
    
    df = pd.DataFrame([data_dict])

    df = pd.get_dummies(df)
    df = df.reindex(columns=columns, fill_value=0)

    df = imputer.transform(df)
    df = scaler.transform(df)
    df = pca.transform(df)

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return prediction, probability


# 🔥 ADD THIS FUNCTION
def risk_category(probability):
    if probability < 0.3:
        return "Low Risk"
    elif probability < 0.7:
        return "Medium Risk"
    else:
        return "High Risk"