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


# 🔥 Prediction Function ONLY
def predict_loan(data_dict):
    
    df = pd.DataFrame([data_dict])

    # Encoding
    df = pd.get_dummies(df)

    # Align columns with training
    df = df.reindex(columns=columns, fill_value=0)

    # Apply preprocessing
    df = imputer.transform(df)
    df = scaler.transform(df)
    df = pca.transform(df)

    # Prediction
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return prediction, probability