import pandas as pd
import joblib
import os

# Load models
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model = joblib.load(os.path.join(BASE_DIR, "models", "model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "models", "scaler.pkl"))
imputer = joblib.load(os.path.join(BASE_DIR, "models", "imputer.pkl"))
columns = joblib.load(os.path.join(BASE_DIR, "models", "columns.pkl"))

# ─── PREDICTION FUNCTION ───────────────────────────────────────
def predict_loan(data_dict):
    
    df = pd.DataFrame([data_dict])

    # ✅ ADD IMPORTANT FEATURES (VERY IMPORTANT 🔥)
    df['loan_income_ratio'] = df['loan_amnt'] / (df['person_income'] + 1)
    df['income_to_loan'] = df['person_income'] / (df['loan_amnt'] + 1)

    # Encoding
    df = pd.get_dummies(df)
    df = df.reindex(columns=columns, fill_value=0)

    # Preprocessing
    df = imputer.transform(df)
    df = scaler.transform(df)

    # Prediction
    probability = model.predict_proba(df)[0][1]
    risk = risk_category(probability)

    return risk, probability


# ─── RISK CATEGORY ─────────────────────────────────────────────
def risk_category(prob):
    if prob < 0.3:
        return "Low Risk 🟢"
    elif prob < 0.6:
        return "Medium Risk ⚠️"
    else:
        return "High Risk 🔴"