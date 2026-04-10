from flask import Flask, request, jsonify
from src.predict import predict_loan, risk_category

app = Flask(__name__)

@app.route("/")
def home():
    return "Credit Risk API Running 🚀"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["features"]

    pred, prob = predict_loan(data)
    risk = risk_category(prob)

    return jsonify({
        "prediction": int(pred),
        "probability": float(prob),
        "risk_level": risk
    })


if __name__ == "__main__":
    app.run(debug=True)