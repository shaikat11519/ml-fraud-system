import joblib
import numpy as np
import pandas as pd
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "fraud_model.pkl"
FEATURE_NAMES = [
    "login_frequency",
    "session_duration",
    "transaction_amount",
    "device_changes",
    "failed_logins",
]


def load_model(model_path: str | Path = MODEL_PATH):
    artifact = joblib.load(model_path)
    return artifact["model"], artifact["scaler"]


def predict(features: list) -> dict:
    model, scaler = load_model()
    X = pd.DataFrame([features], columns=FEATURE_NAMES)
    X_scaled = scaler.transform(X)
    prediction = int(model.predict(X_scaled)[0])
    probability = float(model.predict_proba(X_scaled)[0][1])
    return {"prediction": prediction, "fraud_probability": round(probability, 4)}


def read_features_from_terminal() -> list[float]:
    print("Enter feature values for fraud prediction:")
    values = []
    for feature_name in FEATURE_NAMES:
        while True:
            raw = input(f"{feature_name}: ").strip()
            try:
                values.append(float(raw))
                break
            except ValueError:
                print("Invalid number. Please enter a numeric value.")
    return values


if __name__ == "__main__":
    features = read_features_from_terminal()
    result = predict(features)
    label = "Fraud" if result["prediction"] == 1 else "Not Fraud"

    print("\nPrediction Result")
    print(f"class: {label} ({result['prediction']})")
    print(f"fraud_probability: {result['fraud_probability']}")
