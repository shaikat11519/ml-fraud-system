import joblib
import numpy as np

MODEL_PATH = "../models/fraud_model.pkl"


def load_model(model_path: str = MODEL_PATH):
    artifact = joblib.load(model_path)
    return artifact["model"], artifact["scaler"]


def predict(features: list) -> dict:
    model, scaler = load_model()
    X = np.array(features).reshape(1, -1)
    X_scaled = scaler.transform(X)
    prediction = int(model.predict(X_scaled)[0])
    probability = float(model.predict_proba(X_scaled)[0][1])
    return {"prediction": prediction, "fraud_probability": round(probability, 4)}


if __name__ == "__main__":
    sample = [0.0] * 30  # placeholder feature vector
    print(predict(sample))
