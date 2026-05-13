import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from preprocess import load_data, preprocess

DATA_PATH = "data/dataset.csv"
MODEL_PATH = "models/fraud_model.pkl"


def train():
    df = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test, scaler = preprocess(df)

    model = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    joblib.dump({"model": model, "scaler": scaler}, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train()
