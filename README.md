# Fraud Detection System - Codebase Summary

## Main Goal
Build an end-to-end fraud detection pipeline that:
1. Generates synthetic fraud data
2. Trains a Logistic Regression model
3. Serves predictions through a FastAPI endpoint

## Architecture Overview

```text
src/generate_dataset.py  ->  data/dataset.csv
         |
src/preprocess.py        ->  scaling + train/test split
         |
src/train.py             ->  models/fraud_model.pkl
         |
src/predict.py           ->  inference from saved model
         |
api/main.py              ->  REST API (/health, /predict)
```

## File-by-File Explanation

### src/generate_dataset.py
- Creates synthetic fraud dataset with 5 features:
  - login_frequency
  - session_duration
  - transaction_amount
  - device_changes
  - failed_logins
- Generates `Class` labels using a sigmoid-based probability score.
- Saves data to `data/dataset.csv`.
- Default setup creates 10,000 rows with about 20% fraud rate.

### src/preprocess.py
- Loads CSV into pandas DataFrame.
- Splits data into features (`X`) and label (`y`).
- Applies `StandardScaler` transformation.
- Uses `train_test_split(..., stratify=y)` to preserve class ratio.

### src/train.py
- Loads and preprocesses dataset.
- Trains `LogisticRegression` with:
  - `max_iter=1000`
  - `class_weight="balanced"` (helps with imbalanced data)
- Prints classification report (precision, recall, F1, accuracy).
- Saves model + scaler artifact to `models/fraud_model.pkl`.

### src/predict.py
- Loads saved model artifact.
- Accepts a feature list, scales it, predicts:
  - `prediction` (0/1)
  - `fraud_probability` (0.0 to 1.0)
- Note: placeholder in `__main__` currently uses 30 values and should be changed to 5 features.

### api/main.py
- FastAPI app for inference service.
- Endpoints:
  - `GET /health` for status check
  - `POST /predict` for fraud prediction
- Uses request/response models with Pydantic.

### src/test.py
- Currently a smoke check only (`print(...)`).
- Does not yet include unit/integration tests.

## What Is Complete
- Synthetic dataset generation
- Feature preprocessing and scaling
- Logistic Regression training pipeline
- Model persistence to disk
- Prediction utility from saved model
- API endpoints for health and prediction

## Remaining Improvements
- Fix `src/predict.py` sample input in `__main__` from 30 features to 5.
- Add proper tests for preprocess, train output artifact, and prediction behavior.
- Add API tests for `/health` and `/predict`.
- Optionally expand evaluation with ROC-AUC, confusion matrix plots, and threshold tuning.

## How It Works End-to-End
1. Run dataset generator to create `data/dataset.csv`.
2. Train model with `src/train.py`.
3. Saved artifact (`model` + `scaler`) is loaded by `src/predict.py`.
4. FastAPI endpoint in `api/main.py` calls `predict()` for real-time inference.

## Run Commands

```bash
cd ml-fraud-system
source .venv/bin/activate
PYTHONPATH=src python3 src/predict.py
```

## How prediction is calculated

This project uses a trained Logistic Regression model. The prediction for a single input is computed in these steps:

1. Feature order: provide values in this exact order: `login_frequency`, `session_duration`, `transaction_amount`, `device_changes`, `failed_logins`.
2. The script builds a single-row pandas `DataFrame` with the above column names.
3. The saved `StandardScaler` (fitted during training) transforms the row: `X_scaled = scaler.transform(X)`.
4. The logistic model maintains coefficients (`model.coef_`) and intercept (`model.intercept_`).
  - Compute the linear score (log-odds):
    - `z = intercept + X_scaled.dot(coef_.T)`
5. Convert log-odds to probability with the logistic (sigmoid) function:
  - `prob = 1 / (1 + exp(-z))`.
6. Convert probability to a class using a threshold (default 0.5):
  - `pred = 1 if prob >= 0.5 else 0`.

In code (manual calculation using the saved artifact):

```python
import joblib
import numpy as np
import pandas as pd

artifact = joblib.load('models/fraud_model.pkl')
model, scaler = artifact['model'], artifact['scaler']

features = [2, 12.5, 250.0, 1, 0]
X = pd.DataFrame([features], columns=['login_frequency','session_duration','transaction_amount','device_changes','failed_logins'])
X_scaled = scaler.transform(X)

# linear score (log-odds)
z = model.intercept_[0] + X_scaled.dot(model.coef_.T)[0]
prob = 1.0 / (1.0 + np.exp(-z))
pred = int(prob >= 0.5)

print('probability:', prob)
print('prediction:', pred)

# shortcut using sklearn
print(model.predict_proba(X_scaled)[0][1], model.predict(X_scaled)[0])
```

Notes:
- The model returned probability and class using `model.predict_proba` and `model.predict`. Use those in production to avoid mistakes with shapes.
- If you want to prefer fewer false positives or fewer false negatives, change the threshold from `0.5` to a value that suits your trade-off.

## Terminal example (interactive)

Run the interactive predictor to enter values from the terminal and see the prediction printed immediately:

```bash
source .venv/bin/activate
PYTHONPATH=src python3 src/predict.py
```

Example session (user input shown):

```
Enter feature values for fraud prediction:
login_frequency: 2
session_duration: 12.5
transaction_amount: 250
device_changes: 1
failed_logins: 0

Prediction Result
class: Fraud (1)
fraud_probability: 0.9949
```

This output means the model computed a probability of approximately `0.9949` for the fraud class and therefore returned class `1` (Fraud) at the default 0.5 threshold.

If you'd like, I can also add a one-shot script that accepts arguments and prints the same output without interactive prompts.
