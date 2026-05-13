import numpy as np
import pandas as pd


def _sigmoid(x):
    # Converts a linear score (log-odds) into a probability in (0, 1).
    return 1 / (1 + np.exp(-x))


def generate_dataset(n_samples: int = 10000, random_state: int = 42) -> pd.DataFrame:
    np.random.seed(random_state)

    login_frequency    = np.random.poisson(2, n_samples)              # count of logins
    session_duration   = np.random.exponential(scale=10, size=n_samples)  # minutes
    transaction_amount = np.random.exponential(scale=50, size=n_samples)  # dollars
    device_changes     = np.random.poisson(0.2, n_samples)            # device switches
    failed_logins      = np.random.poisson(0.1, n_samples)            # failed attempts

    X = np.column_stack(
        [login_frequency, session_duration, transaction_amount, device_changes, failed_logins]
    )

    # Standardize so features are on the same scale before computing scores.
    X_z = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)

    # Positive weight = increases fraud risk; negative = decreases it.
    weights   = np.array([-0.3, -0.6, 1.2, 1.0, 0.9])
    intercept = -2.0  # keeps base fraud rate ~20%

    probs = _sigmoid(intercept + X_z.dot(weights))
    y     = np.random.binomial(1, probs)  # stochastic binary labels

    return pd.DataFrame(
        {
            "login_frequency":    login_frequency,
            "session_duration":   session_duration,
            "transaction_amount": transaction_amount,
            "device_changes":     device_changes,
            "failed_logins":      failed_logins,
            "Class":              y,
        }
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic fraud dataset")
    parser.add_argument("--n",    type=int, default=10000)
    parser.add_argument("--out",  default="../data/dataset.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = generate_dataset(n_samples=args.n, random_state=args.seed)
    df.to_csv(args.out, index=False)
    print(f"Generated {df.shape[0]} rows -> fraud rate: {df['Class'].mean():.4f}")


# ---------------------------------------------------------------------------
# Explanation (using only the ML topics you requested)
# The following points explain how this generated dataset relates to training
# and evaluation of a classifier, using only your listed topics.
# ---------------------------------------------------------------------------
# Datasets & Labels:
# - This script produces a dataset with features and a binary label `Class`.
# - Labels are sampled from a Bernoulli distribution driven by a logistic
#   probability; therefore labels contain realistic overlap and noise.
#
# Imbalanced datasets:
# - The `intercept` and `weights` control the base fraud rate; with
#   the current intercept the dataset is imbalanced (~20% fraud in default).
# - Imbalance affects metric choice: prefer precision/recall or PR curves over
#   raw accuracy when fraud is the minority class.
#
# Train/test split:
# - When training, split the generated data into train and test (e.g., 80/20).
# - Use `stratify=y` in `train_test_split` to preserve class ratio in both sets.
#
# Data transformation:
# - The code standardizes features before computing scores; likewise, apply
#   consistent scaling (e.g., `StandardScaler`) during model training and
#   persist the scaler for production inference to avoid distribution shift.
#
# Logistic Regression (probability, loss, regularization):
# - The generator creates probabilities using a linear score passed through
#   a sigmoid (logistic) function — the same mapping logistic regression
#   learns from data.
# - Training a logistic regression minimizes the cross-entropy (log loss),
#   which measures the divergence between predicted probabilities and labels.
# - Use L2 regularization to control model complexity and reduce overfitting.
#
# Classification (thresholds, confusion matrix, metrics):
# - After training, convert predicted probabilities to labels via a threshold
#   (default 0.5) or adjust threshold to optimize precision/recall balance.
# - Evaluate with a confusion matrix to compute accuracy, precision, recall.
# - For imbalanced data, prioritize recall (catching fraud) and precision
#   (reducing false alarms); use ROC & AUC and PR curves to assess trade-offs.
# - Be mindful of prediction bias: check per-group performance if groups exist.

# Data & Model Concepts (generalization, overfitting, complexity, loss curves):
# - Logistic regression is a low-complexity linear model; it's less prone to
#   overfitting than high-capacity models but still benefits from regularization.
# - Monitor training/validation loss curves: a rising gap indicates overfitting.
# - Use techniques like regularization (L2), cross-validation, and simpler
#   model choices to improve generalization.
# ---------------------------------------------------------------------------
