import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from preprocess import load_data, preprocess

DATA_PATH = "data/dataset.csv"
OUT_DIR = "outputs"
OUT_FILE = "train_class_distribution_pie.png"


def main():
    # Load dataset and obtain the training split labels
    df = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test, scaler = preprocess(df)

    # Count classes in the training set (0 = not fraud, 1 = fraud)
    counts = y_train.value_counts().reindex([0, 1], fill_value=0)
    sizes = [int(counts.loc[0]), int(counts.loc[1])]
    labels = [f"Not Fraud (0)", f"Fraud (1)"]

    # Make output directory
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, OUT_FILE)

    # Plot pie chart with percentages and counts
    explode = (0, 0.08)  # emphasize the fraud slice
    plt.figure(figsize=(6, 6))
    def autopct(pct):
        total = sum(sizes)
        count = int(round(pct * total / 100.0))
        return f"{pct:.1f}%\n({count})"

    colors = ["#66b3ff", "#ff6666"]
    plt.pie(sizes, labels=labels, autopct=autopct, startangle=90, explode=explode, colors=colors)
    plt.title("Training set class distribution")
    plt.axis("equal")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

    print(f"Saved pie chart to {out_path}")
    print(f"Counts: Not Fraud (0) = {sizes[0]}, Fraud (1) = {sizes[1]}")


if __name__ == "__main__":
    main()
