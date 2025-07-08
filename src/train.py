"""
Train & evaluate a decision-tree classifier on the Iris dataset.

Usage:
    python src/train.py --test-size 0.2 --random-state 42
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def train(test_size: float = 0.2, random_state: int = 42) -> float:
    """Train the model and return accuracy on the held‑out test set."""
    # 1 Load data
    iris = load_iris()
    X, y = iris.data, iris.target

    # 2 Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    # 3 Model
    model = DecisionTreeClassifier(random_state=random_state)
    model.fit(X_train, y_train)

    # 4 Predict & evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # 5 Persist artefacts
    out_dir = Path(__file__).resolve().parent.parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, out_dir / "model.joblib")

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=iris.target_names)

    fig, ax = plt.subplots(figsize=(4, 4))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(f"Decision Tree - accuracy {acc:.2%}")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    return acc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a Decision Tree on the Iris dataset."
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data reserved for testing.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Seed for deterministic behaviour.",
    )
    args = parser.parse_args()

    accuracy = train(test_size=args.test_size, random_state=args.random_state)
    print(f"Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()