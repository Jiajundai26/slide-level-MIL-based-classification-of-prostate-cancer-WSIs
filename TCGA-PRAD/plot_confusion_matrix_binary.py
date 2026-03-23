#!/usr/bin/env python3
"""
Plot confusion matrix for binary classification predictions.

Usage:
    python plot_confusion_matrix_binary.py \
        --predictions output/supervised_attention/your_run/test_predictions.csv \
        --output output/supervised_attention/your_run/confusion_matrix.png
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def main(args):
    df = pd.read_csv(args.predictions)
    df = df.dropna(subset=["true_label", "predicted"]).copy()
    df["true_label"] = df["true_label"].astype(int)
    df["predicted"] = df["predicted"].astype(int)

    y_true = df["true_label"].values
    y_pred = df["predicted"].values

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal", "Cancer"])
    ax.set_yticklabels(["Normal", "Cancer"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (Binary)")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, dpi=200)
    plt.close()
    print(f"Saved confusion matrix to: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="CSV with columns: true_label, predicted",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for confusion matrix image",
    )
    args = parser.parse_args()
    main(args)
