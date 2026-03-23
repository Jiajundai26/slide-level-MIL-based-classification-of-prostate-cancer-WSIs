#!/usr/bin/env python3
"""
Plot ROC curve for binary classification predictions.

Usage:
    python plot_roc_curve_binary.py \
        --predictions output/supervised_attention/your_run/test_predictions.csv \
        --output output/supervised_attention/your_run/roc_curve.png
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score


def main(args):
    df = pd.read_csv(args.predictions)
    df = df.dropna(subset=["true_label", "probability"]).copy()
    df["true_label"] = df["true_label"].astype(int)

    y_true = df["true_label"].values
    y_prob = df["probability"].values

    if len(np.unique(y_true)) < 2:
        print("Not enough class variety to compute ROC.")
        return

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, label=f"ROC (AUC={auc:.3f})", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title("ROC Curve (Binary)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.output, dpi=200)
    plt.close()
    print(f"Saved ROC plot to: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="CSV with columns: true_label, probability",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for ROC plot",
    )
    args = parser.parse_args()
    main(args)
