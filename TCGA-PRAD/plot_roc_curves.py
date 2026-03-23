"""
Plot OvR ROC curves and 98% sensitivity points.

Usage:
    python plot_roc_curves.py \
        --predictions output/analysis/predictions_with_labels.csv \
        --output output/analysis/roc_curves_ovr.png
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score


CLASS_NAMES = ["G3-dominant", "G4-dominant", "G5-dominant"]
LABEL_MAP = {1: 0, 2: 1, 3: 2}


def main(args):
    df = pd.read_csv(args.predictions)
    labeled = df[df["ground_truth_label"].notna()].copy()
    labeled["ground_truth_label"] = labeled["ground_truth_label"].astype(int)
    labeled["gt_idx"] = labeled["ground_truth_label"].map(LABEL_MAP)

    y_true = labeled["gt_idx"].values
    probs = labeled[["prob_G3", "prob_G4", "prob_G5"]].values

    fig, ax = plt.subplots(figsize=(8, 7))

    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        y_bin = (y_true == cls_idx).astype(int)
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            continue
        fpr, tpr, thresholds = roc_curve(y_bin, probs[:, cls_idx])
        auc = roc_auc_score(y_bin, probs[:, cls_idx])
        ax.plot(fpr, tpr, label=f"{cls_name} (AUC={auc:.3f})")

        # Mark specificity at 98% sensitivity
        idx = np.where(tpr >= 0.98)[0]
        if len(idx) > 0:
            i = idx[0]
            spec = 1 - fpr[i]
            thresh = thresholds[i]
            ax.plot(fpr[i], tpr[i], "o")
            ax.text(
                fpr[i],
                tpr[i],
                f"spec={spec:.2f}\nT={thresh:.2f}",
                fontsize=8,
                ha="left",
                va="bottom",
            )

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title("OvR ROC Curves (Labeled Set)")
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
        default="output/analysis/predictions_with_labels.csv",
        help="CSV with predictions and labels",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/analysis/roc_curves_ovr.png",
        help="Output path for ROC plot",
    )
    args = parser.parse_args()
    main(args)
