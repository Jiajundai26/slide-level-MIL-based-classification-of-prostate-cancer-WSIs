#!/usr/bin/env python3
"""
Recompute the ROC curve for the fine-tuned GTEx model on the full combined test set.

The original roc_curve_finetune_gtex.png showed AUC=nan because the fine-tune
evaluation only used GTEx test slides (all benign, label=0). This script evaluates
the fine-tuned model on the full test set (TCGA cancer + GTEx benign) to produce
a meaningful ROC curve.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score

# ============================================================================
# Model architecture (must match training script)
# ============================================================================
import torch.nn as nn

class GatedAttention(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128, dropout=0.25):
        super().__init__()
        self.attention_a = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh(), nn.Dropout(dropout))
        self.attention_b = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Sigmoid(), nn.Dropout(dropout))
        self.attention_c = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = self.attention_c(a * b).squeeze(-1)
        return torch.softmax(A, dim=1)

class ABMIL(nn.Module):
    def __init__(self, input_dim=1536, hidden_dim=256, attention_dim=128, num_classes=1, dropout=0.25):
        super().__init__()
        self.feature_projection = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.attention = GatedAttention(input_dim=hidden_dim, hidden_dim=attention_dim, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x, return_attention=False):
        if isinstance(x, dict):
            x = x['features']
        h = self.feature_projection(x)
        A = self.attention(h)
        M = torch.bmm(A.unsqueeze(1), h).squeeze(1)
        logits = self.classifier(M)
        if return_attention:
            return logits, A
        return logits


def main():
    run_dir = Path(__file__).resolve().parent / \
        "output/supervised_attention/supervised_attention_20260202_003005"

    # --- load config ---
    with open(run_dir / "config.json") as f:
        config = json.load(f)

    tcga_feats_dir = Path(config["tcga_feats_dir"])
    gtex_feats_dir = Path(config["gtex_feats_dir"])

    # --- determine which checkpoint to use ---
    finetuned_ckpt = run_dir / "best_model_finetuned_gtex.pt"
    if not finetuned_ckpt.exists():
        finetuned_ckpt = run_dir / "latest_checkpoint_finetune.pt"
    if not finetuned_ckpt.exists():
        print("ERROR: No fine-tuned checkpoint found.")
        sys.exit(1)
    print(f"Using checkpoint: {finetuned_ckpt}")

    # --- read the combined test set slide IDs + labels from test_predictions.csv ---
    combined_df = pd.read_csv(run_dir / "test_predictions.csv")
    slide_ids = combined_df["slide_id"].tolist()
    true_labels = combined_df["true_label"].astype(int).tolist()
    print(f"Full test set: {len(slide_ids)} slides "
          f"(cancer={sum(true_labels)}, benign={len(true_labels)-sum(true_labels)})")

    # --- load model ---
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    checkpoint = torch.load(finetuned_ckpt, map_location=device, weights_only=False)
    ckpt_config = checkpoint.get("config", config)

    model = ABMIL(
        input_dim=ckpt_config.get("input_dim", 1536),
        hidden_dim=ckpt_config.get("hidden_dim", 256),
        attention_dim=ckpt_config.get("attention_dim", 128),
        num_classes=1,
        dropout=0.0,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print("Model loaded.")

    # --- run inference on full test set ---
    probabilities = []
    valid_labels = []
    valid_ids = []

    for sid, label in zip(slide_ids, true_labels):
        # Find feature file
        h5_path = None
        for feats_dir in [tcga_feats_dir, gtex_feats_dir]:
            candidate = feats_dir / f"{sid}.h5"
            if candidate.exists():
                h5_path = candidate
                break
        if h5_path is None:
            print(f"  WARNING: features not found for {sid}, skipping.")
            continue

        with h5py.File(h5_path, "r") as f:
            features = torch.from_numpy(f["features"][:]).float()

        with torch.no_grad():
            features = features.unsqueeze(0).to(device)
            logits = model(features)
            prob = torch.sigmoid(logits).squeeze().cpu().item()

        probabilities.append(prob)
        valid_labels.append(label)
        valid_ids.append(sid)

    y_true = np.array(valid_labels)
    y_prob = np.array(probabilities)
    y_pred = (y_prob > 0.5).astype(int)

    print(f"\nEvaluated {len(valid_ids)} slides with the fine-tuned model.")
    print(f"  Cancer: {(y_true == 1).sum()}, Benign: {(y_true == 0).sum()}")

    # --- compute metrics ---
    try:
        auc_score = roc_auc_score(y_true, y_prob)
    except ValueError:
        print("ERROR: Still only one class present. Cannot compute ROC.")
        sys.exit(1)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    accuracy = (y_pred == y_true).mean()
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  AUC-ROC:  {roc_auc:.4f}")

    # --- save updated predictions CSV ---
    pred_df = pd.DataFrame({
        "slide_id": valid_ids,
        "true_label": valid_labels,
        "predicted": y_pred.tolist(),
        "probability": probabilities,
    })
    pred_df.to_csv(run_dir / "test_predictions_finetune_gtex.csv", index=False)
    print(f"\nUpdated predictions saved to: {run_dir / 'test_predictions_finetune_gtex.csv'}")

    # --- update results JSON ---
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, precision_recall_fscore_support,
        confusion_matrix, average_precision_score
    )
    
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    ap = average_precision_score(y_true, y_prob)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    else:
        sensitivity = specificity = 0

    results = {
        "loss": None,
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_acc),
        "auc": float(roc_auc),
        "average_precision": float(ap),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "sensitivity": int(sensitivity) if isinstance(sensitivity, (int, np.integer)) else float(sensitivity),
        "specificity": int(specificity) if isinstance(specificity, (int, np.integer)) else float(specificity),
        "confusion_matrix": cm.tolist(),
        "predictions": {
            "slide_ids": valid_ids,
            "labels": [float(l) for l in valid_labels],
            "predictions": [float(p) for p in y_pred],
            "probabilities": [float(p) for p in probabilities],
        }
    }
    with open(run_dir / "test_results_finetune_gtex.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Updated results saved to: {run_dir / 'test_results_finetune_gtex.json'}")

    # --- plot ROC curve ---
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.4f}')
    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve - Supervised Attention (GTEx fine-tune)', fontsize=14)
    ax.legend(loc='lower right', fontsize=12)
    ax.set_aspect('equal')
    plt.tight_layout()

    out_png = run_dir / "roc_curve_finetune_gtex.png"
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nROC curve saved to: {out_png}")


if __name__ == "__main__":
    main()
