#!/usr/bin/env python3
"""
Generate ROC curve for GTEx prostate inference results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from pathlib import Path

# Paths
GTEX_LABELS_PATH = "/sci-it/projects/cedmav/data/ARPA-H/Prostate_HnE/preprocessing/GTEx_Prostate/GTEx_prostate_labels.csv"
INFERENCE_RESULTS_PATH = Path.home() / "output/combined_inference_noncancer_vs_g45/inference_results.csv"
OUTPUT_DIR = Path("output")

CLASS_NAMES = ['Non-Cancer (Benign)', 'Cancer']

def load_gtex_labels():
    """Load GTEx ground truth labels."""
    df = pd.read_csv(GTEX_LABELS_PATH)
    
    # Rename file_name to slide_id if needed
    if "file_name" in df.columns and "slide_id" not in df.columns:
        df = df.rename(columns={"file_name": "slide_id"})
    
    # Ensure label is numeric and filter discard (-1)
    df['label'] = pd.to_numeric(df['label'], errors='coerce')
    df = df[df['label'].isin([0, 1])].reset_index(drop=True)
    
    # Create dictionary mapping slide_id to label
    labels_dict = {row['slide_id']: int(row['label']) for _, row in df.iterrows()}
    
    print(f"Loaded GTEx labels: {len(labels_dict)} slides")
    print(f"  Benign (0): {sum(1 for v in labels_dict.values() if v == 0)}")
    print(f"  Cancer (1): {sum(1 for v in labels_dict.values() if v == 1)}")
    
    return labels_dict

def load_inference_results():
    """Load inference results CSV."""
    df = pd.read_csv(INFERENCE_RESULTS_PATH)
    print(f"\nLoaded inference results: {len(df)} slides")
    
    # Filter to GTEx slides only (start with GTEX-)
    gtex_df = df[df['slide_id'].str.startswith('GTEX-')].copy()
    print(f"  GTEx slides: {len(gtex_df)}")
    
    return gtex_df

def plot_gtex_roc():
    """Generate ROC curve and confusion matrix for GTEx prostate predictions."""
    
    # Load data
    gtex_labels = load_gtex_labels()
    inference_df = load_inference_results()
    
    # Match predictions with ground truth labels
    inference_df['true_label'] = inference_df['slide_id'].map(gtex_labels)
    
    # Filter out slides without ground truth
    matched_df = inference_df[inference_df['true_label'].notna()].copy()
    print(f"\nMatched slides with ground truth: {len(matched_df)}")
    
    if len(matched_df) == 0:
        print("ERROR: No matched slides found!")
        return
    
    # Extract data for metrics
    y_true = matched_df['true_label'].values.astype(int)
    y_prob = matched_df['probability'].values
    y_pred = matched_df['predicted_class'].values
    
    # Check if both classes are present
    unique_labels = np.unique(y_true)
    print(f"Unique true labels: {unique_labels}")
    
    if len(unique_labels) < 2:
        print(f"WARNING: Only one class present in ground truth: {unique_labels}")
        print("Cannot generate ROC curve - need both classes.")
        return
    
    # Compute metrics
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, 
                                 precision_recall_fscore_support, roc_auc_score,
                                 average_precision_score)
    
    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)
    avg_precision = average_precision_score(y_true, y_prob)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    print(f"\n{'='*80}")
    print("GTEx Prostate Inference Results")
    print(f"{'='*80}")
    print(f"Test samples:      {len(matched_df)}")
    print(f"  Benign (0):      {sum(y_true == 0)}")
    print(f"  Cancer (1):      {sum(y_true == 1)}")
    print(f"\nMetrics:")
    print(f"  Accuracy:        {accuracy:.4f}")
    print(f"  Balanced Acc:    {balanced_acc:.4f}")
    print(f"  AUC-ROC:         {roc_auc:.4f}")
    print(f"  Avg Precision:   {avg_precision:.4f}")
    print(f"  Precision:       {precision:.4f}")
    print(f"  Recall (Sens):   {sensitivity:.4f}")
    print(f"  Specificity:     {specificity:.4f}")
    print(f"  F1 Score:        {f1:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"                   Predicted Benign  Predicted Cancer")
    print(f"Actual Benign:     {cm[0][0]:>17}  {cm[0][1]:>16}")
    print(f"Actual Cancer:     {cm[1][0]:>17}  {cm[1][1]:>16}")
    print(f"{'='*80}\n")
    
    # ========================================================================
    # Plot 1: Confusion Matrix
    # ========================================================================
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    
    disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    
    title = f"GTEx Prostate — Confusion Matrix\nAcc={accuracy:.3f}, Bal.Acc={balanced_acc:.3f}, AUC={roc_auc:.3f}"
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'gtex_confusion_matrix.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("Saved: output/gtex_confusion_matrix.png")
    
    # ========================================================================
    # Plot 2: ROC Curve
    # ========================================================================
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    
    ax.plot(fpr, tpr, color='#2ca02c', lw=2.5, 
            label=f'GTEx Prostate (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random (AUC = 0.500)')
    
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=13)
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=13)
    ax.set_title('ROC Curve — GTEx Prostate', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'gtex_roc_curve.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("Saved: output/gtex_roc_curve.png")
    
    # Save metrics to file
    metrics_dict = {
        'dataset': 'GTEx Prostate',
        'n_samples': len(matched_df),
        'n_benign': int(sum(y_true == 0)),
        'n_cancer': int(sum(y_true == 1)),
        'accuracy': float(accuracy),
        'balanced_accuracy': float(balanced_acc),
        'auc_roc': float(roc_auc),
        'average_precision': float(avg_precision),
        'precision': float(precision),
        'recall': float(recall),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'f1_score': float(f1),
        'confusion_matrix': cm.tolist()
    }
    
    import json
    with open(OUTPUT_DIR / 'gtex_metrics.json', 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    print("Saved: output/gtex_metrics.json")

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)
    plot_gtex_roc()
