#!/usr/bin/env python3
"""
Generate combined ROC curves for TCGA-PRAD and GTEx on the same plot.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from pathlib import Path

# Paths
TCGA_TEST_RESULTS = "output/combined_binary_noncancer_vs_g45/test_results.json"
GTEX_LABELS_PATH = "/sci-it/projects/cedmav/data/ARPA-H/Prostate_HnE/preprocessing/GTEx_Prostate/GTEx_prostate_labels.csv"
GTEX_INFERENCE_PATH = Path.home() / "output/combined_inference_noncancer_vs_g45/inference_results.csv"
OUTPUT_DIR = Path("output")

def load_tcga_results():
    """Load TCGA test results."""
    with open(TCGA_TEST_RESULTS, 'r') as f:
        data = json.load(f)
    
    preds = data['test_results']['predictions']
    labels = np.array(preds['labels'])
    probs = np.array(preds['probabilities'])
    
    print(f"Loaded TCGA test results: {len(labels)} samples")
    print(f"  Non-Cancer: {sum(labels == 0)}, Cancer: {sum(labels == 1)}")
    
    return labels, probs

def load_gtex_results():
    """Load GTEx inference results with ground truth."""
    # Load ground truth labels
    gtex_labels_df = pd.read_csv(GTEX_LABELS_PATH)
    if "file_name" in gtex_labels_df.columns:
        gtex_labels_df = gtex_labels_df.rename(columns={"file_name": "slide_id"})
    gtex_labels_df['label'] = pd.to_numeric(gtex_labels_df['label'], errors='coerce')
    gtex_labels_df = gtex_labels_df[gtex_labels_df['label'].isin([0, 1])].reset_index(drop=True)
    labels_dict = {row['slide_id']: int(row['label']) for _, row in gtex_labels_df.iterrows()}
    
    # Load inference results
    inference_df = pd.read_csv(GTEX_INFERENCE_PATH)
    gtex_df = inference_df[inference_df['slide_id'].str.startswith('GTEX-')].copy()
    gtex_df['true_label'] = gtex_df['slide_id'].map(labels_dict)
    matched_df = gtex_df[gtex_df['true_label'].notna()].copy()
    
    labels = matched_df['true_label'].values.astype(int)
    probs = matched_df['probability'].values
    
    print(f"\nLoaded GTEx results: {len(labels)} samples")
    print(f"  Benign: {sum(labels == 0)}, Cancer: {sum(labels == 1)}")
    
    return labels, probs

def plot_combined_roc():
    """Generate combined ROC curve plot."""
    
    # Load data
    tcga_labels, tcga_probs = load_tcga_results()
    gtex_labels, gtex_probs = load_gtex_results()
    
    # Compute ROC curves
    tcga_fpr, tcga_tpr, _ = roc_curve(tcga_labels, tcga_probs)
    tcga_auc = auc(tcga_fpr, tcga_tpr)
    
    gtex_fpr, gtex_tpr, _ = roc_curve(gtex_labels, gtex_probs)
    gtex_auc = auc(gtex_fpr, gtex_tpr)
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(9, 8))
    
    # Combined model test set (TCGA-PRAD + GTEx combined training)
    ax.plot(tcga_fpr, tcga_tpr, color='#ff7f0e', lw=3,
            label=f'Combined Model Test Set (AUC = {tcga_auc:.3f}, n={len(tcga_labels)})')
    
    # GTEx inference
    ax.plot(gtex_fpr, gtex_tpr, color='#2ca02c', lw=3,
            label=f'GTEx Prostate Inference (AUC = {gtex_auc:.3f}, n={len(gtex_labels)})')
    
    # Random baseline
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.5, label='Random (AUC = 0.500)')
    
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=14, fontweight='bold')
    ax.set_title('ROC Curves — Combined Model (TCGA+GTEx Training)\nPerformance on Test Set and GTEx Inference', 
                 fontsize=15, fontweight='bold', pad=15)
    ax.legend(loc='lower right', fontsize=12, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_aspect('equal')
    
    # Add text box with summary
    textstr = f'Combined Model Test Set:\n  {sum(tcga_labels==0)} Non-Cancer, {sum(tcga_labels==1)} Cancer\n'
    textstr += f'GTEx Inference:\n  {sum(gtex_labels==0)} Benign, {sum(gtex_labels==1)} Cancer'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.55, 0.15, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'combined_roc_curves.png', dpi=250, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"\n{'='*80}")
    print("ROC Curve Summary")
    print(f"{'='*80}")
    print(f"Combined Model Test Set: AUC = {tcga_auc:.4f}  (n={len(tcga_labels)}: {sum(tcga_labels==0)} Non-Cancer, {sum(tcga_labels==1)} Cancer)")
    print(f"GTEx Inference:          AUC = {gtex_auc:.4f}  (n={len(gtex_labels)}: {sum(gtex_labels==0)} Benign, {sum(gtex_labels==1)} Cancer)")
    print(f"{'='*80}\n")
    print("Saved: output/combined_roc_curves.png")

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)
    plot_combined_roc()
