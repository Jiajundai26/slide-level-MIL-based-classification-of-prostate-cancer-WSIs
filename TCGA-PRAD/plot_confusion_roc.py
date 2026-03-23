#!/usr/bin/env python3
"""
Generate confusion matrix and ROC curve plots for trained models.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from pathlib import Path

# ============================================================================
# Model results directories
# ============================================================================

MODELS = {
    'TCGA-PRAD Only': 'output/binary_noncancer_vs_g45',
    'Combined (TCGA+GTEx)': 'output/combined_binary_noncancer_vs_g45',
    'TCGA-PRAD Only (v2)': 'output/TCGA_binary_noncancer_vs_g45',
}

CLASS_NAMES = ['Non-Cancer', 'Cancer (G4/G5)']

def load_results(results_dir):
    """Load test results from a model output directory."""
    results_path = Path(results_dir) / 'test_results.json'
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    preds = data['test_results']['predictions']
    labels = np.array(preds['labels'])
    probs = np.array(preds['probabilities'])
    predictions = np.array(preds['predictions'])
    
    metrics = data['metrics']
    return labels, probs, predictions, metrics


def plot_all():
    """Generate confusion matrices and ROC curves for all models."""
    
    # Load all model results
    model_data = {}
    for name, path in MODELS.items():
        try:
            labels, probs, predictions, metrics = load_results(path)
            model_data[name] = {
                'labels': labels,
                'probs': probs,
                'predictions': predictions,
                'metrics': metrics
            }
            print(f"Loaded {name}: {len(labels)} test samples")
        except Exception as e:
            print(f"Warning: Could not load {name}: {e}")
    
    n_models = len(model_data)
    if n_models == 0:
        print("No models loaded!")
        return
    
    # ========================================================================
    # 1. Confusion Matrices (side by side)
    # ========================================================================
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]
    
    for ax, (name, data) in zip(axes, model_data.items()):
        cm = confusion_matrix(data['labels'], data['predictions'])
        disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
        disp.plot(ax=ax, cmap='Blues', colorbar=False, values_format='d')
        
        acc = data['metrics']['accuracy']
        bal_acc = data['metrics']['balanced_accuracy']
        auc_val = data['metrics'].get('auc', None)
        
        title = f"{name}\nAcc={acc:.3f}, Bal.Acc={bal_acc:.3f}"
        if auc_val is not None and not (isinstance(auc_val, float) and np.isnan(auc_val)):
            title += f", AUC={auc_val:.3f}"
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=11)
        ax.set_ylabel('True Label', fontsize=11)
    
    plt.suptitle('Confusion Matrices — Non-Cancer vs Cancer (G4/G5)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig('output/confusion_matrices.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("\nSaved: output/confusion_matrices.png")
    
    # ========================================================================
    # 2. ROC Curves (overlaid)
    # ========================================================================
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for (name, data), color in zip(model_data.items(), colors):
        labels = data['labels']
        probs = data['probs']
        
        # Check if both classes are present
        if len(np.unique(labels)) < 2:
            print(f"  Skipping ROC for '{name}': only one class in test set")
            continue
        
        fpr, tpr, thresholds = roc_curve(labels, probs)
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, color=color, lw=2.5,
                label=f'{name} (AUC = {roc_auc:.3f})')
    
    # Diagonal reference
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random (AUC = 0.500)')
    
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=13)
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=13)
    ax.set_title('ROC Curves — Non-Cancer vs Cancer (G4/G5)', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    fig.savefig('output/roc_curves.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("Saved: output/roc_curves.png")
    
    # ========================================================================
    # 3. Print summary table
    # ========================================================================
    print("\n" + "=" * 90)
    print(f"{'Model':<28s} {'Acc':>7s} {'Bal.Acc':>8s} {'AUC':>7s} {'Prec':>7s} {'Recall':>7s} {'Spec':>7s} {'F1':>7s}")
    print("-" * 90)
    for name, data in model_data.items():
        m = data['metrics']
        auc_str = f"{m['auc']:.4f}" if not (isinstance(m['auc'], float) and np.isnan(m['auc'])) else "N/A"
        print(f"{name:<28s} {m['accuracy']:7.4f} {m['balanced_accuracy']:8.4f} {auc_str:>7s} "
              f"{m['precision']:7.4f} {m['recall']:7.4f} {m['specificity']:7.4f} {m['f1']:7.4f}")
    print("=" * 90)


if __name__ == "__main__":
    plot_all()
