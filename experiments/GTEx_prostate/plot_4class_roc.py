#!/usr/bin/env python3
"""
Quick script to generate ROC curves for 4-class Gleason classification.
Runs inference on the combined evaluation set and plots ROC curves.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# Configuration
CHECKPOINT_PATH = "/local/data/magicscan/HnE/GTEx_prostate/output/trident_4class/4class_focal_20260123_001634/best_model.pt"
TCGA_FEATS_DIR = "/sci-it/projects/cedmav/data/ARPA-H/Prostate_HnE/preprocessing/TCGA-PRAD/20x_256px_0px_overlap/features_uni_v2"
TCGA_LABELS_PATH = "/sci-it/projects/cedmav/data/ARPA-H/Prostate_HnE/preprocessing/TCGA-PRAD/labels_MIL/slide_labels.tsv"
GTEX_FEATS_DIR = "/sci-it/projects/cedmav/data/ARPA-H/Prostate_HnE/preprocessing/GTEx_Prostate/20x_256px_0px_overlap/features_uni_v2"
GTEX_LABELS_PATH = "/sci-it/projects/cedmav/data/ARPA-H/Prostate_HnE/preprocessing/GTEx_Prostate/GTEx_prostate_labels.csv"
OUTPUT_DIR = "/local/data/magicscan/HnE/GTEx_prostate/output/4class_inference"

NUM_CLASSES = 4
CLASS_NAMES = {
    0: 'Normal',
    1: 'G3 (Low-grade)',
    2: 'G4 (Intermediate)',
    3: 'G5 (High-grade)'
}
TCGA_LABEL_TO_4CLASS = {1: 1, 2: 2, 3: 3}


def load_gtex_benign_labels(labels_path: str) -> pd.DataFrame:
    """
    Load GTEx labels and keep only benign/normal (label == 0).
    Supports CSV (GTEx_prostate_labels.csv) and TSV with slide_id/label.
    """
    labels_path = Path(labels_path)
    if labels_path.suffix.lower() == ".csv":
        df = pd.read_csv(labels_path)
        if "label" not in df.columns:
            raise ValueError("GTEx labels CSV must contain a 'label' column.")
        slide_col = "file_name" if "file_name" in df.columns else "slide_id"
        if slide_col not in df.columns:
            raise ValueError("GTEx labels CSV must contain 'file_name' or 'slide_id'.")
        df = df[[slide_col, "label"]].rename(columns={slide_col: "slide_id"})
    else:
        df = pd.read_csv(labels_path, sep="\t")
        if "slide_id" not in df.columns or "label" not in df.columns:
            raise ValueError("GTEx labels TSV must contain 'slide_id' and 'label'.")
        df = df[["slide_id", "label"]]
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    return df[df["label"] == 0]


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


class ABMIL_MultiClass(nn.Module):
    def __init__(self, input_dim=1536, hidden_dim=256, attention_dim=128, num_classes=4, dropout=0.25):
        super().__init__()
        self.feature_projection = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.attention = GatedAttention(hidden_dim, attention_dim, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x):
        h = self.feature_projection(x)
        A = self.attention(h)
        M = torch.bmm(A.unsqueeze(1), h).squeeze(1)
        return self.classifier(M)


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load model
    print("Loading model...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    config = checkpoint.get('config', {'input_dim': 1536, 'hidden_dim': 256, 'attention_dim': 128})
    
    model = ABMIL_MultiClass(
        input_dim=config.get('input_dim', 1536),
        hidden_dim=config.get('hidden_dim', 256),
        attention_dim=config.get('attention_dim', 128),
        num_classes=NUM_CLASSES,
        dropout=0.0
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f"Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Collect slides and labels
    print("\nCollecting slides...")
    slides = []
    
    # TCGA-PRAD
    tcga_df = pd.read_csv(TCGA_LABELS_PATH, sep='\t')
    for _, row in tcga_df.iterrows():
        h5_path = Path(TCGA_FEATS_DIR) / f"{row['slide_id']}.h5"
        if h5_path.exists():
            slides.append({
                'slide_id': row['slide_id'],
                'h5_path': h5_path,
                'true_class': TCGA_LABEL_TO_4CLASS.get(row['label'], row['label']),
                'source': 'TCGA-PRAD'
            })
    print(f"  TCGA-PRAD: {len([s for s in slides if s['source'] == 'TCGA-PRAD'])} slides")
    
    # GTEx
    gtex_df = load_gtex_benign_labels(GTEX_LABELS_PATH)
    for _, row in gtex_df.iterrows():
        h5_path = Path(GTEX_FEATS_DIR) / f"{row['slide_id']}.h5"
        if h5_path.exists():
            slides.append({
                'slide_id': row['slide_id'],
                'h5_path': h5_path,
                'true_class': 0,  # Normal
                'source': 'GTEx'
            })
    print(f"  GTEx: {len([s for s in slides if s['source'] == 'GTEx'])} slides")
    print(f"  Total: {len(slides)} slides")
    
    # Run inference
    print("\nRunning inference...")
    y_true = []
    y_probs = []
    
    with torch.no_grad():
        for slide in tqdm(slides, desc="Inference"):
            with h5py.File(slide['h5_path'], 'r') as f:
                features = torch.from_numpy(f['features'][:]).float().unsqueeze(0).to(device)
            
            logits = model(features)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            
            y_true.append(slide['true_class'])
            y_probs.append(probs)
    
    y_true = np.array(y_true)
    y_probs = np.array(y_probs)
    
    # Plot ROC curves
    print("\nPlotting ROC curves...")
    plot_roc_curves(y_true, y_probs, OUTPUT_DIR)
    print(f"ROC curves saved to: {OUTPUT_DIR}/roc_curves.png")


def plot_roc_curves(y_true, y_probs, output_dir):
    """Plot ROC curves for multi-class classification."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n_classes = y_probs.shape[1]
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
    
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    fpr = {}
    tpr = {}
    roc_auc = {}
    
    present_classes = sorted(set(y_true))
    
    for c in range(n_classes):
        fpr[c], tpr[c], _ = roc_curve(y_true_bin[:, c], y_probs[:, c])
        roc_auc[c] = auc(fpr[c], tpr[c])
        
        linestyle = '-' if c in present_classes else '--'
        alpha = 1.0 if c in present_classes else 0.5
        
        ax.plot(
            fpr[c], tpr[c],
            color=colors[c],
            lw=2,
            linestyle=linestyle,
            alpha=alpha,
            label=f'{CLASS_NAMES[c]} (AUC = {roc_auc[c]:.3f})'
        )
    
    # Micro-average
    y_true_micro = y_true_bin.ravel()
    y_prob_micro = y_probs.ravel()
    fpr_micro, tpr_micro, _ = roc_curve(y_true_micro, y_prob_micro)
    roc_auc_micro = auc(fpr_micro, tpr_micro)
    
    ax.plot(fpr_micro, tpr_micro, color='navy', lw=2, linestyle='--',
            label=f'Micro-average (AUC = {roc_auc_micro:.3f})')
    
    # Macro-average
    all_fpr = np.unique(np.concatenate([fpr[c] for c in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for c in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[c], tpr[c])
    mean_tpr /= n_classes
    roc_auc_macro = auc(all_fpr, mean_tpr)
    
    ax.plot(all_fpr, mean_tpr, color='darkorange', lw=2, linestyle='--',
            label=f'Macro-average (AUC = {roc_auc_macro:.3f})')
    
    # Random classifier
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random (AUC = 0.500)')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves - 4-Class Gleason Classification (Combined Dataset)', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # AUC text box
    auc_text = "Per-class AUC:\n"
    for c in range(n_classes):
        auc_text += f"  {CLASS_NAMES[c]}: {roc_auc[c]:.3f}\n"
    ax.text(0.02, 0.98, auc_text.strip(), transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'roc_curves.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # Print AUC values
    print("\nAUC values:")
    for c in range(n_classes):
        print(f"  {CLASS_NAMES[c]}: {roc_auc[c]:.4f}")
    print(f"  Micro-average: {roc_auc_micro:.4f}")
    print(f"  Macro-average: {roc_auc_macro:.4f}")


if __name__ == "__main__":
    main()
