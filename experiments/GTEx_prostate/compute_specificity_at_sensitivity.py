#!/usr/bin/env python3
"""
Compute specificity at 98% sensitivity for 4-class Gleason classification.
Uses One-vs-Rest approach for multi-class ROC analysis.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_curve
from sklearn.preprocessing import label_binarize

# Configuration
CHECKPOINT_PATH = "/local/data/magicscan/HnE/GTEx_prostate/output/trident_4class/4class_focal_20260123_001634/best_model.pt"
TCGA_FEATS_DIR = "/sci-it/projects/cedmav/data/ARPA-H/Prostate_HnE/preprocessing/TCGA-PRAD/20x_256px_0px_overlap/features_uni_v2"
TCGA_LABELS_PATH = "/sci-it/projects/cedmav/data/ARPA-H/Prostate_HnE/preprocessing/TCGA-PRAD/labels_MIL/slide_labels.tsv"
GTEX_FEATS_DIR = "/sci-it/projects/cedmav/data/ARPA-H/Prostate_HnE/preprocessing/GTEx_Prostate/20x_256px_0px_overlap/features_uni_v2"
GTEX_LABELS_PATH = "/sci-it/projects/cedmav/data/ARPA-H/Prostate_HnE/preprocessing/GTEx_Prostate/GTEx_prostate_labels.csv"

NUM_CLASSES = 4
CLASS_NAMES = {
    0: 'Normal',
    1: 'G3 (Low-grade)',
    2: 'G4 (Intermediate)',
    3: 'G5 (High-grade)'
}
TCGA_LABEL_TO_4CLASS = {1: 1, 2: 2, 3: 3}

TARGET_SENSITIVITY = 0.98


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


def find_specificity_at_sensitivity(fpr, tpr, target_sensitivity=0.98):
    """
    Find specificity at a given sensitivity (TPR) level.
    
    Specificity = 1 - FPR
    Sensitivity = TPR
    """
    # Find the index where TPR is closest to target_sensitivity (but >= target)
    valid_indices = np.where(tpr >= target_sensitivity)[0]
    
    if len(valid_indices) == 0:
        # If we can't reach target sensitivity, return the max sensitivity point
        idx = np.argmax(tpr)
        actual_sensitivity = tpr[idx]
        specificity = 1 - fpr[idx]
        return specificity, actual_sensitivity, "max available"
    else:
        # Find the point with minimum FPR among those meeting the sensitivity threshold
        idx = valid_indices[np.argmin(fpr[valid_indices])]
        specificity = 1 - fpr[idx]
        actual_sensitivity = tpr[idx]
        return specificity, actual_sensitivity, "at threshold"


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
    
    # GTEx
    gtex_df = load_gtex_benign_labels(GTEX_LABELS_PATH)
    for _, row in gtex_df.iterrows():
        h5_path = Path(GTEX_FEATS_DIR) / f"{row['slide_id']}.h5"
        if h5_path.exists():
            slides.append({
                'slide_id': row['slide_id'],
                'h5_path': h5_path,
                'true_class': 0,
                'source': 'GTEx'
            })
    
    print(f"  TCGA-PRAD: {len([s for s in slides if s['source'] == 'TCGA-PRAD'])} slides")
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
    
    # Binarize labels for OvR
    y_true_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))
    
    print("\n" + "=" * 70)
    print(f"SPECIFICITY @ {int(TARGET_SENSITIVITY*100)}% SENSITIVITY (One-vs-Rest)")
    print("=" * 70)
    
    results = {}
    
    for c in range(NUM_CLASSES):
        fpr, tpr, thresholds = roc_curve(y_true_bin[:, c], y_probs[:, c])
        specificity, actual_sens, note = find_specificity_at_sensitivity(fpr, tpr, TARGET_SENSITIVITY)
        
        # Find the threshold at this operating point
        valid_indices = np.where(tpr >= TARGET_SENSITIVITY)[0]
        if len(valid_indices) > 0:
            idx = valid_indices[np.argmin(fpr[valid_indices])]
            threshold = thresholds[idx]
        else:
            idx = np.argmax(tpr)
            threshold = thresholds[idx]
        
        results[c] = {
            'class_name': CLASS_NAMES[c],
            'specificity': specificity,
            'actual_sensitivity': actual_sens,
            'threshold': threshold,
            'note': note
        }
        
        print(f"\n{CLASS_NAMES[c]}:")
        print(f"  Specificity @ {int(TARGET_SENSITIVITY*100)}% Sensitivity: {specificity:.4f} ({specificity*100:.2f}%)")
        print(f"  Actual Sensitivity: {actual_sens:.4f} ({actual_sens*100:.2f}%)")
        print(f"  Threshold: {threshold:.4f}")
    
    # Also compute for Cancer vs Non-Cancer (binary)
    print("\n" + "=" * 70)
    print("CANCER VS NON-CANCER (Binary: Normal vs G3+G4+G5)")
    print("=" * 70)
    
    # Cancer = G3, G4, G5 (classes 1, 2, 3)
    y_true_binary = (y_true > 0).astype(int)  # 0 = Normal, 1 = Cancer
    y_prob_cancer = y_probs[:, 1] + y_probs[:, 2] + y_probs[:, 3]  # Sum of cancer probabilities
    
    fpr, tpr, thresholds = roc_curve(y_true_binary, y_prob_cancer)
    specificity, actual_sens, note = find_specificity_at_sensitivity(fpr, tpr, TARGET_SENSITIVITY)
    
    valid_indices = np.where(tpr >= TARGET_SENSITIVITY)[0]
    if len(valid_indices) > 0:
        idx = valid_indices[np.argmin(fpr[valid_indices])]
        threshold = thresholds[idx]
    else:
        idx = np.argmax(tpr)
        threshold = thresholds[idx]
    
    print(f"\nCancer Detection (Normal vs Cancer):")
    print(f"  Specificity @ {int(TARGET_SENSITIVITY*100)}% Sensitivity: {specificity:.4f} ({specificity*100:.2f}%)")
    print(f"  Actual Sensitivity: {actual_sens:.4f} ({actual_sens*100:.2f}%)")
    print(f"  Threshold: {threshold:.4f}")
    
    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"\n{'Class':<25} {'Specificity':>15} {'Sensitivity':>15}")
    print("-" * 55)
    for c in range(NUM_CLASSES):
        r = results[c]
        print(f"{r['class_name']:<25} {r['specificity']*100:>14.2f}% {r['actual_sensitivity']*100:>14.2f}%")
    print("-" * 55)
    print(f"{'Cancer vs Normal':<25} {specificity*100:>14.2f}% {actual_sens*100:>14.2f}%")


if __name__ == "__main__":
    main()
