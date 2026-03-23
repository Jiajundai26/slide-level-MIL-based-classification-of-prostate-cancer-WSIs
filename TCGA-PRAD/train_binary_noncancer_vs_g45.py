#!/usr/bin/env python3
"""
Binary Classification Training Script: Non-Cancer vs Cancer (G4/G5)

This script trains an ABMIL (Attention-Based Multiple Instance Learning) model
for binary classification distinguishing:
    - Class 0 (Non-Cancer): TCGA G3-dominant + GTEx benign tissue (label=0)
    - Class 1 (Cancer): TCGA G4/G5-dominant + GTEx cancer tissue (label=1)

GTEx_Prostate labels: Benign (0), Cancer (1), Discard (-1, filtered out)

Uses pre-extracted UNI-v2 features (1536-dim) from:
    - TCGA-PRAD: /sci-it/projects/cedmav/data/ARPA-H/Prostate_HnE/preprocessing/TCGA-PRAD/
    - GTEx_Prostate: /sci-it/projects/cedmav/data/ARPA-H/Prostate_HnE/preprocessing/GTEx_Prostate/

Architecture:
    - Patch-level foundation model features (UNI-v2)
    - Attention-Based MIL aggregation (ABMIL)
    - Gated attention mechanism
    - Slide-level binary classification

Usage:
    # Train on TCGA-PRAD only (G3 vs G4/G5)
    python train_binary_noncancer_vs_g45.py --train --epochs 50
    
    # Train combining TCGA-PRAD + GTEx (GTEx benign + TCGA G4/G5 as cancer)
    python train_binary_noncancer_vs_g45.py --train --combine_gtex --epochs 50
    
    # With GPU and custom batch size
    python train_binary_noncancer_vs_g45.py --train --device cuda:0 --batch_size 16 --epochs 50
    
    # Custom paths and output
    python train_binary_noncancer_vs_g45.py --train \\
        --feats_dir /path/to/features --labels_path /path/to/labels.tsv \\
        --output_dir results/noncancer_vs_g45
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import h5py
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from collections import defaultdict
import json
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    average_precision_score
)
from tqdm import tqdm

# ============================================================================
# Configuration
# ============================================================================

SEED = 42

# Default paths for TCGA-PRAD
DEFAULT_TCGA_FEATS_DIR = "/sci-it/projects/cedmav/data/ARPA-H/Prostate_HnE/preprocessing/TCGA-PRAD/20x_256px_0px_overlap/features_uni_v2"
DEFAULT_TCGA_LABELS_PATH = "/sci-it/projects/cedmav/data/ARPA-H/Prostate_HnE/preprocessing/TCGA-PRAD/labels_MIL/slide_labels.tsv"

# Default paths for GTEx Prostate (Benign/Cancer/Discard)
DEFAULT_GTEX_FEATS_DIR = "/sci-it/projects/cedmav/data/ARPA-H/Prostate_HnE/preprocessing/GTEx_Prostate/20x_256px_0px_overlap/features_uni_v2"
DEFAULT_GTEX_LABELS_PATH = "/sci-it/projects/cedmav/data/ARPA-H/Prostate_HnE/preprocessing/GTEx_Prostate/GTEx_prostate_labels.csv"

# Label Mappings
# TCGA-PRAD: 1=G3, 2=G4, 3=G5
# For this binary task: G3 -> Non-Cancer (0), G4/G5 -> Cancer (1)
GLEASON_TO_BINARY = {
    1: 0,  # G3-dominant -> Non-Cancer (0)
    2: 1,  # G4-dominant -> Cancer (1)
    3: 1   # G5-dominant -> Cancer (1)
}

# Class names
CLASS_NAMES = {
    0: 'Non-Cancer (G3/Benign)',
    1: 'Cancer (G4/G5)'
}


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================================
# Model Architecture
# ============================================================================

class GatedAttention(nn.Module):
    """Gated Attention mechanism for ABMIL"""
    def __init__(self, input_dim: int = 256, hidden_dim: int = 128, dropout: float = 0.25):
        super().__init__()
        self.attention_a = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout)
        )
        self.attention_b = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Dropout(dropout)
        )
        self.attention_c = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, num_patches, input_dim]
        Returns:
            attention_weights: [batch, num_patches]
        """
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = self.attention_c(a * b).squeeze(-1)
        A = torch.softmax(A, dim=1)
        return A


class ABMIL(nn.Module):
    """
    Attention-Based Multiple Instance Learning (ABMIL) for binary classification.

    Architecture:
    1. Feature projection: foundation model dimension -> hidden dimension
    2. Gated attention: compute attention weights for each patch
    3. Weighted aggregation: slide-level representation from attended patches
    4. Classification head: binary prediction (logit)
    """
    def __init__(
        self,
        input_dim: int = 1536,      # UNI-v2 feature dimension
        hidden_dim: int = 256,       # Projection dimension
        attention_dim: int = 128,    # Attention hidden dimension
        dropout: float = 0.25
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Feature projection layer
        self.feature_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Gated attention mechanism
        self.attention = GatedAttention(
            input_dim=hidden_dim,
            hidden_dim=attention_dim,
            dropout=dropout
        )

        # Binary classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)  # Output single logit for binary classification
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through ABMIL model.

        Args:
            x: Patch features [batch, num_patches, input_dim]
            return_attention: Whether to return attention weights

        Returns:
            logits: [batch, 1] - binary classification logits
            attention: [batch, num_patches] (optional)
        """
        # Handle dict input (from dataloader)
        if isinstance(x, dict):
            x = x['features']

        # Project features: [B, N, input_dim] -> [B, N, hidden_dim]
        h = self.feature_projection(x)

        # Compute attention weights: [B, N]
        A = self.attention(h)

        # Weighted aggregation: [B, 1, N] x [B, N, hidden_dim] -> [B, hidden_dim]
        M = torch.bmm(A.unsqueeze(1), h).squeeze(1)

        # Classification: [B, hidden_dim] -> [B, 1]
        logits = self.classifier(M)

        if return_attention:
            return logits, A
        return logits


# ============================================================================
# Dataset
# ============================================================================

class SlideFeatureDataset(Dataset):
    """Dataset for loading pre-extracted slide features from H5 files."""

    def __init__(
        self,
        feats_dirs: List[Path],
        slide_ids: List[str],
        labels: Dict[str, int],
        slide_to_dir: Optional[Dict[str, str]] = None,
        max_patches: int = 4096,
        subsample: bool = True
    ):
        """
        Args:
            feats_dirs: List of directories containing H5 feature files
            slide_ids: List of slide IDs to include
            labels: Dictionary mapping slide_id to binary label
            slide_to_dir: Optional mapping of slide_id to specific feature directory
            max_patches: Maximum patches per slide
            subsample: Random subsample (train) or truncate (val/test)
        """
        if isinstance(feats_dirs, (str, Path)):
            feats_dirs = [Path(feats_dirs)]
        self.feats_dirs = [Path(d) for d in feats_dirs]
        self.max_patches = max_patches
        self.subsample = subsample
        self.slide_to_dir = slide_to_dir or {}

        # Find available slides
        self.samples = []
        for slide_id in slide_ids:
            h5_path = None

            # Check slide_to_dir mapping first
            if slide_id in self.slide_to_dir:
                feats_dir = Path(self.slide_to_dir[slide_id])
                candidate_path = feats_dir / f"{slide_id}.h5"
                if candidate_path.exists():
                    h5_path = candidate_path
            else:
                # Search all feature directories
                for feats_dir in self.feats_dirs:
                    candidate_path = feats_dir / f"{slide_id}.h5"
                    if candidate_path.exists():
                        h5_path = candidate_path
                        break

            if h5_path:
                self.samples.append({
                    'slide_id': slide_id,
                    'h5_path': h5_path,
                    'label': labels[slide_id]
                })

        # Statistics
        label_counts = defaultdict(int)
        for s in self.samples:
            label_counts[s['label']] += 1

        print(f"  Dataset: {len(self.samples)} slides")
        print(f"    {CLASS_NAMES[0]}: {label_counts[0]}, {CLASS_NAMES[1]}: {label_counts[1]}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]

        with h5py.File(sample['h5_path'], 'r') as f:
            features = torch.from_numpy(f['features'][:]).float()

        num_patches = features.shape[0]

        # Handle patch subsampling
        if num_patches > self.max_patches:
            if self.subsample:
                # Random subsample for training
                indices = torch.randperm(num_patches)[:self.max_patches]
                features = features[indices]
            else:
                # Deterministic truncation for val/test
                features = features[:self.max_patches]

        return {
            'features': features,
            'label': torch.tensor(sample['label'], dtype=torch.float32),
            'slide_id': sample['slide_id']
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """Collate function for variable-length patch sequences."""
    features = [item['features'] for item in batch]
    labels = torch.stack([item['label'] for item in batch])
    slide_ids = [item['slide_id'] for item in batch]

    # Get dimensions
    max_len = max(f.shape[0] for f in features)
    feature_dim = features[0].shape[1]
    batch_size = len(features)

    # Pad sequences
    padded = torch.zeros(batch_size, max_len, feature_dim)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

    for i, f in enumerate(features):
        length = f.shape[0]
        padded[i, :length] = f
        mask[i, :length] = True

    return {
        'features': padded,
        'labels': labels,
        'mask': mask,
        'slide_ids': slide_ids
    }


# ============================================================================
# Label Loading
# ============================================================================

def load_tcga_labels(labels_path: Path) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Load TCGA-PRAD labels and map to binary classification.

    Mapping: G3 -> 0 (Non-Cancer), G4/G5 -> 1 (Cancer)
    """
    df = pd.read_csv(labels_path, sep='\t')
    print(f"Loaded {len(df)} TCGA-PRAD slides")
    print(f"  Original classes: {df['label'].value_counts().to_dict()}")

    labels = {}
    for _, row in df.iterrows():
        slide_id = row['slide_id']
        original_label = row['label']
        binary_label = GLEASON_TO_BINARY[original_label]
        labels[slide_id] = binary_label

    return df, labels


def load_gtex_labels(labels_path: Path) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Load GTEx labels.

    Expected format: CSV or TSV with columns ['file_name' or 'slide_id', 'label']
    where label in {0 (Benign), 1 (Cancer), -1 (Discard)}

    For this task:
        Benign (0) -> Non-Cancer (class 0)
        Cancer (1) -> Cancer (class 1)
        Discard (-1) -> filtered out
    """
    if labels_path.suffix.lower() == ".csv":
        df = pd.read_csv(labels_path)
    else:
        df = pd.read_csv(labels_path, sep='\t')

    # Rename file_name to slide_id if needed
    if "file_name" in df.columns and "slide_id" not in df.columns:
        df = df.rename(columns={"file_name": "slide_id"})

    # Ensure label is numeric
    df['label'] = pd.to_numeric(df['label'], errors='coerce')

    # Filter out discard (-1) and keep only 0 (benign) and 1 (cancer)
    df = df[df['label'].isin([0, 1])].reset_index(drop=True)
    df['label'] = df['label'].astype(int)

    n_benign = len(df[df['label'] == 0])
    n_cancer = len(df[df['label'] == 1])
    print(f"Loaded {len(df)} GTEx slides (after filtering {n_benign + n_cancer} kept, discards removed)")
    print(f"  Benign (0) -> Non-Cancer: {n_benign}")
    print(f"  Cancer (1) -> Cancer:     {n_cancer}")

    # GTEx benign (0) -> Non-Cancer, GTEx cancer (1) -> Cancer
    labels = {row['slide_id']: int(row['label']) for _, row in df.iterrows()}

    return df, labels


# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Tuple[float, float, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        features = batch['features'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        logits = model(features).squeeze(-1)
        loss = criterion(logits, labels)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        preds = (torch.sigmoid(logits) > 0.5).float()
        all_preds.extend(preds.cpu().detach().numpy())
        all_labels.extend(labels.cpu().detach().numpy())

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)

    return avg_loss, accuracy, balanced_acc


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict:
    """Evaluate the model on validation/test set."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    all_slide_ids = []

    for batch in tqdm(loader, desc="Evaluating"):
        features = batch['features'].to(device)
        labels = batch['labels'].to(device)

        logits = model(features).squeeze(-1)
        loss = criterion(logits, labels)

        total_loss += loss.item()
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_slide_ids.extend(batch['slide_ids'])

    # Convert to numpy
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # Compute metrics
    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0

    try:
        ap = average_precision_score(all_labels, all_probs)
    except ValueError:
        ap = 0.0

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )

    conf_matrix = confusion_matrix(all_labels, all_preds)

    # Sensitivity (recall) and Specificity
    if conf_matrix.shape == (2, 2):
        tn, fp, fn, tp = conf_matrix.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        sensitivity = specificity = 0.0

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'auc': auc,
        'average_precision': ap,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'confusion_matrix': conf_matrix.tolist(),
        'predictions': {
            'slide_ids': all_slide_ids,
            'labels': all_labels.tolist(),
            'predictions': all_preds.tolist(),
            'probabilities': all_probs.tolist()
        }
    }


# ============================================================================
# Main Training Function
# ============================================================================

def train_model(
    feats_dirs: List[Path],
    output_dir: Path,
    labels: Dict[str, int],
    labels_df: Optional[pd.DataFrame] = None,
    slide_to_dir: Optional[Dict[str, str]] = None,
    epochs: int = 50,
    batch_size: int = 8,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    hidden_dim: int = 256,
    attention_dim: int = 128,
    dropout: float = 0.25,
    max_patches: int = 4096,
    device: str = 'cuda:0',
    num_workers: int = 4
):
    """Train ABMIL model for binary classification."""
    set_seed(SEED)

    # Handle feature directories
    if isinstance(feats_dirs, (str, Path)):
        feats_dirs = [Path(feats_dirs)]
    else:
        feats_dirs = [Path(d) for d in feats_dirs]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device setup
    if device.startswith('cuda') and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    device = torch.device(device)
    print(f"Device: {device}\n")

    # Get feature dimension
    sample_h5 = None
    for feats_dir in feats_dirs:
        h5_files = list(feats_dir.glob('*.h5'))
        if h5_files:
            sample_h5 = h5_files[0]
            break

    if sample_h5 is None:
        raise ValueError("No H5 files found in any feature directory")

    with h5py.File(sample_h5, 'r') as f:
        input_dim = f['features'].shape[1]
    print(f"Feature dimension: {input_dim}")
    print(f"Feature directories: {[str(d) for d in feats_dirs]}\n")

    # Prepare train/val/test splits
    if labels_df is not None and 'fold_0' in labels_df.columns:
        print("Using existing train/val/test splits from labels file")
        train_ids = labels_df[labels_df['fold_0'] == 'Training']['slide_id'].tolist()
        val_ids = labels_df[labels_df['fold_0'] == 'Validation']['slide_id'].tolist()
        test_ids = labels_df[labels_df['fold_0'] == 'Testing']['slide_id'].tolist()
    else:
        # Create random splits if not provided
        all_ids = list(labels.keys())
        np.random.shuffle(all_ids)
        n_total = len(all_ids)
        train_end = int(0.7 * n_total)
        val_end = int(0.85 * n_total)

        train_ids = all_ids[:train_end]
        val_ids = all_ids[train_end:val_end]
        test_ids = all_ids[val_end:]

    # Print split statistics
    train_lbls = [labels[s] for s in train_ids]
    val_lbls = [labels[s] for s in val_ids]
    test_lbls = [labels[s] for s in test_ids]

    print(f"Data split:")
    print(f"  Train: {len(train_ids)} ({CLASS_NAMES[0]}: {len(train_lbls) - sum(train_lbls)}, {CLASS_NAMES[1]}: {sum(train_lbls)})")
    print(f"  Val:   {len(val_ids)} ({CLASS_NAMES[0]}: {len(val_lbls) - sum(val_lbls)}, {CLASS_NAMES[1]}: {sum(val_lbls)})")
    print(f"  Test:  {len(test_ids)} ({CLASS_NAMES[0]}: {len(test_lbls) - sum(test_lbls)}, {CLASS_NAMES[1]}: {sum(test_lbls)})\n")

    # Create datasets
    print("Creating datasets...")
    train_dataset = SlideFeatureDataset(
        feats_dirs, train_ids, labels, slide_to_dir, max_patches, subsample=True
    )
    val_dataset = SlideFeatureDataset(
        feats_dirs, val_ids, labels, slide_to_dir, max_patches, subsample=False
    )
    test_dataset = SlideFeatureDataset(
        feats_dirs, test_ids, labels, slide_to_dir, max_patches, subsample=False
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_fn,
        pin_memory=(device.type == 'cuda')
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn
    )

    # Initialize model
    model = ABMIL(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        attention_dim=attention_dim,
        dropout=dropout
    ).to(device)

    print(f"Model: ABMIL (Binary Classification)")
    print(f"  Input dim: {input_dim}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Attention dim: {attention_dim}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    # Class weights for imbalanced data
    n_pos = sum(train_lbls)
    n_neg = len(train_lbls) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)]).to(device)
    print(f"Class balance: {CLASS_NAMES[0]}={n_neg}, {CLASS_NAMES[1]}={n_pos}")
    print(f"Positive weight: {pos_weight.item():.2f}\n")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01
    )

    # Training loop
    best_val_auc = 0.0
    best_epoch = 0
    history = {'train': [], 'val': []}

    print("=" * 80)
    print("ABMIL Binary Classification Training")
    print("=" * 80 + "\n")

    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")
        print("-" * 40)

        # Train
        train_loss, train_acc, train_bal_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_results = evaluate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Log
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | Balanced Acc: {train_bal_acc:.4f}")
        print(f"Val Loss: {val_results['loss']:.4f} | Acc: {val_results['accuracy']:.4f} | "
              f"Balanced Acc: {val_results['balanced_accuracy']:.4f} | AUC: {val_results['auc']:.4f}")
        print(f"LR: {current_lr:.6f}\n")

        history['train'].append({
            'epoch': epoch,
            'loss': train_loss,
            'accuracy': train_acc,
            'balanced_accuracy': train_bal_acc
        })
        history['val'].append({
            'epoch': epoch,
            'loss': val_results['loss'],
            'accuracy': val_results['accuracy'],
            'balanced_accuracy': val_results['balanced_accuracy'],
            'auc': val_results['auc']
        })

        # Save best model
        if val_results['auc'] > best_val_auc:
            best_val_auc = val_results['auc']
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': best_val_auc,
            }, output_dir / 'best_model.pt')
            print(f"  -> New best model saved! (AUC: {best_val_auc:.4f})\n")

        # Save latest checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, output_dir / 'latest_checkpoint.pt')

    # Save training history
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    # Final evaluation on test set
    print("=" * 80)
    print(f"Loading best model from epoch {best_epoch}")
    print("=" * 80 + "\n")

    checkpoint = torch.load(output_dir / 'best_model.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_results = evaluate(model, test_loader, criterion, device)

    print("=" * 80)
    print("TEST RESULTS - Non-Cancer vs Cancer (G4/G5)")
    print("=" * 80)
    print(f"Accuracy:          {test_results['accuracy']:.4f}")
    print(f"Balanced Accuracy: {test_results['balanced_accuracy']:.4f}")
    print(f"AUC-ROC:           {test_results['auc']:.4f}")
    print(f"Average Precision: {test_results['average_precision']:.4f}")
    print(f"Precision:         {test_results['precision']:.4f}")
    print(f"Recall/Sensitivity:{test_results['sensitivity']:.4f} (proportion of true cancer detected)")
    print(f"Specificity:       {test_results['specificity']:.4f} (proportion of true non-cancer detected)")
    print(f"F1 Score:          {test_results['f1']:.4f}\n")

    # Confusion matrix
    cm = test_results['confusion_matrix']
    print(f"Confusion Matrix:")
    print(f"                   Predicted Non-Cancer  Predicted Cancer")
    print(f"Actual Non-Cancer:  {cm[0][0]:>20}  {cm[0][1]:>16}")
    print(f"Actual Cancer:      {cm[1][0]:>20}  {cm[1][1]:>16}\n")

    # Save results
    test_results_save = {
        'metrics': {k: v for k, v in test_results.items() if k != 'predictions'},
        'test_results': test_results
    }
    with open(output_dir / 'test_results.json', 'w') as f:
        json.dump(test_results_save, f, indent=2, default=str)

    # Save predictions
    pred_df = pd.DataFrame({
        'slide_id': test_results['predictions']['slide_ids'],
        'true_label': test_results['predictions']['labels'],
        'predicted': test_results['predictions']['predictions'],
        'probability': test_results['predictions']['probabilities'],
        'true_class': [CLASS_NAMES[int(l)] for l in test_results['predictions']['labels']],
        'predicted_class': [CLASS_NAMES[int(p)] for p in test_results['predictions']['predictions']]
    })
    pred_df.to_csv(output_dir / 'test_predictions.csv', index=False)

    print(f"Results saved to: {output_dir}")
    print("Done!\n")

    return test_results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Binary Classification Training: Non-Cancer vs Cancer (G4/G5)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train on TCGA-PRAD only (G3 vs G4/G5)
    python train_binary_noncancer_vs_g45.py --train --epochs 50
    
    # Train combining TCGA-PRAD + GTEx (GTEx benign->non-cancer, GTEx cancer->cancer)
    python train_binary_noncancer_vs_g45.py --train --combine_gtex --epochs 50
    
    # With GPU and custom parameters
    python train_binary_noncancer_vs_g45.py --train --device cuda:0 \\
        --batch_size 16 --lr 2e-4 --hidden_dim 512 --epochs 100
    
    # Custom paths
    python train_binary_noncancer_vs_g45.py --train \\
        --feats_dir /path/to/tcga/features --labels_path /path/to/tcga/labels.tsv \\
        --output_dir results/binary_classifier
        """
    )

    # Actions
    parser.add_argument('--train', action='store_true', help='Train model')

    # Dataset options
    parser.add_argument('--combine_gtex', action='store_true',
                       help='Combine TCGA-PRAD with GTEx (benign->non-cancer, cancer->cancer)')

    # Paths
    parser.add_argument('--feats_dir', type=str, default=DEFAULT_TCGA_FEATS_DIR,
                       help='Feature directory (TCGA-PRAD)')
    parser.add_argument('--labels_path', type=str, default=DEFAULT_TCGA_LABELS_PATH,
                       help='Labels file (TCGA-PRAD)')
    parser.add_argument('--gtex_feats_dir', type=str, default=DEFAULT_GTEX_FEATS_DIR,
                       help='GTEx feature directory')
    parser.add_argument('--gtex_labels_path', type=str, default=DEFAULT_GTEX_LABELS_PATH,
                       help='GTEx labels file')
    parser.add_argument('--output_dir', type=str, default='output/binary_noncancer_vs_g45',
                       help='Output directory')

    # Model parameters
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='Hidden dimension')
    parser.add_argument('--attention_dim', type=int, default=128,
                       help='Attention dimension')
    parser.add_argument('--dropout', type=float, default=0.25,
                       help='Dropout rate')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--max_patches', type=int, default=4096,
                       help='Maximum patches per slide')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of dataloader workers')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device (cuda:0, cuda:1, cpu, etc.)')

    args = parser.parse_args()

    if not args.train:
        parser.print_help()
        return

    # Load TCGA-PRAD labels
    print("Loading TCGA-PRAD labels...")
    tcga_df, tcga_labels = load_tcga_labels(Path(args.labels_path))

    # Prepare labels and feature directories
    all_labels = tcga_labels.copy()
    feats_dirs = [Path(args.feats_dir)]
    slide_to_dir = {}

    # Add GTEx if requested
    if args.combine_gtex:
        print("\nLoading GTEx labels...")
        gtex_df, gtex_labels = load_gtex_labels(Path(args.gtex_labels_path))
        all_labels.update(gtex_labels)
        feats_dirs.append(Path(args.gtex_feats_dir))

        # Create mapping for GTEx slides
        for slide_id in gtex_labels.keys():
            slide_to_dir[slide_id] = str(Path(args.gtex_feats_dir))

        # Summarize combined dataset
        tcga_noncancer = sum(1 for v in tcga_labels.values() if v == 0)
        tcga_cancer = sum(1 for v in tcga_labels.values() if v == 1)
        gtex_noncancer = sum(1 for v in gtex_labels.values() if v == 0)
        gtex_cancer = sum(1 for v in gtex_labels.values() if v == 1)

        print(f"\nCombined dataset:")
        print(f"  TCGA-PRAD: {len(tcga_labels)} slides (Non-Cancer/G3: {tcga_noncancer}, Cancer/G4+G5: {tcga_cancer})")
        print(f"  GTEx:      {len(gtex_labels)} slides (Benign: {gtex_noncancer}, Cancer: {gtex_cancer})")
        print(f"  Total:     {len(all_labels)} slides (Non-Cancer: {tcga_noncancer + gtex_noncancer}, Cancer: {tcga_cancer + gtex_cancer})")

        # Use combined labels dataframe for splits (random 70/15/15)
        labels_df = None
    else:
        print(f"\nUsing TCGA-PRAD only: {len(tcga_labels)} slides")
        labels_df = tcga_df

    # Train model
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80 + "\n")

    train_model(
        feats_dirs=feats_dirs,
        output_dir=args.output_dir,
        labels=all_labels,
        labels_df=labels_df,
        slide_to_dir=slide_to_dir if slide_to_dir else None,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        attention_dim=args.attention_dim,
        dropout=args.dropout,
        max_patches=args.max_patches,
        device=args.device,
        num_workers=args.num_workers
    )


if __name__ == "__main__":
    main()
