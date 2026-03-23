#!/usr/bin/env python3
"""
TRIDENT-based 4-Class Gleason Classification Training Script

This script implements ABMIL (Attention-Based Multiple Instance Learning) training
following the TRIDENT framework methodology from MahmoodLab:
    https://github.com/mahmoodlab/TRIDENT

4-Class Classification (Prostate Grading):
    - Class 0: Normal (GTEx prostate tissue - non-cancer)
    - Class 1: G3 (Gleason Pattern 3 dominant - low-grade cancer)
    - Class 2: G4 (Gleason Pattern 4 dominant - intermediate-grade cancer)
    - Class 3: G5 (Gleason Pattern 5 dominant - high-grade cancer)

Combines TCGA-PRAD (cancer) and GTEx_Prostate (normal) datasets.

Uses pre-extracted UNI-v2 features (1536-dim) from:
    - TCGA-PRAD: /sci-it/projects/cedmav/data/ARPA-H/Prostate_HnE/preprocessing/TCGA-PRAD/
    - GTEx: /sci-it/projects/cedmav/data/ARPA-H/Prostate_HnE/preprocessing/GTEx_Prostate/

TRIDENT Framework Components Used:
    - Patch-level foundation model features (UNI-v2)
    - Attention-Based MIL aggregation (ABMIL)
    - Gated attention mechanism
    - Slide-level multi-class classification

Usage:
    # Train 4-class classifier
    python train_4class_classification_trident.py --train --epochs 100
    
    # Train with GPU
    python train_4class_classification_trident.py --train --device cuda:0 --epochs 100
    
    # Train with class balancing strategy
    python train_4class_classification_trident.py --train --balance_strategy weighted --epochs 100
    
    # Train with oversampling
    python train_4class_classification_trident.py --train --balance_strategy oversample --epochs 100
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import h5py
import pandas as pd
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, 
    accuracy_score, 
    balanced_accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    f1_score
)
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# Configuration
# ============================================================================

SEED = 42
NUM_CLASSES = 4

# Default paths for pre-extracted features
TCGA_FEATS_DIR = "/sci-it/projects/cedmav/data/ARPA-H/Prostate_HnE/preprocessing/TCGA-PRAD/20x_256px_0px_overlap/features_uni_v2"
TCGA_LABELS_PATH = "/sci-it/projects/cedmav/data/ARPA-H/Prostate_HnE/preprocessing/TCGA-PRAD/labels_MIL/slide_labels.tsv"

# GTEx Prostate paths (Non-Cancer normal tissue)
GTEX_FEATS_DIR = "/sci-it/projects/cedmav/data/ARPA-H/Prostate_HnE/preprocessing/GTEx_Prostate/20x_256px_0px_overlap/features_uni_v2"
GTEX_LABELS_PATH = "/sci-it/projects/cedmav/data/ARPA-H/Prostate_HnE/preprocessing/GTEx_Prostate/labels_MIL/slide_labels.tsv"

# Class names and mapping
CLASS_NAMES = {
    0: 'Normal',
    1: 'G3 (Low-grade)',
    2: 'G4 (Intermediate)',
    3: 'G5 (High-grade)'
}

# TCGA label mapping: TCGA uses 1=G3, 2=G4, 3=G5
# We remap to: 0=Normal, 1=G3, 2=G4, 3=G5
TCGA_LABEL_REMAP = {1: 1, 2: 2, 3: 3}  # G3->1, G4->2, G5->3

# Feature dimensions for different encoders
ENCODER_DIMS = {
    'uni_v2': 1536,
    'uni2_h': 1536,
    'uni': 1024,
    'conch_v1': 512,
    'conch_v15': 512,
    'phikon': 768,
    'phikon_v2': 1024,
    'resnet50': 2048,
    'dinov2_vitb14': 768,
    'dinov2_vitl14': 1024,
    'virchow2': 1280,
}


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================================
# TRIDENT-style ABMIL Model Architecture (Multi-class)
# ============================================================================

class GatedAttention(nn.Module):
    """
    Gated Attention mechanism for MIL (TRIDENT/CLAM style)
    
    Combines tanh and sigmoid gates for attention computation.
    """
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


class ABMIL_MultiClass(nn.Module):
    """
    Attention-Based Multiple Instance Learning (ABMIL) Classifier
    for Multi-class Classification
    
    Following TRIDENT/CLAM architecture:
    1. Feature projection from encoder dimension to hidden dimension
    2. Gated attention mechanism for instance weighting
    3. Weighted aggregation to slide-level representation
    4. Classification head for multi-class prediction
    
    Reference:
        - CLAM: https://github.com/mahmoodlab/CLAM
        - TRIDENT: https://github.com/mahmoodlab/TRIDENT
    """
    def __init__(
        self,
        input_dim: int = 1536,      # UNI-v2 feature dimension
        hidden_dim: int = 256,       # Projection dimension
        attention_dim: int = 128,    # Attention hidden dimension
        num_classes: int = 4,        # Number of classes
        dropout: float = 0.25
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
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
        
        # Classification head for multi-class
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
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
            logits: [batch, num_classes]
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
        
        # Classification: [B, hidden_dim] -> [B, num_classes]
        logits = self.classifier(M)
        
        if return_attention:
            return logits, A
        return logits


# ============================================================================
# Dataset
# ============================================================================

class SlideFeatureDataset4Class(Dataset):
    """
    Dataset for loading pre-extracted slide features from H5 files.
    
    Follows TRIDENT conventions for feature loading and preprocessing.
    Supports multiple feature directories for multi-dataset training.
    """
    def __init__(
        self,
        slide_ids: List[str],
        labels: Dict[str, int],
        slide_to_dir: Dict[str, str],
        max_patches: int = 4096,
        subsample: bool = True,
        num_classes: int = 4
    ):
        """
        Args:
            slide_ids: List of slide IDs to include
            labels: Dictionary mapping slide_id to label (0-3)
            slide_to_dir: Mapping of slide_id to feature directory
            max_patches: Maximum patches per slide
            subsample: Random subsample (train) or truncate (val/test)
            num_classes: Number of classes
        """
        self.max_patches = max_patches
        self.subsample = subsample
        self.num_classes = num_classes
        self.slide_to_dir = slide_to_dir
        
        # Filter to existing slides
        self.samples = []
        for slide_id in slide_ids:
            if slide_id not in labels:
                continue
            if slide_id not in slide_to_dir:
                continue
            
            h5_path = Path(slide_to_dir[slide_id]) / f"{slide_id}.h5"
            if h5_path.exists():
                self.samples.append({
                    'slide_id': slide_id,
                    'label': labels[slide_id],
                    'h5_path': h5_path
                })
        
        # Statistics
        label_counts = defaultdict(int)
        for s in self.samples:
            label_counts[s['label']] += 1
        
        print(f"  Dataset: {len(self.samples)} slides")
        for c in range(num_classes):
            print(f"    Class {c} ({CLASS_NAMES[c]}): {label_counts[c]}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        
        with h5py.File(sample['h5_path'], 'r') as f:
            features = torch.from_numpy(f['features'][:]).float()
        
        num_patches = len(features)
        
        # Handle patch subsampling
        if num_patches > self.max_patches:
            if self.subsample:
                # Random subsample for training
                indices = torch.randperm(num_patches)[:self.max_patches]
                features = features[indices]
            else:
                # Deterministic truncation for validation/test
                features = features[:self.max_patches]
        
        return {
            'features': features,
            'label': torch.tensor(sample['label'], dtype=torch.long),
            'slide_id': sample['slide_id']
        }
    
    def get_class_weights(self) -> torch.Tensor:
        """Compute class weights inversely proportional to class frequencies"""
        label_counts = defaultdict(int)
        for s in self.samples:
            label_counts[s['label']] += 1
        
        total = len(self.samples)
        weights = []
        for c in range(self.num_classes):
            count = label_counts[c] if label_counts[c] > 0 else 1
            weights.append(total / (self.num_classes * count))
        
        return torch.tensor(weights, dtype=torch.float32)
    
    def get_sample_weights(self) -> List[float]:
        """Get per-sample weights for WeightedRandomSampler"""
        label_counts = defaultdict(int)
        for s in self.samples:
            label_counts[s['label']] += 1
        
        total = len(self.samples)
        class_weights = {}
        for c in range(self.num_classes):
            count = label_counts[c] if label_counts[c] > 0 else 1
            class_weights[c] = total / count
        
        return [class_weights[s['label']] for s in self.samples]


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Collate function for variable-length patch sequences.
    
    Pads all sequences to the maximum length in the batch.
    """
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
# Focal Loss for Imbalanced Classification
# ============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced multi-class classification.
    
    Reduces loss contribution from easy examples and focuses on hard examples.
    Particularly useful for our severely imbalanced 4-class problem.
    
    Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
    """
    def __init__(
        self, 
        alpha: Optional[torch.Tensor] = None, 
        gamma: float = 2.0, 
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha  # Class weights
        self.gamma = gamma  # Focusing parameter
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [batch, num_classes]
            targets: [batch] (class indices)
        """
        ce_loss = F.cross_entropy(logits, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_classes: int = 4
) -> Tuple[float, float, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        features = batch['features'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
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
    device: torch.device,
    num_classes: int = 4
) -> Dict:
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    all_slide_ids = []
    
    for batch in tqdm(loader, desc="Evaluating"):
        features = batch['features'].to(device)
        labels = batch['labels'].to(device)
        
        logits = model(features)
        loss = criterion(logits, labels)
        
        total_loss += loss.item()
        probs = F.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)
        
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
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, labels=list(range(num_classes)), zero_division=0
    )
    
    # Macro-averaged metrics
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    
    # Weighted F1
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    # Multi-class AUC (One-vs-Rest)
    try:
        # Only compute if we have all classes represented
        if len(np.unique(all_labels)) == num_classes:
            auc_ovr = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
            auc_per_class = roc_auc_score(all_labels, all_probs, multi_class='ovr', average=None)
        else:
            auc_ovr = 0.0
            auc_per_class = [0.0] * num_classes
    except ValueError:
        auc_ovr = 0.0
        auc_per_class = [0.0] * num_classes
    
    conf_matrix = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    
    # Per-class accuracy
    per_class_acc = conf_matrix.diagonal() / (conf_matrix.sum(axis=1) + 1e-8)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'auc_macro': auc_ovr,
        'per_class': {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'f1': f1.tolist(),
            'support': support.tolist(),
            'accuracy': per_class_acc.tolist(),
            'auc': list(auc_per_class) if isinstance(auc_per_class, np.ndarray) else auc_per_class
        },
        'confusion_matrix': conf_matrix.tolist(),
        'predictions': {
            'slide_ids': all_slide_ids,
            'labels': all_labels.tolist(),
            'predictions': all_preds.tolist(),
            'probabilities': all_probs.tolist()
        }
    }


def plot_confusion_matrix(
    conf_matrix: np.ndarray,
    output_path: Path,
    title: str = "Confusion Matrix"
):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(10, 8))
    
    # Normalize
    conf_matrix_norm = conf_matrix.astype('float') / (conf_matrix.sum(axis=1, keepdims=True) + 1e-8)
    
    # Create labels
    class_labels = [CLASS_NAMES[i] for i in range(len(conf_matrix))]
    
    # Plot
    sns.heatmap(
        conf_matrix_norm, 
        annot=True, 
        fmt='.2%', 
        cmap='Blues',
        xticklabels=class_labels,
        yticklabels=class_labels,
        square=True,
        cbar_kws={'label': 'Proportion'}
    )
    
    # Add raw counts
    for i in range(len(conf_matrix)):
        for j in range(len(conf_matrix)):
            plt.text(
                j + 0.5, i + 0.7, f'({conf_matrix[i, j]})',
                ha='center', va='center', fontsize=8, color='gray'
            )
    
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_training_history(history: Dict, output_path: Path):
    """Plot training history curves."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    epochs = [h['epoch'] for h in history['train']]
    
    # Loss
    ax = axes[0, 0]
    ax.plot(epochs, [h['loss'] for h in history['train']], 'b-', label='Train')
    ax.plot(epochs, [h['loss'] for h in history['val']], 'r-', label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Accuracy
    ax = axes[0, 1]
    ax.plot(epochs, [h['accuracy'] for h in history['train']], 'b-', label='Train')
    ax.plot(epochs, [h['accuracy'] for h in history['val']], 'r-', label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Training and Validation Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Balanced Accuracy
    ax = axes[1, 0]
    ax.plot(epochs, [h['balanced_accuracy'] for h in history['train']], 'b-', label='Train')
    ax.plot(epochs, [h['balanced_accuracy'] for h in history['val']], 'r-', label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Balanced Accuracy')
    ax.set_title('Training and Validation Balanced Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Macro F1
    ax = axes[1, 1]
    ax.plot(epochs, [h.get('macro_f1', 0) for h in history['val']], 'r-', label='Val Macro F1')
    ax.plot(epochs, [h.get('weighted_f1', 0) for h in history['val']], 'g-', label='Val Weighted F1')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('F1 Score')
    ax.set_title('Validation F1 Scores')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# Main Training Function
# ============================================================================

def train_4class(
    output_dir: Path,
    labels: Dict[str, int],
    slide_to_dir: Dict[str, str],
    epochs: int = 100,
    batch_size: int = 8,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    hidden_dim: int = 256,
    attention_dim: int = 128,
    dropout: float = 0.25,
    max_patches: int = 4096,
    device: str = 'cuda:0',
    num_workers: int = 4,
    balance_strategy: str = 'weighted',  # 'weighted', 'oversample', 'focal', 'none'
    focal_gamma: float = 2.0,
    val_split: float = 0.15,
    test_split: float = 0.15
):
    """
    Train 4-class ABMIL model using TRIDENT methodology.
    
    Args:
        output_dir: Directory to save outputs
        labels: Dict mapping slide_id to label (0-3)
        slide_to_dir: Dict mapping slide_id to feature directory
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        weight_decay: Weight decay for optimizer
        hidden_dim: Hidden dimension for ABMIL
        attention_dim: Attention dimension
        dropout: Dropout rate
        max_patches: Maximum patches per slide
        device: Device to train on
        num_workers: Number of data loading workers
        balance_strategy: Strategy to handle class imbalance
        focal_gamma: Gamma parameter for focal loss
        val_split: Validation split ratio
        test_split: Test split ratio
    """
    set_seed(SEED)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Device setup
    if device.startswith('cuda') and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    device = torch.device(device)
    print(f"Device: {device}")
    
    # Get feature dimension from first available H5 file
    sample_slide = list(slide_to_dir.keys())[0]
    sample_h5 = Path(slide_to_dir[sample_slide]) / f"{sample_slide}.h5"
    
    with h5py.File(sample_h5, 'r') as f:
        input_dim = f['features'].shape[1]
    print(f"Feature dimension: {input_dim}")
    
    # Create stratified train/val/test splits
    all_ids = list(labels.keys())
    all_lbls = [labels[s] for s in all_ids]
    
    # First split: train+val vs test
    train_val_ids, test_ids, train_val_lbls, test_lbls = train_test_split(
        all_ids, all_lbls, 
        test_size=test_split, 
        stratify=all_lbls, 
        random_state=SEED
    )
    
    # Second split: train vs val
    val_ratio = val_split / (1 - test_split)
    train_ids, val_ids, train_lbls, val_lbls = train_test_split(
        train_val_ids, train_val_lbls,
        test_size=val_ratio,
        stratify=train_val_lbls,
        random_state=SEED
    )
    
    # Convert lists back to label dict for the split
    train_labels = {s: labels[s] for s in train_ids}
    val_labels = {s: labels[s] for s in val_ids}
    test_labels = {s: labels[s] for s in test_ids}
    
    # Print split statistics
    print(f"\nData split (stratified):")
    for split_name, split_labels in [('Train', train_labels), ('Val', val_labels), ('Test', test_labels)]:
        counts = defaultdict(int)
        for lbl in split_labels.values():
            counts[lbl] += 1
        print(f"  {split_name}: {len(split_labels)} slides")
        for c in range(NUM_CLASSES):
            print(f"    Class {c} ({CLASS_NAMES[c]}): {counts[c]}")
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = SlideFeatureDataset4Class(
        train_ids, labels, slide_to_dir, max_patches, subsample=True, num_classes=NUM_CLASSES
    )
    val_dataset = SlideFeatureDataset4Class(
        val_ids, labels, slide_to_dir, max_patches, subsample=False, num_classes=NUM_CLASSES
    )
    test_dataset = SlideFeatureDataset4Class(
        test_ids, labels, slide_to_dir, max_patches, subsample=False, num_classes=NUM_CLASSES
    )
    
    # Create samplers/dataloaders based on balance strategy
    if balance_strategy == 'oversample':
        # Use weighted random sampler for oversampling minority classes
        sample_weights = train_dataset.get_sample_weights()
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, collate_fn=collate_fn,
            pin_memory=(device.type == 'cuda')
        )
    else:
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
    model = ABMIL_MultiClass(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        attention_dim=attention_dim,
        num_classes=NUM_CLASSES,
        dropout=dropout
    ).to(device)
    
    print(f"\nModel: ABMIL Multi-Class (TRIDENT-style)")
    print(f"  Input dim: {input_dim}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Attention dim: {attention_dim}")
    print(f"  Num classes: {NUM_CLASSES}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Set up loss function based on balance strategy
    class_weights = train_dataset.get_class_weights().to(device)
    print(f"\nClass weights: {class_weights.cpu().numpy()}")
    print(f"Balance strategy: {balance_strategy}")
    
    if balance_strategy == 'focal':
        criterion = FocalLoss(alpha=class_weights, gamma=focal_gamma)
        print(f"Using Focal Loss with gamma={focal_gamma}")
    elif balance_strategy == 'weighted':
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print("Using Weighted CrossEntropyLoss")
    else:  # 'oversample' or 'none'
        criterion = nn.CrossEntropyLoss()
        print("Using standard CrossEntropyLoss")
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=lr * 0.01
    )
    
    # Training loop
    best_val_balanced_acc = 0
    best_val_macro_f1 = 0
    best_epoch = 0
    history = {'train': [], 'val': []}
    patience = 20
    patience_counter = 0
    
    print("\n" + "=" * 70)
    print("TRIDENT ABMIL 4-Class Training")
    print("=" * 70)
    
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        print("-" * 50)
        
        train_loss, train_acc, train_bal_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, NUM_CLASSES
        )
        
        val_results = evaluate(model, val_loader, criterion, device, NUM_CLASSES)
        scheduler.step()
        
        print(f"Train | Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | Bal Acc: {train_bal_acc:.4f}")
        print(f"Val   | Loss: {val_results['loss']:.4f} | Acc: {val_results['accuracy']:.4f} | "
              f"Bal Acc: {val_results['balanced_accuracy']:.4f}")
        print(f"      | Macro F1: {val_results['macro_f1']:.4f} | Weighted F1: {val_results['weighted_f1']:.4f} | "
              f"AUC: {val_results['auc_macro']:.4f}")
        print(f"      | Per-class F1: {[f'{f:.3f}' for f in val_results['per_class']['f1']]}")
        print(f"      | LR: {scheduler.get_last_lr()[0]:.6f}")
        
        history['train'].append({
            'epoch': epoch, 
            'loss': train_loss,
            'accuracy': train_acc, 
            'balanced_accuracy': train_bal_acc
        })
        history['val'].append({
            'epoch': epoch, 
            **{k: v for k, v in val_results.items() if k != 'predictions'}
        })
        
        # Save best model (using balanced accuracy as primary metric)
        current_metric = val_results['balanced_accuracy']
        if current_metric > best_val_balanced_acc:
            best_val_balanced_acc = current_metric
            best_val_macro_f1 = val_results['macro_f1']
            best_epoch = epoch
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_balanced_accuracy': best_val_balanced_acc,
                'val_macro_f1': best_val_macro_f1,
                'config': {
                    'input_dim': input_dim,
                    'hidden_dim': hidden_dim,
                    'attention_dim': attention_dim,
                    'num_classes': NUM_CLASSES
                }
            }, output_dir / 'best_model.pt')
            print(f"  -> New best model! (Bal Acc: {best_val_balanced_acc:.4f}, Macro F1: {best_val_macro_f1:.4f})")
        else:
            patience_counter += 1
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, output_dir / 'latest_checkpoint.pt')
        
        # Early stopping
        if patience_counter >= patience and epoch > 30:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break
    
    # Save history
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Plot training history
    plot_training_history(history, output_dir / 'training_history.png')
    
    # Final evaluation on test set
    print("\n" + "=" * 70)
    print(f"Loading best model (epoch {best_epoch}, Bal Acc: {best_val_balanced_acc:.4f})")
    print("=" * 70)
    
    checkpoint = torch.load(output_dir / 'best_model.pt', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_results = evaluate(model, test_loader, criterion, device, NUM_CLASSES)
    
    print("\n" + "=" * 70)
    print("TEST RESULTS - 4-Class Gleason Classification")
    print("=" * 70)
    print(f"Accuracy:          {test_results['accuracy']:.4f}")
    print(f"Balanced Accuracy: {test_results['balanced_accuracy']:.4f}")
    print(f"Macro F1:          {test_results['macro_f1']:.4f}")
    print(f"Weighted F1:       {test_results['weighted_f1']:.4f}")
    print(f"Macro AUC:         {test_results['auc_macro']:.4f}")
    
    print("\nPer-class Metrics:")
    print(f"{'Class':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC':>10} {'Support':>10}")
    print("-" * 75)
    for c in range(NUM_CLASSES):
        print(f"{CLASS_NAMES[c]:<25} "
              f"{test_results['per_class']['precision'][c]:>10.4f} "
              f"{test_results['per_class']['recall'][c]:>10.4f} "
              f"{test_results['per_class']['f1'][c]:>10.4f} "
              f"{test_results['per_class']['auc'][c]:>10.4f} "
              f"{test_results['per_class']['support'][c]:>10}")
    
    # Plot confusion matrix
    cm = np.array(test_results['confusion_matrix'])
    print(f"\nConfusion Matrix:")
    print(f"{'Actual \\ Predicted':<20}", end='')
    for c in range(NUM_CLASSES):
        print(f"{CLASS_NAMES[c]:>15}", end='')
    print()
    for i in range(NUM_CLASSES):
        print(f"{CLASS_NAMES[i]:<20}", end='')
        for j in range(NUM_CLASSES):
            print(f"{cm[i, j]:>15}", end='')
        print()
    
    plot_confusion_matrix(cm, output_dir / 'confusion_matrix.png', 
                         title='4-Class Gleason Classification - Test Set')
    
    # Save results
    with open(output_dir / 'test_results.json', 'w') as f:
        json.dump(test_results, f, indent=2)
    
    # Save predictions
    pred_df = pd.DataFrame({
        'slide_id': test_results['predictions']['slide_ids'],
        'true_label': test_results['predictions']['labels'],
        'predicted': test_results['predictions']['predictions'],
        'prob_normal': [p[0] for p in test_results['predictions']['probabilities']],
        'prob_g3': [p[1] for p in test_results['predictions']['probabilities']],
        'prob_g4': [p[2] for p in test_results['predictions']['probabilities']],
        'prob_g5': [p[3] for p in test_results['predictions']['probabilities']]
    })
    pred_df['true_class'] = pred_df['true_label'].map(CLASS_NAMES)
    pred_df['predicted_class'] = pred_df['predicted'].map(CLASS_NAMES)
    pred_df.to_csv(output_dir / 'test_predictions.csv', index=False)
    
    print(f"\nResults saved to: {output_dir}")
    
    return test_results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='TRIDENT ABMIL Training for 4-Class Gleason Classification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train 4-class classifier with weighted loss
    python train_4class_classification_trident.py --train --epochs 100
    
    # Train with focal loss for better handling of imbalance
    python train_4class_classification_trident.py --train --balance_strategy focal --epochs 100
    
    # Train with oversampling
    python train_4class_classification_trident.py --train --balance_strategy oversample --epochs 100
    
    # Train with GPU
    python train_4class_classification_trident.py --train --device cuda:0 --epochs 100

Classes:
    0: Normal (GTEx - non-cancer prostate tissue)
    1: G3 (Gleason Pattern 3 dominant - low-grade)
    2: G4 (Gleason Pattern 4 dominant - intermediate)
    3: G5 (Gleason Pattern 5 dominant - high-grade)

Data Distribution:
    Normal (GTEx): ~593 slides
    G3 (TCGA): ~13 slides
    G4 (TCGA): ~48 slides
    G5 (TCGA): ~35 slides
        """
    )
    
    # Actions
    parser.add_argument('--train', action='store_true', help='Train 4-class ABMIL model')
    
    # Paths
    parser.add_argument('--tcga_feats_dir', type=str, default=TCGA_FEATS_DIR,
                       help='Directory with TCGA-PRAD H5 feature files')
    parser.add_argument('--tcga_labels_path', type=str, default=TCGA_LABELS_PATH,
                       help='Path to TCGA-PRAD labels TSV file')
    parser.add_argument('--gtex_feats_dir', type=str, default=GTEX_FEATS_DIR,
                       help='Directory with GTEx H5 feature files')
    parser.add_argument('--gtex_labels_path', type=str, default=GTEX_LABELS_PATH,
                       help='Path to GTEx labels TSV file')
    parser.add_argument('--output_dir', type=str, default='output/trident_4class',
                       help='Output directory')
    
    # Model
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--attention_dim', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.25)
    
    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--max_patches', type=int, default=4096)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda:0')
    
    # Class imbalance handling
    parser.add_argument('--balance_strategy', type=str, default='focal',
                       choices=['weighted', 'oversample', 'focal', 'none'],
                       help='Strategy for handling class imbalance')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                       help='Gamma parameter for focal loss')
    
    # Data splits
    parser.add_argument('--val_split', type=float, default=0.15,
                       help='Validation split ratio')
    parser.add_argument('--test_split', type=float, default=0.15,
                       help='Test split ratio')
    
    args = parser.parse_args()
    
    if not args.train:
        parser.print_help()
        return
    
    base_dir = Path(__file__).resolve().parent
    
    print("\n" + "=" * 70)
    print("TRIDENT ABMIL - 4-CLASS GLEASON CLASSIFICATION")
    print("=" * 70)
    
    # Load labels from both datasets
    labels = {}
    slide_to_dir = {}
    
    # Load TCGA-PRAD data (cancer slides: G3, G4, G5)
    tcga_labels_path = Path(args.tcga_labels_path)
    tcga_feats_dir = Path(args.tcga_feats_dir)
    
    if tcga_labels_path.exists():
        tcga_df = pd.read_csv(tcga_labels_path, sep='\t')
        print(f"\nTCGA-PRAD: {len(tcga_df)} slides loaded")
        
        # Remap TCGA labels: 1->G3, 2->G4, 3->G5
        for _, row in tcga_df.iterrows():
            tcga_label = row['label']  # 1, 2, or 3 in original
            new_label = TCGA_LABEL_REMAP.get(tcga_label)
            if new_label is not None:
                labels[row['slide_id']] = new_label
                slide_to_dir[row['slide_id']] = str(tcga_feats_dir)
        
        # Count per class
        tcga_counts = defaultdict(int)
        for sid in labels:
            tcga_counts[labels[sid]] += 1
        print(f"  G3 (label 1): {tcga_counts[1]}")
        print(f"  G4 (label 2): {tcga_counts[2]}")
        print(f"  G5 (label 3): {tcga_counts[3]}")
    else:
        print(f"Warning: TCGA labels not found: {tcga_labels_path}")
    
    # Load GTEx data (normal slides: label 0)
    gtex_labels_path = Path(args.gtex_labels_path)
    gtex_feats_dir = Path(args.gtex_feats_dir)
    
    if gtex_labels_path.exists():
        gtex_df = pd.read_csv(gtex_labels_path, sep='\t')
        print(f"\nGTEx Prostate: {len(gtex_df)} slides loaded")
        
        # All GTEx slides are normal (label 0)
        for _, row in gtex_df.iterrows():
            labels[row['slide_id']] = 0  # Normal
            slide_to_dir[row['slide_id']] = str(gtex_feats_dir)
        
        gtex_count = sum(1 for v in labels.values() if v == 0)
        print(f"  Normal (label 0): {gtex_count}")
    else:
        print(f"Warning: GTEx labels not found: {gtex_labels_path}")
    
    if not labels:
        print("Error: No labels loaded. Check paths.")
        return
    
    # Print total distribution
    print(f"\nTotal slides: {len(labels)}")
    class_counts = defaultdict(int)
    for lbl in labels.values():
        class_counts[lbl] += 1
    for c in range(NUM_CLASSES):
        print(f"  {CLASS_NAMES[c]}: {class_counts[c]}")
    
    # Output directory
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = base_dir / output_dir
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"4class_{args.balance_strategy}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config = vars(args).copy()
    config['class_names'] = CLASS_NAMES
    config['num_classes'] = NUM_CLASSES
    config['class_distribution'] = dict(class_counts)
    with open(run_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Train
    train_4class(
        output_dir=run_dir,
        labels=labels,
        slide_to_dir=slide_to_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        attention_dim=args.attention_dim,
        dropout=args.dropout,
        max_patches=args.max_patches,
        device=args.device,
        num_workers=args.num_workers,
        balance_strategy=args.balance_strategy,
        focal_gamma=args.focal_gamma,
        val_split=args.val_split,
        test_split=args.test_split
    )


if __name__ == "__main__":
    main()
