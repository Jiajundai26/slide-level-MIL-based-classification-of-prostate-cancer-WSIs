#!/usr/bin/env python3
"""
TRIDENT-based Binary Classification Training Script for TCGA-PRAD

This script implements ABMIL (Attention-Based Multiple Instance Learning) training
following the TRIDENT framework methodology from MahmoodLab:
    https://github.com/mahmoodlab/TRIDENT

Supports two binary classification modes:
    
    MODE 1: Cancer vs Non-Cancer (--mode cancer_vs_noncancer)
        - Non-cancer (label 0): Slides without cancer annotations  
        - Cancer (label 1): Slides with Gleason Pattern 3, 4, or 5 (G3, G4, G5)
    
    MODE 2: Low-Grade vs High-Grade Cancer (--mode low_vs_high) [DEFAULT]
        - Low-grade (label 0): G3-dominant slides
        - High-grade (label 1): G4/G5-dominant slides

Uses pre-extracted UNI-v2 features (1536-dim) from:
    - TCGA-PRAD: /sci-it/projects/cedmav/data/ARPA-H/Prostate_HnE/preprocessing/TCGA-PRAD/
    - GTEx_Prostate: /sci-it/projects/cedmav/data/ARPA-H/Prostate_HnE/preprocessing/GTEx_Prostate/

TRIDENT Framework Components Used:
    - Patch-level foundation model features (UNI-v2)
    - Attention-Based MIL aggregation (ABMIL)
    - Gated attention mechanism
    - Slide-level classification

Usage:
    # Train Low-Grade vs High-Grade classifier (default, uses existing labels)
    python train_binary_classification_trident.py --train --epochs 50
    
    # Train Cancer vs Non-Cancer classifier  
    python train_binary_classification_trident.py --train --mode cancer_vs_noncancer --epochs 50
    
    # Train with GPU
    python train_binary_classification_trident.py --train --device cuda:0 --epochs 50
    
    # Custom paths
    python train_binary_classification_trident.py --train \\
        --feats_dir /path/to/features --labels_path /path/to/labels.tsv
"""

import os
import sys
import json
import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime
from collections import defaultdict, Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import h5py
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
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

# Default paths for pre-extracted features
DEFAULT_FEATS_DIR = "/sci-it/projects/cedmav/data/ARPA-H/Prostate_HnE/preprocessing/TCGA-PRAD/20x_256px_0px_overlap/features_uni_v2"
DEFAULT_LABELS_PATH = "/sci-it/projects/cedmav/data/ARPA-H/Prostate_HnE/preprocessing/TCGA-PRAD/labels_MIL/slide_labels.tsv"
DEFAULT_ANNOTATIONS_DIR = "/local/data/magicscan/HnE/TCGA-PRAD_organize/annotations/geojsons"

# GTEx Prostate paths (Benign/Cancer/Discard from CSV)
GTEX_FEATS_DIR = "/sci-it/projects/cedmav/data/ARPA-H/Prostate_HnE/preprocessing/GTEx_Prostate/20x_256px_0px_overlap/features_uni_v2"
GTEX_LABELS_PATH = "/sci-it/projects/cedmav/data/ARPA-H/Prostate_HnE/preprocessing/GTEx_Prostate/GTEx_prostate_labels.csv"

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
# Label Loading Helpers
# ============================================================================

def load_gtex_labels(labels_path: Path) -> pd.DataFrame:
    """
    Load GTEx labels and filter out discard rows.

    Supports:
      - CSV: columns include ['file_name', 'label'] where label in {0, 1, -1}
      - TSV: columns include ['slide_id', 'label']
    """
    if labels_path.suffix.lower() == ".csv":
        df = pd.read_csv(labels_path)
        if "label" not in df.columns:
            raise ValueError("GTEx labels CSV must contain a 'label' column.")
        slide_col = "file_name" if "file_name" in df.columns else "slide_id"
        if slide_col not in df.columns:
            raise ValueError("GTEx labels CSV must contain 'file_name' or 'slide_id'.")
        df = df[[slide_col, "label"]].rename(columns={slide_col: "slide_id"})
        df["label"] = pd.to_numeric(df["label"], errors="coerce")
        df = df[df["label"].isin([0, 1])]
        return df

    df = pd.read_csv(labels_path, sep="\t")
    if "slide_id" not in df.columns or "label" not in df.columns:
        raise ValueError("GTEx labels TSV must contain 'slide_id' and 'label'.")
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df[df["label"].isin([0, 1])]
    return df


# ============================================================================
# Label Mapping
# ============================================================================

# Mode 1: Cancer vs Non-Cancer (all Gleason grades are cancer)
GLEASON_TO_CANCER = {1: 1, 2: 1, 3: 1}  # G3, G4, G5 -> Cancer

# Mode 2: Low-Grade (G3) vs High-Grade (G4, G5)
GLEASON_TO_GRADE = {1: 0, 2: 1, 3: 1}  # G3 -> Low, G4/G5 -> High

CLASS_NAMES = {
    'cancer_vs_noncancer': {0: 'Non-Cancer', 1: 'Cancer'},
    'low_vs_high': {0: 'Low-Grade (G3)', 1: 'High-Grade (G4/G5)'}
}


# ============================================================================
# TRIDENT-style ABMIL Model Architecture
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


class ABMIL(nn.Module):
    """
    Attention-Based Multiple Instance Learning (ABMIL) Classifier
    
    Following TRIDENT/CLAM architecture:
    1. Feature projection from encoder dimension to hidden dimension
    2. Gated attention mechanism for instance weighting
    3. Weighted aggregation to slide-level representation
    4. Classification head for binary prediction
    
    Reference:
        - CLAM: https://github.com/mahmoodlab/CLAM
        - TRIDENT: https://github.com/mahmoodlab/TRIDENT
    """
    def __init__(
        self,
        input_dim: int = 1536,      # UNI-v2 feature dimension
        hidden_dim: int = 256,       # Projection dimension
        attention_dim: int = 128,    # Attention hidden dimension
        num_classes: int = 1,        # 1 for binary (sigmoid), >1 for multi-class
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
        
        # Classification head
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

class SlideFeatureDataset(Dataset):
    """
    Dataset for loading pre-extracted slide features from H5 files.
    
    Follows TRIDENT conventions for feature loading and preprocessing.
    Supports multiple feature directories for multi-dataset training.
    """
    def __init__(
        self,
        feats_dirs: List[str],  # Can be single dir or list of dirs
        slide_ids: List[str],
        labels: Dict[str, int],
        slide_to_dir: Optional[Dict[str, str]] = None,  # Maps slide_id to feats_dir
        max_patches: int = 4096,
        subsample: bool = True
    ):
        """
        Args:
            feats_dirs: Directory or list of directories containing H5 feature files
            slide_ids: List of slide IDs to include
            labels: Dictionary mapping slide_id to label
            slide_to_dir: Optional mapping of slide_id to specific feature directory
            max_patches: Maximum patches per slide
            subsample: Random subsample (train) or truncate (val/test)
        """
        # Handle single dir or list of dirs
        if isinstance(feats_dirs, (str, Path)):
            feats_dirs = [feats_dirs]
        self.feats_dirs = [Path(d) for d in feats_dirs]
        self.max_patches = max_patches
        self.subsample = subsample
        self.slide_to_dir = slide_to_dir or {}
        
        # Filter to existing slides
        self.samples = []
        for slide_id in slide_ids:
            if slide_id not in labels:
                continue
            
            # Find the H5 file in any of the feature directories
            h5_path = None
            
            # First check if we have a specific mapping
            if slide_id in self.slide_to_dir:
                candidate = Path(self.slide_to_dir[slide_id]) / f"{slide_id}.h5"
                if candidate.exists():
                    h5_path = candidate
            
            # Otherwise search all directories
            if h5_path is None:
                for feats_dir in self.feats_dirs:
                    candidate = feats_dir / f"{slide_id}.h5"
                    if candidate.exists():
                        h5_path = candidate
                        break
            
            if h5_path is not None:
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
        print(f"    Class 0: {label_counts[0]}, Class 1: {label_counts[1]}")
    
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
            'label': torch.tensor(sample['label'], dtype=torch.float32),
            'slide_id': sample['slide_id']
        }


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
    total_loss = 0
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
    device: torch.device
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
        ap = average_precision_score(all_labels, all_probs)
    except ValueError:
        auc, ap = 0.5, 0.5
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )
    
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    # Sensitivity and Specificity
    if conf_matrix.shape == (2, 2):
        tn, fp, fn, tp = conf_matrix.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    else:
        sensitivity = specificity = 0
    
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
# Label Creation
# ============================================================================

def create_binary_labels_from_annotations(
    feats_dir: Path,
    annotations_dir: Path,
    output_path: Path
) -> pd.DataFrame:
    """
    Create binary labels based on presence of cancer annotations.
    
    Slides with Gleason pattern annotations -> Cancer (1)
    Slides without -> Non-Cancer (0)
    """
    feats_dir = Path(feats_dir)
    annotations_dir = Path(annotations_dir)
    
    h5_files = list(feats_dir.glob('*.h5'))
    print(f"Found {len(h5_files)} feature files")
    
    # Map annotation files
    annotation_map = {g.stem: g for g in annotations_dir.glob('*.geojson')}
    print(f"Found {len(annotation_map)} annotation files")
    
    cancer_patterns = [
        'gleason', 'g3', 'g4', 'g5', 'pattern 3', 'pattern 4', 'pattern 5',
        '3+3', '3+4', '3+5', '4+3', '4+4', '4+5', '5+3', '5+4', '5+5'
    ]
    
    labels = []
    for h5_path in tqdm(h5_files, desc="Creating labels"):
        slide_id = h5_path.stem
        ann_path = annotation_map.get(slide_id)
        
        is_cancer = False
        found_patterns = []
        
        if ann_path and ann_path.exists():
            try:
                with open(ann_path, 'r') as f:
                    geojson = json.load(f)
                
                for feat in geojson.get('features', []):
                    if 'properties' in feat:
                        props = feat['properties']
                        class_name = (props.get('classification', {}).get('name', '') or
                                     props.get('class', '') or props.get('name', '') or '')
                        
                        for pattern in cancer_patterns:
                            if pattern in class_name.lower():
                                is_cancer = True
                                if class_name not in found_patterns:
                                    found_patterns.append(class_name)
                                break
            except Exception as e:
                print(f"  Error reading {ann_path.name}: {e}")
        
        labels.append({
            'slide_id': slide_id,
            'label': 1 if is_cancer else 0,
            'class_name': 'Cancer' if is_cancer else 'Non-Cancer',
            'found_patterns': ','.join(found_patterns)
        })
    
    df = pd.DataFrame(labels)
    print(f"\nLabel distribution:\n{df['class_name'].value_counts()}")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, sep='\t', index=False)
    print(f"Saved to {output_path}")
    
    return df


# ============================================================================
# Main Training Function
# ============================================================================

def train_abmil(
    feats_dirs: List[Path],  # List of feature directories
    output_dir: Path,
    labels_df: pd.DataFrame,
    labels: Dict[str, int],
    slide_to_dir: Optional[Dict[str, str]] = None,
    mode: str = 'low_vs_high',
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
    """
    Train ABMIL model using TRIDENT methodology.
    
    Supports multiple feature directories for multi-dataset training
    (e.g., combining TCGA-PRAD cancer with GTEx non-cancer).
    """
    set_seed(SEED)
    
    # Handle single or multiple feature dirs
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
    print(f"Device: {device}")
    
    # Get feature dimension from first available H5 file
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
    print(f"Feature directories: {[str(d) for d in feats_dirs]}")
    
    # Use existing train/val/test splits from labels file
    if 'fold_0' in labels_df.columns:
        print("Using existing train/val/test splits")
        train_ids = labels_df[labels_df['fold_0'] == 'Training']['slide_id'].tolist()
        val_ids = labels_df[labels_df['fold_0'] == 'Validation']['slide_id'].tolist()
        test_ids = labels_df[labels_df['fold_0'] == 'Testing']['slide_id'].tolist()
        
        # Filter to slides with labels
        train_ids = [s for s in train_ids if s in labels]
        val_ids = [s for s in val_ids if s in labels]
        test_ids = [s for s in test_ids if s in labels]
    else:
        # Create new splits
        print("Creating new train/val/test splits")
        all_ids = list(labels.keys())
        all_lbls = [labels[s] for s in all_ids]

        def can_stratify(lbls: List[int]) -> bool:
            counts = Counter(lbls)
            return all(c >= 2 for c in counts.values())

        stratify_all = all_lbls if can_stratify(all_lbls) else None
        if stratify_all is None:
            print("Warning: too few samples for stratified split; using random split")

        train_ids, temp_ids, train_lbls, temp_lbls = train_test_split(
            all_ids, all_lbls, test_size=0.3, stratify=stratify_all, random_state=SEED
        )

        stratify_temp = temp_lbls if can_stratify(temp_lbls) else None
        if stratify_temp is None:
            print("Warning: too few samples for stratified val/test split; using random split")

        val_ids, test_ids, _, _ = train_test_split(
            temp_ids, temp_lbls, test_size=0.5, stratify=stratify_temp, random_state=SEED
        )
    
    # Print split statistics
    train_lbls = [labels[s] for s in train_ids]
    val_lbls = [labels[s] for s in val_ids]
    test_lbls = [labels[s] for s in test_ids]
    
    class_names = CLASS_NAMES[mode]
    print(f"\nData split:")
    print(f"  Train: {len(train_ids)} ({class_names[0]}: {len(train_lbls) - sum(train_lbls)}, {class_names[1]}: {sum(train_lbls)})")
    print(f"  Val:   {len(val_ids)} ({class_names[0]}: {len(val_lbls) - sum(val_lbls)}, {class_names[1]}: {sum(val_lbls)})")
    print(f"  Test:  {len(test_ids)} ({class_names[0]}: {len(test_lbls) - sum(test_lbls)}, {class_names[1]}: {sum(test_lbls)})")
    
    # Create datasets
    print("\nCreating datasets...")
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
    
    # Initialize ABMIL model
    model = ABMIL(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        attention_dim=attention_dim,
        num_classes=1,
        dropout=dropout
    ).to(device)
    
    print(f"\nModel: ABMIL (TRIDENT-style)")
    print(f"  Input dim: {input_dim}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Attention dim: {attention_dim}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Class weights for imbalanced data
    n_pos = sum(train_lbls)
    n_neg = len(train_lbls) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)]).to(device)
    print(f"\nClass balance: {class_names[0]}={n_neg}, {class_names[1]}={n_pos}")
    print(f"Positive weight: {pos_weight.item():.2f}")
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    
    # Training loop
    best_val_auc = 0
    best_epoch = 0
    history = {'train': [], 'val': []}
    
    print("\n" + "=" * 70)
    print("TRIDENT ABMIL Training")
    print("=" * 70)
    
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        print("-" * 50)
        
        train_loss, train_acc, train_bal_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        val_results = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        
        print(f"Train | Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | Bal Acc: {train_bal_acc:.4f}")
        print(f"Val   | Loss: {val_results['loss']:.4f} | Acc: {val_results['accuracy']:.4f} | "
              f"AUC: {val_results['auc']:.4f} | F1: {val_results['f1']:.4f}")
        print(f"      | Sens: {val_results['sensitivity']:.4f} | Spec: {val_results['specificity']:.4f} | "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")
        
        history['train'].append({
            'epoch': epoch, 'loss': train_loss,
            'accuracy': train_acc, 'balanced_accuracy': train_bal_acc
        })
        history['val'].append({
            'epoch': epoch, **{k: v for k, v in val_results.items() if k != 'predictions'}
        })
        
        # Save best model
        if val_results['auc'] > best_val_auc:
            best_val_auc = val_results['auc']
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_auc': best_val_auc,
                'config': {
                    'input_dim': input_dim,
                    'hidden_dim': hidden_dim,
                    'attention_dim': attention_dim,
                    'mode': mode
                }
            }, output_dir / 'best_model.pt')
            print(f"  -> New best model! (AUC: {best_val_auc:.4f})")
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, output_dir / 'latest_checkpoint.pt')
    
    # Save history
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Final evaluation
    print("\n" + "=" * 70)
    print(f"Loading best model (epoch {best_epoch}, AUC: {best_val_auc:.4f})")
    print("=" * 70)
    
    checkpoint = torch.load(output_dir / 'best_model.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_results = evaluate(model, test_loader, criterion, device)
    
    print("\n" + "=" * 70)
    print(f"TEST RESULTS - {mode}")
    print("=" * 70)
    print(f"Accuracy:          {test_results['accuracy']:.4f}")
    print(f"Balanced Accuracy: {test_results['balanced_accuracy']:.4f}")
    print(f"AUC-ROC:           {test_results['auc']:.4f}")
    print(f"Average Precision: {test_results['average_precision']:.4f}")
    print(f"Precision:         {test_results['precision']:.4f}")
    print(f"Recall/Sens:       {test_results['recall']:.4f}")
    print(f"Specificity:       {test_results['specificity']:.4f}")
    print(f"F1 Score:          {test_results['f1']:.4f}")
    
    cm = test_results['confusion_matrix']
    print(f"\nConfusion Matrix:")
    print(f"  Predicted:  {class_names[0]:>12}  {class_names[1]:>12}")
    print(f"  Actual {class_names[0]:>12}:  {cm[0][0]:>12}  {cm[0][1]:>12}")
    print(f"  Actual {class_names[1]:>12}:  {cm[1][0]:>12}  {cm[1][1]:>12}")
    
    # Save results
    with open(output_dir / 'test_results.json', 'w') as f:
        json.dump(test_results, f, indent=2)
    
    pred_df = pd.DataFrame({
        'slide_id': test_results['predictions']['slide_ids'],
        'true_label': test_results['predictions']['labels'],
        'predicted': test_results['predictions']['predictions'],
        'probability': test_results['predictions']['probabilities']
    })
    pred_df.to_csv(output_dir / 'test_predictions.csv', index=False)
    
    print(f"\nResults saved to: {output_dir}")
    
    return test_results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='TRIDENT ABMIL Training for Binary Classification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train Low-Grade vs High-Grade classifier (TCGA-PRAD only)
    python train_binary_classification_trident.py --train --mode low_vs_high --epochs 50
    
    # Train Cancer vs Non-Cancer combining TCGA-PRAD + GTEx
    python train_binary_classification_trident.py --train --mode cancer_vs_noncancer --combine_gtex --epochs 50
    
    # Train on GTEx only (benign/cancer; discards filtered)
    python train_binary_classification_trident.py --train --dataset gtex --epochs 50
    
    # With GPU
    python train_binary_classification_trident.py --train --device cuda:0 --epochs 50

Datasets:
    TCGA-PRAD: Gleason-graded cancer slides
    GTEx_Prostate: Benign/Cancer/Discard (CSV labels; discards filtered)
        """
    )
    
    # Actions
    parser.add_argument('--train', action='store_true', help='Train ABMIL model')
    parser.add_argument('--create_labels', action='store_true', help='Create binary labels from annotations')
    
    # Dataset selection
    parser.add_argument('--dataset', type=str, default='tcga',
                       choices=['tcga', 'gtex', 'combined'],
                       help='Dataset to use: tcga, gtex, or combined (default: tcga)')
    parser.add_argument('--combine_gtex', action='store_true',
                       help='Combine TCGA-PRAD with GTEx labels for cancer_vs_noncancer mode')
    
    # Classification mode
    parser.add_argument('--mode', type=str, default='low_vs_high',
                       choices=['cancer_vs_noncancer', 'low_vs_high'],
                       help='Classification mode (default: low_vs_high)')
    
    # Paths
    parser.add_argument('--feats_dir', type=str, default=DEFAULT_FEATS_DIR,
                       help='Directory with H5 feature files (TCGA-PRAD)')
    parser.add_argument('--gtex_feats_dir', type=str, default=GTEX_FEATS_DIR,
                       help='Directory with GTEx H5 feature files')
    parser.add_argument('--labels_path', type=str, default=DEFAULT_LABELS_PATH,
                       help='Path to TCGA-PRAD labels TSV file')
    parser.add_argument('--gtex_labels_path', type=str, default=GTEX_LABELS_PATH,
                       help='Path to GTEx labels CSV/TSV file')
    parser.add_argument('--annotations_dir', type=str, default=DEFAULT_ANNOTATIONS_DIR,
                       help='Directory with GeoJSON annotations')
    parser.add_argument('--output_dir', type=str, default='output/trident_training',
                       help='Output directory')
    
    # Model
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--attention_dim', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.25)
    
    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--max_patches', type=int, default=4096)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda:0')
    
    args = parser.parse_args()
    
    base_dir = Path(__file__).resolve().parent
    feats_dir = Path(args.feats_dir)
    annotations_dir = Path(args.annotations_dir)
    
    if not args.train and not args.create_labels:
        parser.print_help()
        return
    
    # Create labels from annotations
    if args.create_labels:
        output_path = base_dir / 'output' / 'binary_labels.tsv'
        create_binary_labels_from_annotations(feats_dir, annotations_dir, output_path)
        if not args.train:
            return
    
    # Training
    if args.train:
        mode = args.mode
        dataset = args.dataset
        combine_gtex = args.combine_gtex
        
        # Auto-set dataset to combined if --combine_gtex is used
        if combine_gtex:
            dataset = 'combined'
            mode = 'cancer_vs_noncancer'
        
        print("\n" + "=" * 70)
        print(f"TRIDENT ABMIL - {mode.upper()}")
        print(f"Dataset: {dataset.upper()}")
        print("=" * 70)
        
        # Initialize data structures
        labels = {}
        all_dfs = []
        feats_dirs = []
        slide_to_dir = {}
        
        # Load TCGA-PRAD data
        if dataset in ['tcga', 'combined']:
            tcga_labels_path = Path(args.labels_path)
            tcga_feats_dir = Path(args.feats_dir)
            
            if tcga_labels_path.exists():
                tcga_df = pd.read_csv(tcga_labels_path, sep='\t')
                print(f"TCGA-PRAD: {len(tcga_df)} slides loaded")
                
                if mode == 'low_vs_high':
                    # G3 -> 0 (Low-grade), G4/G5 -> 1 (High-grade)
                    for _, row in tcga_df.iterrows():
                        labels[row['slide_id']] = GLEASON_TO_GRADE[row['label']]
                        slide_to_dir[row['slide_id']] = str(tcga_feats_dir)
                else:  # cancer_vs_noncancer
                    # All TCGA slides are cancer (label=1)
                    for _, row in tcga_df.iterrows():
                        labels[row['slide_id']] = 1
                        slide_to_dir[row['slide_id']] = str(tcga_feats_dir)
                
                all_dfs.append(tcga_df)
                feats_dirs.append(tcga_feats_dir)
            else:
                print(f"Warning: TCGA labels not found: {tcga_labels_path}")
        
        # Load GTEx data
        if dataset in ['gtex', 'combined']:
            gtex_labels_path = Path(args.gtex_labels_path)
            gtex_feats_dir = Path(args.gtex_feats_dir)
            
            if gtex_labels_path.exists():
                gtex_df = load_gtex_labels(gtex_labels_path)
                print(f"GTEx Prostate: {len(gtex_df)} slides loaded (labels 0/1)")
                
                # Use provided labels (0=benign, 1=cancer), discard -1 already filtered
                for _, row in gtex_df.iterrows():
                    labels[row['slide_id']] = int(row['label'])
                    slide_to_dir[row['slide_id']] = str(gtex_feats_dir)
                
                all_dfs.append(gtex_df)
                feats_dirs.append(gtex_feats_dir)
            else:
                print(f"Warning: GTEx labels not found: {gtex_labels_path}")
        
        if not labels:
            print("Error: No labels loaded. Check paths.")
            return
        
        # Combine dataframes
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        # Print label distribution
        n_class0 = sum(1 for v in labels.values() if v == 0)
        n_class1 = sum(1 for v in labels.values() if v == 1)
        class_names = CLASS_NAMES[mode]
        print(f"\nTotal slides: {len(labels)}")
        print(f"  {class_names[0]}: {n_class0}")
        print(f"  {class_names[1]}: {n_class1}")
        
        # Output directory
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = base_dir / output_dir
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = output_dir / f"{dataset}_{mode}_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        with open(run_dir / 'config.json', 'w') as f:
            json.dump(vars(args), f, indent=2)
        
        # Train
        train_abmil(
            feats_dirs=feats_dirs,
            output_dir=run_dir,
            labels_df=combined_df,
            labels=labels,
            slide_to_dir=slide_to_dir,
            mode=mode,
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
