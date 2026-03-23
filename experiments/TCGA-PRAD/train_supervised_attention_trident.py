#!/usr/bin/env python3
"""
TRIDENT-based Training with Supervised Attention for TCGA-PRAD

This script extends the standard ABMIL training with attention supervision
using GeoJSON cancer annotations. The model learns to focus on annotated
cancer regions rather than arbitrary discriminative features.

Loss = Classification Loss + λ × Attention Supervision Loss

Attention Supervision Loss encourages:
    - High attention on patches overlapping with cancer annotations
    - Low attention on patches in non-cancer regions

Usage:
    python train_supervised_attention_trident.py --train --epochs 50
    
    # Adjust attention supervision weight
    python train_supervised_attention_trident.py --train --attention_lambda 1.0
    
    # With GPU
    python train_supervised_attention_trident.py --train --device cuda:0
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    confusion_matrix,
    average_precision_score
)
from tqdm import tqdm

# For polygon intersection
try:
    from shapely.geometry import Polygon, Point, box
    from shapely.prepared import prep
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False
    print("Warning: shapely not installed. Using simple bounding box check.")
    print("Install with: pip install shapely")


# ============================================================================
# Configuration
# ============================================================================

SEED = 42

# Default paths
DEFAULT_FEATS_DIR = "/sci-it/projects/cedmav/data/ARPA-H/Prostate_HnE/preprocessing/TCGA-PRAD/20x_256px_0px_overlap/features_uni_v2"
DEFAULT_LABELS_PATH = "/sci-it/projects/cedmav/data/ARPA-H/Prostate_HnE/preprocessing/TCGA-PRAD/labels_MIL/slide_labels.tsv"
DEFAULT_ANNOTATIONS_DIR = "/local/data/magicscan/HnE/TCGA-PRAD_organize/annotations/geojsons"

# GTEx paths (benign/cancer/discard - no cancer annotations)
GTEX_FEATS_DIR = "/sci-it/projects/cedmav/data/ARPA-H/Prostate_HnE/preprocessing/GTEx_Prostate/20x_256px_0px_overlap/features_uni_v2"
GTEX_LABELS_PATH = "/sci-it/projects/cedmav/data/ARPA-H/Prostate_HnE/preprocessing/GTEx_Prostate/GTEx_prostate_labels.csv"

PATCH_SIZE = 256  # Default patch size at extraction


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================================
# Annotation Loading
# ============================================================================

def load_cancer_annotations(geojson_path: Path) -> List[dict]:
    """Load cancer region polygons from GeoJSON file."""
    if not geojson_path.exists():
        return []
    
    with open(geojson_path, 'r') as f:
        data = json.load(f)
    
    annotations = []
    cancer_patterns = ['pattern 3', 'pattern 4', 'pattern 5', 'g3', 'g4', 'g5', 
                       'gleason', 'cancer', 'tumor', 'carcinoma']
    
    for feature in data.get('features', []):
        props = feature.get('properties', {})
        class_info = props.get('classification', {})
        class_name = class_info.get('name', '') or props.get('name', '')
        
        # Check if cancer-related
        is_cancer = any(p in class_name.lower() for p in cancer_patterns)
        
        if is_cancer:
            geometry = feature.get('geometry', {})
            if geometry.get('type') == 'Polygon':
                coords = geometry.get('coordinates', [[]])[0]
                if len(coords) >= 3:
                    annotations.append({
                        'coords': np.array(coords),
                        'class_name': class_name
                    })
    
    return annotations


def create_patch_labels(
    coords: np.ndarray,
    annotations: List[dict],
    patch_size: int = 256
) -> np.ndarray:
    """
    Create binary labels for each patch based on overlap with cancer annotations.
    
    Args:
        coords: Patch coordinates [N, 2] (x, y top-left corners)
        annotations: List of annotation dicts with 'coords' key
        patch_size: Size of each patch
        
    Returns:
        labels: Binary array [N] where 1 = cancer, 0 = non-cancer
    """
    n_patches = len(coords)
    labels = np.zeros(n_patches, dtype=np.float32)
    
    if len(annotations) == 0:
        return labels
    
    if HAS_SHAPELY:
        # Use shapely for accurate polygon intersection
        cancer_polygons = []
        for ann in annotations:
            try:
                poly = Polygon(ann['coords'])
                if poly.is_valid:
                    cancer_polygons.append(prep(poly))
            except:
                continue
        
        if len(cancer_polygons) == 0:
            return labels
        
        for i, (x, y) in enumerate(coords):
            # Create patch bounding box
            patch_box = box(x, y, x + patch_size, y + patch_size)
            
            # Check intersection with any cancer polygon
            for cancer_poly in cancer_polygons:
                if cancer_poly.intersects(patch_box):
                    labels[i] = 1.0
                    break
    else:
        # Simple bounding box check (less accurate but no dependencies)
        for ann in annotations:
            ann_coords = ann['coords']
            min_x, min_y = ann_coords.min(axis=0)
            max_x, max_y = ann_coords.max(axis=0)
            
            for i, (x, y) in enumerate(coords):
                # Check if patch overlaps with annotation bounding box
                if (x + patch_size > min_x and x < max_x and
                    y + patch_size > min_y and y < max_y):
                    labels[i] = 1.0
    
    return labels


# ============================================================================
# ABMIL Model (same as original)
# ============================================================================

class GatedAttention(nn.Module):
    """Gated Attention mechanism for MIL (TRIDENT/CLAM style)"""
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
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = self.attention_c(a * b).squeeze(-1)
        A = torch.softmax(A, dim=1)
        return A


class ABMIL(nn.Module):
    """Attention-Based Multiple Instance Learning (ABMIL) Classifier"""
    def __init__(
        self,
        input_dim: int = 1536,
        hidden_dim: int = 256,
        attention_dim: int = 128,
        num_classes: int = 1,
        dropout: float = 0.25
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        self.feature_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.attention = GatedAttention(
            input_dim=hidden_dim,
            hidden_dim=attention_dim,
            dropout=dropout
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x: torch.Tensor, return_attention: bool = False):
        if isinstance(x, dict):
            x = x['features']
        
        h = self.feature_projection(x)
        A = self.attention(h)
        M = torch.bmm(A.unsqueeze(1), h).squeeze(1)
        logits = self.classifier(M)
        
        if return_attention:
            return logits, A
        return logits


# ============================================================================
# Dataset with Instance Labels
# ============================================================================

class SupervisedAttentionDataset(Dataset):
    """
    Dataset that loads features AND creates instance-level labels
    based on GeoJSON cancer annotations.
    """
    def __init__(
        self,
        feats_dir: str,
        slide_ids: List[str],
        slide_labels: Dict[str, int],
        annotations_dir: str,
        patch_size: int = 256,
        max_patches: int = 4096,
        subsample: bool = True
    ):
        self.feats_dir = Path(feats_dir)
        self.annotations_dir = Path(annotations_dir)
        self.patch_size = patch_size
        self.max_patches = max_patches
        self.subsample = subsample
        
        # Filter to existing slides
        self.samples = []
        n_with_annotations = 0
        
        for slide_id in slide_ids:
            h5_path = self.feats_dir / f"{slide_id}.h5"
            if not h5_path.exists() or slide_id not in slide_labels:
                continue
            
            # Check for annotations
            geojson_path = self.annotations_dir / f"{slide_id}.geojson"
            has_annotations = geojson_path.exists()
            if has_annotations:
                n_with_annotations += 1
            
            self.samples.append({
                'slide_id': slide_id,
                'label': slide_labels[slide_id],
                'h5_path': h5_path,
                'geojson_path': geojson_path if has_annotations else None
            })
        
        # Statistics
        label_counts = defaultdict(int)
        for s in self.samples:
            label_counts[s['label']] += 1
        
        print(f"  Dataset: {len(self.samples)} slides")
        print(f"    Class 0: {label_counts[0]}, Class 1: {label_counts[1]}")
        print(f"    With annotations: {n_with_annotations}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        
        # Load features and coordinates
        with h5py.File(sample['h5_path'], 'r') as f:
            features = torch.from_numpy(f['features'][:]).float()
            coords = f['coords'][:] if 'coords' in f else None
        
        # Load annotations and create instance labels
        if sample['geojson_path'] is not None and coords is not None:
            annotations = load_cancer_annotations(sample['geojson_path'])
            instance_labels = create_patch_labels(coords, annotations, self.patch_size)
            instance_labels = torch.from_numpy(instance_labels).float()
            has_instance_labels = True
        else:
            # For slides without annotations, all patches are labeled same as slide
            # For cancer slides without annotations: assume all patches could be cancer
            # For normal slides: all patches are normal (0)
            instance_labels = torch.zeros(len(features))
            if sample['label'] == 0:  # Normal slide
                instance_labels = torch.zeros(len(features))
            has_instance_labels = False
        
        num_patches = len(features)
        
        # Handle patch subsampling
        if num_patches > self.max_patches:
            if self.subsample:
                indices = torch.randperm(num_patches)[:self.max_patches]
            else:
                indices = torch.arange(self.max_patches)
            
            features = features[indices]
            instance_labels = instance_labels[indices]
        
        return {
            'features': features,
            'label': torch.tensor(sample['label'], dtype=torch.float32),
            'instance_labels': instance_labels,
            'has_instance_labels': has_instance_labels,
            'slide_id': sample['slide_id']
        }


class MultiSourceSupervisedDataset(Dataset):
    """
    Dataset supporting multiple sources (TCGA with annotations, GTEx without).
    """
    def __init__(
        self,
        tcga_feats_dir: str,
        gtex_feats_dir: str,
        tcga_slide_ids: List[str],
        gtex_slide_ids: List[str],
        slide_labels: Dict[str, int],
        annotations_dir: str,
        patch_size: int = 256,
        max_patches: int = 4096,
        subsample: bool = True
    ):
        self.tcga_feats_dir = Path(tcga_feats_dir)
        self.gtex_feats_dir = Path(gtex_feats_dir)
        self.annotations_dir = Path(annotations_dir)
        self.patch_size = patch_size
        self.max_patches = max_patches
        self.subsample = subsample
        
        self.samples = []
        n_with_annotations = 0
        
        # Add TCGA slides (with annotations)
        for slide_id in tcga_slide_ids:
            h5_path = self.tcga_feats_dir / f"{slide_id}.h5"
            if not h5_path.exists() or slide_id not in slide_labels:
                continue
            
            geojson_path = self.annotations_dir / f"{slide_id}.geojson"
            has_annotations = geojson_path.exists()
            if has_annotations:
                n_with_annotations += 1
            
            self.samples.append({
                'slide_id': slide_id,
                'label': slide_labels[slide_id],
                'h5_path': h5_path,
                'geojson_path': geojson_path if has_annotations else None,
                'source': 'tcga'
            })
        
        # Add GTEx slides (no annotations, benign/cancer)
        for slide_id in gtex_slide_ids:
            h5_path = self.gtex_feats_dir / f"{slide_id}.h5"
            if not h5_path.exists() or slide_id not in slide_labels:
                continue
            
            self.samples.append({
                'slide_id': slide_id,
                'label': slide_labels[slide_id],
                'h5_path': h5_path,
                'geojson_path': None,
                'source': 'gtex'
            })
        
        # Statistics
        label_counts = defaultdict(int)
        source_counts = defaultdict(int)
        for s in self.samples:
            label_counts[s['label']] += 1
            source_counts[s['source']] += 1
        
        print(f"  Dataset: {len(self.samples)} slides")
        print(f"    TCGA: {source_counts['tcga']}, GTEx: {source_counts['gtex']}")
        print(f"    Class 0 (Normal): {label_counts[0]}, Class 1 (Cancer): {label_counts[1]}")
        print(f"    With cancer annotations: {n_with_annotations}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        
        with h5py.File(sample['h5_path'], 'r') as f:
            features = torch.from_numpy(f['features'][:]).float()
            coords = f['coords'][:] if 'coords' in f else None
        
        # Create instance labels
        if sample['geojson_path'] is not None and coords is not None:
            annotations = load_cancer_annotations(sample['geojson_path'])
            instance_labels = create_patch_labels(coords, annotations, self.patch_size)
            instance_labels = torch.from_numpy(instance_labels).float()
            has_instance_labels = True
        else:
            # GTEx or TCGA without annotations
            if sample['label'] == 0:  # Normal tissue
                instance_labels = torch.zeros(len(features))
                has_instance_labels = True  # We know all patches are normal
            else:  # Cancer slide without annotations
                instance_labels = torch.zeros(len(features))
                has_instance_labels = False
        
        num_patches = len(features)
        
        if num_patches > self.max_patches:
            if self.subsample:
                indices = torch.randperm(num_patches)[:self.max_patches]
            else:
                indices = torch.arange(self.max_patches)
            
            features = features[indices]
            instance_labels = instance_labels[indices]
        
        return {
            'features': features,
            'label': torch.tensor(sample['label'], dtype=torch.float32),
            'instance_labels': instance_labels,
            'has_instance_labels': has_instance_labels,
            'slide_id': sample['slide_id']
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """Collate function with instance labels."""
    features = [item['features'] for item in batch]
    labels = torch.stack([item['label'] for item in batch])
    instance_labels = [item['instance_labels'] for item in batch]
    has_instance_labels = [item['has_instance_labels'] for item in batch]
    slide_ids = [item['slide_id'] for item in batch]
    
    max_len = max(f.shape[0] for f in features)
    feature_dim = features[0].shape[1]
    batch_size = len(features)
    
    # Pad features
    padded_features = torch.zeros(batch_size, max_len, feature_dim)
    padded_instance_labels = torch.zeros(batch_size, max_len)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    
    for i, (f, il) in enumerate(zip(features, instance_labels)):
        length = f.shape[0]
        padded_features[i, :length] = f
        padded_instance_labels[i, :length] = il
        mask[i, :length] = True
    
    return {
        'features': padded_features,
        'labels': labels,
        'instance_labels': padded_instance_labels,
        'has_instance_labels': has_instance_labels,
        'mask': mask,
        'slide_ids': slide_ids
    }


# ============================================================================
# Attention Supervision Loss
# ============================================================================

def attention_supervision_loss(
    attention: torch.Tensor,
    instance_labels: torch.Tensor,
    mask: torch.Tensor,
    has_instance_labels: List[bool]
) -> torch.Tensor:
    """
    Compute attention supervision loss.
    
    Encourages:
        - High attention on cancer patches (instance_label = 1)
        - Low attention on non-cancer patches (instance_label = 0)
    
    Args:
        attention: [B, N] attention weights (sum to 1)
        instance_labels: [B, N] binary labels (1 = cancer, 0 = normal)
        mask: [B, N] valid patch mask
        has_instance_labels: List of bools indicating which samples have labels
        
    Returns:
        loss: Scalar attention supervision loss
    """
    batch_size = attention.shape[0]
    total_loss = 0.0
    n_valid = 0
    
    for i in range(batch_size):
        if not has_instance_labels[i]:
            continue
        
        valid_mask = mask[i]
        att = attention[i][valid_mask]
        labels = instance_labels[i][valid_mask]
        
        n_cancer = labels.sum()
        n_normal = (1 - labels).sum()
        
        if n_cancer > 0 and n_normal > 0:
            # We want attention on cancer patches to be higher than on normal patches
            # Use a margin-based loss
            cancer_attention = (att * labels).sum() / n_cancer
            normal_attention = (att * (1 - labels)).sum() / n_normal
            
            # Margin loss: cancer attention should be higher by margin
            margin = 0.1
            loss = F.relu(normal_attention - cancer_attention + margin)
            
            # Also add BCE loss to directly supervise attention
            # Normalize instance_labels as target attention distribution
            target_attention = labels / (labels.sum() + 1e-8)
            bce_loss = F.binary_cross_entropy(att, target_attention.detach())
            
            total_loss += loss + 0.5 * bce_loss
            n_valid += 1
        
        elif n_cancer == 0 and n_normal > 0:
            # Normal slide - attention should be uniform (low entropy)
            # Don't strongly penalize as we want flexibility
            pass
    
    if n_valid > 0:
        return total_loss / n_valid
    else:
        return torch.tensor(0.0, device=attention.device)


# ============================================================================
# Label Loading
# ============================================================================

def _pick_slide_id_column(df: pd.DataFrame) -> str:
    """Select slide id column from a labels dataframe."""
    lower_map = {c.lower(): c for c in df.columns}
    preferred = [
        'slide_id', 'slide', 'wsi', 'wsi_id', 'image_id', 'file_name', 'filename', 'id'
    ]
    for key in preferred:
        if key in lower_map:
            return lower_map[key]
    excluded = {lower_map.get('label'), lower_map.get('class-name'), lower_map.get('class_name')}
    candidates = [c for c in df.columns if c not in excluded]
    if candidates:
        return candidates[0]
    raise ValueError("Could not determine slide id column in GTEx labels CSV.")


def load_gtex_labels_csv(labels_path: Path) -> Tuple[List[str], Dict[str, int]]:
    """
    Load GTEx labels from GTEx_prostate_labels.csv.

    Expects columns including:
      - label: {0=Benign, 1=Cancer, -1=Discard}
      - class-name: optional, human-readable class name
      - slide identifier column (auto-detected)
    """
    df = pd.read_csv(labels_path)
    if df.empty:
        return [], {}
    slide_id_col = _pick_slide_id_column(df)
    label_col = None
    for col in df.columns:
        if col.lower() == 'label':
            label_col = col
            break
    if label_col is None:
        raise ValueError("GTEx labels CSV must contain a 'label' column.")
    labels_numeric = pd.to_numeric(df[label_col], errors='coerce')
    df = df.assign(_label=labels_numeric)
    df = df[df['_label'].isin([0, 1])]
    slide_ids = df[slide_id_col].astype(str).tolist()
    labels = {sid: int(lbl) for sid, lbl in zip(slide_ids, df['_label'].tolist())}
    return slide_ids, labels


# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    attention_lambda: float = 0.5
) -> Tuple[float, float, float, float]:
    """Train for one epoch with attention supervision."""
    model.train()
    total_cls_loss = 0
    total_att_loss = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        features = batch['features'].to(device)
        labels = batch['labels'].to(device)
        instance_labels = batch['instance_labels'].to(device)
        mask = batch['mask'].to(device)
        has_instance_labels = batch['has_instance_labels']
        
        optimizer.zero_grad()
        
        # Forward with attention
        logits, attention = model(features, return_attention=True)
        logits = logits.squeeze(-1)
        
        # Classification loss
        cls_loss = criterion(logits, labels)
        
        # Attention supervision loss
        att_loss = attention_supervision_loss(
            attention, instance_labels, mask, has_instance_labels
        )
        
        # Total loss
        loss = cls_loss + attention_lambda * att_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_cls_loss += cls_loss.item()
        total_att_loss += att_loss.item()
        
        preds = (torch.sigmoid(logits) > 0.5).float()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({
            'cls': f'{cls_loss.item():.4f}',
            'att': f'{att_loss.item():.4f}'
        })
    
    avg_cls_loss = total_cls_loss / len(loader)
    avg_att_loss = total_att_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    
    return avg_cls_loss, avg_att_loss, accuracy, balanced_acc


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
    
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
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
# Main Training
# ============================================================================

def train_supervised_attention(
    tcga_feats_dir: Path,
    gtex_feats_dir: Path,
    annotations_dir: Path,
    output_dir: Path,
    tcga_slide_ids: List[str],
    gtex_slide_ids: List[str],
    labels: Dict[str, int],
    epochs: int = 50,
    batch_size: int = 8,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    attention_lambda: float = 0.5,
    hidden_dim: int = 256,
    attention_dim: int = 128,
    dropout: float = 0.25,
    max_patches: int = 4096,
    device: str = 'cuda:0',
    num_workers: int = 4
):
    """Train ABMIL with supervised attention."""
    set_seed(SEED)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if device.startswith('cuda') and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    device = torch.device(device)
    print(f"Device: {device}")
    
    # Get feature dimension
    sample_h5 = None
    for h5_file in tcga_feats_dir.glob('*.h5'):
        sample_h5 = h5_file
        break
    if sample_h5 is None:
        for h5_file in gtex_feats_dir.glob('*.h5'):
            sample_h5 = h5_file
            break
    
    with h5py.File(sample_h5, 'r') as f:
        input_dim = f['features'].shape[1]
    print(f"Feature dimension: {input_dim}")
    
    # Combine slide IDs
    all_ids = tcga_slide_ids + gtex_slide_ids
    all_labels = [labels[s] for s in all_ids]
    
    # Create splits
    train_ids, temp_ids, train_lbls, temp_lbls = train_test_split(
        all_ids, all_labels, test_size=0.3, stratify=all_labels, random_state=SEED
    )
    val_ids, test_ids, _, _ = train_test_split(
        temp_ids, temp_lbls, test_size=0.5, stratify=temp_lbls, random_state=SEED
    )
    
    # Split back into TCGA and GTEx
    tcga_set = set(tcga_slide_ids)
    gtex_set = set(gtex_slide_ids)
    
    train_tcga = [s for s in train_ids if s in tcga_set]
    train_gtex = [s for s in train_ids if s in gtex_set]
    val_tcga = [s for s in val_ids if s in tcga_set]
    val_gtex = [s for s in val_ids if s in gtex_set]
    test_tcga = [s for s in test_ids if s in tcga_set]
    test_gtex = [s for s in test_ids if s in gtex_set]
    
    print(f"\nData split:")
    print(f"  Train: {len(train_ids)} (TCGA: {len(train_tcga)}, GTEx: {len(train_gtex)})")
    print(f"  Val:   {len(val_ids)} (TCGA: {len(val_tcga)}, GTEx: {len(val_gtex)})")
    print(f"  Test:  {len(test_ids)} (TCGA: {len(test_tcga)}, GTEx: {len(test_gtex)})")
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = MultiSourceSupervisedDataset(
        tcga_feats_dir, gtex_feats_dir, train_tcga, train_gtex,
        labels, annotations_dir, PATCH_SIZE, max_patches, subsample=True
    )
    val_dataset = MultiSourceSupervisedDataset(
        tcga_feats_dir, gtex_feats_dir, val_tcga, val_gtex,
        labels, annotations_dir, PATCH_SIZE, max_patches, subsample=False
    )
    test_dataset = MultiSourceSupervisedDataset(
        tcga_feats_dir, gtex_feats_dir, test_tcga, test_gtex,
        labels, annotations_dir, PATCH_SIZE, max_patches, subsample=False
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
        num_classes=1,
        dropout=dropout
    ).to(device)
    
    print(f"\nModel: ABMIL with Supervised Attention")
    print(f"  Input dim: {input_dim}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Attention lambda: {attention_lambda}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Class weights
    train_lbls = [labels[s] for s in train_ids]
    n_pos = sum(train_lbls)
    n_neg = len(train_lbls) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)]).to(device)
    print(f"\nClass balance: Normal={n_neg}, Cancer={n_pos}")
    print(f"Positive weight: {pos_weight.item():.2f}")
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    
    # Training loop
    best_val_auc = 0
    best_epoch = 0
    history = {'train': [], 'val': []}
    
    print("\n" + "=" * 70)
    print("TRIDENT ABMIL Training with Supervised Attention")
    print("=" * 70)
    
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        print("-" * 50)
        
        cls_loss, att_loss, train_acc, train_bal_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, attention_lambda
        )
        
        val_results = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        
        print(f"Train | Cls Loss: {cls_loss:.4f} | Att Loss: {att_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"Val   | Loss: {val_results['loss']:.4f} | Acc: {val_results['accuracy']:.4f} | "
              f"AUC: {val_results['auc']:.4f} | F1: {val_results['f1']:.4f}")
        
        history['train'].append({
            'epoch': epoch, 'cls_loss': cls_loss, 'att_loss': att_loss,
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
                    'attention_lambda': attention_lambda,
                    'mode': 'cancer_vs_noncancer'
                }
            }, output_dir / 'best_model.pt')
            print(f"  -> New best model! (AUC: {best_val_auc:.4f})")
        
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
    print("TEST RESULTS - Supervised Attention")
    print("=" * 70)
    print(f"Accuracy:          {test_results['accuracy']:.4f}")
    print(f"Balanced Accuracy: {test_results['balanced_accuracy']:.4f}")
    print(f"AUC-ROC:           {test_results['auc']:.4f}")
    print(f"F1 Score:          {test_results['f1']:.4f}")
    print(f"Sensitivity:       {test_results['sensitivity']:.4f}")
    print(f"Specificity:       {test_results['specificity']:.4f}")
    
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
        description='TRIDENT ABMIL Training with Supervised Attention',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--train', action='store_true', help='Train model')
    
    # Paths
    parser.add_argument('--tcga_feats_dir', type=str, default=DEFAULT_FEATS_DIR)
    parser.add_argument('--gtex_feats_dir', type=str, default=GTEX_FEATS_DIR)
    parser.add_argument('--tcga_labels_path', type=str, default=DEFAULT_LABELS_PATH)
    parser.add_argument('--gtex_labels_path', type=str, default=GTEX_LABELS_PATH)
    parser.add_argument('--annotations_dir', type=str, default=DEFAULT_ANNOTATIONS_DIR)
    parser.add_argument('--output_dir', type=str, default='output/supervised_attention')
    
    # Model
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--attention_dim', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.25)
    
    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--attention_lambda', type=float, default=0.5,
                        help='Weight for attention supervision loss')
    parser.add_argument('--max_patches', type=int, default=4096)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda:0')
    
    args = parser.parse_args()
    
    if not args.train:
        parser.print_help()
        return
    
    base_dir = Path(__file__).resolve().parent
    
    print("\n" + "=" * 70)
    print("TRIDENT ABMIL with Supervised Attention")
    print("=" * 70)
    
    # Load TCGA labels (cancer)
    tcga_labels_path = Path(args.tcga_labels_path)
    if tcga_labels_path.exists():
        tcga_df = pd.read_csv(tcga_labels_path, sep='\t')
        tcga_slide_ids = tcga_df['slide_id'].tolist()
        labels = {row['slide_id']: 1 for _, row in tcga_df.iterrows()}  # All TCGA = cancer
        print(f"Loaded {len(tcga_slide_ids)} TCGA slides (cancer)")
    else:
        print(f"Error: TCGA labels not found: {tcga_labels_path}")
        return
    
    # Load GTEx labels (benign/cancer, discard = -1)
    gtex_labels_path = Path(args.gtex_labels_path)
    if gtex_labels_path.exists():
        try:
            gtex_slide_ids, gtex_labels = load_gtex_labels_csv(gtex_labels_path)
            labels.update(gtex_labels)
            n_benign = sum(1 for v in gtex_labels.values() if v == 0)
            n_cancer = sum(1 for v in gtex_labels.values() if v == 1)
            print(f"Loaded {len(gtex_slide_ids)} GTEx slides (benign/cancer, discard excluded)")
            print(f"  GTEx benign: {n_benign}, GTEx cancer: {n_cancer}")
        except Exception as exc:
            print(f"Error reading GTEx labels: {exc}")
            return
    else:
        print(f"Warning: GTEx labels not found, training on TCGA only")
        gtex_slide_ids = []
    
    # Output directory
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = base_dir / output_dir
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"supervised_attention_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(run_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Train
    train_supervised_attention(
        tcga_feats_dir=Path(args.tcga_feats_dir),
        gtex_feats_dir=Path(args.gtex_feats_dir),
        annotations_dir=Path(args.annotations_dir),
        output_dir=run_dir,
        tcga_slide_ids=tcga_slide_ids,
        gtex_slide_ids=gtex_slide_ids,
        labels=labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        attention_lambda=args.attention_lambda,
        hidden_dim=args.hidden_dim,
        attention_dim=args.attention_dim,
        dropout=args.dropout,
        max_patches=args.max_patches,
        device=args.device,
        num_workers=args.num_workers
    )


if __name__ == "__main__":
    main()
