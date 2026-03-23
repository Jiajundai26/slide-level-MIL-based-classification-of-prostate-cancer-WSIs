#!/usr/bin/env python3
"""
TRIDENT-based 4-Class Gleason Classification Inference Script

Runs inference on WSI feature files using a trained ABMIL 4-class classifier.
Supports attention heatmap generation and WSI overlay visualization.

4-Class Classification:
    - Class 0: Normal (non-cancer)
    - Class 1: G3 (Gleason Pattern 3 dominant - low-grade)
    - Class 2: G4 (Gleason Pattern 4 dominant - intermediate)
    - Class 3: G5 (Gleason Pattern 5 dominant - high-grade)

Usage:
    # Basic inference on all TCGA-PRAD features
    python inference_4class_trident.py \\
        --checkpoint /path/to/4class_model/best_model.pt

    # Inference with attention heatmaps
    python inference_4class_trident.py \\
        --checkpoint /path/to/4class_model/best_model.pt \\
        --save_attention \\
        --generate_heatmaps

    # Inference on specific slides
    python inference_4class_trident.py \\
        --checkpoint /path/to/4class_model/best_model.pt \\
        --slide_ids TCGA-XX-XXXX,TCGA-YY-YYYY

    # Evaluate against ground truth labels
    python inference_4class_trident.py \\
        --checkpoint /path/to/4class_model/best_model.pt \\
        --labels_path labels/slide_labels.tsv
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
import h5py
import pandas as pd
from tqdm import tqdm

# Optional imports for heatmap generation
try:
    import openslide
    HAS_OPENSLIDE = True
except ImportError:
    HAS_OPENSLIDE = False

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.cm import ScalarMappable
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    from scipy.ndimage import gaussian_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ============================================================================
# Configuration
# ============================================================================

# Default paths for TCGA-PRAD
TCGA_FEATS_DIR = "/sci-it/projects/cedmav/data/ARPA-H/Prostate_HnE/preprocessing/TCGA-PRAD/20x_256px_0px_overlap/features_uni_v2"
TCGA_WSI_DIR = "/local/data/magicscan/HnE/TCGA-PRAD_organize/WSI"
TCGA_LABELS_PATH = "/sci-it/projects/cedmav/data/ARPA-H/Prostate_HnE/preprocessing/TCGA-PRAD/labels_MIL/slide_labels.tsv"

# Default paths for GTEx_Prostate
GTEX_FEATS_DIR = "/sci-it/projects/cedmav/data/ARPA-H/Prostate_HnE/preprocessing/GTEx_Prostate/20x_256px_0px_overlap/features_uni_v2"
GTEX_WSI_DIR = "/local/data/magicscan/HnE/GTEx_prostate/histology_images_prostate"
GTEX_LABELS_PATH = "/sci-it/projects/cedmav/data/ARPA-H/Prostate_HnE/preprocessing/GTEx_Prostate/GTEx_prostate_labels.csv"

# Default checkpoint
DEFAULT_CHECKPOINT = "/local/data/magicscan/HnE/GTEx_prostate/output/trident_4class/4class_focal_20260123_001634/best_model.pt"

# 4-class labels
NUM_CLASSES = 4
CLASS_NAMES = {
    0: 'Normal',
    1: 'G3 (Low-grade)',
    2: 'G4 (Intermediate)',
    3: 'G5 (High-grade)'
}

# Short class names for display
CLASS_NAMES_SHORT = {
    0: 'Normal',
    1: 'G3',
    2: 'G4',
    3: 'G5'
}


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
    raise ValueError("Could not determine slide id column in GTEx labels file.")


def _normalize_class_names(raw: dict) -> dict:
    """Normalize class_names from config.json (string keys) to int keys."""
    normalized = {}
    for k, v in raw.items():
        try:
            idx = int(k)
        except (TypeError, ValueError):
            continue
        normalized[idx] = str(v)
    return normalized

# TCGA label to 4-class mapping (TCGA uses 1=G3, 2=G4, 3=G5)
TCGA_LABEL_TO_4CLASS = {1: 1, 2: 2, 3: 3}  # G3->1, G4->2, G5->3


# ============================================================================
# Model Architecture (same as training script)
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


class ABMIL_MultiClass(nn.Module):
    """Attention-Based Multiple Instance Learning (ABMIL) Classifier for Multi-class"""
    def __init__(
        self,
        input_dim: int = 1536,
        hidden_dim: int = 256,
        attention_dim: int = 128,
        num_classes: int = 4,
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
# Model Loading
# ============================================================================

def load_model(checkpoint_path: str, device: torch.device) -> Tuple[ABMIL_MultiClass, dict]:
    """
    Load trained 4-class ABMIL model from checkpoint.
    
    Args:
        checkpoint_path: Path to best_model.pt
        device: Device to load model on
        
    Returns:
        model: Loaded ABMIL model in eval mode
        config: Model configuration dict
    """
    checkpoint_path = Path(checkpoint_path)
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get config from checkpoint or config.json
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        config = {}
    config_path = checkpoint_path.parent / 'config.json'
    if config_path.exists():
        with open(config_path, 'r') as f:
            file_config = json.load(f)
        # Backfill any missing fields from config.json
        for k, v in file_config.items():
            config.setdefault(k, v)
    if not config:
        print("Warning: No config found, using defaults")
        config = {
            'input_dim': 1536,
            'hidden_dim': 256,
            'attention_dim': 128,
            'num_classes': 4
        }
    
    # Initialize model
    model = ABMIL_MultiClass(
        input_dim=config.get('input_dim', 1536),
        hidden_dim=config.get('hidden_dim', 256),
        attention_dim=config.get('attention_dim', 128),
        num_classes=config.get('num_classes', NUM_CLASSES),
        dropout=0.0  # No dropout during inference
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Print info
    epoch = checkpoint.get('epoch', 'unknown')
    val_bal_acc = checkpoint.get('val_balanced_accuracy', 'N/A')
    val_macro_f1 = checkpoint.get('val_macro_f1', 'N/A')
    
    print(f"  Model loaded from epoch {epoch}")
    print(f"  Validation Balanced Accuracy: {val_bal_acc}")
    print(f"  Validation Macro F1: {val_macro_f1}")
    print(f"  Input dim: {config.get('input_dim', 1536)}")
    print(f"  Hidden dim: {config.get('hidden_dim', 256)}")
    print(f"  Num classes: {config.get('num_classes', NUM_CLASSES)}")
    
    return model, config


def load_features(h5_path: Path, max_patches: Optional[int] = None) -> Tuple[torch.Tensor, np.ndarray, dict]:
    """
    Load features and coordinates from H5 file.
    
    Args:
        h5_path: Path to H5 feature file
        max_patches: Maximum patches to load (None = all)
        
    Returns:
        features: Tensor of shape [num_patches, feature_dim]
        coords: Array of patch coordinates [num_patches, 2]
        attrs: Additional metadata from H5 file
    """
    with h5py.File(h5_path, 'r') as f:
        features = torch.from_numpy(f['features'][:]).float()
        
        if 'coords' in f:
            coords = f['coords'][:]
        else:
            coords = None
            
        attrs = {}
        for key in f.attrs:
            attrs[key] = f.attrs[key]
        
        if 'coords' in f:
            for key in f['coords'].attrs:
                attrs[key] = f['coords'].attrs[key]
    
    if max_patches and features.shape[0] > max_patches:
        features = features[:max_patches]
        if coords is not None:
            coords = coords[:max_patches]
    
    return features, coords, attrs


# ============================================================================
# Inference Functions
# ============================================================================

@torch.no_grad()
def predict_single(
    model: ABMIL_MultiClass,
    h5_path: Path,
    device: torch.device,
    max_patches: Optional[int] = None,
    return_attention: bool = False
) -> dict:
    """
    Run 4-class inference on a single slide.
    
    Args:
        model: Trained ABMIL model
        h5_path: Path to H5 feature file
        device: Computation device
        max_patches: Maximum patches to use
        return_attention: Whether to return attention weights
        
    Returns:
        result: Dictionary with prediction results
    """
    # Load features
    features, coords, attrs = load_features(h5_path, max_patches)
    features = features.unsqueeze(0).to(device)
    
    # Run model
    if return_attention:
        logits, attention = model(features, return_attention=True)
        attention = attention.squeeze(0).cpu().numpy()
    else:
        logits = model(features)
        attention = None
    
    # Get prediction probabilities
    probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    pred_class = int(logits.argmax(dim=1).cpu().item())
    confidence = float(probs[pred_class])
    
    result = {
        'slide_id': h5_path.stem,
        'predicted_class': pred_class,
        'predicted_name': CLASS_NAMES[pred_class],
        'predicted_short': CLASS_NAMES_SHORT[pred_class],
        'confidence': confidence,
        'prob_normal': float(probs[0]),
        'prob_g3': float(probs[1]),
        'prob_g4': float(probs[2]),
        'prob_g5': float(probs[3]),
        'num_patches': features.shape[1]
    }
    
    if return_attention:
        attention_norm = (attention - attention.min()) / (attention.max() - attention.min() + 1e-8)
        result['attention'] = attention
        result['attention_norm'] = attention_norm
        result['coords'] = coords
        result['attrs'] = attrs
    
    return result


def predict_batch(
    model: ABMIL_MultiClass,
    h5_paths: List[Path],
    device: torch.device,
    max_patches: Optional[int] = None,
    return_attention: bool = False,
    output_dir: Optional[Path] = None
) -> List[dict]:
    """
    Run 4-class inference on multiple slides.
    
    Args:
        model: Trained ABMIL model
        h5_paths: List of H5 feature file paths
        device: Computation device
        max_patches: Maximum patches per slide
        return_attention: Whether to save attention weights
        output_dir: Directory to save attention files
        
    Returns:
        results: List of prediction dictionaries
    """
    results = []
    
    for h5_path in tqdm(h5_paths, desc="Running inference"):
        try:
            result = predict_single(
                model, h5_path, device, max_patches, return_attention
            )
            
            # Save attention weights if requested
            if return_attention and output_dir and result.get('attention') is not None:
                attention_dir = output_dir / 'attention'
                attention_dir.mkdir(parents=True, exist_ok=True)
                
                attention_path = attention_dir / f"{result['slide_id']}_attention.npz"
                np.savez_compressed(
                    attention_path,
                    attention=result['attention'],
                    attention_norm=result['attention_norm'],
                    coords=result['coords'],
                    predicted_class=result['predicted_class'],
                    predicted_name=result['predicted_name'],
                    prob_normal=result['prob_normal'],
                    prob_g3=result['prob_g3'],
                    prob_g4=result['prob_g4'],
                    prob_g5=result['prob_g5'],
                    **{k: v for k, v in result.get('attrs', {}).items() if isinstance(v, (int, float, str))}
                )
                
                # Remove from result dict to save memory
                del result['attention']
                del result['attention_norm']
                del result['coords']
                if 'attrs' in result:
                    del result['attrs']
            
            results.append(result)
            
        except Exception as e:
            print(f"Error processing {h5_path.name}: {e}")
            results.append({
                'slide_id': h5_path.stem,
                'error': str(e)
            })
    
    return results


# ============================================================================
# Heatmap Generation
# ============================================================================

def get_4class_colormap(predicted_class: int):
    """Return a class-specific colormap for attention visualization."""
    class_cmaps = {
        0: 'Blues',    # Normal
        1: 'Greens',   # G3
        2: 'Oranges',  # G4
        3: 'Reds'      # G5
    }
    cmap_name = class_cmaps.get(int(predicted_class), 'viridis')
    return plt.get_cmap(cmap_name)


def generate_attention_heatmap(
    attention_path: Path,
    wsi_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    thumbnail_size: int = 2048,
    alpha: float = 0.5,
    cmap: str = 'class',
    gaussian_sigma: float = 10.0,
    smooth: bool = True,
    show_colorbar: bool = True,
    show_prediction: bool = False
) -> Optional[np.ndarray]:
    """
    Generate attention heatmap overlay on WSI thumbnail with colorbar.
    
    Args:
        attention_path: Path to .npz file with attention weights
        wsi_path: Path to WSI file (optional, for overlay)
        output_path: Path to save heatmap image
        thumbnail_size: Maximum dimension for thumbnail
        alpha: Opacity of heatmap overlay
        cmap: Colormap for heatmap ('class' or 'auto' for class-specific palettes)
        gaussian_sigma: Sigma for Gaussian smoothing
        smooth: Whether to apply Gaussian smoothing
        show_colorbar: Whether to add a colorbar
        show_prediction: Whether to show prediction info on image
        
    Returns:
        heatmap: Heatmap image as numpy array (or None if failed)
    """
    if not HAS_PIL or not HAS_MATPLOTLIB:
        print("Warning: PIL and matplotlib required for heatmap generation")
        return None
    
    # Load attention data
    data = np.load(attention_path)
    attention = data['attention_norm']
    coords = data['coords']
    
    # Get prediction info
    predicted_class = int(data.get('predicted_class', -1))
    predicted_name = str(data.get('predicted_name', 'Unknown'))
    predicted_short = CLASS_NAMES_SHORT.get(predicted_class, predicted_name)
    prob_normal = float(data.get('prob_normal', 0))
    prob_g3 = float(data.get('prob_g3', 0))
    prob_g4 = float(data.get('prob_g4', 0))
    prob_g5 = float(data.get('prob_g5', 0))
    
    # Get patch info
    patch_size = int(data.get('patch_size', 256))
    
    # Determine heatmap dimensions
    if coords is not None and len(coords) > 0:
        max_x = coords[:, 0].max() + patch_size
        max_y = coords[:, 1].max() + patch_size
    else:
        print(f"Warning: No coordinates in {attention_path.name}")
        return None
    
    # Calculate scale factor
    scale = min(thumbnail_size / max_x, thumbnail_size / max_y)
    heatmap_width = int(max_x * scale)
    heatmap_height = int(max_y * scale)
    
    # Create heatmap
    heatmap = np.zeros((heatmap_height, heatmap_width), dtype=np.float32)
    count_map = np.zeros((heatmap_height, heatmap_width), dtype=np.float32)
    
    scaled_patch = max(int(patch_size * scale), 1)
    
    for i, (coord, att) in enumerate(zip(coords, attention)):
        x, y = coord
        sx, sy = int(x * scale), int(y * scale)
        ex, ey = min(sx + scaled_patch, heatmap_width), min(sy + scaled_patch, heatmap_height)
        
        heatmap[sy:ey, sx:ex] += att
        count_map[sy:ey, sx:ex] += 1
    
    # Average overlapping regions
    count_map[count_map == 0] = 1
    heatmap = heatmap / count_map
    
    # Apply Gaussian smoothing
    if smooth and HAS_SCIPY and gaussian_sigma > 0:
        mask = (count_map > 0).astype(np.float32)
        heatmap_smooth = gaussian_filter(heatmap, sigma=gaussian_sigma)
        mask_smooth = gaussian_filter(mask, sigma=gaussian_sigma)
        mask_smooth[mask_smooth < 0.01] = 1
        heatmap = heatmap_smooth / mask_smooth
        heatmap = np.clip(heatmap, 0, 1)
    
    # Normalize
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    
    # Apply colormap (optionally class-specific)
    if cmap in ['class', 'auto']:
        colormap = get_4class_colormap(predicted_class)
    else:
        colormap = plt.get_cmap(cmap)
    heatmap_colored = colormap(heatmap)[:, :, :3]
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
    
    # Overlay on WSI thumbnail if available
    if wsi_path and wsi_path.exists() and HAS_OPENSLIDE:
        try:
            slide = openslide.OpenSlide(str(wsi_path))
            thumb = slide.get_thumbnail((heatmap_width, heatmap_height))
            thumb = np.array(thumb.convert('RGB'))
            output = (alpha * heatmap_colored + (1 - alpha) * thumb).astype(np.uint8)
            slide.close()
        except Exception as e:
            print(f"Warning: Could not load WSI for overlay: {e}")
            output = heatmap_colored
    else:
        output = heatmap_colored
    
    # Save with colorbar and prediction info
    if output_path:
        if show_colorbar or show_prediction:
            fig, ax = plt.subplots(1, 1, figsize=(14, 10))
            
            ax.imshow(output)
            ax.axis('off')
            
            # Add colorbar
            if show_colorbar:
                sm = ScalarMappable(cmap=colormap, norm=Normalize(vmin=0, vmax=1))
                sm.set_array([])
                cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, aspect=30)
                cbar.set_ticks([0.0, 1.0])
                cbar.set_ticklabels([
                    f'Low\n{predicted_short}',
                    f'High\n{predicted_short}'
                ])
                cbar.ax.tick_params(labelsize=10)
            
            # Prediction info intentionally omitted from the image title
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            plt.close(fig)
        else:
            Image.fromarray(output).save(output_path)
    
    return output


def generate_heatmaps_batch(
    attention_dir: Path,
    wsi_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    thumbnail_size: int = 2048,
    alpha: float = 0.5,
    cmap: str = 'class',
    gaussian_sigma: float = 10.0,
    smooth: bool = True,
    show_colorbar: bool = True,
    show_prediction: bool = True
):
    """Generate heatmaps for all attention files in a directory."""
    if not HAS_PIL or not HAS_MATPLOTLIB:
        print("Error: PIL and matplotlib required for heatmap generation")
        return
    
    attention_files = list(attention_dir.glob('*_attention.npz'))
    print(f"Found {len(attention_files)} attention files")
    
    if output_dir is None:
        output_dir = attention_dir.parent / 'heatmaps'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for att_path in tqdm(attention_files, desc="Generating heatmaps"):
        slide_id = att_path.stem.replace('_attention', '')
        
        # Find WSI
        wsi_path = None
        if wsi_dir:
            for ext in ['.svs', '.ndpi', '.tif', '.tiff', '.mrxs']:
                candidate = wsi_dir / f"{slide_id}{ext}"
                if candidate.exists():
                    wsi_path = candidate
                    break
        
        output_path = output_dir / f"{slide_id}_heatmap.png"
        
        try:
            generate_attention_heatmap(
                att_path, wsi_path, output_path,
                thumbnail_size, alpha, cmap,
                gaussian_sigma, smooth,
                show_colorbar, show_prediction
            )
        except Exception as e:
            print(f"Error generating heatmap for {slide_id}: {e}")


# ============================================================================
# Evaluation Functions
# ============================================================================

def load_gtex_benign_labels(labels_path: Path) -> Dict[str, int]:
    """
    Load GTEx labels and keep only benign/normal (label == 0).
    Supports CSV (GTEx_prostate_labels.csv) and TSV with slide_id/label.
    """
    if labels_path.suffix.lower() == ".csv":
        df = pd.read_csv(labels_path)
    else:
        df = pd.read_csv(labels_path, sep="\t")
    if "label" not in [c.lower() for c in df.columns]:
        raise ValueError("GTEx labels file must contain a 'label' column.")
    slide_col = _pick_slide_id_column(df)
    label_col = None
    for col in df.columns:
        if col.lower() == "label":
            label_col = col
            break
    df = df[[slide_col, label_col]].rename(columns={slide_col: "slide_id", label_col: "label"})
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df[df["label"] == 0]
    return {sid: 0 for sid in df["slide_id"].astype(str).tolist()}

def evaluate_predictions(
    predictions_df: pd.DataFrame,
    labels_df: pd.DataFrame
) -> dict:
    """
    Evaluate predictions against ground truth.
    
    Args:
        predictions_df: DataFrame with predictions
        labels_df: DataFrame with ground truth labels
        
    Returns:
        metrics: Dictionary with evaluation metrics
    """
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score,
        precision_recall_fscore_support, confusion_matrix,
        f1_score, roc_auc_score
    )
    
    # Merge predictions with labels
    merged = predictions_df.merge(
        labels_df[['slide_id', 'label']],
        on='slide_id',
        how='inner'
    )
    
    if len(merged) == 0:
        print("Warning: No matching slides for evaluation")
        return {}
    
    # Map TCGA labels (1=G3, 2=G4, 3=G5) to our 4-class scheme
    # Note: TCGA doesn't have Normal class, so we only evaluate on cancer classes
    merged['true_class'] = merged['label'].map(TCGA_LABEL_TO_4CLASS)
    
    y_true = merged['true_class'].values
    y_pred = merged['predicted_class'].values
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    
    # Per-class metrics (only for classes present in TCGA: 1, 2, 3)
    present_classes = sorted(set(y_true) | set(y_pred))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=present_classes, average=None, zero_division=0
    )
    
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=present_classes)
    
    # Multi-class AUC (if we have probabilities)
    if 'prob_g3' in merged.columns:
        # Get probabilities for classes present in the data (TCGA: G3/G4/G5)
        probs = merged[['prob_g3', 'prob_g4', 'prob_g5']].values
        try:
            auc = roc_auc_score(y_true, probs, multi_class='ovr', average='macro', labels=[1, 2, 3])
        except ValueError:
            auc = None
    else:
        auc = None
    
    metrics = {
        'num_evaluated': len(merged),
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'auc_macro': auc,
        'per_class': {},
        'confusion_matrix': cm.tolist(),
        'present_classes': present_classes
    }
    
    for i, c in enumerate(present_classes):
        metrics['per_class'][CLASS_NAMES[c]] = {
            'precision': precision[i],
            'recall': recall[i],
            'f1': f1[i],
            'support': int(support[i])
        }
    
    return metrics


def plot_roc_curves(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    classes: List[int],
    output_path: Path,
    title: str = "ROC Curves",
    class_ids: Optional[List[int]] = None
):
    """
    Plot ROC curves for multi-class classification (One-vs-Rest).
    
    Args:
        y_true: True labels [n_samples]
        y_probs: Predicted probabilities [n_samples, n_classes]
        classes: List of class indices present in the data
        output_path: Path to save the plot
        title: Plot title
    """
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    
    # Binarize the labels for OvR
    n_classes = y_probs.shape[1]
    if class_ids is None:
        class_ids = list(range(n_classes))
    y_true_bin = label_binarize(y_true, classes=class_ids)
    
    # If binary (only 2 classes in data), expand to 2D
    if y_true_bin.ndim == 1:
        y_true_bin = np.column_stack([1 - y_true_bin, y_true_bin])
    
    # Colors for each class
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']  # Green, Blue, Red, Purple
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Compute ROC curve and AUC for each class
    fpr = {}
    tpr = {}
    roc_auc = {}
    
    for i, c in enumerate(classes):
        if c < y_true_bin.shape[1] and c < y_probs.shape[1]:
            class_id = class_ids[c]
            fpr[c], tpr[c], _ = roc_curve(y_true_bin[:, c], y_probs[:, c])
            roc_auc[c] = auc(fpr[c], tpr[c])
            
            ax.plot(
                fpr[c], tpr[c],
                color=colors[class_id % len(colors)],
                lw=2,
                label=f'{CLASS_NAMES[class_id]} (AUC = {roc_auc[c]:.3f})'
            )
    
    # Compute micro-average ROC curve
    relevant_classes = [c for c in classes if c < n_classes]
    if len(relevant_classes) > 1:
        y_true_micro = y_true_bin[:, relevant_classes].ravel()
        y_prob_micro = y_probs[:, relevant_classes].ravel()
        fpr_micro, tpr_micro, _ = roc_curve(y_true_micro, y_prob_micro)
        roc_auc_micro = auc(fpr_micro, tpr_micro)
        
        ax.plot(
            fpr_micro, tpr_micro,
            color='navy',
            lw=2,
            linestyle='--',
            label=f'Micro-average (AUC = {roc_auc_micro:.3f})'
        )
    
    # Compute macro-average ROC curve
    if len(relevant_classes) > 1:
        all_fpr = np.unique(np.concatenate([fpr[c] for c in relevant_classes]))
        mean_tpr = np.zeros_like(all_fpr)
        for c in relevant_classes:
            mean_tpr += np.interp(all_fpr, fpr[c], tpr[c])
        mean_tpr /= len(relevant_classes)
        
        fpr_macro = all_fpr
        tpr_macro = mean_tpr
        roc_auc_macro = auc(fpr_macro, tpr_macro)
        
        ax.plot(
            fpr_macro, tpr_macro,
            color='darkorange',
            lw=2,
            linestyle='--',
            label=f'Macro-average (AUC = {roc_auc_macro:.3f})'
        )
    
    # Plot diagonal (random classifier)
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random (AUC = 0.500)')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add AUC values as text box
    auc_text = "Per-class AUC:\n"
    for c in classes:
        if c in roc_auc:
            auc_text += f"  {CLASS_NAMES[class_ids[c]]}: {roc_auc[c]:.3f}\n"
    
    ax.text(0.02, 0.98, auc_text.strip(), transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_confusion_matrix(
    conf_matrix: np.ndarray,
    output_path: Path,
    class_labels: List[str],
    title: str = "Confusion Matrix"
):
    """Plot and save confusion matrix."""
    import seaborn as sns
    
    plt.figure(figsize=(10, 8))
    
    # Normalize
    conf_matrix_norm = conf_matrix.astype('float') / (conf_matrix.sum(axis=1, keepdims=True) + 1e-8)
    
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


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='TRIDENT ABMIL 4-Class Gleason Classification Inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Inference on combined dataset (TCGA-PRAD + GTEx)
    python inference_4class_trident.py --dataset combined

    # Inference on TCGA-PRAD only
    python inference_4class_trident.py --dataset tcga

    # Inference on GTEx only
    python inference_4class_trident.py --dataset gtex

    # With attention heatmaps
    python inference_4class_trident.py --dataset combined --save_attention --generate_heatmaps

    # Evaluate against ground truth
    python inference_4class_trident.py --dataset combined --evaluate

Classes:
    0: Normal (benign-only GTEx prostate tissue)
    1: G3 (Gleason Pattern 3 dominant - low-grade)
    2: G4 (Gleason Pattern 4 dominant - intermediate)
    3: G5 (Gleason Pattern 5 dominant - high-grade)
        """
    )
    
    # Model
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT,
                        help='Path to model checkpoint (best_model.pt)')
    
    # Dataset selection
    parser.add_argument('--dataset', type=str, default='combined',
                        choices=['tcga', 'gtex', 'combined'],
                        help='Dataset to run inference on: tcga, gtex, or combined (default: combined)')
    
    # Input paths (override defaults if provided)
    parser.add_argument('--tcga_feats_dir', type=str, default=TCGA_FEATS_DIR,
                        help='Directory containing TCGA-PRAD H5 feature files')
    parser.add_argument('--gtex_feats_dir', type=str, default=GTEX_FEATS_DIR,
                        help='Directory containing GTEx H5 feature files')
    parser.add_argument('--slide_ids', type=str, default=None,
                        help='Comma-separated list of slide IDs (optional)')
    parser.add_argument('--slide_list', type=str, default=None,
                        help='Path to file with slide IDs (one per line)')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='output/4class_inference',
                        help='Output directory for predictions')
    
    # Model options
    parser.add_argument('--max_patches', type=int, default=None,
                        help='Maximum patches per slide (None = use all)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use (cuda:0, cpu)')
    
    # Attention/Heatmap options
    parser.add_argument('--save_attention', action='store_true',
                        help='Save attention weights for each slide')
    parser.add_argument('--generate_heatmaps', action='store_true',
                        help='Generate attention heatmap images')
    parser.add_argument('--tcga_wsi_dir', type=str, default=TCGA_WSI_DIR,
                        help='Directory containing TCGA WSI files')
    parser.add_argument('--gtex_wsi_dir', type=str, default=GTEX_WSI_DIR,
                        help='Directory containing GTEx WSI files')
    parser.add_argument('--thumbnail_size', type=int, default=2048,
                        help='Maximum dimension for heatmap thumbnails')
    parser.add_argument('--heatmap_alpha', type=float, default=0.5,
                        help='Opacity of heatmap overlay')
    parser.add_argument('--heatmap_cmap', type=str, default='class',
                        help="Colormap for heatmaps ('class' for class-specific palettes)")
    parser.add_argument('--gaussian_sigma', type=float, default=10.0,
                        help='Sigma for Gaussian smoothing')
    parser.add_argument('--no_smooth', action='store_true',
                        help='Disable Gaussian smoothing')
    parser.add_argument('--no_colorbar', action='store_true',
                        help='Disable colorbar on heatmaps')
    parser.add_argument('--no_prediction_label', action='store_true',
                        help='Disable prediction label on heatmaps (title already omitted)')
    
    # Evaluation
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate predictions against ground truth labels')
    parser.add_argument('--tcga_labels_path', type=str, default=TCGA_LABELS_PATH,
                        help='Path to TCGA-PRAD labels TSV')
    parser.add_argument('--gtex_labels_path', type=str, default=GTEX_LABELS_PATH,
                        help='Path to GTEx labels CSV/TSV')
    
    args = parser.parse_args()
    
    # Setup
    base_dir = Path(__file__).resolve().parent
    checkpoint_path = Path(args.checkpoint)
    
    if not checkpoint_path.is_absolute():
        checkpoint_path = base_dir / checkpoint_path
    
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = base_dir / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Device
    device = args.device
    if device.startswith('cuda') and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    device = torch.device(device)
    print(f"Device: {device}")
    
    # Load model
    print("\n" + "=" * 70)
    print("Loading 4-Class Model")
    print("=" * 70)
    model, config = load_model(checkpoint_path, device)
    
    # Update class names from config.json if available
    if isinstance(config, dict) and 'class_names' in config:
        updated = _normalize_class_names(config['class_names'])
        if updated:
            CLASS_NAMES.update(updated)
            for k, v in updated.items():
                # Keep short names consistent
                CLASS_NAMES_SHORT[k] = str(v).split(' ')[0]
    
    # Collect feature paths and labels based on dataset selection
    print("\n" + "=" * 70)
    print(f"Dataset: {args.dataset.upper()}")
    print("=" * 70)
    
    h5_paths = []
    slide_to_source = {}  # Track which dataset each slide comes from
    ground_truth_labels = {}  # For evaluation
    wsi_dirs = {}  # For heatmap generation
    
    # Handle specific slide IDs if provided
    if args.slide_ids:
        requested_ids = [s.strip() for s in args.slide_ids.split(',')]
    elif args.slide_list:
        with open(args.slide_list, 'r') as f:
            requested_ids = [line.strip() for line in f if line.strip()]
    else:
        requested_ids = None
    
    # Load TCGA-PRAD data
    if args.dataset in ['tcga', 'combined']:
        tcga_feats_dir = Path(args.tcga_feats_dir)
        if tcga_feats_dir.exists():
            tcga_h5_files = sorted(tcga_feats_dir.glob('*.h5'))
            
            if requested_ids:
                tcga_h5_files = [f for f in tcga_h5_files if f.stem in requested_ids]
            
            for h5_path in tcga_h5_files:
                h5_paths.append(h5_path)
                slide_to_source[h5_path.stem] = 'TCGA-PRAD'
            
            print(f"TCGA-PRAD: {len(tcga_h5_files)} slides")
            wsi_dirs['TCGA-PRAD'] = Path(args.tcga_wsi_dir)
            
            # Load TCGA labels for evaluation
            if args.evaluate:
                tcga_labels_path = Path(args.tcga_labels_path)
                if tcga_labels_path.exists():
                    tcga_df = pd.read_csv(tcga_labels_path, sep='\t')
                    for _, row in tcga_df.iterrows():
                        # TCGA labels: 1=G3, 2=G4, 3=G5 -> map to our 4-class: 1, 2, 3
                        ground_truth_labels[row['slide_id']] = TCGA_LABEL_TO_4CLASS.get(row['label'], row['label'])
        else:
            print(f"Warning: TCGA features directory not found: {tcga_feats_dir}")
    
    # Load GTEx data
    if args.dataset in ['gtex', 'combined']:
        gtex_feats_dir = Path(args.gtex_feats_dir)
        if gtex_feats_dir.exists():
            gtex_h5_files = sorted(gtex_feats_dir.glob('*.h5'))
            
            if requested_ids:
                gtex_h5_files = [f for f in gtex_h5_files if f.stem in requested_ids]
            
            for h5_path in gtex_h5_files:
                h5_paths.append(h5_path)
                slide_to_source[h5_path.stem] = 'GTEx'
            
            print(f"GTEx Prostate: {len(gtex_h5_files)} slides")
            wsi_dirs['GTEx'] = Path(args.gtex_wsi_dir)
            
            # Load GTEx labels for evaluation (benign-only; cancer/discard excluded)
            if args.evaluate:
                gtex_labels_path = Path(args.gtex_labels_path)
                if gtex_labels_path.exists():
                    gtex_labels = load_gtex_benign_labels(gtex_labels_path)
                    ground_truth_labels.update(gtex_labels)
        else:
            print(f"Warning: GTEx features directory not found: {gtex_feats_dir}")
    
    print(f"\nTotal slides: {len(h5_paths)}")
    
    if len(h5_paths) == 0:
        print("No H5 files found!")
        return
    
    # Run inference
    print("\n" + "=" * 70)
    print("Running 4-Class Inference")
    print("=" * 70)
    
    results = predict_batch(
        model=model,
        h5_paths=h5_paths,
        device=device,
        max_patches=args.max_patches,
        return_attention=args.save_attention,
        output_dir=output_dir
    )
    
    # Create results DataFrame
    df_results = pd.DataFrame([
        {
            'slide_id': r.get('slide_id'),
            'source': slide_to_source.get(r.get('slide_id'), 'Unknown'),
            'predicted_class': r.get('predicted_class'),
            'predicted_name': r.get('predicted_name'),
            'predicted_short': r.get('predicted_short'),
            'confidence': r.get('confidence'),
            'prob_normal': r.get('prob_normal'),
            'prob_g3': r.get('prob_g3'),
            'prob_g4': r.get('prob_g4'),
            'prob_g5': r.get('prob_g5'),
            'num_patches': r.get('num_patches'),
            'error': r.get('error')
        }
        for r in results
    ])
    
    # Print summary
    print("\n" + "=" * 70)
    print("INFERENCE RESULTS")
    print("=" * 70)
    
    valid_results = df_results[df_results['error'].isna()]
    if len(valid_results) > 0:
        print(f"\nProcessed {len(valid_results)} slides successfully")
        
        # Per-source breakdown
        print(f"\nBy source:")
        for source in valid_results['source'].unique():
            source_df = valid_results[valid_results['source'] == source]
            print(f"  {source}: {len(source_df)} slides")
        
        print(f"\nPrediction distribution:")
        print(valid_results['predicted_name'].value_counts())
        
        # Per-source prediction breakdown
        if args.dataset == 'combined':
            print(f"\nPredictions by source:")
            for source in valid_results['source'].unique():
                source_df = valid_results[valid_results['source'] == source]
                print(f"\n  {source}:")
                for pred_name, count in source_df['predicted_name'].value_counts().items():
                    pct = 100 * count / len(source_df)
                    print(f"    {pred_name}: {count} ({pct:.1f}%)")
        
        print(f"\nMean confidence: {valid_results['confidence'].mean():.4f}")
        print(f"\nClass probability means:")
        print(f"  P(Normal): {valid_results['prob_normal'].mean():.4f}")
        print(f"  P(G3):     {valid_results['prob_g3'].mean():.4f}")
        print(f"  P(G4):     {valid_results['prob_g4'].mean():.4f}")
        print(f"  P(G5):     {valid_results['prob_g5'].mean():.4f}")
    
    if df_results['error'].notna().any():
        n_errors = df_results['error'].notna().sum()
        print(f"\n{n_errors} slides had errors")
    
    # Evaluate against ground truth if requested
    if args.evaluate and ground_truth_labels:
        print("\n" + "=" * 70)
        print(f"EVALUATION METRICS - {args.dataset.upper()}")
        print("=" * 70)
        
        # Add ground truth to results
        valid_results['true_class'] = valid_results['slide_id'].map(ground_truth_labels)
        eval_df = valid_results.dropna(subset=['true_class'])
        eval_df['true_class'] = eval_df['true_class'].astype(int)
        
        if len(eval_df) > 0:
            from sklearn.metrics import (
                accuracy_score, balanced_accuracy_score,
                precision_recall_fscore_support, confusion_matrix,
                f1_score, roc_auc_score
            )
            
            y_true = eval_df['true_class'].values
            y_pred = eval_df['predicted_class'].values
            
            # Get probabilities for AUC/ROC calculation (use cancer classes only for TCGA)
            probs = eval_df[['prob_normal', 'prob_g3', 'prob_g4', 'prob_g5']].values
            
            # Basic metrics
            accuracy = accuracy_score(y_true, y_pred)
            balanced_acc = balanced_accuracy_score(y_true, y_pred)
            macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
            weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            print(f"\nSlides evaluated: {len(eval_df)}")
            print(f"  TCGA-PRAD: {len(eval_df[eval_df['source'] == 'TCGA-PRAD'])}")
            print(f"  GTEx: {len(eval_df[eval_df['source'] == 'GTEx'])}")
            print(f"\nAccuracy: {accuracy:.4f}")
            print(f"Balanced Accuracy: {balanced_acc:.4f}")
            print(f"Macro F1: {macro_f1:.4f}")
            print(f"Weighted F1: {weighted_f1:.4f}")
            
            # Multi-class AUC
            try:
                present_classes = sorted(set(y_true) | set(y_pred))
                if len(present_classes) > 1:
                    if args.dataset == 'tcga':
                        probs_auc = probs[:, [1, 2, 3]]
                        auc = roc_auc_score(
                            y_true, probs_auc, multi_class='ovr',
                            average='macro', labels=[1, 2, 3]
                        )
                    else:
                        probs_auc = probs
                        auc = roc_auc_score(y_true, probs_auc, multi_class='ovr', average='macro')
                    print(f"Macro AUC: {auc:.4f}")
            except Exception as e:
                print(f"AUC: Could not compute ({e})")
            
            # Per-class metrics
            present_classes = [int(c) for c in sorted(set(y_true) | set(y_pred))]
            precision, recall, f1, support = precision_recall_fscore_support(
                y_true, y_pred, labels=present_classes, average=None, zero_division=0
            )
            
            print("\nPer-class Metrics:")
            print(f"{'Class':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
            print("-" * 60)
            for i, c in enumerate(present_classes):
                print(f"{CLASS_NAMES[c]:<20} {precision[i]:>10.4f} {recall[i]:>10.4f} "
                      f"{f1[i]:>10.4f} {int(support[i]):>10}")
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred, labels=present_classes)
            class_labels = [CLASS_NAMES[c] for c in present_classes]
            
            print(f"\nConfusion Matrix:")
            print(f"{'Actual \\ Predicted':<20}", end='')
            for label in class_labels:
                print(f"{label:>15}", end='')
            print()
            for i, label in enumerate(class_labels):
                print(f"{label:<20}", end='')
                for j in range(len(class_labels)):
                    print(f"{cm[i, j]:>15}", end='')
                print()
            
            # Save confusion matrix plot
            try:
                plot_confusion_matrix(
                    cm, output_dir / 'confusion_matrix.png',
                    class_labels, f"4-Class Gleason Classification - {args.dataset.upper()}"
                )
                print(f"\nConfusion matrix saved to: {output_dir / 'confusion_matrix.png'}")
            except Exception as e:
                print(f"Warning: Could not save confusion matrix plot: {e}")
            
            # Save metrics
            # Convert all numpy types to native Python types for JSON serialization
            metrics = {
                'dataset': str(args.dataset),
                'num_evaluated': int(len(eval_df)),
                'num_tcga': int(len(eval_df[eval_df['source'] == 'TCGA-PRAD'])),
                'num_gtex': int(len(eval_df[eval_df['source'] == 'GTEx'])),
                'accuracy': float(accuracy),
                'balanced_accuracy': float(balanced_acc),
                'macro_f1': float(macro_f1),
                'weighted_f1': float(weighted_f1),
                'per_class': {
                    str(CLASS_NAMES[c]): {
                        'precision': float(precision[i]),
                        'recall': float(recall[i]),
                        'f1': float(f1[i]),
                        'support': int(support[i])
                    }
                    for i, c in enumerate(present_classes)
                },
                'confusion_matrix': [[int(x) for x in row] for row in cm],
                'present_classes': present_classes  # Already converted to int above
            }
            
            with open(output_dir / 'evaluation_metrics.json', 'w') as f:
                json.dump(metrics, f, indent=2)
            
            # Plot ROC curves
            try:
                if args.dataset == 'tcga':
                    probs_plot = probs[:, [1, 2, 3]]
                    class_ids = [1, 2, 3]
                    classes = list(range(len(class_ids)))
                    plot_roc_curves(
                        y_true, probs_plot, classes,
                        output_dir / 'roc_curves.png',
                        title=f"ROC Curves - 4-Class Gleason Classification ({args.dataset.upper()})",
                        class_ids=class_ids
                    )
                else:
                    plot_roc_curves(
                        y_true, probs, present_classes,
                        output_dir / 'roc_curves.png',
                        title=f"ROC Curves - 4-Class Gleason Classification ({args.dataset.upper()})"
                    )
                print(f"ROC curves saved to: {output_dir / 'roc_curves.png'}")
            except Exception as e:
                print(f"Warning: Could not plot ROC curves: {e}")
        else:
            print("Warning: No matching slides found for evaluation")
    
    # Save results
    output_csv = output_dir / 'predictions.csv'
    df_results.to_csv(output_csv, index=False)
    print(f"\nResults saved to: {output_csv}")
    
    # Save full results as JSON
    output_json = output_dir / 'predictions.json'
    json_results = []
    for r in results:
        jr = {k: v for k, v in r.items()
              if k not in ['attention', 'attention_norm', 'coords', 'attrs']}
        jr['source'] = slide_to_source.get(r.get('slide_id'), 'Unknown')
        json_results.append(jr)
    
    # Compute summary statistics
    pred_counts = defaultdict(int)
    source_counts = defaultdict(int)
    for jr in json_results:
        if jr.get('predicted_name'):
            pred_counts[jr['predicted_name']] += 1
        source_counts[jr['source']] += 1
    
    with open(output_json, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'checkpoint': str(checkpoint_path),
            'dataset': args.dataset,
            'num_classes': NUM_CLASSES,
            'class_names': CLASS_NAMES,
            'num_slides': len(results),
            'num_by_source': dict(source_counts),
            'prediction_distribution': dict(pred_counts),
            'predictions': json_results
        }, f, indent=2)
    print(f"Full results saved to: {output_json}")
    
    # Generate heatmaps if requested
    if args.generate_heatmaps and args.save_attention:
        print("\n" + "=" * 70)
        print("Generating Heatmaps")
        print("=" * 70)
        
        attention_dir = output_dir / 'attention'
        heatmap_output_dir = output_dir / 'heatmaps'
        heatmap_output_dir.mkdir(parents=True, exist_ok=True)
        
        attention_files = list(attention_dir.glob('*_attention.npz'))
        print(f"Found {len(attention_files)} attention files")
        
        for att_path in tqdm(attention_files, desc="Generating heatmaps"):
            slide_id = att_path.stem.replace('_attention', '')
            source = slide_to_source.get(slide_id, 'Unknown')
            
            # Find WSI in the appropriate directory
            wsi_path = None
            if source in wsi_dirs:
                wsi_dir = wsi_dirs[source]
                if wsi_dir and wsi_dir.exists():
                    for ext in ['.svs', '.ndpi', '.tif', '.tiff', '.mrxs']:
                        candidate = wsi_dir / f"{slide_id}{ext}"
                        if candidate.exists():
                            wsi_path = candidate
                            break
            
            output_path = heatmap_output_dir / f"{slide_id}_heatmap.png"
            
            try:
                generate_attention_heatmap(
                    att_path, wsi_path, output_path,
                    args.thumbnail_size, args.heatmap_alpha, args.heatmap_cmap,
                    args.gaussian_sigma, not args.no_smooth,
                    not args.no_colorbar, False
                )
            except Exception as e:
                print(f"Error generating heatmap for {slide_id}: {e}")
        
        print(f"Heatmaps saved to: {heatmap_output_dir}")
    
    # Print sample predictions
    print("\nSample predictions:")
    display_cols = ['slide_id', 'predicted_short', 'confidence', 'prob_normal', 'prob_g3', 'prob_g4', 'prob_g5']
    available_cols = [c for c in display_cols if c in df_results.columns]
    print(df_results[available_cols].head(10).to_string(index=False))
    
    print("\n" + "=" * 70)
    print("Inference Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
