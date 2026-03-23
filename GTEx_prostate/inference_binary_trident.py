#!/usr/bin/env python3
"""
TRIDENT-based Binary Classification Inference Script for TCGA-PRAD, GTEx_Prostate, or Combined

Runs inference on WSI feature files using a trained ABMIL binary classifier.
Supports attention heatmap generation and WSI overlay visualization.

Usage:
    # Basic inference on all features
    python inference_binary_trident.py \
        --checkpoint output/trident_training/low_vs_high_XXXX/best_model.pt \
        --feats_dir /path/to/features

    # Inference with attention heatmaps
    python inference_binary_trident.py \
        --checkpoint output/trident_training/low_vs_high_XXXX/best_model.pt \
        --feats_dir /path/to/features \
        --save_attention

    # Inference with WSI heatmap overlays
    python inference_binary_trident.py \
        --checkpoint output/trident_training/low_vs_high_XXXX/best_model.pt \
        --feats_dir /path/to/features \
        --save_attention \
        --wsi_dir WSI/ \
        --generate_heatmaps

    # Inference on specific slides
    python inference_binary_trident.py \
        --checkpoint output/trident_training/low_vs_high_XXXX/best_model.pt \
        --feats_dir /path/to/features \
        --slide_ids TCGA-XX-XXXX,TCGA-YY-YYYY
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
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
    from matplotlib.colors import LinearSegmentedColormap
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

# Default paths
DEFAULT_FEATS_DIR = "/sci-it/projects/cedmav/data/ARPA-H/Prostate_HnE/preprocessing/GTEx_Prostate/20x_256px_0px_overlap/features_uni_v2"
DEFAULT_WSI_DIR = "/local/data/magicscan/HnE/TCGA-PRAD_organize/WSI"
DEFAULT_LABELS_PATH = "/sci-it/projects/cedmav/data/ARPA-H/Prostate_HnE/preprocessing/GTEx_Prostate/GTEx_prostate_labels.csv"

# Class names for different modes
CLASS_NAMES = {
    'cancer_vs_noncancer': {0: 'Benign', 1: 'Cancer'},
    'low_vs_high': {0: 'Low-Grade', 1: 'High-Grade'}
}


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
# Model Loading
# ============================================================================

def load_model(checkpoint_path: str, device: torch.device) -> Tuple[ABMIL, dict]:
    """
    Load trained ABMIL model from checkpoint.
    
    Args:
        checkpoint_path: Path to best_model.pt
        device: Device to load model on
        
    Returns:
        model: Loaded ABMIL model in eval mode
        config: Model configuration dict
    """
    checkpoint_path = Path(checkpoint_path)
    
    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get config from checkpoint or config.json
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        config_path = checkpoint_path.parent / 'config.json'
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            print("Warning: No config found, using defaults")
            config = {
                'input_dim': 1536,
                'hidden_dim': 256,
                'attention_dim': 128,
                'mode': 'cancer_vs_noncancer'
            }
    
    # Initialize model
    model = ABMIL(
        input_dim=config.get('input_dim', 1536),
        hidden_dim=config.get('hidden_dim', 256),
        attention_dim=config.get('attention_dim', 128),
        num_classes=1,
        dropout=0.0  # No dropout during inference
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Print info
    epoch = checkpoint.get('epoch', 'unknown')
    val_auc = checkpoint.get('val_auc', 'N/A')
    mode = config.get('mode', 'cancer_vs_noncancer')
    
    print(f"  Model loaded from epoch {epoch}")
    print(f"  Validation AUC: {val_auc}")
    print(f"  Classification mode: {mode}")
    print(f"  Input dim: {config.get('input_dim', 1536)}")
    print(f"  Hidden dim: {config.get('hidden_dim', 256)}")
    
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
            
        # Get additional attributes
        attrs = {}
        for key in f.attrs:
            attrs[key] = f.attrs[key]
        
        # Try to get patch size from coords attributes
        if 'coords' in f:
            for key in f['coords'].attrs:
                attrs[key] = f['coords'].attrs[key]
    
    if max_patches and features.shape[0] > max_patches:
        features = features[:max_patches]
        if coords is not None:
            coords = coords[:max_patches]
    
    return features, coords, attrs


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
    raise ValueError("Could not determine slide id column in labels file.")


def load_labels_file(labels_path: Path, mode: str) -> pd.DataFrame:
    """
    Load labels for evaluation.

    Supports:
      - GTEx_prostate_labels.csv: label {0=Benign, 1=Cancer, -1=Discard}
      - TCGA TSV: slide_id + label
    """
    if labels_path.suffix.lower() == '.csv':
        df = pd.read_csv(labels_path)
        if df.empty:
            return df
        slide_id_col = _pick_slide_id_column(df)
        label_col = None
        for col in df.columns:
            if col.lower() == 'label':
                label_col = col
                break
        if label_col is None:
            raise ValueError("Labels CSV must contain a 'label' column.")
        labels_numeric = pd.to_numeric(df[label_col], errors='coerce')
        df = df.assign(label=labels_numeric)
        df = df[df['label'].isin([0, 1])]
        df = df[[slide_id_col, 'label']].rename(columns={slide_id_col: 'slide_id'})
        return df
    else:
        df = pd.read_csv(labels_path, sep='\t')
        if 'slide_id' not in df.columns or 'label' not in df.columns:
            raise ValueError("Labels TSV must contain 'slide_id' and 'label' columns.")
        return df[['slide_id', 'label']]


# ============================================================================
# Inference Functions
# ============================================================================

@torch.no_grad()
def predict_single(
    model: ABMIL,
    h5_path: Path,
    device: torch.device,
    mode: str = 'low_vs_high',
    max_patches: Optional[int] = None,
    return_attention: bool = False
) -> dict:
    """
    Run inference on a single slide.
    
    Args:
        model: Trained ABMIL model
        h5_path: Path to H5 feature file
        device: Computation device
        mode: Classification mode
        max_patches: Maximum patches to use
        return_attention: Whether to return attention weights
        
    Returns:
        result: Dictionary with prediction results
    """
    # Load features
    features, coords, attrs = load_features(h5_path, max_patches)
    features = features.unsqueeze(0).to(device)  # Add batch dimension
    
    # Run model
    if return_attention:
        logits, attention = model(features, return_attention=True)
        attention = attention.squeeze(0).cpu().numpy()
    else:
        logits = model(features)
        attention = None
    
    # Get prediction
    prob = torch.sigmoid(logits).squeeze().cpu().item()
    pred_class = int(prob > 0.5)
    
    class_names = CLASS_NAMES.get(mode, CLASS_NAMES['low_vs_high'])
    
    result = {
        'slide_id': h5_path.stem,
        'predicted_class': pred_class,
        'predicted_name': class_names[pred_class],
        'probability': prob,
        'confidence': prob if pred_class == 1 else 1 - prob,
        'num_patches': features.shape[1]
    }
    
    if return_attention:
        # Normalize attention for visualization
        attention_norm = (attention - attention.min()) / (attention.max() - attention.min() + 1e-8)
        
        result['attention'] = attention
        result['attention_norm'] = attention_norm
        result['coords'] = coords
        result['attrs'] = attrs
    
    return result


def predict_batch(
    model: ABMIL,
    h5_paths: List[Path],
    device: torch.device,
    mode: str = 'low_vs_high',
    max_patches: Optional[int] = None,
    return_attention: bool = False,
    output_dir: Optional[Path] = None
) -> List[dict]:
    """
    Run inference on multiple slides.
    
    Args:
        model: Trained ABMIL model
        h5_paths: List of H5 feature file paths
        device: Computation device
        mode: Classification mode
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
                model, h5_path, device, mode, max_patches, return_attention
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
                    probability=result['probability'],
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

def generate_attention_heatmap(
    attention_path: Path,
    wsi_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    thumbnail_size: int = 2048,
    alpha: float = 0.5,
    cmap: str = 'jet',
    gaussian_sigma: float = 10.0,
    smooth: bool = True,
    mode: str = 'low_vs_high',
    show_colorbar: bool = True
) -> Optional[np.ndarray]:
    """
    Generate attention heatmap overlay on WSI thumbnail with colorbar.
    
    Args:
        attention_path: Path to .npz file with attention weights
        wsi_path: Path to WSI file (optional, for overlay)
        output_path: Path to save heatmap image
        thumbnail_size: Maximum dimension for thumbnail
        alpha: Opacity of heatmap overlay
        cmap: Colormap for heatmap
        gaussian_sigma: Sigma for Gaussian smoothing (higher = smoother)
        smooth: Whether to apply Gaussian smoothing
        mode: Classification mode for colorbar labels
        show_colorbar: Whether to add a colorbar with labels
        
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
    
    # Get patch info
    patch_size = int(data.get('patch_size', 256))
    
    # Determine heatmap dimensions from coords
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
    
    # Apply Gaussian smoothing for continuous appearance
    if smooth and HAS_SCIPY and gaussian_sigma > 0:
        # Create a mask of where we have data
        mask = (count_map > 0).astype(np.float32)
        
        # Smooth both heatmap and mask
        heatmap_smooth = gaussian_filter(heatmap, sigma=gaussian_sigma)
        mask_smooth = gaussian_filter(mask, sigma=gaussian_sigma)
        
        # Normalize by smoothed mask to avoid edge artifacts
        mask_smooth[mask_smooth < 0.01] = 1  # Avoid division by zero
        heatmap = heatmap_smooth / mask_smooth
        
        # Clip to valid range
        heatmap = np.clip(heatmap, 0, 1)
    elif smooth and not HAS_SCIPY:
        print("Warning: scipy not available for Gaussian smoothing, using raw heatmap")
    
    # Normalize
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    
    # Apply colormap
    colormap = plt.get_cmap(cmap)
    heatmap_colored = colormap(heatmap)[:, :, :3]  # RGB
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
    
    # Overlay on WSI thumbnail if available
    if wsi_path and wsi_path.exists() and HAS_OPENSLIDE:
        try:
            slide = openslide.OpenSlide(str(wsi_path))
            # Get thumbnail
            thumb = slide.get_thumbnail((heatmap_width, heatmap_height))
            thumb = np.array(thumb.convert('RGB'))
            
            # Blend
            output = (alpha * heatmap_colored + (1 - alpha) * thumb).astype(np.uint8)
            slide.close()
        except Exception as e:
            print(f"Warning: Could not load WSI for overlay: {e}")
            output = heatmap_colored
    else:
        output = heatmap_colored
    
    # Save with colorbar
    if output_path:
        if show_colorbar:
            # Create figure with colorbar
            fig, ax = plt.subplots(1, 1, figsize=(12, 10))
            
            # Show the blended image
            im = ax.imshow(output)
            ax.axis('off')
            
            # Create a separate axes for colorbar
            # We need to create a ScalarMappable for the colorbar
            from matplotlib.cm import ScalarMappable
            from matplotlib.colors import Normalize
            
            sm = ScalarMappable(cmap=colormap, norm=Normalize(vmin=0, vmax=1))
            sm.set_array([])
            
            # Add colorbar
            cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, aspect=30)
            
            # Set colorbar labels for binary cancer vs non-cancer
            cbar.set_ticks([0, 0.5, 1.0])
            cbar.set_ticklabels([
                'Noncancer',
                'Medium',
                'Cancer'
            ])
            
            cbar.ax.tick_params(labelsize=10)
            
            # Save figure
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close(fig)
        else:
            # Save without colorbar
            Image.fromarray(output).save(output_path)
    
    return output


def generate_heatmaps_batch(
    attention_dir: Path,
    wsi_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    thumbnail_size: int = 2048,
    alpha: float = 0.5,
    cmap: str = 'jet',
    gaussian_sigma: float = 10.0,
    smooth: bool = True,
    mode: str = 'low_vs_high',
    show_colorbar: bool = True
):
    """Generate heatmaps for all attention files in a directory."""
    if not HAS_PIL or not HAS_MATPLOTLIB:
        print("Error: PIL and matplotlib required for heatmap generation")
        return
    
    attention_files = list(attention_dir.glob('*_attention.npz'))
    print(f"Found {len(attention_files)} attention files")
    print(f"Gaussian smoothing: {'enabled (sigma={})'.format(gaussian_sigma) if smooth else 'disabled'}")
    print(f"Colorbar: {'enabled' if show_colorbar else 'disabled'}")
    
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
                mode, show_colorbar
            )
        except Exception as e:
            print(f"Error generating heatmap for {slide_id}: {e}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='TRIDENT ABMIL Binary Classification Inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic inference
    python inference_binary_trident.py \\
        --checkpoint output/trident_training/low_vs_high_XXXX/best_model.pt \\
        --feats_dir /path/to/features
    
    # With attention heatmaps
    python inference_binary_trident.py \\
        --checkpoint output/trident_training/low_vs_high_XXXX/best_model.pt \\
        --feats_dir /path/to/features \\
        --save_attention --generate_heatmaps --wsi_dir WSI/
    
    # Specific slides only
    python inference_binary_trident.py \\
        --checkpoint output/trident_training/low_vs_high_XXXX/best_model.pt \\
        --feats_dir /path/to/features \\
        --slide_ids TCGA-XX-XXXX,TCGA-YY-YYYY
        """
    )
    
    # Required
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (best_model.pt)')
    
    # Input
    parser.add_argument('--feats_dir', type=str, default=DEFAULT_FEATS_DIR,
                        help='Directory containing H5 feature files')
    parser.add_argument('--slide_ids', type=str, default=None,
                        help='Comma-separated list of slide IDs (optional, infer all if not provided)')
    parser.add_argument('--slide_list', type=str, default=None,
                        help='Path to file with slide IDs (one per line)')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='output/binary_inference',
                        help='Output directory for predictions')
    
    # Model options
    parser.add_argument('--mode', type=str, default=None,
                        choices=['cancer_vs_noncancer', 'low_vs_high'],
                        help='Classification mode (auto-detected from checkpoint if not specified)')
    parser.add_argument('--max_patches', type=int, default=None,
                        help='Maximum patches per slide (None = use all)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use (cuda:0, cpu)')
    
    # Attention/Heatmap options
    parser.add_argument('--save_attention', action='store_true',
                        help='Save attention weights for each slide')
    parser.add_argument('--generate_heatmaps', action='store_true',
                        help='Generate attention heatmap images')
    parser.add_argument('--wsi_dir', type=str, default=DEFAULT_WSI_DIR,
                        help='Directory containing WSI files (for heatmap overlays)')
    parser.add_argument('--thumbnail_size', type=int, default=2048,
                        help='Maximum dimension for heatmap thumbnails')
    parser.add_argument('--heatmap_alpha', type=float, default=0.5,
                        help='Opacity of heatmap overlay')
    parser.add_argument('--heatmap_cmap', type=str, default='jet',
                        help='Colormap for heatmaps')
    parser.add_argument('--gaussian_sigma', type=float, default=10.0,
                        help='Sigma for Gaussian smoothing (higher = smoother, 0 = disabled)')
    parser.add_argument('--no_smooth', action='store_true',
                        help='Disable Gaussian smoothing (show raw patch squares)')
    parser.add_argument('--no_colorbar', action='store_true',
                        help='Disable colorbar legend on heatmaps')
    
    # Labels for evaluation
    parser.add_argument('--labels_path', type=str, default=DEFAULT_LABELS_PATH,
                        help='Path to labels file for evaluation (CSV or TSV)')
    
    args = parser.parse_args()
    
    # Setup
    base_dir = Path(__file__).resolve().parent
    feats_dir = Path(args.feats_dir)
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
    print("Loading Model")
    print("=" * 70)
    model, config = load_model(checkpoint_path, device)
    
    # Get mode from config or argument
    mode = args.mode or config.get('mode', 'low_vs_high')
    print(f"Mode: {mode}")
    
    # Get slide IDs
    if args.slide_ids:
        slide_ids = [s.strip() for s in args.slide_ids.split(',')]
    elif args.slide_list:
        with open(args.slide_list, 'r') as f:
            slide_ids = [line.strip() for line in f if line.strip()]
    else:
        # All H5 files in feats_dir
        slide_ids = None
    
    # Get H5 paths
    if slide_ids:
        h5_paths = [feats_dir / f"{sid}.h5" for sid in slide_ids]
        h5_paths = [p for p in h5_paths if p.exists()]
        print(f"\nFound {len(h5_paths)} / {len(slide_ids)} requested slides")
    else:
        h5_paths = sorted(feats_dir.glob('*.h5'))
        print(f"\nFound {len(h5_paths)} H5 feature files")
    
    if len(h5_paths) == 0:
        print("No H5 files found!")
        return
    
    # Run inference
    print("\n" + "=" * 70)
    print("Running Inference")
    print("=" * 70)
    
    results = predict_batch(
        model=model,
        h5_paths=h5_paths,
        device=device,
        mode=mode,
        max_patches=args.max_patches,
        return_attention=args.save_attention,
        output_dir=output_dir
    )
    
    # Create results DataFrame
    df_results = pd.DataFrame([
        {
            'slide_id': r.get('slide_id'),
            'predicted_class': r.get('predicted_class'),
            'predicted_name': r.get('predicted_name'),
            'probability': r.get('probability'),
            'confidence': r.get('confidence'),
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
        print(f"\nPrediction distribution:")
        print(valid_results['predicted_name'].value_counts())
        print(f"\nMean confidence: {valid_results['confidence'].mean():.4f}")
        print(f"Mean probability (class 1): {valid_results['probability'].mean():.4f}")
    
    if df_results['error'].notna().any():
        n_errors = df_results['error'].notna().sum()
        print(f"\n{n_errors} slides had errors")
    
    # Load labels for evaluation if provided
    if args.labels_path:
        labels_path = Path(args.labels_path)
        if labels_path.exists():
            try:
                labels_df = load_labels_file(labels_path, mode)
            except Exception as exc:
                print(f"Warning: Could not load labels: {exc}")
                labels_df = None
            
            # Merge with predictions
            if labels_df is not None:
                df_eval = valid_results.merge(
                    labels_df[['slide_id', 'label']], 
                    on='slide_id', 
                    how='inner'
                )
            else:
                df_eval = pd.DataFrame()
            
            if len(df_eval) > 0:
                # Map labels based on mode
                df_eval['label'] = pd.to_numeric(df_eval['label'], errors='coerce')
                df_eval = df_eval[df_eval['label'].notna()]
                
                if mode == 'low_vs_high':
                    # G3 -> 0, G4/G5 -> 1
                    label_map = {1: 0, 2: 1, 3: 1}
                    df_eval['true_class'] = df_eval['label'].map(label_map)
                else:
                    # cancer_vs_noncancer: GTEx CSV uses {0,1}; TCGA uses {1,2,3}
                    if df_eval['label'].max() > 1:
                        df_eval['true_class'] = (df_eval['label'] > 0).astype(int)
                    else:
                        df_eval['true_class'] = df_eval['label'].astype(int)
                
                # Calculate metrics
                from sklearn.metrics import (
                    accuracy_score, balanced_accuracy_score, 
                    roc_auc_score, confusion_matrix
                )
                
                y_true = df_eval['true_class'].values
                y_pred = df_eval['predicted_class'].values
                y_prob = df_eval['probability'].values
                
                print("\n" + "=" * 70)
                print("EVALUATION METRICS")
                print("=" * 70)
                print(f"Slides evaluated: {len(df_eval)}")
                print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
                print(f"Balanced Accuracy: {balanced_accuracy_score(y_true, y_pred):.4f}")
                
                try:
                    auc = roc_auc_score(y_true, y_prob)
                    print(f"AUC-ROC: {auc:.4f}")
                except ValueError:
                    print("AUC-ROC: N/A (single class in ground truth)")
                
                cm = confusion_matrix(y_true, y_pred)
                class_names = CLASS_NAMES[mode]
                print(f"\nConfusion Matrix:")
                print(f"  Predicted:  {class_names[0]:>12}  {class_names[1]:>12}")
                print(f"  Actual {class_names[0]:>12}:  {cm[0][0]:>12}  {cm[0][1]:>12}")
                if cm.shape[0] > 1:
                    print(f"  Actual {class_names[1]:>12}:  {cm[1][0]:>12}  {cm[1][1]:>12}")
    
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
        json_results.append(jr)
    
    with open(output_json, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'checkpoint': str(checkpoint_path),
            'mode': mode,
            'num_slides': len(results),
            'predictions': json_results
        }, f, indent=2)
    print(f"Full results saved to: {output_json}")
    
    # Generate heatmaps if requested
    if args.generate_heatmaps and args.save_attention:
        print("\n" + "=" * 70)
        print("Generating Heatmaps")
        print("=" * 70)
        
        wsi_dir = Path(args.wsi_dir) if args.wsi_dir else None
        if wsi_dir and not wsi_dir.is_absolute():
            wsi_dir = base_dir / wsi_dir
        
        generate_heatmaps_batch(
            attention_dir=output_dir / 'attention',
            wsi_dir=wsi_dir,
            output_dir=output_dir / 'heatmaps',
            thumbnail_size=args.thumbnail_size,
            alpha=args.heatmap_alpha,
            cmap=args.heatmap_cmap,
            gaussian_sigma=args.gaussian_sigma,
            smooth=not args.no_smooth,
            mode=mode,
            show_colorbar=not args.no_colorbar
        )
        print(f"Heatmaps saved to: {output_dir / 'heatmaps'}")
    
    # Print sample predictions
    print("\nSample predictions:")
    print(df_results.head(10).to_string(index=False))
    
    print("\n" + "=" * 70)
    print("Inference Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
