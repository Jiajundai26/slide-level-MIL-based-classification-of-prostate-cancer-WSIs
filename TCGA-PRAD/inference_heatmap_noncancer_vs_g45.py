#!/usr/bin/env python3
"""
Inference & Attention Heatmap Generation for Binary ABMIL Model

Generates patch-level attention heatmaps overlaid on whole-slide images (WSIs).
Supports both thumbnail overlay and full-resolution heatmap outputs.

Inputs:
    - Trained ABMIL checkpoint (best_model.pt)
    - Pre-extracted H5 features with coords (features + coords keys)
    - (Optional) WSI files (.svs) for overlay visualization

Outputs per slide:
    - Attention heatmap image (standalone + WSI overlay if WSI available)
    - Slide-level prediction with probability
    - CSV of per-patch attention scores with coordinates

Usage:
    # Run on all slides from features dir
    python inference_heatmap.py \\
        --checkpoint output/binary_noncancer_vs_g45_coarse/best_model.pt \\
        --feats_dir /path/to/features_uni_v2

    # Run on specific slides
    python inference_heatmap.py \\
        --checkpoint output/binary_noncancer_vs_g45_coarse/best_model.pt \\
        --feats_dir /path/to/features_uni_v2 \\
        --slide_ids TCGA-2A-A8VL-01A-02-TS2.xxx TCGA-2A-A8VO-01A-01-TSA.yyy

    # With WSI overlay
    python inference_heatmap.py \\
        --checkpoint output/binary_noncancer_vs_g45_coarse/best_model.pt \\
        --feats_dir /path/to/features_uni_v2 \\
        --wsi_dir WSI/hug_WSI \\
        --overlay

    # With labels file (to show ground truth in output)
    python inference_heatmap.py \\
        --checkpoint output/binary_noncancer_vs_g45_coarse/best_model.pt \\
        --feats_dir /path/to/features_uni_v2 \\
        --labels_path /path/to/slide_labels.tsv
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import h5py
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm as mpl_cm
from PIL import Image

# Try to import openslide for WSI overlay
try:
    import openslide
    HAS_OPENSLIDE = True
except ImportError:
    HAS_OPENSLIDE = False
    print("Warning: openslide not available. WSI overlay will be disabled.")

try:
    from scipy.ndimage import gaussian_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_FEATS_DIR = "/sci-it/projects/cedmav/data/ARPA-H/Prostate_HnE/preprocessing/TCGA-PRAD/20x_256px_0px_overlap/features_uni_v2"
DEFAULT_WSI_DIR = "/local/data/magicscan/HnE/TCGA-PRAD_organize/WSI/hug_WSI"
DEFAULT_LABELS_PATH = "/sci-it/projects/cedmav/data/ARPA-H/Prostate_HnE/preprocessing/TCGA-PRAD/labels_MIL/slide_labels.tsv"

# Binary class names matching the training script
GLEASON_TO_BINARY = {1: 0, 2: 1, 3: 1}
CLASS_NAMES = {0: 'Non-Cancer', 1: 'Cancer (G4/G5)'}

# Patch extraction parameters (from directory name: 20x_256px_0px_overlap)
PATCH_SIZE_AT_EXTRACTION = 256  # pixels at extraction magnification
# Actual coordinate spacing is 512 (level-0 coords for 256px patches at 20x on a 40x WSI)


# ============================================================================
# Model (must match training script exactly)
# ============================================================================

class GatedAttention(nn.Module):
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

    def forward(self, x: torch.Tensor, return_raw: bool = False):
        a = self.attention_a(x)
        b = self.attention_b(x)
        raw = self.attention_c(a * b).squeeze(-1)  # pre-softmax scores
        A = torch.softmax(raw, dim=1)
        if return_raw:
            return A, raw
        return A


class ABMIL(nn.Module):
    def __init__(
        self,
        input_dim: int = 1536,
        hidden_dim: int = 256,
        attention_dim: int = 128,
        dropout: float = 0.25
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

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
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        if isinstance(x, dict):
            x = x['features']
        h = self.feature_projection(x)
        if return_attention:
            A, raw_attn = self.attention(h, return_raw=True)
            M = torch.bmm(A.unsqueeze(1), h).squeeze(1)
            logits = self.classifier(M)
            return logits, raw_attn  # return raw pre-softmax scores for visualization
        else:
            A = self.attention(h)
            M = torch.bmm(A.unsqueeze(1), h).squeeze(1)
            logits = self.classifier(M)
            return logits


# ============================================================================
# Inference
# ============================================================================

@torch.no_grad()
def run_inference(
    model: ABMIL,
    h5_path: Path,
    device: torch.device
) -> Dict:
    """
    Run inference on a single slide and extract attention weights.

    Returns dict with:
        - slide_id, logit, probability, predicted_class
        - attention: (N,) array of attention weights
        - coords: (N, 2) array of patch coordinates
        - features shape info
    """
    model.eval()

    with h5py.File(h5_path, 'r') as f:
        features = torch.from_numpy(f['features'][:]).float()
        coords = f['coords'][:] if 'coords' in f else None

    # Add batch dimension: [1, N, D]
    features = features.unsqueeze(0).to(device)

    logits, raw_attn = model(features, return_attention=True)

    logit = logits.squeeze().cpu().item()
    prob = torch.sigmoid(logits).squeeze().cpu().item()
    raw_scores = raw_attn.squeeze(0).cpu().numpy()  # (N,) raw pre-softmax scores

    # Percentile-based normalization to [0, 1] for visualization
    # This spreads values across the full colormap range
    p_low, p_high = np.percentile(raw_scores, [2, 98])
    if p_high - p_low > 1e-8:
        attn_normalized = (raw_scores - p_low) / (p_high - p_low)
    else:
        attn_normalized = raw_scores - raw_scores.min()
        r = attn_normalized.max()
        if r > 1e-8:
            attn_normalized /= r
    attn_normalized = np.clip(attn_normalized, 0.0, 1.0)

    predicted_class = 1 if prob > 0.5 else 0

    return {
        'slide_id': h5_path.stem,
        'logit': logit,
        'probability': prob,
        'predicted_class': predicted_class,
        'predicted_label': CLASS_NAMES[predicted_class],
        'attention': attn_normalized,       # normalized [0,1] for heatmap
        'attention_raw': raw_scores,         # raw pre-softmax scores
        'coords': coords,
        'num_patches': len(attn_normalized)
    }


# ============================================================================
# Heatmap Generation
# ============================================================================

def generate_attention_heatmap(
    attention: np.ndarray,
    coords: np.ndarray,
    output_path: Path,
    slide_id: str = '',
    predicted_label: str = '',
    probability: float = 0.0,
    true_label: str = None,
    patch_spacing: int = 512,
    cmap: str = 'jet',
    smooth_sigma: float = 1.0,
    figsize: Tuple[int, int] = (12, 10),
    wsi_dimensions: Tuple[int, int] = None,
):
    """
    Generate a standalone attention heatmap from patch coordinates and attention weights.

    Args:
        attention: (N,) attention weights per patch
        coords: (N, 2) patch coordinates (x, y) at level-0
        output_path: where to save the figure
        slide_id: slide ID for title
        predicted_label: predicted class name
        probability: prediction probability
        true_label: ground truth label if available
        patch_spacing: spacing between patches in level-0 coordinates
        cmap: matplotlib colormap
        smooth_sigma: Gaussian smoothing sigma (0 to disable)
        figsize: figure size
        wsi_dimensions: (width, height) in level-0 pixels. If provided, the grid
            spans the full WSI instead of just the tight bounding-box of patches.
    """
    if coords is None or len(coords) == 0:
        print(f"  Warning: No coords for {slide_id}, skipping heatmap.")
        return

    x_coords = coords[:, 0]
    y_coords = coords[:, 1]

    if wsi_dimensions is not None:
        # Use the full WSI coordinate space  (origin at 0, 0)
        wsi_w, wsi_h = wsi_dimensions
        grid_x = (x_coords / patch_spacing).astype(int)
        grid_y = (y_coords / patch_spacing).astype(int)
        grid_w = max(grid_x.max() + 1, int(np.ceil(wsi_w / patch_spacing)))
        grid_h = max(grid_y.max() + 1, int(np.ceil(wsi_h / patch_spacing)))
    else:
        # Fall back to coord-origin (0, 0) so that absolute position is preserved
        grid_x = (x_coords / patch_spacing).astype(int)
        grid_y = (y_coords / patch_spacing).astype(int)
        grid_w = grid_x.max() + 1
        grid_h = grid_y.max() + 1

    # Create attention grid (NaN for missing patches)
    attn_grid = np.full((grid_h, grid_w), np.nan)
    for i in range(len(attention)):
        attn_grid[grid_y[i], grid_x[i]] = attention[i]

    # Optional smoothing (only on non-NaN regions)
    if smooth_sigma > 0 and HAS_SCIPY:
        mask = ~np.isnan(attn_grid)
        attn_filled = np.where(mask, attn_grid, 0)
        weight = mask.astype(float)
        attn_smooth = gaussian_filter(attn_filled, sigma=smooth_sigma)
        weight_smooth = gaussian_filter(weight, sigma=smooth_sigma)
        with np.errstate(divide='ignore', invalid='ignore'):
            attn_grid_display = np.where(weight_smooth > 0, attn_smooth / weight_smooth, np.nan)
    else:
        attn_grid_display = attn_grid

    # Plot – scale figure size to match grid aspect ratio
    aspect = grid_w / max(grid_h, 1)
    fig_h = figsize[1]
    fig_w = max(figsize[0], fig_h * aspect)
    fig_w = min(fig_w, 30)  # cap at 30 inches
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))

    # Use masked array for proper NaN handling
    masked_grid = np.ma.masked_invalid(attn_grid_display)

    im = ax.imshow(
        masked_grid,
        cmap=cmap,
        interpolation='nearest',
        aspect='equal',
        vmin=0.0,
        vmax=1.0  # attention is already percentile-normalized to [0, 1]
    )

    # Set background color for missing patches
    ax.set_facecolor('white')

    # Title
    title = f"{slide_id}\nPred: {predicted_label} (p={probability:.3f})"
    if true_label is not None:
        title += f" | GT: {true_label}"
    ax.set_title(title, fontsize=14, fontweight='bold')

    ax.set_xlabel('X grid')
    ax.set_ylabel('Y grid')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Attention Weight', fontsize=12)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def generate_wsi_overlay(
    attention: np.ndarray,
    coords: np.ndarray,
    wsi_path: Path,
    output_path: Path,
    slide_id: str = '',
    predicted_label: str = '',
    probability: float = 0.0,
    true_label: str = None,
    patch_spacing: int = 512,
    cmap: str = 'jet',
    alpha: float = 0.45,
    thumb_longest: int = 2048,
):
    """
    Generate attention heatmap overlaid on a WSI thumbnail.

    Args:
        attention: (N,) attention weights
        coords: (N, 2) level-0 coordinates
        wsi_path: path to WSI file (.svs)
        output_path: output figure path
        patch_spacing: spacing between patches at level 0
        cmap: colormap
        alpha: overlay transparency
        thumb_longest: longest dimension of thumbnail
    """
    if not HAS_OPENSLIDE:
        print("  openslide not available, skipping WSI overlay.")
        return

    if coords is None or len(coords) == 0:
        print(f"  Warning: No coords for {slide_id}, skipping overlay.")
        return

    # Open WSI and get thumbnail
    wsi = openslide.OpenSlide(str(wsi_path))
    wsi_w, wsi_h = wsi.dimensions  # level-0 dimensions

    # Compute thumbnail size
    scale = thumb_longest / max(wsi_w, wsi_h)
    thumb_w = int(wsi_w * scale)
    thumb_h = int(wsi_h * scale)
    thumbnail = wsi.get_thumbnail((thumb_w, thumb_h))
    thumbnail = thumbnail.convert('RGB')
    thumb_arr = np.array(thumbnail)

    # Create heatmap at thumbnail resolution
    heatmap = np.zeros((thumb_h, thumb_w), dtype=np.float64)
    count_map = np.zeros((thumb_h, thumb_w), dtype=np.float64)

    # Scale patch coordinates to thumbnail space
    patch_size_thumb = max(1, int(patch_spacing * scale))

    for i in range(len(attention)):
        x0 = int(coords[i, 0] * scale)
        y0 = int(coords[i, 1] * scale)
        x1 = min(x0 + patch_size_thumb, thumb_w)
        y1 = min(y0 + patch_size_thumb, thumb_h)
        x0 = max(0, x0)
        y0 = max(0, y0)
        if x1 > x0 and y1 > y0:
            heatmap[y0:y1, x0:x1] += attention[i]
            count_map[y0:y1, x0:x1] += 1

    # Average overlapping regions
    with np.errstate(divide='ignore', invalid='ignore'):
        heatmap = np.where(count_map > 0, heatmap / count_map, 0)

    # Normalize to [0, 1]
    attn_max = heatmap.max()
    if attn_max > 0:
        heatmap_norm = heatmap / attn_max
    else:
        heatmap_norm = heatmap

    # Optional smoothing
    if HAS_SCIPY:
        smooth_px = max(1, patch_size_thumb // 2)
        mask = count_map > 0
        heatmap_filled = np.where(mask, heatmap_norm, 0)
        weight = mask.astype(float)
        heatmap_smooth = gaussian_filter(heatmap_filled, sigma=smooth_px)
        weight_smooth = gaussian_filter(weight, sigma=smooth_px)
        with np.errstate(divide='ignore', invalid='ignore'):
            heatmap_norm = np.where(weight_smooth > 0, heatmap_smooth / weight_smooth, 0)

    # Apply colormap to heatmap
    colormap = mpl_cm.get_cmap(cmap)
    heatmap_rgba = colormap(heatmap_norm)  # (H, W, 4)
    heatmap_rgb = (heatmap_rgba[:, :, :3] * 255).astype(np.uint8)

    # Create tissue mask (avoid overlaying on background)
    tissue_mask = count_map > 0
    if HAS_SCIPY:
        tissue_mask = gaussian_filter(tissue_mask.astype(float), sigma=smooth_px) > 0.01

    # Blend
    overlay = thumb_arr.copy().astype(np.float64)
    for c in range(3):
        overlay[:, :, c] = np.where(
            tissue_mask,
            (1 - alpha) * thumb_arr[:, :, c] + alpha * heatmap_rgb[:, :, c],
            thumb_arr[:, :, c]
        )
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    # Plot side-by-side: thumbnail + overlay
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    axes[0].imshow(thumb_arr)
    axes[0].set_title('WSI Thumbnail', fontsize=14)
    axes[0].axis('off')

    axes[1].imshow(overlay)
    title = f"Attention Heatmap\nPred: {predicted_label} (p={probability:.3f})"
    if true_label is not None:
        title += f" | GT: {true_label}"
    axes[1].set_title(title, fontsize=14, fontweight='bold')
    axes[1].axis('off')

    # Add colorbar
    norm = mcolors.Normalize(vmin=0, vmax=1)
    sm = mpl_cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.6, pad=0.02)
    cbar.set_label('Attention Weight (normalized)', fontsize=12)

    fig.suptitle(slide_id, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    wsi.close()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Inference & Attention Heatmap Generation for Binary ABMIL',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic inference with heatmaps on all slides
    python inference_heatmap.py \\
        --checkpoint output/binary_noncancer_vs_g45_coarse/best_model.pt \\
        --feats_dir /path/to/features_uni_v2

    # Specific slides with WSI overlay
    python inference_heatmap.py \\
        --checkpoint output/binary_noncancer_vs_g45_coarse/best_model.pt \\
        --feats_dir /path/to/features_uni_v2 \\
        --wsi_dir WSI/hug_WSI --overlay \\
        --slide_ids TCGA-XX-XXXX-01A.XXXXX

    # Include ground truth labels
    python inference_heatmap.py \\
        --checkpoint output/binary_noncancer_vs_g45_coarse/best_model.pt \\
        --feats_dir /path/to/features_uni_v2 \\
        --labels_path /path/to/slide_labels.tsv
        """
    )

    # Required
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint (best_model.pt)')
    parser.add_argument('--feats_dir', type=str, default=DEFAULT_FEATS_DIR,
                       help='Directory containing H5 feature files')

    # Optional input
    parser.add_argument('--slide_ids', nargs='+', default=None,
                       help='Specific slide IDs to process (default: all in feats_dir)')
    parser.add_argument('--labels_path', type=str, default=None,
                       help='Path to labels TSV for ground truth display')
    parser.add_argument('--wsi_dir', type=str, default=DEFAULT_WSI_DIR,
                       help='Directory containing WSI files (.svs)')

    # Output
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: ~/output/inference_noncancer_vs_g45 '
                            'or ~/output/combined_inference_noncancer_vs_g45 with --combine_gtex)')
    parser.add_argument('--combine_gtex', action='store_true',
                       help='Also process GTEx slides (uses combined output dir)')
    parser.add_argument('--overlay', action='store_true',
                       help='Generate WSI overlay heatmaps (requires openslide)')

    # Model parameters (must match training)
    parser.add_argument('--input_dim', type=int, default=1536,
                       help='Feature dimension')
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='Hidden dimension')
    parser.add_argument('--attention_dim', type=int, default=128,
                       help='Attention dimension')
    parser.add_argument('--dropout', type=float, default=0.25,
                       help='Dropout rate')

    # Heatmap options
    parser.add_argument('--patch_spacing', type=int, default=512,
                       help='Patch spacing in level-0 coordinates')
    parser.add_argument('--cmap', type=str, default='jet',
                       help='Colormap for heatmap (jet, coolwarm, inferno, etc.)')
    parser.add_argument('--smooth_sigma', type=float, default=1.0,
                       help='Gaussian smooth sigma for grid heatmap (0=no smoothing)')
    parser.add_argument('--alpha', type=float, default=0.45,
                       help='Overlay transparency (0=WSI only, 1=heatmap only)')
    parser.add_argument('--thumb_size', type=int, default=2048,
                       help='Longest dimension of WSI thumbnail')

    # Device
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device for inference')

    # Top-K analysis
    parser.add_argument('--top_k', type=int, default=20,
                       help='Number of top-attention patches to highlight')

    args = parser.parse_args()

    # Set default output directory based on dataset mode
    if args.output_dir is None:
        if args.combine_gtex:
            args.output_dir = os.path.expanduser('~/output/combined_inference_noncancer_vs_g45')
        else:
            args.output_dir = os.path.expanduser('~/output/inference_noncancer_vs_g45')

    # Setup
    feats_dir = Path(args.feats_dir)
    print(f"Features directory: {feats_dir}")
    if not feats_dir.exists():
        print(f"ERROR: feats_dir does not exist: {feats_dir}")
        sys.exit(1)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'grid_heatmaps').mkdir(exist_ok=True)
    if args.overlay:
        (output_dir / 'wsi_overlays').mkdir(exist_ok=True)

    # Device
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    device = torch.device(args.device)
    print(f"Device: {device}")

    # Load model
    print(f"Loading model from: {args.checkpoint}")
    model = ABMIL(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        attention_dim=args.attention_dim,
        dropout=args.dropout
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"  Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}, "
          f"val AUC: {checkpoint.get('val_auc', '?')}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    # Load labels if provided
    gt_labels = {}
    if args.labels_path:
        labels_df = pd.read_csv(args.labels_path, sep='\t')
        for _, row in labels_df.iterrows():
            binary = GLEASON_TO_BINARY.get(row['label'], None)
            if binary is not None:
                gt_labels[row['slide_id']] = CLASS_NAMES[binary]
        print(f"Loaded ground truth for {len(gt_labels)} slides\n")

    # Find WSI files
    wsi_map = {}
    if args.overlay and args.wsi_dir:
        wsi_dir = Path(args.wsi_dir)
        if wsi_dir.exists():
            for ext in ['*.svs', '*.tiff', '*.tif', '*.ndpi', '*.mrxs']:
                for wsi_path in wsi_dir.glob(ext):
                    wsi_map[wsi_path.stem] = wsi_path
            print(f"Found {len(wsi_map)} WSI files in {args.wsi_dir}\n")

    # Find slides to process
    if args.slide_ids:
        h5_files = [feats_dir / f"{sid}.h5" for sid in args.slide_ids]
        h5_files = [f for f in h5_files if f.exists()]
    else:
        h5_files = sorted(feats_dir.glob('*.h5'))

    print(f"Processing {len(h5_files)} slides...\n")

    # Run inference
    all_results = []

    for h5_path in tqdm(h5_files, desc="Inference"):
        slide_id = h5_path.stem

        # Run model
        result = run_inference(model, h5_path, device)

        true_label = gt_labels.get(slide_id, None)

        # Save per-patch attention scores
        patch_df = pd.DataFrame({
            'patch_idx': np.arange(result['num_patches']),
            'coord_x': result['coords'][:, 0] if result['coords'] is not None else 0,
            'coord_y': result['coords'][:, 1] if result['coords'] is not None else 0,
            'attention_raw': result['attention_raw'],
            'attention_normalized': result['attention']
        })
        patch_df = patch_df.sort_values('attention_normalized', ascending=False)
        patch_df.to_csv(output_dir / f'{slide_id}_attention.csv', index=False)

        # Resolve WSI dimensions for proper grid sizing
        wsi_dims = None
        if slide_id in wsi_map:
            try:
                _wsi = openslide.OpenSlide(str(wsi_map[slide_id]))
                wsi_dims = _wsi.dimensions  # (width, height) at level-0
                _wsi.close()
            except Exception:
                pass

        # Generate grid heatmap (always)
        generate_attention_heatmap(
            attention=result['attention'],
            coords=result['coords'],
            output_path=output_dir / 'grid_heatmaps' / f'{slide_id}_heatmap.png',
            slide_id=slide_id,
            predicted_label=result['predicted_label'],
            probability=result['probability'],
            true_label=true_label,
            patch_spacing=args.patch_spacing,
            cmap=args.cmap,
            smooth_sigma=args.smooth_sigma,
            wsi_dimensions=wsi_dims,
        )

        # Generate WSI overlay if requested and WSI exists
        if args.overlay and slide_id in wsi_map:
            generate_wsi_overlay(
                attention=result['attention'],
                coords=result['coords'],
                wsi_path=wsi_map[slide_id],
                output_path=output_dir / 'wsi_overlays' / f'{slide_id}_overlay.png',
                slide_id=slide_id,
                predicted_label=result['predicted_label'],
                probability=result['probability'],
                true_label=true_label,
                patch_spacing=args.patch_spacing,
                cmap=args.cmap,
                alpha=args.alpha,
                thumb_longest=args.thumb_size,
            )

        # Collect summary
        summary = {
            'slide_id': slide_id,
            'predicted_class': result['predicted_class'],
            'predicted_label': result['predicted_label'],
            'probability': round(result['probability'], 4),
            'logit': round(result['logit'], 4),
            'num_patches': result['num_patches'],
            'top_attention_mean': round(float(np.sort(result['attention'])[-args.top_k:].mean()), 6),
            'attention_std': round(float(result['attention'].std()), 6),
        }
        if true_label is not None:
            summary['true_label'] = true_label
        all_results.append(summary)

    # Save summary
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_dir / 'inference_results.csv', index=False)

    # Print summary statistics
    print(f"\n{'='*70}")
    print("INFERENCE SUMMARY")
    print(f"{'='*70}")
    print(f"Total slides processed: {len(all_results)}")
    if all_results:
        n_cancer = sum(1 for r in all_results if r['predicted_class'] == 1)
        n_noncancer = len(all_results) - n_cancer
        print(f"  Predicted Non-Cancer: {n_noncancer}")
        print(f"  Predicted Cancer:     {n_cancer}")

        probs = [r['probability'] for r in all_results]
        print(f"\nProbability statistics:")
        print(f"  Mean: {np.mean(probs):.4f}")
        print(f"  Std:  {np.std(probs):.4f}")
        print(f"  Min:  {np.min(probs):.4f}")
        print(f"  Max:  {np.max(probs):.4f}")

        if gt_labels:
            correct = sum(1 for r in all_results
                         if 'true_label' in r and r['true_label'] == r['predicted_label'])
            total_with_gt = sum(1 for r in all_results if 'true_label' in r)
            if total_with_gt > 0:
                print(f"\nAccuracy (on slides with GT): {correct}/{total_with_gt} = {correct/total_with_gt:.4f}")

    print(f"\nResults saved to: {output_dir}")
    print(f"  - inference_results.csv (slide-level predictions)")
    print(f"  - *_attention.csv (per-patch attention scores)")
    print(f"  - grid_heatmaps/ (attention grid heatmaps)")
    if args.overlay:
        print(f"  - wsi_overlays/ (WSI overlay heatmaps)")
    print("Done!")


if __name__ == "__main__":
    main()
