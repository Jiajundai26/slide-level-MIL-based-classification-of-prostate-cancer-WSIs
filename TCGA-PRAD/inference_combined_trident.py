#!/usr/bin/env python3
"""
TRIDENT ABMIL Inference for Combined TCGA-PRAD + GTEx

Runs inference with a binary TRIDENT model across both datasets.
Generates attention heatmaps with a colorbar labeled:
  0 = Normal, 1 = Cancer
No title text is added to the heatmap images.
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import h5py
import pandas as pd
from tqdm import tqdm

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
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

try:
    import openslide
    HAS_OPENSLIDE = True
except ImportError:
    HAS_OPENSLIDE = False


DEFAULT_TCGA_FEATS_DIR = "/sci-it/projects/cedmav/data/ARPA-H/Prostate_HnE/preprocessing/TCGA-PRAD/20x_256px_0px_overlap/features_uni_v2"
DEFAULT_GTEX_FEATS_DIR = "/sci-it/projects/cedmav/data/ARPA-H/Prostate_HnE/preprocessing/GTEx_Prostate/20x_256px_0px_overlap/features_uni_v2"
DEFAULT_TCGA_WSI_DIR = "/local/data/magicscan/HnE/TCGA-PRAD_organize/WSI"
DEFAULT_GTEX_WSI_DIR = "/local/data/magicscan/HnE/GTEx_prostate/histology_images_prostate"


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = self.attention_c(a * b).squeeze(-1)
        A = torch.softmax(A, dim=1)
        return A


class ABMIL(nn.Module):
    def __init__(
        self,
        input_dim: int = 1536,
        hidden_dim: int = 256,
        attention_dim: int = 128,
        num_classes: int = 1,
        dropout: float = 0.25
    ):
        super().__init__()
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


def load_model(checkpoint_path: Path, device: torch.device) -> Tuple[ABMIL, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        config_path = checkpoint_path.parent / 'config.json'
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            config = {'input_dim': 1536, 'hidden_dim': 256, 'attention_dim': 128}

    model = ABMIL(
        input_dim=config.get('input_dim', 1536),
        hidden_dim=config.get('hidden_dim', 256),
        attention_dim=config.get('attention_dim', 128),
        num_classes=1,
        dropout=0.0
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model, config


def load_features(h5_path: Path, max_patches: Optional[int] = None):
    with h5py.File(h5_path, 'r') as f:
        features = torch.from_numpy(f['features'][:]).float()
        coords = f['coords'][:] if 'coords' in f else None
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


@torch.no_grad()
def predict_single(
    model: ABMIL,
    h5_path: Path,
    device: torch.device,
    max_patches: Optional[int] = None,
    return_attention: bool = False
) -> dict:
    features, coords, attrs = load_features(h5_path, max_patches)
    features = features.unsqueeze(0).to(device)

    if return_attention:
        logits, attention = model(features, return_attention=True)
        attention = attention.squeeze(0).cpu().numpy()
    else:
        logits = model(features)
        attention = None

    prob = torch.sigmoid(logits).squeeze().cpu().item()
    pred_class = int(prob > 0.5)

    result = {
        'slide_id': h5_path.stem,
        'predicted_class': pred_class,
        'predicted_name': 'Cancer' if pred_class == 1 else 'Normal',
        'probability': prob,
        'confidence': prob if pred_class == 1 else 1 - prob,
        'num_patches': features.shape[1]
    }

    if return_attention:
        attention_norm = (attention - attention.min()) / (attention.max() - attention.min() + 1e-8)
        result['attention'] = attention
        result['attention_norm'] = attention_norm
        result['coords'] = coords
        result['attrs'] = attrs

    return result


def save_attention_npz(attention_dir: Path, result: dict):
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


def generate_attention_heatmap(
    attention_path: Path,
    wsi_path: Optional[Path],
    output_path: Path,
    thumbnail_size: int = 2048,
    alpha: float = 0.5,
    cmap: str = 'jet',
    gaussian_sigma: float = 10.0,
    smooth: bool = True
):
    if not HAS_PIL or not HAS_MATPLOTLIB:
        return

    data = np.load(attention_path)
    attention = data['attention_norm']
    coords = data['coords']
    patch_size = int(data.get('patch_size', 256))

    if coords is None or len(coords) == 0:
        return

    max_x = coords[:, 0].max() + patch_size
    max_y = coords[:, 1].max() + patch_size
    scale = min(thumbnail_size / max_x, thumbnail_size / max_y)
    heatmap_width = int(max_x * scale)
    heatmap_height = int(max_y * scale)

    heatmap = np.zeros((heatmap_height, heatmap_width), dtype=np.float32)
    count_map = np.zeros((heatmap_height, heatmap_width), dtype=np.float32)
    scaled_patch = max(int(patch_size * scale), 1)

    for coord, att in zip(coords, attention):
        x, y = coord
        sx, sy = int(x * scale), int(y * scale)
        ex, ey = min(sx + scaled_patch, heatmap_width), min(sy + scaled_patch, heatmap_height)
        heatmap[sy:ey, sx:ex] += att
        count_map[sy:ey, sx:ex] += 1

    count_map[count_map == 0] = 1
    heatmap = heatmap / count_map

    if smooth and HAS_SCIPY and gaussian_sigma > 0:
        mask = (count_map > 0).astype(np.float32)
        heatmap_smooth = gaussian_filter(heatmap, sigma=gaussian_sigma)
        mask_smooth = gaussian_filter(mask, sigma=gaussian_sigma)
        mask_smooth[mask_smooth < 0.01] = 1
        heatmap = heatmap_smooth / mask_smooth
        heatmap = np.clip(heatmap, 0, 1)

    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    colormap = plt.get_cmap(cmap)
    heatmap_colored = colormap(heatmap)[:, :, :3]
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)

    if wsi_path and wsi_path.exists() and HAS_OPENSLIDE:
        try:
            slide = openslide.OpenSlide(str(wsi_path))
            thumb = slide.get_thumbnail((heatmap_width, heatmap_height))
            thumb = np.array(thumb.convert('RGB'))
            output = (alpha * heatmap_colored + (1 - alpha) * thumb).astype(np.uint8)
            slide.close()
        except Exception:
            output = heatmap_colored
    else:
        output = heatmap_colored

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.imshow(output)
    ax.axis('off')

    sm = ScalarMappable(cmap=colormap, norm=Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, aspect=30)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['0 (Normal)', '1 (Cancer)'])
    cbar.ax.tick_params(labelsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)


def find_wsi_path(slide_id: str, wsi_dir: Path) -> Optional[Path]:
    for ext in ['.svs', '.ndpi', '.tif', '.tiff', '.mrxs']:
        candidate = wsi_dir / f"{slide_id}{ext}"
        if candidate.exists():
            return candidate
    return None


def main():
    parser = argparse.ArgumentParser(description="Inference for combined TCGA + GTEx")
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--tcga_feats_dir', type=str, default=DEFAULT_TCGA_FEATS_DIR)
    parser.add_argument('--gtex_feats_dir', type=str, default=DEFAULT_GTEX_FEATS_DIR)
    parser.add_argument('--tcga_wsi_dir', type=str, default=DEFAULT_TCGA_WSI_DIR)
    parser.add_argument('--gtex_wsi_dir', type=str, default=DEFAULT_GTEX_WSI_DIR)
    parser.add_argument('--output_dir', type=str, default='output/combined_inference')
    parser.add_argument('--max_patches', type=int, default=None)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--save_attention', action='store_true')
    parser.add_argument('--generate_heatmaps', action='store_true')
    parser.add_argument('--thumbnail_size', type=int, default=2048)
    parser.add_argument('--heatmap_alpha', type=float, default=0.5)
    parser.add_argument('--heatmap_cmap', type=str, default='jet')
    parser.add_argument('--gaussian_sigma', type=float, default=10.0)
    parser.add_argument('--no_smooth', action='store_true')
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = base_dir / checkpoint_path

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = base_dir / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device.startswith('cuda') and not torch.cuda.is_available():
        device = 'cpu'
    device = torch.device(device)

    model, _ = load_model(checkpoint_path, device)

    tcga_feats_dir = Path(args.tcga_feats_dir)
    gtex_feats_dir = Path(args.gtex_feats_dir)
    tcga_wsi_dir = Path(args.tcga_wsi_dir)
    gtex_wsi_dir = Path(args.gtex_wsi_dir)

    tcga_h5 = sorted(tcga_feats_dir.glob('*.h5'))
    gtex_h5 = sorted(gtex_feats_dir.glob('*.h5'))

    all_items = [('tcga', p) for p in tcga_h5] + [('gtex', p) for p in gtex_h5]
    if len(all_items) == 0:
        print("No H5 files found.")
        return

    results = []
    attention_dir = output_dir / 'attention'
    heatmap_dir = output_dir / 'heatmaps'
    if args.generate_heatmaps:
        heatmap_dir.mkdir(parents=True, exist_ok=True)

    for source, h5_path in tqdm(all_items, desc="Running inference"):
        result = predict_single(
            model, h5_path, device, args.max_patches, args.save_attention
        )
        result['source'] = source
        results.append(result)

        if args.save_attention:
            save_attention_npz(attention_dir, result)

        if args.generate_heatmaps and args.save_attention:
            wsi_dir = tcga_wsi_dir if source == 'tcga' else gtex_wsi_dir
            wsi_path = find_wsi_path(result['slide_id'], wsi_dir)
            output_path = heatmap_dir / f"{result['slide_id']}_heatmap.png"
            generate_attention_heatmap(
                attention_dir / f"{result['slide_id']}_attention.npz",
                wsi_path,
                output_path,
                thumbnail_size=args.thumbnail_size,
                alpha=args.heatmap_alpha,
                cmap=args.heatmap_cmap,
                gaussian_sigma=args.gaussian_sigma,
                smooth=not args.no_smooth
            )

    df = pd.DataFrame([{
        'slide_id': r.get('slide_id'),
        'source': r.get('source'),
        'predicted_class': r.get('predicted_class'),
        'predicted_name': r.get('predicted_name'),
        'probability': r.get('probability'),
        'confidence': r.get('confidence'),
        'num_patches': r.get('num_patches')
    } for r in results])

    output_csv = output_dir / 'predictions.csv'
    df.to_csv(output_csv, index=False)

    output_json = output_dir / 'predictions.json'
    with open(output_json, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'checkpoint': str(checkpoint_path),
            'num_slides': len(results),
            'predictions': results
        }, f, indent=2)

    print(f"Saved: {output_csv}")
    print(f"Saved: {output_json}")


if __name__ == "__main__":
    main()
