#!/usr/bin/env python3
"""
Compare Attention Heatmaps vs Ground Truth Cancer Annotations

This script creates a side-by-side comparison showing:
1. Model attention heatmap
2. Ground truth cancer annotations
3. Overlay showing alignment

Usage:
    python compare_attention_vs_gt.py --slide_id TCGA-2A-A8VL-01Z-00-DX1.2C2BD6EF-EC17-4117-AE89-A22B67AFB233
"""

import os
import json
import argparse
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon
    from matplotlib.collections import PatchCollection
except ImportError:
    print("matplotlib required: pip install matplotlib")
    exit(1)

try:
    from PIL import Image
except ImportError:
    print("PIL required: pip install pillow")
    exit(1)

try:
    import openslide
    HAS_OPENSLIDE = True
except ImportError:
    HAS_OPENSLIDE = False
    print("Warning: openslide not available, will use attention data only")

try:
    from scipy.ndimage import gaussian_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# Default paths
DEFAULT_WSI_DIR = "/local/data/magicscan/HnE/TCGA-PRAD_organize/WSI"
DEFAULT_ATTENTION_DIR = "/local/data/magicscan/HnE/TCGA-PRAD_organize/output/binary_inference/attention"
DEFAULT_ANNOTATIONS_DIR = "/local/data/magicscan/HnE/TCGA-PRAD_organize/annotations/geojsons"
DEFAULT_OUTPUT_DIR = "/local/data/magicscan/HnE/TCGA-PRAD_organize/output/attention_vs_gt"


def load_attention(attention_path: Path, thumbnail_size: int = 2048):
    """Load attention weights and create heatmap."""
    data = np.load(attention_path)
    attention = data['attention_norm']
    coords = data['coords']
    patch_size = int(data.get('patch_size', 256))
    
    # Determine dimensions
    max_x = coords[:, 0].max() + patch_size
    max_y = coords[:, 1].max() + patch_size
    
    scale = min(thumbnail_size / max_x, thumbnail_size / max_y)
    heatmap_width = int(max_x * scale)
    heatmap_height = int(max_y * scale)
    
    # Create heatmap
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
    
    # Smooth
    if HAS_SCIPY:
        mask = (count_map > 0).astype(np.float32)
        heatmap_smooth = gaussian_filter(heatmap, sigma=10)
        mask_smooth = gaussian_filter(mask, sigma=10)
        mask_smooth[mask_smooth < 0.01] = 1
        heatmap = heatmap_smooth / mask_smooth
        heatmap = np.clip(heatmap, 0, 1)
    
    # Normalize
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    
    return heatmap, scale, (max_x, max_y)


def load_annotations(geojson_path: Path):
    """Load cancer annotations from GeoJSON."""
    with open(geojson_path, 'r') as f:
        data = json.load(f)
    
    annotations = []
    for feature in data.get('features', []):
        props = feature.get('properties', {})
        class_info = props.get('classification', {})
        class_name = class_info.get('name', '')
        
        # Check if cancer-related
        cancer_patterns = ['pattern 3', 'pattern 4', 'pattern 5', 'g3', 'g4', 'g5', 'gleason']
        is_cancer = any(p in class_name.lower() for p in cancer_patterns)
        
        if is_cancer:
            geometry = feature.get('geometry', {})
            if geometry.get('type') == 'Polygon':
                coords = geometry.get('coordinates', [[]])[0]
                annotations.append({
                    'coords': np.array(coords),
                    'class_name': class_name
                })
    
    return annotations


def create_comparison(
    slide_id: str,
    wsi_dir: Path,
    attention_dir: Path,
    annotations_dir: Path,
    output_dir: Path,
    thumbnail_size: int = 2048
):
    """Create side-by-side comparison of attention vs ground truth."""
    
    attention_path = attention_dir / f"{slide_id}_attention.npz"
    geojson_path = annotations_dir / f"{slide_id}.geojson"
    
    if not attention_path.exists():
        print(f"Attention file not found: {attention_path}")
        return
    
    if not geojson_path.exists():
        print(f"Annotation file not found: {geojson_path}")
        return
    
    # Load attention
    print(f"Loading attention for {slide_id}...")
    heatmap, scale, (max_x, max_y) = load_attention(attention_path, thumbnail_size)
    heatmap_height, heatmap_width = heatmap.shape
    
    # Load annotations
    print("Loading annotations...")
    annotations = load_annotations(geojson_path)
    print(f"  Found {len(annotations)} cancer annotations")
    
    # Load WSI thumbnail if available
    thumbnail = None
    wsi_path = None
    for ext in ['.svs', '.ndpi', '.tif', '.tiff']:
        candidate = wsi_dir / f"{slide_id}{ext}"
        if candidate.exists():
            wsi_path = candidate
            break
    
    if wsi_path and HAS_OPENSLIDE:
        print(f"Loading WSI thumbnail...")
        slide = openslide.OpenSlide(str(wsi_path))
        thumbnail = slide.get_thumbnail((heatmap_width, heatmap_height))
        thumbnail = np.array(thumbnail.convert('RGB'))
        slide.close()
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Panel 1: Attention Heatmap
    ax1 = axes[0]
    if thumbnail is not None:
        ax1.imshow(thumbnail)
        cmap = plt.get_cmap('jet')
        heatmap_colored = cmap(heatmap)[:, :, :3]
        heatmap_overlay = (0.5 * heatmap_colored * 255 + 0.5 * thumbnail).astype(np.uint8)
        ax1.imshow(heatmap_overlay)
    else:
        ax1.imshow(heatmap, cmap='jet')
    ax1.set_title('Model Attention', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Panel 2: Ground Truth Annotations
    ax2 = axes[1]
    if thumbnail is not None:
        ax2.imshow(thumbnail)
    else:
        ax2.imshow(np.ones((heatmap_height, heatmap_width, 3)))
    
    # Draw annotation polygons
    patches = []
    for ann in annotations:
        coords = ann['coords'] * scale
        if len(coords) >= 3:
            polygon = MplPolygon(coords, closed=True)
            patches.append(polygon)
    
    if patches:
        pc = PatchCollection(patches, facecolor='red', edgecolor='darkred', 
                            alpha=0.5, linewidth=1)
        ax2.add_collection(pc)
    
    ax2.set_xlim(0, heatmap_width)
    ax2.set_ylim(heatmap_height, 0)
    ax2.set_title('Ground Truth Cancer Regions', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    # Panel 3: Overlay comparison
    ax3 = axes[2]
    if thumbnail is not None:
        ax3.imshow(thumbnail, alpha=0.3)
    
    # Show attention as contours
    ax3.contourf(heatmap, levels=10, cmap='Blues', alpha=0.6)
    
    # Overlay annotation outlines
    for ann in annotations:
        coords = ann['coords'] * scale
        if len(coords) >= 3:
            coords_closed = np.vstack([coords, coords[0]])
            ax3.plot(coords_closed[:, 0], coords_closed[:, 1], 
                    'r-', linewidth=2, label='GT Cancer')
    
    ax3.set_xlim(0, heatmap_width)
    ax3.set_ylim(heatmap_height, 0)
    ax3.set_title('Overlay: Attention (blue) vs GT (red outline)', fontsize=14, fontweight='bold')
    ax3.axis('off')
    
    # Calculate overlap statistics
    gt_mask = np.zeros((heatmap_height, heatmap_width), dtype=np.float32)
    for ann in annotations:
        coords = ann['coords'] * scale
        if len(coords) >= 3:
            from matplotlib.path import Path as MplPath
            path = MplPath(coords)
            x, y = np.meshgrid(np.arange(heatmap_width), np.arange(heatmap_height))
            points = np.column_stack([x.ravel(), y.ravel()])
            mask = path.contains_points(points).reshape((heatmap_height, heatmap_width))
            gt_mask = np.maximum(gt_mask, mask.astype(np.float32))
    
    # Compute statistics
    high_attention_mask = heatmap > 0.5
    
    if gt_mask.sum() > 0:
        # What fraction of high attention is inside GT cancer regions?
        attention_in_cancer = (high_attention_mask & (gt_mask > 0)).sum()
        total_high_attention = high_attention_mask.sum()
        precision = attention_in_cancer / max(total_high_attention, 1)
        
        # What fraction of GT cancer regions have high attention?
        cancer_with_attention = (high_attention_mask & (gt_mask > 0)).sum()
        total_cancer = (gt_mask > 0).sum()
        recall = cancer_with_attention / max(total_cancer, 1)
        
        stats_text = f"Attention-Cancer Overlap:\n"
        stats_text += f"  Precision: {precision:.1%} (high attention in cancer)\n"
        stats_text += f"  Recall: {recall:.1%} (cancer with high attention)"
    else:
        stats_text = "No cancer annotations found"
    
    fig.text(0.5, 0.02, stats_text, ha='center', fontsize=11, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle(f'{slide_id}', fontsize=12)
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    
    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{slide_id}_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"Saved: {output_path}")
    print(stats_text)


def main():
    parser = argparse.ArgumentParser(description='Compare Attention vs Ground Truth')
    parser.add_argument('--slide_id', type=str, required=True,
                        help='Slide ID to analyze')
    parser.add_argument('--wsi_dir', type=str, default=DEFAULT_WSI_DIR)
    parser.add_argument('--attention_dir', type=str, default=DEFAULT_ATTENTION_DIR)
    parser.add_argument('--annotations_dir', type=str, default=DEFAULT_ANNOTATIONS_DIR)
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--thumbnail_size', type=int, default=2048)
    parser.add_argument('--all', action='store_true',
                        help='Process all slides with both attention and annotations')
    
    args = parser.parse_args()
    
    wsi_dir = Path(args.wsi_dir)
    attention_dir = Path(args.attention_dir)
    annotations_dir = Path(args.annotations_dir)
    output_dir = Path(args.output_dir)
    
    if args.all:
        # Find all slides with both attention and annotations
        attention_files = list(attention_dir.glob('*_attention.npz'))
        for att_file in attention_files:
            slide_id = att_file.stem.replace('_attention', '')
            geojson_path = annotations_dir / f"{slide_id}.geojson"
            if geojson_path.exists():
                print(f"\nProcessing {slide_id}...")
                create_comparison(
                    slide_id, wsi_dir, attention_dir, annotations_dir,
                    output_dir, args.thumbnail_size
                )
    else:
        create_comparison(
            args.slide_id, wsi_dir, attention_dir, annotations_dir,
            output_dir, args.thumbnail_size
        )


if __name__ == "__main__":
    main()
