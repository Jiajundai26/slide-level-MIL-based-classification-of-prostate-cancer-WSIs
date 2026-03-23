#!/usr/bin/env python
"""
Generate ground-truth overlay maps by drawing GeoJSON annotations
on top of a WSI thumbnail.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import openslide

from extract_features_tcga_prad import parse_geojson_annotations, CLASS_NAMES


LABEL_COLORS = {
    0: (200, 200, 200, 80),   # Normal/Benign (light gray)
    1: (102, 204, 102, 110),  # Gleason 3 (green)
    2: (0, 170, 170, 110),    # Gleason 4 (teal, contrasts with pink)
    3: (106, 90, 205, 110),   # Gleason 5 (purple)
}


def polygon_to_scaled_coords(poly, scale):
    coords = np.array(poly.exterior.coords)
    coords = coords / scale
    return [tuple(pt) for pt in coords]

def get_font(image_size):
    font_size = max(24, int(min(image_size) * 0.04))
    try:
        return ImageFont.truetype("DejaVuSans.ttf", font_size)
    except OSError:
        return ImageFont.load_default()


def render_wsi_thumbnail(wsi_path, target_size, max_dim):
    slide = openslide.OpenSlide(str(wsi_path))
    width, height = slide.dimensions

    if target_size is not None:
        target_w, target_h = target_size
        scale = max(width / float(target_w), height / float(target_h))
    else:
        scale = max(width, height) / float(max_dim)

    if scale < 1.0:
        scale = 1.0

    level = slide.get_best_level_for_downsample(scale)
    level_dims = slide.level_dimensions[level]
    base = slide.read_region((0, 0), level, level_dims).convert("RGBA")

    if target_size is not None and base.size != target_size:
        base = base.resize(target_size, Image.BILINEAR)

    downsample = width / float(base.size[0])
    return base, downsample


def draw_gleason_labels(draw, polygon_map, downsample, font):
    for class_name, poly_list in polygon_map.items():
        for poly, label in poly_list:
            if label is None:
                continue
            text = class_name or CLASS_NAMES.get(label, f"Gleason {label}")
            centroid = poly.centroid
            x = centroid.x / downsample
            y = centroid.y / downsample
            draw.text((x, y), text, fill=(0, 0, 0, 255), font=font, anchor="mm")


def render_overlay_from_geojson(wsi_path, annotation_path, output_path, max_dim, add_labels=False):
    base, downsample = render_wsi_thumbnail(wsi_path, target_size=None, max_dim=max_dim)
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")

    polygon_map = parse_geojson_annotations(Path(annotation_path))
    for _, poly_list in polygon_map.items():
        for poly, label in poly_list:
            color = LABEL_COLORS.get(label, (255, 255, 255, 80))
            coords = polygon_to_scaled_coords(poly, downsample)
            if len(coords) >= 3:
                draw.polygon(coords, fill=color, outline=color)

    if add_labels:
        font = get_font(base.size)
        draw_gleason_labels(draw, polygon_map, downsample, font)

    combined = Image.alpha_composite(base, overlay)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.save(output_path)


def render_overlay_from_label_png(wsi_path, label_png_path, output_path, max_dim, label_alpha, annotation_path=None):
    label_img = Image.open(label_png_path).convert("RGBA")
    label_w, label_h = label_img.size

    base, downsample = render_wsi_thumbnail(wsi_path, target_size=(label_w, label_h), max_dim=max_dim)

    label_arr = np.array(label_img)
    mask = (label_arr[:, :, :3].sum(axis=2) > 0)
    alpha = (mask.astype(np.uint8) * int(255 * label_alpha))
    color = (255, 255, 255)
    dominant_label = None
    if annotation_path:
        polygon_map = parse_geojson_annotations(Path(annotation_path))
        labels = [label for poly_list in polygon_map.values() for _, label in poly_list if label is not None]
        if labels:
            dominant_label = max(labels)
            color = LABEL_COLORS.get(dominant_label, (255, 255, 255, 80))[:3]
    label_arr[:, :, 0] = color[0]
    label_arr[:, :, 1] = color[1]
    label_arr[:, :, 2] = color[2]
    label_arr[:, :, 3] = alpha
    label_img = Image.fromarray(label_arr, mode="RGBA")

    combined = Image.alpha_composite(base, label_img)

    if annotation_path:
        polygon_map = parse_geojson_annotations(Path(annotation_path))
        draw = ImageDraw.Draw(combined, "RGBA")
        font = get_font(combined.size)
        draw_gleason_labels(draw, polygon_map, downsample, font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.save(output_path)


def render_overlay(wsi_path, annotation_path, output_path, max_dim, label_png_path=None, label_alpha=0.4):
    if label_png_path:
        render_overlay_from_label_png(wsi_path, label_png_path, output_path, max_dim, label_alpha, annotation_path)
    else:
        render_overlay_from_geojson(wsi_path, annotation_path, output_path, max_dim, add_labels=True)


def main():
    parser = argparse.ArgumentParser(description="Generate GT overlay maps from GeoJSON")
    parser.add_argument("--wsi_path", type=str, required=True, help="Path to WSI .svs file")
    parser.add_argument("--annotation_path", type=str, default=None, help="Path to GeoJSON annotation")
    parser.add_argument("--label_png", type=str, default=None, help="Path to low_quality label PNG")
    parser.add_argument("--label_alpha", type=float, default=0.4, help="Alpha for label PNG overlay")
    parser.add_argument("--output_path", type=str, required=True, help="Output PNG path")
    parser.add_argument("--max_dim", type=int, default=4096, help="Max dimension of output image")
    args = parser.parse_args()

    if args.label_png is None and args.annotation_path is None:
        raise ValueError("Provide --annotation_path or --label_png")

    render_overlay(
        wsi_path=Path(args.wsi_path),
        annotation_path=Path(args.annotation_path) if args.annotation_path else None,
        output_path=Path(args.output_path),
        max_dim=args.max_dim,
        label_png_path=Path(args.label_png) if args.label_png else None,
        label_alpha=args.label_alpha
    )


if __name__ == "__main__":
    main()
