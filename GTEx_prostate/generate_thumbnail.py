#!/usr/bin/env python3
"""
Generate thumbnail maps for whole slide images (WSI).
Creates a ground truth visualization of the slide.

Usage:
    python generate_thumbnail.py --slide histology_images_prostate/GTEX-1A8G6-2126.svs
    python generate_thumbnail.py --slide histology_images_prostate/GTEX-1A8G6-2126.svs --output output/thumbnails/
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np

try:
    import openslide
    HAS_OPENSLIDE = True
except ImportError:
    HAS_OPENSLIDE = False
    print("Error: openslide-python is required. Install with: pip install openslide-python")
    sys.exit(1)

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("Error: Pillow is required. Install with: pip install Pillow")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as patheffects
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def generate_thumbnail(
    slide_path: Path,
    output_path: Path = None,
    thumbnail_size: int = 2048,
    add_info: bool = True,
    add_scale_bar: bool = True
):
    """
    Generate a thumbnail map for a whole slide image.
    
    Args:
        slide_path: Path to the SVS/WSI file
        output_path: Path to save the thumbnail (optional)
        thumbnail_size: Maximum dimension for thumbnail
        add_info: Whether to add slide information text
        add_scale_bar: Whether to add a scale bar
        
    Returns:
        thumbnail: PIL Image of the thumbnail
    """
    slide_path = Path(slide_path)
    
    if not slide_path.exists():
        raise FileNotFoundError(f"Slide not found: {slide_path}")
    
    # Open the slide
    print(f"Opening slide: {slide_path.name}")
    slide = openslide.OpenSlide(str(slide_path))
    
    # Get slide dimensions
    width, height = slide.dimensions
    print(f"  Dimensions: {width:,} x {height:,} pixels")
    
    # Get other properties
    mpp_x = slide.properties.get(openslide.PROPERTY_NAME_MPP_X)
    mpp_y = slide.properties.get(openslide.PROPERTY_NAME_MPP_Y)
    magnification = slide.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER)
    vendor = slide.properties.get(openslide.PROPERTY_NAME_VENDOR, 'Unknown')
    
    if mpp_x:
        print(f"  Microns per pixel: {float(mpp_x):.4f}")
    if magnification:
        print(f"  Objective magnification: {magnification}x")
    print(f"  Vendor: {vendor}")
    print(f"  Level count: {slide.level_count}")
    
    # Calculate thumbnail dimensions maintaining aspect ratio
    scale = min(thumbnail_size / width, thumbnail_size / height)
    thumb_width = int(width * scale)
    thumb_height = int(height * scale)
    
    print(f"  Generating thumbnail: {thumb_width} x {thumb_height}")
    
    # Get thumbnail
    thumbnail = slide.get_thumbnail((thumb_width, thumb_height))
    thumbnail = thumbnail.convert('RGB')
    
    # Create output with matplotlib if info is requested
    if add_info and HAS_MATPLOTLIB:
        fig, ax = plt.subplots(1, 1, figsize=(14, 12))
        
        # Display thumbnail
        ax.imshow(np.array(thumbnail))
        ax.axis('off')
        
        # No title - clean image only
        
        # Add scale bar if MPP is available
        if add_scale_bar and mpp_x:
            # Calculate 1mm in thumbnail pixels
            mm_in_pixels = (1000 / float(mpp_x)) * scale
            
            # Choose appropriate scale bar length
            if mm_in_pixels > 50:
                bar_mm = 1
            else:
                bar_mm = 5
            
            bar_pixels = bar_mm * mm_in_pixels
            
            # Position scale bar in bottom-right corner
            bar_x = thumb_width - bar_pixels - 20
            bar_y = thumb_height - 30
            
            # Draw scale bar
            ax.plot([bar_x, bar_x + bar_pixels], [bar_y, bar_y], 
                   'w-', linewidth=4)
            ax.plot([bar_x, bar_x + bar_pixels], [bar_y, bar_y], 
                   'k-', linewidth=2)
            ax.text(bar_x + bar_pixels/2, bar_y - 15, f'{bar_mm} mm',
                   ha='center', va='bottom', fontsize=10, 
                   fontweight='bold', color='white',
                   path_effects=[patheffects.withStroke(
                       linewidth=2, foreground='black')])
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print(f"  Saved to: {output_path}")
        
        plt.close(fig)
    else:
        # Save simple thumbnail
        if output_path:
            thumbnail.save(output_path)
            print(f"  Saved to: {output_path}")
    
    slide.close()
    return thumbnail


def main():
    parser = argparse.ArgumentParser(
        description='Generate thumbnail maps for whole slide images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python generate_thumbnail.py --slide histology_images_prostate/GTEX-1A8G6-2126.svs
    python generate_thumbnail.py --slide histology_images_prostate/GTEX-1A8G6-2126.svs --output output/thumbnails/
    python generate_thumbnail.py --slide histology_images_prostate/GTEX-1A8G6-2126.svs --size 4096
        """
    )
    
    parser.add_argument('--slide', type=str, required=True,
                        help='Path to the SVS/WSI file')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path (file or directory)')
    parser.add_argument('--size', type=int, default=2048,
                        help='Maximum thumbnail dimension (default: 2048)')
    parser.add_argument('--no-info', action='store_true',
                        help='Do not add slide info and scale bar')
    
    args = parser.parse_args()
    
    slide_path = Path(args.slide)
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
        if output_path.is_dir() or str(args.output).endswith('/'):
            output_path.mkdir(parents=True, exist_ok=True)
            output_path = output_path / f"{slide_path.stem}_thumbnail.png"
    else:
        # Default: save in output/thumbnails/
        output_dir = Path('output/thumbnails')
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{slide_path.stem}_thumbnail.png"
    
    generate_thumbnail(
        slide_path=slide_path,
        output_path=output_path,
        thumbnail_size=args.size,
        add_info=not args.no_info,
        add_scale_bar=not args.no_info
    )
    
    print("\nDone!")


if __name__ == '__main__':
    main()
