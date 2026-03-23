"""
Feature extraction script for TCGA-PRAD dataset.

This script extracts patch-level features from WSI files using a pre-trained encoder
and saves them as .h5 files. Supports both local WSI files and downloading from
HuggingFace.

TCGA-PRAD Gleason Grading Labels (0-indexed for PyTorch):
    - 0: Normal/Benign
    - 1: Gleason Pattern 3
    - 2: Gleason Pattern 4
    - 3: Gleason Pattern 5

Usage:
    # Extract features from local WSI directory
    python extract_features_tcga_prad.py --wsi_dir ./WSI --output_dir ./features
    
    # With annotation-guided extraction (filters patches by GeoJSON annotations)
    python extract_features_tcga_prad.py --wsi_dir ./WSI --output_dir ./features \\
        --annotations_dir ./annotations/geojsons
    
    # Custom encoder and parameters
    python extract_features_tcga_prad.py --wsi_dir ./WSI --output_dir ./features \\
        --encoder_name uni_v2 --patch_size 256 --magnification 20
"""

import sys
import os
import json
import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
import torch
import numpy as np
from tqdm import tqdm

# For phikon models via transformers
try:
    from transformers import AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Add PatchMIL to path for encoder loading
current_dir = Path(__file__).resolve().parent
patchmil_dir = Path('/local/data/magicscan/PatchMIL')
if str(patchmil_dir) not in sys.path:
    sys.path.insert(0, str(patchmil_dir))


# TCGA-PRAD Gleason Pattern Mapping (0-indexed for PyTorch)
GLEASON_PATTERN_MAP = {
    # Non-cancerous
    'Normal': 0, 'Benign': 0, 'Stroma': 0,
    'normal': 0, 'benign': 0, 'stroma': 0,
    
    # Gleason Pattern 3
    'Gleason Pattern 3': 1, 'Gleason Pattern 3+3': 1, 'Gleason Pattern 3+4': 1,
    'G3': 1, 'GP3': 1, '3+3': 1, '3+4': 1,
    
    # Gleason Pattern 4
    'Gleason Pattern 4': 2, 'Gleason Pattern 4+3': 2, 'Gleason Pattern 4+4': 2,
    'Gleason Pattern 4+5': 2, 'G4': 2, 'GP4': 2, '4+3': 2, '4+4': 2, '4+5': 2,
    
    # Gleason Pattern 5
    'Gleason Pattern 5': 3, 'Gleason Pattern 5+4': 3, 'Gleason Pattern 5+5': 3,
    'G5': 3, 'GP5': 3, '5+4': 3, '5+5': 3,
}

CLASS_NAMES = {
    0: 'Normal/Benign',
    1: 'Gleason 3',
    2: 'Gleason 4',
    3: 'Gleason 5',
}

# Import shapely for polygon operations
try:
    from shapely.geometry import Polygon, Point, box
    from shapely import prepared
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False
    print("Warning: shapely not available. Annotation-guided extraction will be disabled.")


def parse_gleason_pattern(pattern_name: str) -> Optional[int]:
    """Parse a Gleason pattern name to a label."""
    if pattern_name in GLEASON_PATTERN_MAP:
        return GLEASON_PATTERN_MAP[pattern_name]
    
    pattern_lower = pattern_name.lower()
    for key, value in GLEASON_PATTERN_MAP.items():
        if key.lower() in pattern_lower:
            return value
    
    # Try to parse "Gleason Pattern X+Y" format
    if 'gleason' in pattern_lower:
        numbers = re.findall(r'\d+', pattern_name)
        if numbers:
            primary = int(numbers[0])
            if primary == 3:
                return 1
            elif primary == 4:
                return 2
            elif primary == 5:
                return 3
            elif primary <= 2:
                return 0
    
    return None


def parse_geojson_annotations(geojson_path: Path) -> Dict[str, List]:
    """
    Parse GeoJSON annotation file and return polygons grouped by Gleason grade.
    
    Args:
        geojson_path: Path to GeoJSON annotation file
        
    Returns:
        Dictionary mapping class names to lists of Polygon objects
    """
    if not SHAPELY_AVAILABLE:
        raise RuntimeError("shapely is required for annotation parsing")
    
    with open(geojson_path, 'r') as f:
        geojson_data = json.load(f)
    
    features = []
    if 'features' in geojson_data:
        features = geojson_data['features']
    elif isinstance(geojson_data, list):
        features = geojson_data
    
    polygon_map = {}
    for feature in features:
        class_name = None
        if 'properties' in feature:
            props = feature['properties']
            class_name = props.get('classification', {}).get('name') or \
                       props.get('class') or \
                       props.get('name') or \
                       props.get('label')
        
        if class_name is None:
            continue
        
        label = parse_gleason_pattern(class_name)
        if label is None:
            continue
        
        # Use standardized class name
        standardized_name = CLASS_NAMES.get(label, class_name)
        
        if 'geometry' in feature:
            geom = feature['geometry']
            if geom['type'] == 'Polygon':
                polygon_coords = geom['coordinates'][0]
                try:
                    poly = Polygon(polygon_coords)
                    if poly.is_valid:
                        if standardized_name not in polygon_map:
                            polygon_map[standardized_name] = []
                        polygon_map[standardized_name].append((poly, label))
                except Exception as e:
                    print(f"  Warning: Invalid polygon: {e}")
    
    return polygon_map


def get_patches_in_annotations(
    coords: np.ndarray,
    polygon_map: Dict[str, List],
    patch_size: int = 256
) -> Dict[str, Set[int]]:
    """
    Get patch indices that fall within each annotation category.
    
    Args:
        coords: Array of patch coordinates (N, 2) [x, y]
        polygon_map: Dictionary mapping class names to (polygon, label) tuples
        patch_size: Size of patches in pixels
        
    Returns:
        Dictionary mapping class names to sets of patch indices
    """
    if not SHAPELY_AVAILABLE:
        raise RuntimeError("shapely is required")
    
    category_indices = {class_name: set() for class_name in polygon_map.keys()}
    
    prepared_polygons = {}
    for class_name, poly_list in polygon_map.items():
        prepared_polygons[class_name] = [(prepared.prep(p), l) for p, l in poly_list]
    
    half_patch = patch_size // 2
    
    for idx, (x, y) in enumerate(coords):
        patch_center = Point(x + half_patch, y + half_patch)
        
        for class_name, poly_list in polygon_map.items():
            prep_polys = prepared_polygons[class_name]
            
            for prep_poly, label in prep_polys:
                if prep_poly.contains(patch_center):
                    category_indices[class_name].add(idx)
                    break
    
    return category_indices


def filter_patches_by_annotations(
    coords: np.ndarray,
    polygon_map: Dict[str, List],
    patch_size: int = 256,
    min_patch_coverage: float = 0.10,
    min_annotation_coverage: float = 0.30
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Filter patches based on annotation coverage.
    
    Args:
        coords: Array of patch coordinates (N, 2)
        polygon_map: Dictionary mapping class names to (polygon, label) tuples
        patch_size: Size of patches in pixels
        min_patch_coverage: Minimum patch area coverage by annotation
        min_annotation_coverage: Minimum annotation area coverage by patch
        
    Returns:
        Tuple of (filtered_indices, filtered_labels, filtered_class_names)
    """
    if not SHAPELY_AVAILABLE:
        raise RuntimeError("shapely is required")
    
    filtered_indices = []
    filtered_labels = []
    filtered_class_names = []
    
    patch_area = patch_size * patch_size
    
    prepared_polygons = {}
    for class_name, poly_list in polygon_map.items():
        prepared_polygons[class_name] = [(prepared.prep(p), p, l) for p, l in poly_list]
    
    for idx, (x, y) in enumerate(coords):
        patch_box = box(x, y, x + patch_size, y + patch_size)
        
        max_label = -1
        assigned_class = None
        highest_intersecting_label = -1
        
        for class_name, poly_list in polygon_map.items():
            prep_polys = prepared_polygons[class_name]
            
            for prep_poly, poly, label in prep_polys:
                if prep_poly.intersects(patch_box):
                    if label > highest_intersecting_label:
                        highest_intersecting_label = label
                    
                    intersection = patch_box.intersection(poly)
                    intersection_area = intersection.area
                    
                    patch_coverage = intersection_area / patch_area
                    annotation_coverage = intersection_area / poly.area if poly.area > 0 else 0
                    
                    if patch_coverage >= min_patch_coverage or annotation_coverage >= min_annotation_coverage:
                        if label > max_label:
                            max_label = label
                            assigned_class = class_name
                        break
        
        if assigned_class is not None and highest_intersecting_label <= max_label:
            filtered_indices.append(idx)
            filtered_labels.append(max_label)
            filtered_class_names.append(assigned_class)
    
    return np.array(filtered_indices), np.array(filtered_labels), filtered_class_names


def extract_features_from_wsi(
    wsi_path: Path,
    output_dir: Path,
    encoder_name: str = 'uni_v2',
    patch_size: int = 256,
    target_magnification: int = 20,
    overlap: int = 0,
    batch_size: int = 512,
    device: str = 'cuda:0',
    save_coords: bool = True,
    default_mpp: float = 0.5,
    annotations_path: Optional[Path] = None,
    min_patch_coverage: float = 0.10,
    min_annotation_coverage: float = 0.30,
    save_patch_labels: bool = True
) -> bool:
    """
    Extract features from a single WSI file.
    
    Args:
        wsi_path: Path to WSI file
        output_dir: Directory to save feature .h5 files
        encoder_name: Name of the encoder (e.g., 'uni_v2')
        patch_size: Size of patches in pixels
        target_magnification: Target magnification level
        overlap: Overlap between patches in pixels
        batch_size: Batch size for feature extraction
        device: Device to use
        save_coords: Whether to save patch coordinates
        default_mpp: Default MPP when metadata is missing
        annotations_path: Optional path to GeoJSON annotation file
        min_patch_coverage: Minimum patch coverage threshold
        min_annotation_coverage: Minimum annotation coverage threshold
        save_patch_labels: Whether to save patch-level labels
        
    Returns:
        True if successful, False otherwise
    """
    from src.inference.wsi_objects import load_wsi
    from src.inference.wsi_objects.IO import get_weights_path
    
    print(f"Loading WSI: {wsi_path.name}")
    try:
        try:
            wsi = load_wsi(str(wsi_path), lazy_init=False)
        except Exception as e:
            error_msg = str(e)
            if "Unable to extract MPP" in error_msg or "MPP" in error_msg:
                print(f"⚠️  MPP metadata missing, using default MPP={default_mpp}")
                wsi = load_wsi(str(wsi_path), lazy_init=False, mpp=default_mpp)
            else:
                raise
    except Exception as e:
        print(f"❌ Error loading WSI {wsi_path.name}: {e}")
        return False
    
    # Load encoder
    print(f"Loading encoder: {encoder_name}")
    try:
        import timm
        from torchvision import transforms
        
        # Encoder configurations
        # PUBLIC encoders (no approval required):
        #   - phikon, phikon_v2: Owkin's pathology foundation models
        #   - resnet50, resnet101: ImageNet pretrained CNNs
        #   - dinov2_vitb14, dinov2_vitl14: Meta's DINOv2 models
        # GATED encoders (require HuggingFace approval):
        #   - uni, uni_v2, uni2_h: MahmoodLab's UNI models
        #   - virchow2: Paige AI's Virchow2 model
        
        encoder_configs = {
            # ===== PUBLIC ENCODERS (No approval required) =====
            'phikon': {
                'model_name': 'hf-hub:owkin/phikon',
                'feature_dim': 768,
                'input_size': 224,
                'public': True,
            },
            'phikon_v2': {
                'model_name': 'hf-hub:owkin/phikon-v2',
                'feature_dim': 1024,
                'input_size': 224,
                'public': True,
            },
            'resnet50': {
                'model_name': 'resnet50',
                'feature_dim': 2048,
                'input_size': 224,
                'public': True,
            },
            'resnet101': {
                'model_name': 'resnet101',
                'feature_dim': 2048,
                'input_size': 224,
                'public': True,
            },
            'dinov2_vitb14': {
                'model_name': 'vit_base_patch14_dinov2',
                'feature_dim': 768,
                'input_size': 224,
                'public': True,
            },
            'dinov2_vitl14': {
                'model_name': 'vit_large_patch14_dinov2',
                'feature_dim': 1024,
                'input_size': 224,
                'public': True,
            },
            
            # ===== GATED ENCODERS (Require HuggingFace approval) =====
            'uni': {
                'model_name': 'hf-hub:MahmoodLab/uni',
                'feature_dim': 1024,
                'input_size': 224,
                'public': False,
            },
            'uni_v2': {
                'model_name': 'hf-hub:MahmoodLab/uni',
                'feature_dim': 1024,
                'input_size': 224,
                'public': False,
            },
            'uni2_h': {
                'model_name': 'hf-hub:MahmoodLab/UNI2-h',
                'feature_dim': 1536,
                'input_size': 224,
                'public': False,
            },
            'virchow2': {
                'model_name': 'hf-hub:paige-ai/Virchow2',
                'feature_dim': 1280,
                'input_size': 224,
                'public': False,
            },
        }
        
        config = encoder_configs.get(encoder_name.lower(), None)
        
        if config:
            model_name = config['model_name']
            feature_dim = config.get('feature_dim', 1024)
            input_size = config.get('input_size', 224)
            
            # Special handling for phikon models using transformers
            if encoder_name.lower() in ['phikon', 'phikon_v2']:
                if not TRANSFORMERS_AVAILABLE:
                    raise ImportError("transformers library required for phikon models. Install with: pip install transformers")
                
                hf_model_id = 'owkin/phikon-v2' if encoder_name.lower() == 'phikon_v2' else 'owkin/phikon'
                print(f"Loading encoder from HuggingFace (transformers): {hf_model_id}")
                
                try:
                    patch_encoder = AutoModel.from_pretrained(hf_model_id, trust_remote_code=True)
                    print(f"  - Model loaded successfully via transformers")
                    print(f"  - Feature dimension: {feature_dim}")
                except Exception as e:
                    print(f"❌ Could not load {hf_model_id}: {e}")
                    print("   Make sure you have HuggingFace access and are logged in")
                    raise
            else:
                # Use timm for other models
                print(f"Loading encoder from HuggingFace: {model_name}")
                
                try:
                    patch_encoder = timm.create_model(
                        model_name, 
                        pretrained=True, 
                        num_classes=0,
                        dynamic_img_size=True
                    )
                    print(f"  - Model loaded successfully")
                    print(f"  - Feature dimension: {feature_dim}")
                except Exception as e:
                    print(f"❌ Could not load {model_name}: {e}")
                    print("   Make sure you have HuggingFace access and are logged in")
                    raise
        else:
            encoder_weights_path = get_weights_path('patch', encoder_name)
            
            if encoder_weights_path and os.path.exists(encoder_weights_path):
                print(f"Loading encoder from local weights: {encoder_weights_path}")
                try:
                    patch_encoder = timm.create_model(encoder_name, pretrained=True, num_classes=0)
                except:
                    patch_encoder = timm.create_model('resnet50', pretrained=True, num_classes=0)
                    state_dict = torch.load(encoder_weights_path, map_location='cpu')
                    patch_encoder.load_state_dict(state_dict, strict=False)
            else:
                print(f"Loading encoder: {encoder_name}")
                try:
                    patch_encoder = timm.create_model(encoder_name, pretrained=True, num_classes=0)
                except Exception as e:
                    print(f"⚠️  Could not load {encoder_name}: {e}")
                    print(f"   Falling back to resnet50")
                    patch_encoder = timm.create_model('resnet50', pretrained=True, num_classes=0)
            
            input_size = 224
        
        patch_encoder = patch_encoder.to(device)
        patch_encoder.eval()
        
        patch_encoder.eval_transforms = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        patch_encoder.precision = torch.float16 if device.startswith('cuda') else torch.float32
        patch_encoder.enc_name = encoder_name
        
        print(f"✓ Encoder loaded and moved to {device}")
        
    except Exception as e:
        print(f"❌ Error loading encoder: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse annotations if provided
    polygon_map = None
    if annotations_path is not None and SHAPELY_AVAILABLE:
        if annotations_path.exists():
            print(f"Loading annotations from: {annotations_path.name}")
            try:
                polygon_map = parse_geojson_annotations(annotations_path)
                if polygon_map:
                    total_polygons = sum(len(polys) for polys in polygon_map.values())
                    print(f"  - Found {total_polygons} annotation polygons across {len(polygon_map)} classes")
                    for class_name, polys in polygon_map.items():
                        print(f"    - {class_name}: {len(polys)} polygons")
                else:
                    print(f"  - No valid annotations found")
            except Exception as e:
                print(f"⚠️  Error parsing annotations: {e}")
                polygon_map = None
        else:
            print(f"⚠️  Annotations file not found: {annotations_path}")
    
    # Extract features
    print(f"Extracting features at {target_magnification}x magnification, patch size {patch_size}px...")
    try:
        patcher = wsi.create_patcher(
            patch_size=patch_size,
            src_mag=wsi.mag,
            dst_mag=target_magnification,
            overlap=overlap,
            coords_only=False,
            pil=True
        )
        
        from src.inference.wsi_objects.WSIPatcherDataset import WSIPatcherDataset
        from torch.utils.data import DataLoader
        
        patch_transforms = patch_encoder.eval_transforms
        dataset = WSIPatcherDataset(patcher, patch_transforms)
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            num_workers=4, 
            pin_memory=(device != 'cpu')
        )
        
        features = []
        coords = []
        
        precision = getattr(patch_encoder, 'precision', torch.float32)
        use_amp = precision != torch.float32 and device.startswith('cuda')
        
        with torch.no_grad():
            for batch_patches, batch_coords in tqdm(dataloader, desc=f"Extracting {wsi.name}"):
                batch_patches = batch_patches.to(device)
                
                if use_amp:
                    with torch.autocast(device_type='cuda', dtype=precision):
                        batch_features = patch_encoder(batch_patches)
                else:
                    batch_features = patch_encoder(batch_patches)
                
                # Handle transformers output (BaseModelOutputWithPooling) vs timm tensor output
                if hasattr(batch_features, 'pooler_output') and batch_features.pooler_output is not None:
                    # Transformers model with pooler output (e.g., phikon)
                    batch_features = batch_features.pooler_output
                elif hasattr(batch_features, 'last_hidden_state'):
                    # Transformers model - use CLS token
                    batch_features = batch_features.last_hidden_state[:, 0, :]
                
                features.append(batch_features.float().cpu().numpy())
                
                if save_coords:
                    if isinstance(batch_coords, torch.Tensor):
                        coords.append(batch_coords.cpu().numpy())
                    elif isinstance(batch_coords, (list, tuple)):
                        for coord in batch_coords:
                            if isinstance(coord, torch.Tensor):
                                coords.append(coord.cpu().numpy())
                            elif isinstance(coord, (list, tuple)):
                                coords.append(np.array(coord))
                            else:
                                coords.append(coord)
                    else:
                        coords.append(np.array(batch_coords))
        
        features = np.concatenate(features, axis=0)
        
        if coords:
            try:
                coords_array = np.stack(coords)
            except (ValueError, TypeError):
                try:
                    coords_array = np.vstack(coords)
                except (ValueError, TypeError):
                    coords_array = np.concatenate([c.reshape(-1, 2) if c.ndim == 1 else c for c in coords], axis=0)
        else:
            coords_array = None
        
        # Apply annotation-based filtering if available
        patch_labels = None
        category_indices = None
        
        if polygon_map and coords_array is not None:
            print(f"Filtering patches by annotations...")
            
            category_indices = get_patches_in_annotations(
                coords_array, polygon_map, patch_size
            )
            
            total_in_annotations = sum(len(indices) for indices in category_indices.values())
            print(f"  - Total patches: {len(features)}")
            print(f"  - Patches in annotations: {total_in_annotations}")
            for class_name, indices in category_indices.items():
                print(f"    - {class_name}: {len(indices)} patches")
            
            if save_patch_labels:
                filtered_indices, filtered_labels, filtered_class_names = filter_patches_by_annotations(
                    coords_array, polygon_map, patch_size,
                    min_patch_coverage, min_annotation_coverage
                )
                print(f"  - Patches meeting coverage thresholds: {len(filtered_indices)}")
                
                patch_labels = np.full(len(features), -1, dtype=np.int32)
                patch_labels[filtered_indices] = filtered_labels
        
        # Save to .h5 file
        output_file = output_dir / f"{wsi.name}.h5"
        
        import h5py
        with h5py.File(output_file, 'w') as f:
            f.create_dataset('features', data=features, compression='gzip')
            
            if save_coords and coords_array is not None:
                f.create_dataset('coords', data=coords_array, compression='gzip')
            
            if patch_labels is not None:
                f.create_dataset('patch_labels', data=patch_labels, compression='gzip')
            
            if category_indices is not None:
                cat_group = f.create_group('category_indices')
                for class_name, indices in category_indices.items():
                    if len(indices) > 0:
                        cat_group.create_dataset(
                            class_name, 
                            data=np.array(sorted(indices), dtype=np.int32),
                            compression='gzip'
                        )
            
            f.attrs['encoder'] = encoder_name
            f.attrs['patch_size'] = patch_size
            f.attrs['target_magnification'] = target_magnification
            f.attrs['overlap'] = overlap
            f.attrs['num_patches'] = len(features)
            f.attrs['feature_dim'] = features.shape[1] if len(features.shape) > 1 else features.shape[0]
            f.attrs['dataset'] = 'TCGA-PRAD'
            
            if polygon_map:
                f.attrs['has_annotations'] = True
                f.attrs['annotation_classes'] = list(polygon_map.keys())
            else:
                f.attrs['has_annotations'] = False
        
        print(f"✓ Saved features to: {output_file}")
        print(f"  - Features shape: {features.shape}")
        print(f"  - Number of patches: {len(features)}")
        if patch_labels is not None:
            labeled_count = np.sum(patch_labels >= 0)
            print(f"  - Labeled patches: {labeled_count}")
        return True
        
    except Exception as e:
        print(f"❌ Error extracting features: {e}")
        import traceback
        traceback.print_exc()
        return False


def extract_features_from_directory(
    wsi_dir: Path,
    output_dir: Path,
    encoder_name: str = 'uni_v2',
    patch_size: int = 256,
    target_magnification: int = 20,
    overlap: int = 0,
    batch_size: int = 512,
    device: str = 'cuda:0',
    wsi_extensions: tuple = ('.svs', '.tif', '.tiff', '.ndpi', '.mrxs', '.scn', '.bif'),
    default_mpp: float = 0.5,
    annotations_dir: Optional[Path] = None,
    min_patch_coverage: float = 0.10,
    min_annotation_coverage: float = 0.30,
    save_patch_labels: bool = True
):
    """
    Extract features from all WSI files in a directory.
    
    Args:
        wsi_dir: Directory containing WSI files
        output_dir: Directory to save feature .h5 files
        encoder_name: Name of the encoder
        patch_size: Size of patches in pixels
        target_magnification: Target magnification level
        overlap: Overlap between patches
        batch_size: Batch size for extraction
        device: Device to use
        wsi_extensions: Valid WSI file extensions
        default_mpp: Default MPP when metadata is missing
        annotations_dir: Optional directory with GeoJSON annotations
        min_patch_coverage: Minimum patch coverage threshold
        min_annotation_coverage: Minimum annotation coverage threshold
        save_patch_labels: Whether to save patch-level labels
    """
    wsi_dir = Path(wsi_dir)
    output_dir = Path(output_dir)
    
    # Find all WSI files
    wsi_files = []
    for ext in wsi_extensions:
        wsi_files.extend(wsi_dir.glob(f"*{ext}"))
        wsi_files.extend(wsi_dir.glob(f"*{ext.upper()}"))
    
    if not wsi_files:
        print(f"❌ No WSI files found in {wsi_dir}")
        return
    
    # Build annotation mapping
    annotation_map = {}
    if annotations_dir is not None:
        annotations_dir = Path(annotations_dir)
        if annotations_dir.exists():
            geojson_files = list(annotations_dir.rglob("*.geojson"))
            for geojson_file in geojson_files:
                slide_id = geojson_file.stem
                annotation_map[slide_id] = geojson_file
            print(f"Found {len(annotation_map)} annotation files")
    
    print(f"Found {len(wsi_files)} WSI files")
    print(f"Output directory: {output_dir}")
    print(f"Encoder: {encoder_name}")
    print(f"Parameters: {patch_size}px patches at {target_magnification}x, overlap={overlap}px")
    print("=" * 80)
    
    success_count = 0
    failed_files = []
    
    for wsi_path in tqdm(wsi_files, desc="Processing WSI files"):
        slide_id = wsi_path.stem
        
        # Try multiple ways to match annotation files
        annotations_path = None
        
        # Direct match by slide ID
        if slide_id in annotation_map:
            annotations_path = annotation_map[slide_id]
        else:
            # Try matching without UUID suffix (TCGA format)
            base_id = slide_id.split('.')[0] if '.' in slide_id else slide_id
            for ann_id, ann_path in annotation_map.items():
                if base_id in ann_id or ann_id in base_id:
                    annotations_path = ann_path
                    break
        
        if annotations_dir and not annotations_path:
            print(f"⚠️  No annotations found for {slide_id}")
        
        success = extract_features_from_wsi(
            wsi_path=wsi_path,
            output_dir=output_dir,
            encoder_name=encoder_name,
            patch_size=patch_size,
            target_magnification=target_magnification,
            overlap=overlap,
            batch_size=batch_size,
            device=device,
            default_mpp=default_mpp,
            annotations_path=annotations_path,
            min_patch_coverage=min_patch_coverage,
            min_annotation_coverage=min_annotation_coverage,
            save_patch_labels=save_patch_labels
        )
        
        if success:
            success_count += 1
        else:
            failed_files.append(wsi_path.name)
        
        print()
    
    # Summary
    print("=" * 80)
    print("FEATURE EXTRACTION SUMMARY")
    print("=" * 80)
    print(f"Total WSI files: {len(wsi_files)}")
    print(f"Successfully processed: {success_count}")
    print(f"Failed: {len(failed_files)}")
    
    if failed_files:
        print(f"\nFailed files:")
        for fname in failed_files:
            print(f"  - {fname}")
    
    print(f"\n✓ Features saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract patch-level features from TCGA-PRAD WSI files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic extraction from local WSI directory
  python extract_features_tcga_prad.py --wsi_dir ./WSI --output_dir ./features
  
  # With custom encoder and parameters
  python extract_features_tcga_prad.py --wsi_dir ./WSI --output_dir ./features \\
      --encoder_name uni_v2 --patch_size 256 --magnification 20
  
  # Annotation-guided extraction
  python extract_features_tcga_prad.py --wsi_dir ./WSI --output_dir ./features \\
      --annotations_dir ./annotations/geojsons

TCGA-PRAD Gleason Pattern Labels (0-indexed for PyTorch):
  0: Normal/Benign
  1: Gleason Pattern 3 (well-differentiated)
  2: Gleason Pattern 4 (moderately differentiated)  
  3: Gleason Pattern 5 (poorly differentiated)

Available Encoders:
  PUBLIC (no approval required):
    - phikon, phikon_v2: Owkin pathology foundation models
    - resnet50, resnet101: ImageNet pretrained CNNs
    - dinov2_vitb14, dinov2_vitl14: Meta DINOv2 models
  
  GATED (require HuggingFace approval):
    - uni, uni_v2, uni2_h: MahmoodLab UNI models
    - virchow2: Paige AI Virchow2 model
        """
    )
    
    parser.add_argument('--wsi_dir', type=str, default=None,
                       help='Directory containing WSI files')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save feature .h5 files')
    parser.add_argument('--annotations_dir', type=str, default=None,
                       help='Directory containing GeoJSON annotation files')
    parser.add_argument('--encoder_name', type=str, default='phikon_v2',
                       help='Encoder name (default: phikon_v2). Public: phikon, phikon_v2, resnet50, dinov2_vitb14. Gated: uni_v2, uni2_h, virchow2')
    parser.add_argument('--patch_size', type=int, default=256,
                       help='Patch size in pixels (default: 256)')
    parser.add_argument('--magnification', type=int, default=20, dest='target_magnification',
                       help='Target magnification level (default: 20)')
    parser.add_argument('--overlap', type=int, default=0,
                       help='Overlap between patches in pixels (default: 0)')
    parser.add_argument('--batch_size', type=int, default=512,
                       help='Batch size for feature extraction (default: 512)')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use (default: cuda:0)')
    parser.add_argument('--default_mpp', type=float, default=0.5,
                       help='Default MPP when metadata is missing (default: 0.5)')
    parser.add_argument('--min_patch_coverage', type=float, default=0.10,
                       help='Minimum patch coverage by annotation (default: 0.10)')
    parser.add_argument('--min_annotation_coverage', type=float, default=0.30,
                       help='Minimum annotation coverage by patch (default: 0.30)')
    parser.add_argument('--no_patch_labels', action='store_true', default=False,
                       help='Do not save patch-level labels')
    
    args = parser.parse_args()
    
    # Set defaults
    base_dir = Path(__file__).resolve().parent
    wsi_dir = Path(args.wsi_dir) if args.wsi_dir else base_dir / "WSI"
    output_dir = Path(args.output_dir) if args.output_dir else base_dir / "output" / "features"
    annotations_dir = Path(args.annotations_dir) if args.annotations_dir else base_dir / "annotations" / "geojsons"
    
    # Validate inputs
    if not wsi_dir.exists():
        print(f"❌ WSI directory not found: {wsi_dir}")
        return
    
    if not annotations_dir.exists():
        print(f"⚠️  Annotations directory not found: {annotations_dir}")
        annotations_dir = None
    
    # Check device
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        print(f"⚠️  CUDA not available, switching to CPU")
        args.device = 'cpu'
    
    # Run extraction
    extract_features_from_directory(
        wsi_dir=wsi_dir,
        output_dir=output_dir,
        encoder_name=args.encoder_name,
        patch_size=args.patch_size,
        target_magnification=args.target_magnification,
        overlap=args.overlap,
        batch_size=args.batch_size,
        device=args.device,
        default_mpp=args.default_mpp,
        annotations_dir=annotations_dir,
        min_patch_coverage=args.min_patch_coverage,
        min_annotation_coverage=args.min_annotation_coverage,
        save_patch_labels=not args.no_patch_labels
    )


if __name__ == "__main__":
    main()
