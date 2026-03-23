"""
TCGA-PRAD HuggingFace Dataloader.

This module provides dataloaders for the TCGA-PRAD (Prostate Adenocarcinoma) dataset
from HuggingFace. It supports loading both locally available data and downloading
from the Codatta/Refined-TCGA-PRAD-Prostate-Cancer-Pathology-Dataset on HuggingFace.

TCGA-PRAD Gleason Grading Labels (0-indexed for PyTorch):
    - 0: Normal/Benign (non-cancerous tissue)
    - 1: Gleason Pattern 3 (well-differentiated)
    - 2: Gleason Pattern 4 (moderately differentiated)
    - 3: Gleason Pattern 5 (poorly differentiated)

Usage:
    # Load local data with pre-extracted features
    from tcga_prad_hf_dataloader import TCGAPRADDataset, create_tcgaprad_dataloaders
    
    loaders = create_tcgaprad_dataloaders(
        feats_path='./features',
        labels_tsv='./labels.tsv',
        batch_size=1
    )
    
    # Download from HuggingFace
    from tcga_prad_hf_dataloader import download_tcgaprad_from_huggingface
    
    download_tcgaprad_from_huggingface(output_dir='./data')
"""

import os
import sys
import json
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Set
from dataclasses import dataclass


# TCGA-PRAD Gleason Pattern to Label Mapping (0-indexed for PyTorch)
# Note: The HuggingFace dataset uses combined Gleason scores like "3+3", "3+4", etc.
# We extract the primary pattern for classification
GLEASON_PATTERN_MAP = {
    # Non-cancerous
    'Normal': 0,
    'Benign': 0,
    'Stroma': 0,
    'normal': 0,
    'benign': 0,
    'stroma': 0,
    
    # Gleason Pattern 3
    'Gleason Pattern 3': 1,
    'Gleason Pattern 3+3': 1,
    'Gleason Pattern 3+4': 1,  # Primary is 3
    'G3': 1,
    'GP3': 1,
    '3+3': 1,
    '3+4': 1,
    
    # Gleason Pattern 4
    'Gleason Pattern 4': 2,
    'Gleason Pattern 4+3': 2,  # Primary is 4
    'Gleason Pattern 4+4': 2,
    'Gleason Pattern 4+5': 2,  # Primary is 4
    'G4': 2,
    'GP4': 2,
    '4+3': 2,
    '4+4': 2,
    '4+5': 2,
    
    # Gleason Pattern 5
    'Gleason Pattern 5': 3,
    'Gleason Pattern 5+4': 3,  # Primary is 5
    'Gleason Pattern 5+5': 3,
    'G5': 3,
    'GP5': 3,
    '5+4': 3,
    '5+5': 3,
}

# Reverse mapping for class names
CLASS_NAMES = {
    0: 'Normal/Benign',
    1: 'Gleason 3',
    2: 'Gleason 4',
    3: 'Gleason 5',
}

# Priority hierarchy for label assignment (higher = more priority)
CLASS_PRIORITY = {
    'Normal/Benign': 0,
    'Normal': 0,
    'Benign': 0,
    'Stroma': 0,
    'Gleason 3': 1,
    'Gleason 4': 2,
    'Gleason 5': 3,
}


def parse_gleason_pattern(pattern_name: str) -> Optional[int]:
    """
    Parse a Gleason pattern name to a label.
    
    Args:
        pattern_name: Gleason pattern name from annotation
        
    Returns:
        Integer label (0-3) or None if not recognized
    """
    # Direct match
    if pattern_name in GLEASON_PATTERN_MAP:
        return GLEASON_PATTERN_MAP[pattern_name]
    
    # Try to extract pattern from string
    pattern_lower = pattern_name.lower()
    
    # Look for "5+5", "5+4", etc.
    for key, value in GLEASON_PATTERN_MAP.items():
        if key.lower() in pattern_lower:
            return value
    
    # Try to parse "Gleason Pattern X+Y" format
    if 'gleason' in pattern_lower and 'pattern' in pattern_lower:
        # Extract numbers
        import re
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


def download_tcgaprad_from_huggingface(
    output_dir: str,
    repo_id: str = "Codatta/Refined-TCGA-PRAD-Prostate-Cancer-Pathology-Dataset",
    download_wsi: bool = False,
    download_annotations: bool = True,
    verbose: bool = True
) -> Dict[str, Path]:
    """
    Download TCGA-PRAD dataset from HuggingFace.
    
    The dataset includes:
    - PRAD.csv: Slide-level metadata and Gleason grading information
    - GeoJSON files: Spatial annotations for tumor regions
    - WSI files: Full-resolution whole slide images (optional, large download)
    
    Args:
        output_dir: Directory to save downloaded files
        repo_id: HuggingFace repository ID
        download_wsi: Whether to download WSI files (large, ~100GB+)
        download_annotations: Whether to download GeoJSON annotations
        verbose: Print download progress
        
    Returns:
        Dictionary with paths to downloaded files/directories
    """
    try:
        from huggingface_hub import hf_hub_download, snapshot_download, list_repo_files
    except ImportError:
        raise ImportError(
            "huggingface_hub is required. Install with: pip install huggingface_hub"
        )
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    result_paths = {}
    
    if verbose:
        print(f"Downloading TCGA-PRAD dataset from: {repo_id}")
        print(f"Output directory: {output_dir}")
    
    # List all files in the repository
    try:
        files = list_repo_files(repo_id, repo_type="dataset")
    except Exception as e:
        print(f"Error listing repository files: {e}")
        print("Make sure you have access to the dataset on HuggingFace.")
        print("You may need to: huggingface-cli login")
        raise
    
    # Download metadata CSV
    csv_files = [f for f in files if f.endswith('.csv')]
    for csv_file in csv_files:
        if verbose:
            print(f"Downloading: {csv_file}")
        try:
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=csv_file,
                repo_type="dataset",
                local_dir=output_dir
            )
            result_paths['metadata_csv'] = Path(local_path)
        except Exception as e:
            print(f"  Warning: Could not download {csv_file}: {e}")
    
    # Download GeoJSON annotations
    if download_annotations:
        geojson_files = [f for f in files if f.endswith('.geojson')]
        annotations_dir = output_dir / "annotations"
        annotations_dir.mkdir(exist_ok=True)
        
        if verbose:
            print(f"Found {len(geojson_files)} GeoJSON annotation files")
        
        for geojson_file in geojson_files:
            try:
                local_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=geojson_file,
                    repo_type="dataset",
                    local_dir=output_dir
                )
                if verbose:
                    print(f"  Downloaded: {geojson_file}")
            except Exception as e:
                print(f"  Warning: Could not download {geojson_file}: {e}")
        
        result_paths['annotations_dir'] = annotations_dir
    
    # Download WSI files (optional - very large)
    if download_wsi:
        wsi_extensions = ('.svs', '.tif', '.tiff', '.ndpi')
        wsi_files = [f for f in files if any(f.endswith(ext) for ext in wsi_extensions)]
        
        if verbose:
            print(f"Found {len(wsi_files)} WSI files (this may take a long time)")
        
        wsi_dir = output_dir / "WSI"
        wsi_dir.mkdir(exist_ok=True)
        
        for wsi_file in wsi_files:
            try:
                local_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=wsi_file,
                    repo_type="dataset",
                    local_dir=output_dir
                )
                if verbose:
                    print(f"  Downloaded: {wsi_file}")
            except Exception as e:
                print(f"  Warning: Could not download {wsi_file}: {e}")
        
        result_paths['wsi_dir'] = wsi_dir
    
    if verbose:
        print(f"\nDownload complete! Files saved to: {output_dir}")
    
    return result_paths


class TCGAPRADLabels:
    """
    Label generation for TCGA-PRAD dataset.
    
    Supports creating labels at different granularities:
    - Slide-level: One label per WSI based on highest Gleason grade
    - Annotation-level: One label per annotated region
    - Patch-level: Labels for 256px patches based on annotation overlap
    """
    
    def __init__(self):
        """Initialize TCGAPRADLabels."""
        self.class_names = CLASS_NAMES
        self.class_priority = CLASS_PRIORITY
    
    def create_slide_labels(
        self,
        annotations_dir: str,
        metadata_csv: Optional[str] = None,
        save_path: Optional[str] = None,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        seed: int = 42
    ) -> pd.DataFrame:
        """
        Create slide-level labels for TCGA-PRAD dataset.
        
        The slide-level label is determined by the highest Gleason grade annotation
        present in the GeoJSON file (G5 > G4 > G3 > Normal).
        
        Args:
            annotations_dir: Path to directory containing GeoJSON annotation files
            metadata_csv: Optional path to metadata CSV from HuggingFace
            save_path: Optional path to save the labels TSV file
            train_ratio: Ratio of slides for training (default: 0.7)
            val_ratio: Ratio of slides for validation (default: 0.15)
            seed: Random seed for reproducibility
            
        Returns:
            DataFrame with slide-level labels
        """
        annotations_dir = Path(annotations_dir)
        
        if not annotations_dir.exists():
            raise ValueError(f"Annotations directory not found: {annotations_dir}")
        
        # Find all GeoJSON files
        geojson_files = list(annotations_dir.rglob("*.geojson"))
        
        if not geojson_files:
            raise ValueError(f"No GeoJSON files found in {annotations_dir}")
        
        print(f"Found {len(geojson_files)} GeoJSON annotation files")
        
        all_slides = []
        
        for geojson_file in geojson_files:
            # Extract slide ID from filename
            slide_id = geojson_file.stem
            
            # Parse GeoJSON to find highest priority annotation
            try:
                with open(geojson_file, 'r') as f:
                    geojson_data = json.load(f)
            except Exception as e:
                print(f"Warning: Could not parse {geojson_file}: {e}")
                continue
            
            # Get features from GeoJSON
            features = []
            if 'features' in geojson_data:
                features = geojson_data['features']
            elif isinstance(geojson_data, list):
                features = geojson_data
            
            # Track the highest priority class found
            max_priority = -1
            found_classes = set()
            
            for feature in features:
                class_name = None
                if 'properties' in feature:
                    props = feature['properties']
                    class_name = props.get('classification', {}).get('name') or \
                               props.get('class') or \
                               props.get('name') or \
                               props.get('label')
                
                if class_name:
                    label = parse_gleason_pattern(class_name)
                    if label is not None:
                        found_classes.add(class_name)
                        if label > max_priority:
                            max_priority = label
            
            # Only add slides with valid annotations
            if max_priority >= 0:
                all_slides.append({
                    'slide_id': slide_id,
                    'label': max_priority,
                    'class_name': CLASS_NAMES.get(max_priority, 'Unknown'),
                    'found_classes': ','.join(sorted(found_classes))
                })
            else:
                print(f"Warning: No valid Gleason annotations found in {geojson_file.name}")
        
        if not all_slides:
            raise ValueError(f"No valid slides with annotations found")
        
        df = pd.DataFrame(all_slides)
        
        # Create train/val/test splits
        np.random.seed(seed)
        indices = np.random.permutation(len(df))
        
        n_train = int(len(df) * train_ratio)
        n_val = int(len(df) * val_ratio)
        
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]
        
        df['fold_0'] = 'Training'
        df.loc[df.index[val_indices], 'fold_0'] = 'Validation'
        df.loc[df.index[test_indices], 'fold_0'] = 'Testing'
        
        # Reorder columns
        cols = ['slide_id', 'label', 'fold_0', 'class_name', 'found_classes']
        df = df[cols]
        
        print(f"\nSlide-level label summary:")
        print(f"  Total slides: {len(df)}")
        print(f"  Training: {len(df[df['fold_0'] == 'Training'])}")
        print(f"  Validation: {len(df[df['fold_0'] == 'Validation'])}")
        print(f"  Testing: {len(df[df['fold_0'] == 'Testing'])}")
        print(f"\nLabel distribution:")
        for label, name in CLASS_NAMES.items():
            count = len(df[df['label'] == label])
            print(f"  {label} ({name}): {count}")
        
        # Save if path provided
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(save_path, sep='\t', index=False)
            print(f"\nSaved to: {save_path}")
        
        return df


class TCGAPRADSlideDataset(Dataset):
    """
    Dataset for TCGA-PRAD slide-level MIL.
    
    Each item returns all patch features for a slide and the slide-level label.
    """
    
    def __init__(
        self,
        feats_path: str,
        df: pd.DataFrame,
        split: str = "Training",
        feature_key: str = "features",
        num_features: Optional[int] = None,
        random_sampling: bool = True,
        cache_h5: bool = True,
        seed: int = 42
    ):
        """
        Initialize the dataset.
        
        Args:
            feats_path: Path to directory containing .h5 feature files
            df: DataFrame with columns: ['slide_id', 'label', 'fold_0']
            split: Data split ("Training", "Validation", "Testing")
            feature_key: Key for features in H5 file
            num_features: Maximum number of features to sample (None = use all)
            random_sampling: Whether to randomly sample features during training
            cache_h5: Whether to cache H5 file handles
            seed: Random seed for reproducibility
        """
        self.feats_path = Path(feats_path)
        self.feature_key = feature_key
        self.num_features = num_features
        self.random_sampling = random_sampling
        self.cache_h5 = cache_h5
        self.seed = seed
        
        # Filter by split
        self.df = df[df['fold_0'] == split].reset_index(drop=True)
        
        if len(self.df) == 0:
            raise ValueError(f"No samples found for split '{split}'")
        
        print(f"[TCGAPRADSlideDataset - {split}] Loaded {len(self.df)} slides")
        
        # H5 file cache
        self._h5_cache = {}
    
    def _get_h5(self, slide_id: str) -> h5py.File:
        """Get H5 file handle (with optional caching)."""
        if self.cache_h5:
            if slide_id not in self._h5_cache:
                h5_path = self.feats_path / f"{slide_id}.h5"
                if not h5_path.exists():
                    raise FileNotFoundError(f"Feature file not found: {h5_path}")
                self._h5_cache[slide_id] = h5py.File(h5_path, 'r')
            return self._h5_cache[slide_id]
        else:
            h5_path = self.feats_path / f"{slide_id}.h5"
            return h5py.File(h5_path, 'r')
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Get slide at index.
        
        Returns:
            features: Tensor (num_patches, feature_dim)
            label: Tensor (class label)
            meta: Dict with slide_id, num_patches
        """
        row = self.df.iloc[idx]
        slide_id = row['slide_id']
        label = torch.tensor(row['label'], dtype=torch.long)
        
        # Load features
        h5 = self._get_h5(slide_id)
        features = h5[self.feature_key][:]
        
        # Sample features if needed
        if self.num_features is not None and len(features) > self.num_features:
            if self.random_sampling:
                indices = np.random.choice(len(features), self.num_features, replace=False)
            else:
                indices = np.linspace(0, len(features) - 1, self.num_features, dtype=int)
            features = features[indices]
        
        features = torch.from_numpy(features).float()
        
        meta = {
            'slide_id': slide_id,
            'num_patches': features.shape[0],
        }
        
        return features, label, meta
    
    def __del__(self):
        """Close cached H5 files."""
        for h5 in self._h5_cache.values():
            try:
                h5.close()
            except:
                pass


def mil_collate(batch: List[Tuple]) -> Tuple[List[torch.Tensor], torch.Tensor, List[Dict]]:
    """
    Collate function for MIL with variable-length bags.
    
    Args:
        batch: List of (features, label, meta) tuples
        
    Returns:
        feats_list: List of Tensors (variable num_patches per slide)
        labels: Tensor (batch_size,)
        metas: List of metadata dicts
    """
    feats_list = [item[0] for item in batch]
    labels = torch.stack([item[1] for item in batch], dim=0)
    metas = [item[2] for item in batch]
    return feats_list, labels, metas


def create_tcgaprad_dataloaders(
    feats_path: str,
    labels_tsv: str,
    batch_size: int = 1,
    num_workers: int = 4,
    num_features: Optional[int] = None,
    random_sampling: bool = True,
    seed: int = 42
) -> Dict[str, DataLoader]:
    """
    Create TCGA-PRAD dataloaders for training, validation, and testing.
    
    Args:
        feats_path: Path to directory containing .h5 feature files
        labels_tsv: Path to labels TSV file with columns: ['slide_id', 'label', 'fold_0']
        batch_size: Batch size (default: 1 for MIL)
        num_workers: Number of data loading workers
        num_features: Maximum features per slide (None = use all)
        random_sampling: Whether to randomly sample features during training
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with 'train', 'val', 'test' DataLoaders
        
    Example:
        >>> loaders = create_tcgaprad_dataloaders(
        ...     feats_path='./features',
        ...     labels_tsv='./labels.tsv'
        ... )
        >>> for features, labels, metas in loaders['train']:
        ...     # features is a list of tensors (one per slide)
        ...     # labels is a tensor of shape (batch_size,)
        ...     pass
    """
    # Load labels
    df = pd.read_csv(labels_tsv, sep='\t')
    
    # Validate columns
    required_cols = ['slide_id', 'label', 'fold_0']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in {labels_tsv}")
    
    # Create datasets
    train_dataset = TCGAPRADSlideDataset(
        feats_path=feats_path,
        df=df,
        split='Training',
        num_features=num_features,
        random_sampling=random_sampling,
        seed=seed
    )
    
    val_dataset = TCGAPRADSlideDataset(
        feats_path=feats_path,
        df=df,
        split='Validation',
        num_features=num_features,
        random_sampling=False,  # No random sampling for validation
        seed=seed
    )
    
    test_dataset = TCGAPRADSlideDataset(
        feats_path=feats_path,
        df=df,
        split='Testing',
        num_features=num_features,
        random_sampling=False,  # No random sampling for testing
        seed=seed
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=mil_collate,
        pin_memory=True,
        worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=mil_collate,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=mil_collate,
        pin_memory=True
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }


class TCGAPRADInferenceDataset(Dataset):
    """
    Dataset for TCGA-PRAD inference (no labels required).
    """
    
    def __init__(
        self,
        feats_path: str,
        slide_list: Optional[List[str]] = None,
        feature_key: str = "features",
        cache_h5: bool = True
    ):
        """
        Initialize inference dataset.
        
        Args:
            feats_path: Path to directory containing .h5 feature files
            slide_list: Optional list of slide IDs (None = use all in feats_path)
            feature_key: Key for features in H5 file
            cache_h5: Whether to cache H5 file handles
        """
        self.feats_path = Path(feats_path)
        self.feature_key = feature_key
        self.cache_h5 = cache_h5
        
        # Get slide list
        if slide_list is not None:
            self.slide_ids = slide_list
        else:
            # Find all H5 files
            h5_files = list(self.feats_path.glob("*.h5"))
            self.slide_ids = [f.stem for f in h5_files]
        
        if len(self.slide_ids) == 0:
            raise ValueError(f"No H5 files found in {feats_path}")
        
        print(f"[TCGAPRADInferenceDataset] Loaded {len(self.slide_ids)} slides")
        
        self._h5_cache = {}
    
    def _get_h5(self, slide_id: str) -> h5py.File:
        """Get H5 file handle (with optional caching)."""
        if self.cache_h5:
            if slide_id not in self._h5_cache:
                h5_path = self.feats_path / f"{slide_id}.h5"
                if not h5_path.exists():
                    raise FileNotFoundError(f"Feature file not found: {h5_path}")
                self._h5_cache[slide_id] = h5py.File(h5_path, 'r')
            return self._h5_cache[slide_id]
        else:
            h5_path = self.feats_path / f"{slide_id}.h5"
            return h5py.File(h5_path, 'r')
    
    def __len__(self) -> int:
        return len(self.slide_ids)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Get slide at index."""
        slide_id = self.slide_ids[idx]
        
        h5 = self._get_h5(slide_id)
        features = h5[self.feature_key][:]
        features = torch.from_numpy(features).float()
        
        # No label for inference
        label = torch.tensor(-1, dtype=torch.long)
        
        meta = {
            'slide_id': slide_id,
            'num_patches': features.shape[0],
        }
        
        return features, label, meta
    
    def __del__(self):
        for h5 in self._h5_cache.values():
            try:
                h5.close()
            except:
                pass


def create_tcgaprad_inference_dataloader(
    feats_path: str,
    slide_list: Optional[Union[List[str], str]] = None,
    batch_size: int = 1,
    num_workers: int = 4,
    cache_h5: bool = True
) -> DataLoader:
    """
    Create TCGA-PRAD dataloader for inference.
    
    Args:
        feats_path: Path to directory containing .h5 feature files
        slide_list: Optional list of slide IDs or path to TSV file with slide_id column
        batch_size: Batch size
        num_workers: Number of data loading workers
        cache_h5: Whether to cache H5 files
        
    Returns:
        DataLoader for inference
        
    Example:
        >>> loader = create_tcgaprad_inference_dataloader(
        ...     feats_path='./features',
        ...     slide_list=['TCGA-EJ-7791-01Z-00-DX1', 'TCGA-EJ-7123-01Z-00-DX1']
        ... )
    """
    # Parse slide_list
    slides = None
    if slide_list is not None:
        if isinstance(slide_list, str):
            # Load from TSV
            df = pd.read_csv(slide_list, sep='\t')
            if 'slide_id' not in df.columns:
                raise ValueError(f"TSV must contain 'slide_id' column")
            slides = df['slide_id'].tolist()
        else:
            slides = slide_list
    
    dataset = TCGAPRADInferenceDataset(
        feats_path=feats_path,
        slide_list=slides,
        cache_h5=cache_h5
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=mil_collate,
        pin_memory=True
    )
    
    return loader


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='TCGA-PRAD HuggingFace Dataloader')
    parser.add_argument('--download', action='store_true', help='Download dataset from HuggingFace')
    parser.add_argument('--output_dir', type=str, default='./data', help='Output directory for downloads')
    parser.add_argument('--annotations_dir', type=str, help='Directory with GeoJSON annotations')
    parser.add_argument('--create_labels', action='store_true', help='Create slide-level labels from annotations')
    parser.add_argument('--labels_output', type=str, default='./labels/slide_labels.tsv', help='Output path for labels')
    
    args = parser.parse_args()
    
    if args.download:
        print("Downloading TCGA-PRAD dataset from HuggingFace...")
        paths = download_tcgaprad_from_huggingface(
            output_dir=args.output_dir,
            download_wsi=False,  # WSIs are large, don't download by default
            download_annotations=True
        )
        print(f"\nDownloaded files: {paths}")
    
    if args.create_labels and args.annotations_dir:
        print("\nCreating slide-level labels...")
        labels_gen = TCGAPRADLabels()
        df = labels_gen.create_slide_labels(
            annotations_dir=args.annotations_dir,
            save_path=args.labels_output
        )
