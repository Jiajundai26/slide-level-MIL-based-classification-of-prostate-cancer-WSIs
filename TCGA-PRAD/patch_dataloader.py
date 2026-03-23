"""
Patch dataloader for MIL training and inference.

This dataloader loads 256px patches within each larger patch region
and creates bags of features for the MIL model.

The patch size is configurable (e.g., 2mm @ 20x = 4096x4096 pixels).
"""

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional, Tuple, Dict, List

from .base import BaseInferenceDataset, BaseTrainingDataset
from .utils.patch_utils import load_patch_mapping_h5


class PatchMILInferenceDataset(BaseInferenceDataset):
    """
    Dataset for patch-level MIL inference.

    Args:
        feats_path: Path to directory containing .h5 feature files
        df: DataFrame with columns: ['slide_id', 'h5_idx'] or just slide IDs
        patch_mapping_dir: Optional path to directory with patch mapping H5 files.
                          If None, uses all patches from feature H5 files.
        feature_key: Key for features in H5 file. Default: "features"
        cache_h5: Whether to cache H5 file handles. Default: True

    Returns:
        Tuple of (features, label, metadata_dict) where:
            - features: torch.Tensor of shape [N, D] where N is num 256px patches, D is feature dim
            - label: torch.Tensor with value -1 (no label for inference)
            - metadata_dict: Dict with 'slide_id', 'h5_idx', 'num_patches'
    """

    def __init__(
        self,
        feats_path: str,
        df: pd.DataFrame,
        patch_mapping_dir: Optional[str] = None,
        feature_key: str = "features",
        cache_h5: bool = True,
    ):
        super().__init__(feats_path, cache_h5)

        self.feature_key = feature_key
        self.df = df.reset_index(drop=True)
        self.data = self.df

        # Load patch mapping if provided
        if patch_mapping_dir is not None:
            from .utils.patch_utils import load_patch_mapping_h5
            slide_ids = df['slide_id'].unique().tolist()
            self.patch_mapping = load_patch_mapping_h5(patch_mapping_dir, slide_ids)
        else:
            self.patch_mapping = None

        print(f"[PatchMILInferenceDataset] Loaded {len(self.df)} patches for inference")
        if self.patch_mapping is not None:
            print(f"[PatchMILInferenceDataset] Using patch mapping from {patch_mapping_dir}")
        else:
            print(f"[PatchMILInferenceDataset] Using all patches from feature H5 files")

    def _get_h5_path(self, slide_id: str) -> Path:
        """Get path to H5 file for slide."""
        return self.features_dir / f"{slide_id}.h5"

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Get patch at index.

        Returns:
            features: Tensor (num_256px_patches, feature_dim)
            label: Tensor (-1 for inference mode)
            meta: Dict with slide_id, h5_idx, num_patches
        """
        row = self.df.iloc[idx]
        slide_id = row['slide_id']

        # Support 'h5_idx' (new unified column) and legacy columns for backward compatibility
        if 'h5_idx' in row:
            h5_idx = int(row['h5_idx'])
        elif 'patch_idx' in row:
            h5_idx = int(row['patch_idx'])
        elif 'patch_2mm_idx' in row:
            h5_idx = int(row['patch_2mm_idx'])
        else:
            h5_idx = idx

        # Load features from H5 file
        h5 = self._get_h5(slide_id)

        if self.patch_mapping is not None:
            # Use mapping to extract specific patch
            # Check if slide_id exists in mapping
            if slide_id not in self.patch_mapping:
                raise KeyError(
                    f"Slide '{slide_id}' not found in patch mapping. "
                    f"Available slides: {list(self.patch_mapping.keys())[:5]}... "
                    f"(showing first 5 of {len(self.patch_mapping)} slides). "
                    f"Ensure the patch mapping directory contains an H5 file for this slide."
                )

            # Check if h5_idx is valid
            slide_mapping = self.patch_mapping[slide_id]
            if h5_idx >= len(slide_mapping):
                raise IndexError(
                    f"h5_idx {h5_idx} out of range for slide '{slide_id}' "
                    f"(has {len(slide_mapping)} rows in H5 mapping). "
                    f"Check that the label file h5_idx values match the mapping file."
                )

            # Get the patch indices for this row (works for both 2mm and category formats)
            patch_indices = slide_mapping[h5_idx]

            # Extract valid indices (remove -1 padding)
            # Both formats use -1 padding, so this works universally
            valid_mask = patch_indices != -1
            valid_indices = patch_indices[valid_mask]

            if len(valid_indices) == 0:
                # Empty patch - return single zero vector
                features = torch.zeros((1, 1024), dtype=torch.float32)
                print(f"Warning: Empty patch for {slide_id} at h5_idx {h5_idx}")
            else:
                # Get features for these patch indices
                features_np = h5[self.feature_key][valid_indices.astype(int), :]
                features = torch.from_numpy(features_np).float()
        else:
            # No mapping - use all patches from feature H5
            features_np = h5[self.feature_key][:]
            features = torch.from_numpy(features_np).float()

        # Label is -1 for inference (no labels)
        label = torch.tensor(-1, dtype=torch.long)

        # Metadata
        meta = {
            'slide_id': slide_id,
            'h5_idx': h5_idx,
            'num_patches': features.shape[0],
        }

        return features, label, meta


def mil_collate(batch: List[Tuple]) -> Tuple[List[torch.Tensor], torch.Tensor, List[Dict]]:
    """
    Collate function for MIL with variable-length bags.

    Args:
        batch: List of (features, label, meta) tuples

    Returns:
        feats_list: List of Tensors (variable num_patches per patch)
        labels: Tensor (batch_size,)
        metas: List of metadata dicts
    """
    feats_list = [item[0] for item in batch]
    labels = torch.stack([item[1] for item in batch], dim=0)
    metas = [item[2] for item in batch]
    return feats_list, labels, metas


def create_patchmil_inference_dataloader(
    feats_path: str,
    df: pd.DataFrame,
    batch_size: int = 8,
    num_workers: int = 4,
    patch_mapping_dir: Optional[str] = None,
    cache_h5: bool = True,
) -> DataLoader:
    """
    Create patch MIL dataloader for inference.

    Args:
        feats_path: Path to directory containing .h5 feature files
        df: DataFrame with columns: ['slide_id', 'h5_idx'] or just slide IDs
        batch_size: Batch size for dataloaders. Default: 8
        num_workers: Number of worker processes for data loading. Default: 4
        patch_mapping_dir: Optional path to directory with patch mapping H5 files. Default: None
        cache_h5: Whether to cache H5 files for faster iteration. Default: True

    Returns:
        DataLoader for patch-level MIL inference

    Example:
        >>> df = pd.DataFrame({'slide_id': ['BRACS_001', 'BRACS_002'], 'h5_idx': [0, 0]})
        >>> loader = create_patchmil_inference_dataloader(
        ...     feats_path='./features/20x_256px_0px_overlap/features_uni_v2',
        ...     df=df,
        ...     batch_size=8,
        ...     patch_mapping_dir='./patch_mapping'
        ... )
    """
    dataset = PatchMILInferenceDataset(
        feats_path=feats_path,
        df=df,
        patch_mapping_dir=patch_mapping_dir,
        cache_h5=cache_h5,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=mil_collate,
        pin_memory=True,
    )

    return loader


class PatchMILTrainingDataset(BaseTrainingDataset):
    """
    Dataset for patch-level MIL training.

    Args:
        feats_path: Path to directory containing .h5 feature files
        df: DataFrame with columns: ['slide_id', 'h5_idx', 'label', 'fold_0']
        split: Data split ("Training", "Validation", "Testing")
        patch_mapping_dir: Optional path to directory with patch mapping H5 files. Default: None
        feature_key: Key for features in H5 file. Default: "features"
        cache_h5: Whether to cache H5 file handles. Default: True
        seed: Random seed for reproducibility. Default: 42
        split_column: Column name to use for filtering by split. Default: "fold_0"

    Returns:
        Tuple of (features, label, metadata_dict) where:
            - features: torch.Tensor of shape [N, D] where N is num 256px patches, D is feature dim
            - label: torch.Tensor with integer class label
            - metadata_dict: Dict with 'slide_id', 'h5_idx', 'num_patches'
    """

    def __init__(
        self,
        feats_path: str,
        df: pd.DataFrame,
        split: str = "Training",
        patch_mapping_dir: Optional[str] = None,
        feature_key: str = "features",
        cache_h5: bool = True,
        seed: int = 42,
        split_column: str = "fold_0",
    ):
        super().__init__(feats_path, split, cache_h5)

        self.feature_key = feature_key
        self.seed = seed

        # Filter by split
        self.df = df[df[split_column] == split].reset_index(drop=True)

        if len(self.df) == 0:
            raise ValueError(f"No samples found for split '{split}' in the provided DataFrame")

        print(f"[PatchMILTrainingDataset - {split}] Loaded {len(self.df)} patches")

        # Load patch mapping if provided
        if patch_mapping_dir is not None:
            from .utils.patch_utils import load_patch_mapping_h5
            slide_ids = self.df['slide_id'].unique().tolist()
            self.patch_mapping = load_patch_mapping_h5(patch_mapping_dir, slide_ids)
            print(f"[PatchMILTrainingDataset - {split}] Using patch mapping from {patch_mapping_dir}")
        else:
            self.patch_mapping = None
            print(f"[PatchMILTrainingDataset - {split}] Using all patches from feature H5 files")

        self.data = self.df

    def _get_h5_path(self, slide_id: str) -> Path:
        """Get path to H5 file for slide."""
        return self.features_dir / f"{slide_id}.h5"

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Get patch at index.

        Returns:
            features: Tensor (num_256px_patches, feature_dim)
            label: Tensor (class label)
            meta: Dict with slide_id, h5_idx, num_patches
        """
        row = self.df.iloc[idx]
        slide_id = row['slide_id']

        # Support 'h5_idx' (new unified column) and legacy columns for backward compatibility
        if 'h5_idx' in row:
            h5_idx = int(row['h5_idx'])
        elif 'patch_idx' in row:
            h5_idx = int(row['patch_idx'])
        elif 'patch_2mm_idx' in row:
            h5_idx = int(row['patch_2mm_idx'])
        else:
            h5_idx = idx

        # Get label
        label = torch.tensor(row['label'], dtype=torch.long)

        # Load features from H5 file
        h5 = self._get_h5(slide_id)

        if self.patch_mapping is not None:
            # Use mapping to extract specific patch
            # Check if slide_id exists in mapping
            if slide_id not in self.patch_mapping:
                raise KeyError(
                    f"Slide '{slide_id}' not found in patch mapping. "
                    f"Available slides: {list(self.patch_mapping.keys())[:5]}... "
                    f"(showing first 5 of {len(self.patch_mapping)} slides). "
                    f"Ensure the patch mapping directory contains an H5 file for this slide."
                )

            # Check if h5_idx is valid
            slide_mapping = self.patch_mapping[slide_id]
            if h5_idx >= len(slide_mapping):
                raise IndexError(
                    f"h5_idx {h5_idx} out of range for slide '{slide_id}' "
                    f"(has {len(slide_mapping)} rows in H5 mapping). "
                    f"Check that the label file h5_idx values match the mapping file."
                )

            # Get the patch indices for this row (works for both 2mm and category formats)
            patch_indices = slide_mapping[h5_idx]

            # Extract valid indices (remove -1 padding)
            # Both formats use -1 padding, so this works universally
            valid_mask = patch_indices != -1
            valid_indices = patch_indices[valid_mask]

            if len(valid_indices) == 0:
                # Empty patch - return single zero vector
                features = torch.zeros((1, 1024), dtype=torch.float32)
                print(f"Warning: Empty patch for {slide_id} at h5_idx {h5_idx}")
            else:
                # Get features for these patch indices
                features_np = h5[self.feature_key][valid_indices.astype(int), :]
                features = torch.from_numpy(features_np).float()
        else:
            # No mapping - use all patches from feature H5
            features_np = h5[self.feature_key][:]
            features = torch.from_numpy(features_np).float()

        # Metadata
        meta = {
            'slide_id': slide_id,
            'h5_idx': h5_idx,
            'num_patches': features.shape[0],
        }

        return features, label, meta


def create_patchmil_training_dataloaders(
    feats_path: str,
    labels_tsv: str,
    batch_size: int = 1,
    num_workers: int = 4,
    patch_mapping_dir: Optional[str] = None,
    seed: int = 42,
) -> Dict[str, DataLoader]:
    """
    Create patch MIL training dataloaders.

    Args:
        feats_path: Path to directory containing .h5 feature files
        labels_tsv: Path to TSV file with columns: ['slide_id', 'h5_idx', 'label', 'fold_0']
                   (also supports legacy 'patch_idx' and 'patch_2mm_idx' column names)
        batch_size: Batch size for dataloaders. Default: 1 (recommended for MIL)
        num_workers: Number of worker processes for data loading. Default: 4
        patch_mapping_dir: Optional path to directory with patch mapping H5 files. Default: None
        seed: Random seed for reproducibility. Default: 42

    Returns:
        Dictionary with keys 'train', 'val', 'test' containing respective DataLoaders

    Example:
        >>> loaders = create_patchmil_training_dataloaders(
        ...     feats_path='./features/20x_256px_0px_overlap/features_uni_v2',
        ...     labels_tsv='./patches_labels.tsv',
        ...     batch_size=1,
        ...     patch_mapping_dir='./patch_mapping'
        ... )
    """
    # Load labels DataFrame
    df = pd.read_csv(labels_tsv, sep='\t')

    # Validate required columns (support both new and legacy naming)
    if 'slide_id' not in df.columns:
        raise ValueError(f"Column 'slide_id' not found in {labels_tsv}")
    if 'label' not in df.columns:
        raise ValueError(f"Column 'label' not found in {labels_tsv}")
    if 'fold_0' not in df.columns:
        raise ValueError(f"Column 'fold_0' not found in {labels_tsv}")

    # Check for h5_idx column (support new and legacy naming)
    if 'h5_idx' not in df.columns and 'patch_idx' not in df.columns and 'patch_2mm_idx' not in df.columns:
        raise ValueError(
            f"None of 'h5_idx', 'patch_idx', or 'patch_2mm_idx' columns found in {labels_tsv}. "
            f"One of these columns is required."
        )

    # Create datasets for each split
    train_dataset = PatchMILTrainingDataset(
        feats_path=feats_path,
        df=df,
        split='Training',
        patch_mapping_dir=patch_mapping_dir,
        cache_h5=True,
        seed=seed,
    )

    val_dataset = PatchMILTrainingDataset(
        feats_path=feats_path,
        df=df,
        split='Validation',
        patch_mapping_dir=patch_mapping_dir,
        cache_h5=True,
        seed=seed,
    )

    test_dataset = PatchMILTrainingDataset(
        feats_path=feats_path,
        df=df,
        split='Testing',
        patch_mapping_dir=patch_mapping_dir,
        cache_h5=True,
        seed=seed,
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
        pin_memory=True,
        worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id)
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=mil_collate,
        pin_memory=True,
        worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id)
    )

    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }
