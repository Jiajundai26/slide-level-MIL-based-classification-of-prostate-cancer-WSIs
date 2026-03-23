"""
GTEx Prostate feature dataloaders.

Loads pre-extracted UNI-v2 features from:
  /sci-it/projects/cedmav/data/ARPA-H/Prostate_HnE/preprocessing/GTEx_Prostate/

Expected labels TSV format (default):
  slide_id	label	fold_0	class_name	dataset
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

try:
    import h5py
except ImportError as exc:
    raise ImportError(
        "h5py is required for GTExProstate dataloaders. "
        "Install with: pip install h5py"
    ) from exc


DEFAULT_FEATS_DIR = (
    "/sci-it/projects/cedmav/data/ARPA-H/Prostate_HnE/preprocessing/"
    "GTEx_Prostate/20x_256px_0px_overlap/features_uni_v2"
)
DEFAULT_LABELS_TSV = (
    "/sci-it/projects/cedmav/data/ARPA-H/Prostate_HnE/preprocessing/"
    "GTEx_Prostate/labels_MIL/slide_labels.tsv"
)

CLASS_NAMES = {0: "Non-Cancer"}


def _validate_labels_df(df: pd.DataFrame, split_column: str) -> None:
    required_cols = ["slide_id", "label", split_column]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Labels TSV is missing columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )


class GTExProstateSlideDataset(Dataset):
    """
    Slide-level dataset for GTEx prostate features.

    Each item returns:
      - features: Tensor [num_patches, feature_dim]
      - label: Tensor scalar
      - meta: dict with slide_id and num_patches
    """

    def __init__(
        self,
        feats_path: str,
        df: pd.DataFrame,
        split: Optional[str] = "Training",
        feature_key: str = "features",
        num_features: Optional[int] = None,
        random_sampling: bool = True,
        cache_h5: bool = True,
        split_column: str = "fold_0",
        seed: int = 42,
    ):
        self.features_dir = Path(feats_path)
        self.feature_key = feature_key
        self.num_features = num_features
        self.random_sampling = random_sampling
        self.cache_h5 = cache_h5
        self.split_column = split_column
        self.seed = seed

        _validate_labels_df(df, split_column=split_column)

        if split is None:
            self.df = df.reset_index(drop=True)
        else:
            self.df = df[df[split_column] == split].reset_index(drop=True)

        if len(self.df) == 0:
            split_info = "all splits" if split is None else f"split '{split}'"
            raise ValueError(f"No samples found for {split_info}")

        print(f"[GTExProstateSlideDataset - {split}] Loaded {len(self.df)} slides")
        self._h5_cache: Dict[str, h5py.File] = {}

    def _get_h5(self, slide_id: str) -> h5py.File:
        if self.cache_h5:
            if slide_id not in self._h5_cache:
                h5_path = self.features_dir / f"{slide_id}.h5"
                if not h5_path.exists():
                    raise FileNotFoundError(f"Feature file not found: {h5_path}")
                self._h5_cache[slide_id] = h5py.File(h5_path, "r")
            return self._h5_cache[slide_id]
        h5_path = self.features_dir / f"{slide_id}.h5"
        return h5py.File(h5_path, "r")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        row = self.df.iloc[idx]
        slide_id = row["slide_id"]
        label = torch.tensor(row["label"], dtype=torch.long)

        h5 = self._get_h5(slide_id)
        if self.feature_key not in h5:
            raise KeyError(
                f"Key '{self.feature_key}' not found in {slide_id}.h5. "
                f"Available keys: {list(h5.keys())}"
            )
        features = h5[self.feature_key][:]

        if self.num_features is not None and len(features) > self.num_features:
            if self.random_sampling:
                rng = np.random.default_rng(self.seed + idx)
                indices = rng.choice(len(features), self.num_features, replace=False)
            else:
                indices = np.linspace(
                    0, len(features) - 1, self.num_features, dtype=int
                )
            features = features[indices]

        features = torch.from_numpy(features).float()

        meta = {
            "slide_id": slide_id,
            "num_patches": features.shape[0],
        }

        return features, label, meta

    def __del__(self) -> None:
        for h5 in self._h5_cache.values():
            try:
                h5.close()
            except Exception:
                pass


def mil_collate(
    batch: List[Tuple[torch.Tensor, torch.Tensor, Dict]]
) -> Tuple[List[torch.Tensor], torch.Tensor, List[Dict]]:
    feats_list = [item[0] for item in batch]
    labels = torch.stack([item[1] for item in batch], dim=0)
    metas = [item[2] for item in batch]
    return feats_list, labels, metas


def create_gtex_prostate_dataloaders(
    feats_path: str = DEFAULT_FEATS_DIR,
    labels_tsv: str = DEFAULT_LABELS_TSV,
    batch_size: int = 1,
    num_workers: int = 4,
    num_features: Optional[int] = None,
    random_sampling: bool = True,
    split_column: str = "fold_0",
    seed: int = 42,
) -> Dict[str, DataLoader]:
    """
    Create GTEx prostate dataloaders for training/validation/testing.
    """
    df = pd.read_csv(labels_tsv, sep="\t")
    _validate_labels_df(df, split_column=split_column)

    train_dataset = GTExProstateSlideDataset(
        feats_path=feats_path,
        df=df,
        split="Training",
        num_features=num_features,
        random_sampling=random_sampling,
        split_column=split_column,
        seed=seed,
    )
    val_dataset = GTExProstateSlideDataset(
        feats_path=feats_path,
        df=df,
        split="Validation",
        num_features=num_features,
        random_sampling=False,
        split_column=split_column,
        seed=seed,
    )
    test_dataset = GTExProstateSlideDataset(
        feats_path=feats_path,
        df=df,
        split="Testing",
        num_features=num_features,
        random_sampling=False,
        split_column=split_column,
        seed=seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=mil_collate,
        pin_memory=True,
        worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=mil_collate,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=mil_collate,
        pin_memory=True,
    )

    return {"train": train_loader, "val": val_loader, "test": test_loader}


class GTExProstateInferenceDataset(Dataset):
    """
    Inference-only dataset (no labels required).
    """

    def __init__(
        self,
        feats_path: str,
        slide_list: Optional[List[str]] = None,
        feature_key: str = "features",
        cache_h5: bool = True,
    ):
        self.features_dir = Path(feats_path)
        self.feature_key = feature_key
        self.cache_h5 = cache_h5

        if slide_list is not None:
            self.slide_ids = slide_list
        else:
            self.slide_ids = [p.stem for p in self.features_dir.glob("*.h5")]

        if len(self.slide_ids) == 0:
            raise ValueError(f"No H5 files found in {self.features_dir}")

        print(f"[GTExProstateInferenceDataset] Loaded {len(self.slide_ids)} slides")
        self._h5_cache: Dict[str, h5py.File] = {}

    def _get_h5(self, slide_id: str) -> h5py.File:
        if self.cache_h5:
            if slide_id not in self._h5_cache:
                h5_path = self.features_dir / f"{slide_id}.h5"
                if not h5_path.exists():
                    raise FileNotFoundError(f"Feature file not found: {h5_path}")
                self._h5_cache[slide_id] = h5py.File(h5_path, "r")
            return self._h5_cache[slide_id]
        h5_path = self.features_dir / f"{slide_id}.h5"
        return h5py.File(h5_path, "r")

    def __len__(self) -> int:
        return len(self.slide_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        slide_id = self.slide_ids[idx]
        h5 = self._get_h5(slide_id)
        if self.feature_key not in h5:
            raise KeyError(
                f"Key '{self.feature_key}' not found in {slide_id}.h5. "
                f"Available keys: {list(h5.keys())}"
            )
        features = torch.from_numpy(h5[self.feature_key][:]).float()
        label = torch.tensor(-1, dtype=torch.long)
        meta = {"slide_id": slide_id, "num_patches": features.shape[0]}
        return features, label, meta

    def __del__(self) -> None:
        for h5 in self._h5_cache.values():
            try:
                h5.close()
            except Exception:
                pass


def create_gtex_prostate_inference_dataloader(
    feats_path: str = DEFAULT_FEATS_DIR,
    slide_list: Optional[Union[List[str], str]] = None,
    batch_size: int = 1,
    num_workers: int = 4,
    cache_h5: bool = True,
) -> DataLoader:
    """
    Create inference DataLoader for GTEx prostate.

    slide_list can be:
      - list of slide IDs
      - path to TSV with a 'slide_id' column
      - None (use all H5 files in feats_path)
    """
    slides = None
    if slide_list is not None:
        if isinstance(slide_list, str):
            df = pd.read_csv(slide_list, sep="\t")
            if "slide_id" not in df.columns:
                raise ValueError("TSV must contain a 'slide_id' column")
            slides = df["slide_id"].tolist()
        else:
            slides = slide_list

    dataset = GTExProstateInferenceDataset(
        feats_path=feats_path,
        slide_list=slides,
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


if __name__ == "__main__":
    # Quick sanity check: list counts for each split
    df = pd.read_csv(DEFAULT_LABELS_TSV, sep="\t")
    for split in ["Training", "Validation", "Testing"]:
        print(split, len(df[df["fold_0"] == split]))
