#!/usr/bin/env python3
"""
Verify slide-level labels for GTEx prostate WSIs.

- Checks that every .svs file in histology_images_prostate/
  has a corresponding row in GTEx_Portal_Prostate.csv.
- Checks that every Tissue Sample ID in the CSV has a .svs file.
- Prints summary statistics and lists mismatches.
"""

import csv
import os
from pathlib import Path

CSV_FILE = "GTEx_Portal_Prostate.csv"
IMAGES_DIR = "histology_images_prostate"
ID_COLUMN = "Tissue Sample ID"


def load_csv_ids(csv_path: Path):
    """Load Tissue Sample IDs and full rows from CSV."""
    ids = {}
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_id = row.get(ID_COLUMN, "").strip().strip('"')
            if not raw_id:
                continue
            if raw_id in ids:
                # duplicate ID in CSV
                print(f"[WARN] Duplicate ID in CSV: {raw_id}")
            ids[raw_id] = row
    return ids


def load_image_ids(images_dir: Path):
    """Load slide IDs from .svs filenames."""
    image_ids = set()
    if not images_dir.exists():
        print(f"[ERROR] Images directory does not exist: {images_dir}")
        return image_ids

    for p in images_dir.glob("*.svs"):
        # filename is GTEX-XXXX-YYYY.svs -> id = GTEX-XXXX-YYYY
        image_ids.add(p.stem)
    return image_ids


def main():
    base_dir = Path(".").resolve()
    csv_path = base_dir / CSV_FILE
    images_dir = base_dir / IMAGES_DIR

    print(f"Base directory       : {base_dir}")
    print(f"CSV file             : {csv_path}")
    print(f"Images directory     : {images_dir}")
    print(f"ID column (CSV)      : {ID_COLUMN}")
    print("-" * 60)

    if not csv_path.exists():
        print(f"[ERROR] CSV file not found: {csv_path}")
        return

    # 1. Load IDs from CSV
    csv_ids = load_csv_ids(csv_path)
    print(f"Total IDs in CSV     : {len(csv_ids)}")

    # 2. Load IDs from .svs filenames
    image_ids = load_image_ids(images_dir)
    print(f"Total .svs files     : {len(image_ids)}")

    # 3. Compute set differences
    ids_only_in_csv = sorted(set(csv_ids.keys()) - image_ids)
    ids_only_in_images = sorted(image_ids - set(csv_ids.keys()))

    print("-" * 60)
    print(f"IDs only in CSV (no .svs file): {len(ids_only_in_csv)}")
    print(f"IDs only in images (no CSV row): {len(ids_only_in_images)}")

    # 4. Show first few examples for sanity
    if ids_only_in_csv:
        print("\nExamples of IDs present in CSV but missing .svs:")
        for id_ in ids_only_in_csv[:10]:
            print("  ", id_)

    if ids_only_in_images:
        print("\nExamples of IDs present as .svs but missing in CSV:")
        for id_ in ids_only_in_images[:10]:
            print("  ", id_)

    # 5. Optional: inspect one specific slide’s labels
    #    If you want, set TARGET_ID and rerun.
    TARGET_ID = None  # e.g. "GTEX-111CU-1526"
    if TARGET_ID:
        row = csv_ids.get(TARGET_ID)
        if row:
            print("\nLabels for slide", TARGET_ID)
            for k, v in row.items():
                print(f"  {k}: {v}")
        else:
            print(f"\n[INFO] Target ID {TARGET_ID} not found in CSV.")

    print("\nVerification complete.")


if __name__ == "__main__":
    main()
