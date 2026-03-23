import csv
import os
import time
from pathlib import Path

import requests
from tqdm import tqdm


def load_histology_ids(csv_file, id_column="Tissue Sample ID"):
    """Load histology IDs from GTEx portal CSV."""
    ids = []
    with open(csv_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if id_column in row and row[id_column]:
                hist_id = row[id_column].strip().strip('"')
                if hist_id:
                    ids.append(hist_id)
    return ids


def download_one_slide(
    histology_id,
    output_dir,
    session,
    timeout=(10, 1200),
    chunk_size=1024 * 1024,
):
    """
    Download a single slide using streaming.

    timeout: (connect_timeout, read_timeout) in seconds
    chunk_size: bytes per chunk
    """
    url = f"https://brd.nci.nih.gov/brd/imagedownload/{histology_id}"
    filename = f"{histology_id}.svs"
    filepath = os.path.join(output_dir, filename)

    # Skip if already downloaded and non-empty
    if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
        return True, filepath, "exists"

    try:
        with session.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()

            total = r.headers.get("Content-Length")
            total = int(total) if total is not None else None

            # Per-file progress bar
            with open(filepath, "wb") as f, tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                desc=histology_id,
                leave=False,
            ) as pbar:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if not chunk:
                        continue
                    f.write(chunk)
                    pbar.update(len(chunk))

        size_bytes = os.path.getsize(filepath)
        if size_bytes == 0:
            # Zero-byte file: treat as failure
            os.remove(filepath)
            return False, filepath, "zero-byte download"
        return True, filepath, "downloaded"

    except requests.exceptions.RequestException as e:
        # Clean up empty file if created
        try:
            if os.path.exists(filepath) and os.path.getsize(filepath) == 0:
                os.remove(filepath)
        except OSError:
            pass
        return False, filepath, repr(e)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Streamed downloader for GTEx histology images (Prostate or others)."
    )
    parser.add_argument(
        "--csv-file",
        required=True,
        help="Path to GTEx portal CSV (e.g., GTEx_Portal_Prostate.csv)",
    )
    parser.add_argument(
        "--output-dir",
        default="histology_images",
        help="Directory to save downloaded .svs files",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Download at most N slides (for testing). Default: all.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay in seconds between slide downloads (default: 0.5).",
    )

    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Loading histology IDs from {args.csv_file} ...")
    histology_ids = load_histology_ids(args.csv_file, id_column="Tissue Sample ID")

    if not histology_ids:
        print("No IDs found in 'Tissue Sample ID' column.")
        return

    if args.limit is not None:
        histology_ids = histology_ids[: args.limit]
        print(f"Will download first {len(histology_ids)} slides.")
    else:
        print(f"Will download all {len(histology_ids)} slides.")

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "GTEx-Histology-Downloader/stream/1.0",
        }
    )

    successful = 0
    skipped = 0
    failed = 0

    for i, hist_id in enumerate(
        tqdm(histology_ids, desc="Slides", unit="slide")
    ):
        ok, path, status = download_one_slide(
            hist_id,
            args.output_dir,
            session,
        )

        if ok:
            if status == "exists":
                skipped += 1
                tqdm.write(f"↺ Skipped (exists): {path}")
            else:
                successful += 1
                tqdm.write(f"✓ Downloaded: {path}")
        else:
            failed += 1
            tqdm.write(f"✗ Failed {hist_id}: {status}")

        # polite delay between requests
        if i < len(histology_ids) - 1:
            time.sleep(args.delay)

    print("\nDone.")
    print(f"Successful downloads : {successful}")
    print(f"Already existing     : {skipped}")
    print(f"Failed               : {failed}")
    print(f"Output directory     : {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
