import csv
import requests
import os
from pathlib import Path
import argparse
from tqdm import tqdm
import time

def load_histology_data(csv_file):
    """Load histology data from CSV file"""
    data = []
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data

def download_image(histology_id, output_dir, session):
    """Download a single histology image"""
    url = f"https://brd.nci.nih.gov/brd/imagedownload/{histology_id}"

    try:
        response = session.get(url, timeout=30)
        response.raise_for_status()

        # Create filename
        filename = f"{histology_id}.svs"
        filepath = os.path.join(output_dir, filename)

        # Save the image
        with open(filepath, 'wb') as f:
            f.write(response.content)

        return True, filepath
    except requests.exceptions.RequestException as e:
        return False, str(e)

def main():
    parser = argparse.ArgumentParser(description='Download GTEx histology images')
    parser.add_argument('--csv-file', default='GTEx Portal.csv',
                       help='Path to CSV file containing histology data')
    parser.add_argument('--output-dir', default='histology_images',
                       help='Directory to save downloaded images')
    parser.add_argument('--limit', type=int, default=None,
                       help='Number of images to download (default: all)')
    parser.add_argument('--delay', type=float, default=0.5,
                       help='Delay between downloads in seconds (default: 0.5)')

    args = parser.parse_args()

    # Create output directory
    Path(args.output_dir).mkdir(exist_ok=True)

    # Load histology data
    print(f"Loading data from {args.csv_file}...")
    histology_data = load_histology_data(args.csv_file)

    # Extract Tissue Sample IDs from CSV data
    histology_ids = [entry['Tissue Sample ID'].strip('"') for entry in histology_data if 'Tissue Sample ID' in entry]

    # Apply limit if specified
    if args.limit:
        histology_ids = histology_ids[:args.limit]
        print(f"Downloading first {len(histology_ids)} images...")
    else:
        print(f"Downloading all {len(histology_ids)} images...")

    # Create session for connection reuse
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'GTEx-Histology-Downloader/1.0'
    })

    # Download images
    successful = 0
    failed = 0

    for i, histology_id in enumerate(tqdm(histology_ids, desc="Downloading")):
        success, result = download_image(histology_id, args.output_dir, session)

        if success:
            successful += 1
            tqdm.write(f"✓ Downloaded: {result}")
        else:
            failed += 1
            tqdm.write(f"✗ Failed {histology_id}: {result}")

        # Add delay to be respectful to the server
        if i < len(histology_ids) - 1:  # Don't delay after the last download
            time.sleep(args.delay)

    print(f"\nDownload complete!")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Images saved to: {os.path.abspath(args.output_dir)}")

if __name__ == "__main__":
    main()