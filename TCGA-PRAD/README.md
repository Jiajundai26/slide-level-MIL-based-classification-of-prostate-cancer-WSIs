# TCGA-PRAD Feature Extraction & Dataloader

Scripts for extracting features and loading the TCGA-PRAD (Prostate Adenocarcinoma) dataset from HuggingFace.

## TCGA-PRAD Gleason Pattern Labels (0-indexed for PyTorch)

| Label | Class | Description |
|-------|-------|-------------|
| 0 | Normal/Benign | Non-cancerous tissue |
| 1 | Gleason 3 | Well-differentiated (low grade) |
| 2 | Gleason 4 | Moderately differentiated |
| 3 | Gleason 5 | Poorly differentiated (high grade) |

## Installation

```bash
pip install -r requirements.txt
```

## Available Encoders

### ✅ PUBLIC Encoders (No HuggingFace approval required)

| Encoder | Model | Feature Dim | Notes |
|---------|-------|-------------|-------|
| `phikon` | owkin/phikon | 768 | Owkin pathology model |
| `phikon_v2` | owkin/phikon-v2 | 1024 | Owkin pathology model v2 |
| `resnet50` | resnet50 | 2048 | ImageNet pretrained |
| `resnet101` | resnet101 | 2048 | ImageNet pretrained |
| `dinov2_vitb14` | vit_base_patch14_dinov2 | 768 | Meta DINOv2 Base |
| `dinov2_vitl14` | vit_large_patch14_dinov2 | 1024 | Meta DINOv2 Large |

### 🔒 GATED Encoders (Require HuggingFace approval)

| Encoder | Model | Feature Dim | Request Access |
|---------|-------|-------------|----------------|
| `uni` | MahmoodLab/uni | 1024 | [Request](https://huggingface.co/MahmoodLab/uni) |
| `uni_v2` | MahmoodLab/uni | 1024 | [Request](https://huggingface.co/MahmoodLab/uni) |
| `uni2_h` | MahmoodLab/UNI2-h | 1536 | [Request](https://huggingface.co/MahmoodLab/UNI2-h) |
| `virchow2` | paige-ai/Virchow2 | 1280 | [Request](https://huggingface.co/paige-ai/Virchow2) |

## Quick Start

### Step 1: Create Labels from Annotations

```python
from tcga_prad_hf_dataloader import TCGAPRADLabels

labels = TCGAPRADLabels()
df = labels.create_slide_labels(
    annotations_dir='./annotations/geojsons',
    save_path='./labels/slide_labels.tsv'
)
```

Or from command line:
```bash
python3 tcga_prad_hf_dataloader.py \
    --create_labels \
    --annotations_dir ./annotations/geojsons \
    --labels_output ./labels/slide_labels.tsv
```

### Step 2: Extract Features (using a PUBLIC encoder)

```bash
# Use phikon_v2 (public, no approval required)
python3 extract_features_tcga_prad.py \
    --wsi_dir ./WSI \
    --output_dir ./features/phikon_v2 \
    --encoder_name phikon_v2 \
    --annotations_dir ./annotations/geojsons \
    --device cuda:0

# Or use resnet50 (always available)
python3 extract_features_tcga_prad.py \
    --wsi_dir ./WSI \
    --output_dir ./features/resnet50 \
    --encoder_name resnet50 \
    --device cuda:0
```

### Step 3: Create Dataloaders

```python
from tcga_prad_hf_dataloader import create_tcgaprad_dataloaders

loaders = create_tcgaprad_dataloaders(
    feats_path='./features/phikon_v2',
    labels_tsv='./labels/slide_labels.tsv',
    batch_size=1
)

# Training loop
for features_list, labels, metas in loaders['train']:
    # features_list: List of Tensors [N_patches, feature_dim]
    # labels: Tensor (batch_size,)
    # metas: List of dicts with 'slide_id', 'num_patches'
    pass
```

## What to Do While Waiting for UNI v2 Access

1. **Create slide labels** from your GeoJSON annotations
2. **Extract features** using a public encoder (`phikon_v2` recommended)
3. **Test your training pipeline** with the extracted features
4. **Validate data quality** by checking label distributions
5. **Re-extract with UNI v2** once access is approved (just change `--encoder_name`)

## Download from HuggingFace (Optional)

If you need to download the dataset from HuggingFace:

```python
from tcga_prad_hf_dataloader import download_tcgaprad_from_huggingface

paths = download_tcgaprad_from_huggingface(
    output_dir='./data',
    repo_id='Codatta/Refined-TCGA-PRAD-Prostate-Cancer-Pathology-Dataset',
    download_wsi=False,  # WSIs are large (~100GB+)
    download_annotations=True
)
```

## File Structure

```
TCGA-PRAD_organize/
├── WSI/                          # Whole slide images (.svs files)
├── annotations/
│   └── geojsons/                 # GeoJSON annotation files
├── labels/
│   └── slide_labels.tsv          # Generated slide labels
├── features/
│   ├── phikon_v2/                # Features from phikon_v2
│   ├── resnet50/                 # Features from resnet50
│   └── uni_v2/                   # Features from UNI v2 (when approved)
├── tcga_prad_hf_dataloader.py    # Dataloader and label generation
├── extract_features_tcga_prad.py # Feature extraction script
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Troubleshooting

### "Access denied" for UNI/Virchow models
- Visit the model page on HuggingFace and click "Request access"
- Wait for approval (can take 1-7 days)
- Use `huggingface-cli login` after approval

### Missing MPP metadata
- The script automatically uses `--default_mpp 0.5` when metadata is missing
- You can adjust this with the `--default_mpp` argument

### Out of memory
- Reduce `--batch_size` (default: 512)
- Use CPU with `--device cpu` (slower but uses less GPU memory)
