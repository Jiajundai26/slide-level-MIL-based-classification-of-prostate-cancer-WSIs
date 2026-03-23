# Slide-Level MIL-Based Classification of Prostate Cancer WSIs

Slide-level classification of prostate cancer whole-slide images (WSIs) using **Attention-Based Multiple Instance Learning (ABMIL)**. The project follows the [TRIDENT](https://github.com/mahmoodlab/TRIDENT) framework methodology and leverages pre-extracted patch features from foundation models (UNI-v2, Phikon-v2, DINOv2, etc.) to predict Gleason grades at the slide level.

## Overview

Prostate cancer grading relies on the Gleason scoring system, where pathologists assess tissue architecture to assign a grade. This project automates that process by:

1. **Extracting patch-level features** from digitized H&E-stained whole-slide images using pathology foundation models.
2. **Aggregating patch features** into a slide-level representation via a gated-attention MIL pooling mechanism.
3. **Classifying slides** into clinically meaningful categories (Normal, Gleason 3, Gleason 4, Gleason 5).

Two datasets are used:

| Dataset | Source | Content |
|---------|--------|---------|
| **TCGA-PRAD** | [The Cancer Genome Atlas](https://portal.gdc.cancer.gov/) | Cancer slides with Gleason pattern annotations (G3, G4, G5) |
| **GTEx Prostate** | [Genotype-Tissue Expression Project](https://gtexportal.org/) | Normal/benign prostate tissue and a small number of incidental cancer slides |

## Classification Tasks

### 4-Class Gleason Grading

| Label | Class | Description |
|-------|-------|-------------|
| 0 | Normal/Benign | Non-cancerous tissue (GTEx benign) |
| 1 | Gleason 3 | Well-differentiated (low-grade cancer) |
| 2 | Gleason 4 | Moderately differentiated (intermediate-grade) |
| 3 | Gleason 5 | Poorly differentiated (high-grade cancer) |

### Binary Classification

Two binary formulations are explored:

- **Cancer vs Non-Cancer** — separates benign tissue from all cancer grades.
- **Non-Cancer/Low-Grade vs High-Grade (G4/G5)** — groups benign + G3 as class 0 and G4/G5 as class 1, which is clinically relevant for treatment decisions.

## Model Architecture

All models use the same core architecture:

```
WSI → Patch Tiling (256 px @ 20×) → Foundation Model Encoder → [N, D] features
    → Linear Projection → Gated Attention Pooling → Slide-Level Classifier
```

- **Feature encoder**: UNI-v2 (1536-dim), Phikon-v2 (1024-dim), ResNet-50 (2048-dim), DINOv2 (768/1024-dim), and others.
- **Attention mechanism**: Gated attention (`Tanh` × `Sigmoid` element-wise gating).
- **Classifier head**: Two-layer MLP with ReLU and dropout.

### Supervised Attention

An optional **attention supervision loss** encourages the model to focus on annotated cancer regions (from GeoJSON annotations) rather than learning arbitrary discriminative features:

```
Loss = Classification Loss + λ × Attention Supervision Loss
```

The attention supervision penalises high attention on non-cancer patches and rewards high attention on annotated cancer patches. A separate **normal-slide regularisation** term keeps attention diffuse on benign tissue.

## Repository Structure

```
├── README.md                          # This file
├── TCGA-PRAD/                         # TCGA-PRAD dataset scripts and outputs
│   ├── requirements.txt               # Python dependencies
│   ├── README.md                      # Detailed TCGA-PRAD documentation
│   ├── tcga_prad_hf_dataloader.py     # HuggingFace dataloader & label generation
│   ├── extract_features_tcga_prad.py  # Feature extraction from WSIs
│   ├── patch_dataloader.py            # Patch-level dataloader for MIL
│   │
│   ├── train_abmil.py                          # 3-class ABMIL (G3/G4/G5)
│   ├── train_4class_classification_trident.py   # 4-class TRIDENT training
│   ├── train_binary_noncancer_vs_g45.py         # Binary: non-cancer vs G4/G5
│   ├── train_supervised_attention_trident.py    # Supervised attention training
│   │
│   ├── inference_abmil.py                       # 3-class inference
│   ├── inference_4class_trident.py              # 4-class inference + heatmaps
│   ├── inference_binary_trident.py              # Binary inference + heatmaps
│   ├── inference_combined_trident.py            # Combined TCGA+GTEx inference
│   ├── inference_heatmap_noncancer_vs_g45.py    # Heatmap generation
│   │
│   ├── analyze_predictions.py          # Prediction analysis & visualisation
│   ├── compare_attention_vs_gt.py      # Attention vs ground-truth comparison
│   ├── generate_gt_overlay.py          # Ground-truth annotation overlays
│   ├── plot_roc_curves.py              # One-vs-Rest ROC curves
│   ├── plot_combined_roc.py            # Combined TCGA+GTEx ROC
│   ├── plot_confusion_roc.py           # Confusion matrix & ROC plots
│   ├── plot_confusion_matrix_binary.py # Binary confusion matrices
│   ├── plot_roc_curve_binary.py        # Binary ROC curves
│   ├── plot_gtex_roc.py               # GTEx-specific ROC curves
│   │
│   ├── labels/                         # Slide-level label files
│   │   └── slide_labels.tsv
│   └── output/                         # Trained models, predictions, plots
│
├── GTEx_prostate/                      # GTEx Prostate dataset scripts and outputs
│   ├── requirements.txt               # Python dependencies
│   ├── GTEx_Portal_Prostate.csv       # GTEx portal metadata
│   ├── download_gtex_histology_images.py        # Download WSIs from GTEx
│   ├── download_gtex_histology_images_stream.py # Streaming download variant
│   ├── filter_gtex_prostate.py         # Filter & label GTEx slides
│   ├── verify_gtex_labels.py           # Verify label–slide alignment
│   ├── gtex_prostate_dataloader.py     # GTEx feature dataloader
│   ├── generate_thumbnail.py           # WSI thumbnail generation
│   │
│   ├── train_4class_classification_trident.py   # 4-class TRIDENT training
│   ├── train_binary_classification_trident.py   # Binary TRIDENT training
│   ├── train_supervised_attention_trident.py    # Supervised attention training
│   │
│   ├── inference_4class_trident.py              # 4-class inference + heatmaps
│   ├── inference_binary_trident.py              # Binary inference + heatmaps
│   ├── inference_supervised_attention_trident.py# Supervised attention inference
│   │
│   ├── compute_specificity_at_sensitivity.py    # Specificity @ 98% sensitivity
│   ├── recompute_roc_finetune_gtex.py           # ROC for fine-tuned GTEx model
│   ├── plot_4class_roc.py                       # 4-class ROC plots
│   │
│   ├── labels/                         # GTEx label files
│   │   └── GTEx_prostate_labels.csv
│   └── output/                         # Trained models, predictions, plots
```

## Installation

```bash
# Clone the repository
git clone https://github.com/Jiajundai26/slide-level-MIL-based-classification-of-prostate-cancer-WSIs.git
cd slide-level-MIL-based-classification-of-prostate-cancer-WSIs

# Install dependencies (TCGA-PRAD pipeline)
pip install -r TCGA-PRAD/requirements.txt

# Install dependencies (GTEx pipeline)
pip install -r GTEx_prostate/requirements.txt
```

### Dependencies

| Package | Purpose |
|---------|---------|
| `torch >= 2.0` | Deep learning framework |
| `torchvision` | Image transforms |
| `timm >= 0.9.2` | Vision model zoo (DINOv2, etc.) |
| `transformers >= 4.30` | HuggingFace model hub (Phikon, UNI) |
| `h5py` | HDF5 feature storage |
| `openslide-python` | WSI reading |
| `scikit-learn` | Metrics and data splitting |
| `shapely` | Annotation polygon processing |
| `matplotlib`, `seaborn` | Plotting |
| `pandas`, `numpy` | Data manipulation |

## Usage

### 1. Prepare Labels

**TCGA-PRAD** — generate slide-level labels from GeoJSON annotations:

```bash
python TCGA-PRAD/tcga_prad_hf_dataloader.py \
    --create_labels \
    --annotations_dir ./annotations/geojsons \
    --labels_output ./labels/slide_labels.tsv
```

**GTEx Prostate** — classify slides from pathology notes:

```bash
python GTEx_prostate/filter_gtex_prostate.py
```

### 2. Extract Features

```bash
python TCGA-PRAD/extract_features_tcga_prad.py \
    --wsi_dir ./WSI \
    --output_dir ./features/uni_v2 \
    --encoder_name uni_v2 \
    --annotations_dir ./annotations/geojsons \
    --device cuda:0
```

### 3. Train Models

**4-class Gleason grading** (Normal / G3 / G4 / G5):

```bash
python TCGA-PRAD/train_4class_classification_trident.py \
    --train --epochs 100 --device cuda:0 --balance_strategy focal
```

**Binary classification** (Non-Cancer vs G4/G5):

```bash
python TCGA-PRAD/train_binary_noncancer_vs_g45.py \
    --train --combine_gtex --epochs 50 --device cuda:0
```

**Supervised attention** (with annotation-guided attention loss):

```bash
python TCGA-PRAD/train_supervised_attention_trident.py \
    --train --epochs 50 --attention_lambda 0.5 --device cuda:0
```

**GTEx fine-tuning** (transfer to GTEx after training on TCGA):

```bash
python GTEx_prostate/train_supervised_attention_trident.py \
    --train --dataset combined --finetune_gtex --finetune_epochs 10 --device cuda:0
```

### 4. Run Inference

```bash
# 4-class inference with attention heatmaps
python TCGA-PRAD/inference_4class_trident.py \
    --checkpoint output/trident_4class/best_model.pt \
    --save_attention --generate_heatmaps

# Binary inference
python TCGA-PRAD/inference_binary_trident.py \
    --checkpoint output/binary_noncancer_vs_g45/best_model.pt \
    --feats_dir /path/to/features
```

### 5. Evaluate & Visualise

```bash
# Analyse predictions and generate confusion matrices
python TCGA-PRAD/analyze_predictions.py \
    --predictions output/analysis/predictions_with_labels.csv

# Plot ROC curves
python TCGA-PRAD/plot_roc_curves.py \
    --predictions output/analysis/predictions_with_labels.csv \
    --output output/analysis/roc_curves_ovr.png

# Compare attention heatmaps with ground-truth annotations
python TCGA-PRAD/compare_attention_vs_gt.py \
    --slide_id TCGA-XX-XXXX-01Z-00-DX1.XXXXXXXX
```

## Available Feature Encoders

### Public Encoders (no access request required)

| Encoder | Model | Feature Dim |
|---------|-------|-------------|
| `phikon` | owkin/phikon | 768 |
| `phikon_v2` | owkin/phikon-v2 | 1024 |
| `resnet50` | ImageNet ResNet-50 | 2048 |
| `resnet101` | ImageNet ResNet-101 | 2048 |
| `dinov2_vitb14` | Meta DINOv2 ViT-B/14 | 768 |
| `dinov2_vitl14` | Meta DINOv2 ViT-L/14 | 1024 |

### Gated Encoders (require HuggingFace access)

| Encoder | Model | Feature Dim | Access |
|---------|-------|-------------|--------|
| `uni` | MahmoodLab/uni | 1024 | [Request](https://huggingface.co/MahmoodLab/uni) |
| `uni_v2` | MahmoodLab/uni | 1024 | [Request](https://huggingface.co/MahmoodLab/uni) |
| `uni2_h` | MahmoodLab/UNI2-h | 1536 | [Request](https://huggingface.co/MahmoodLab/UNI2-h) |
| `virchow2` | paige-ai/Virchow2 | 1280 | [Request](https://huggingface.co/paige-ai/Virchow2) |

## Training Configuration

Key hyperparameters used across experiments:

| Parameter | Value |
|-----------|-------|
| Feature dimension | 1536 (UNI2-h) |
| Hidden dimension | 256 |
| Attention dimension | 128 |
| Dropout | 0.25 |
| Learning rate | 1e-4 |
| Weight decay | 1e-4 |
| Batch size | 8 |
| Max patches per slide | 4096 |
| Class balancing | Focal loss (γ = 2.0) |
| Attention supervision λ | 0.5 |
| Seed | 42 |

## Experiment Results

### Supervised Attention — Binary (Non-Cancer vs Cancer)

Evaluated on a combined TCGA-PRAD + GTEx test set:

| Metric | Value |
|--------|-------|
| Accuracy | 1.00 |
| Balanced Accuracy | 1.00 |
| AUC | 1.00 |
| F1 | 1.00 |

### 4-Class Gleason Grading

Evaluated on a combined test set (TCGA-PRAD cancer + GTEx benign):

| Metric | Value |
|--------|-------|
| Accuracy | 0.84 |
| Balanced Accuracy | 0.53 |
| Macro F1 | 0.48 |
| AUC (macro) | 0.91 |

Per-class AUC: Normal = 1.00, G3 = 0.79, G4 = 0.92, G5 = 0.92

### 4-Class on TCGA-PRAD Only (cancer slides)

| Metric | Value |
|--------|-------|
| Accuracy | 0.69 |
| Balanced Accuracy | 0.64 |
| Macro F1 | 0.66 |

## References

- Ilse, M., Tomczak, J. M., & Welling, M. (2018). *Attention-based deep multiple instance learning*. ICML.
- Lu, M. Y., et al. (2021). *Data-efficient and weakly supervised computational pathology on whole-slide images*. Nature Biomedical Engineering.
- Chen, R. J., et al. (2024). *Towards a general-purpose foundation model for computational pathology*. Nature Medicine (UNI).
- MahmoodLab. [TRIDENT](https://github.com/mahmoodlab/TRIDENT) — a framework for training MIL models on computational pathology tasks.

## License

This project is for research purposes.