"""
Analyze ABMIL Predictions

1. Compare predictions against ground truth labels
2. Generate attention heatmaps for specific slides
3. Filter predictions by confidence threshold

Usage:
    python analyze_predictions.py --predictions output/inference/predictions.csv
"""

import os
import argparse
import numpy as np
import pandas as pd
import h5py
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, 
    classification_report, confusion_matrix,
    roc_auc_score
)
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm

# Class mappings
CLASS_NAMES = ['G3-dominant', 'G4-dominant', 'G5-dominant']
ORIGINAL_TO_IDX = {1: 0, 2: 1, 3: 2}


def load_predictions_and_labels(predictions_path, labels_path):
    """Load predictions and merge with ground truth labels"""
    # Load predictions
    preds_df = pd.read_csv(predictions_path)
    
    # Load ground truth labels
    labels_df = pd.read_csv(labels_path, sep='\t')
    
    # Merge on slide_id
    merged = preds_df.merge(
        labels_df[['slide_id', 'label', 'fold_0', 'class_name']], 
        on='slide_id', 
        how='left'
    )
    
    # Rename columns for clarity
    merged = merged.rename(columns={
        'label': 'ground_truth_label',
        'class_name': 'ground_truth_name',
        'fold_0': 'split'
    })
    
    return merged


def compare_with_ground_truth(merged_df, output_dir):
    """Compare predictions against ground truth labels"""
    print("\n" + "="*60)
    print("1. COMPARISON WITH GROUND TRUTH")
    print("="*60)
    
    # Filter to slides with ground truth labels
    labeled = merged_df[merged_df['ground_truth_label'].notna()].copy()
    unlabeled = merged_df[merged_df['ground_truth_label'].isna()]
    
    print(f"\nSlides with ground truth labels: {len(labeled)}")
    print(f"Slides without labels (unlabeled): {len(unlabeled)}")
    
    if len(labeled) == 0:
        print("No labeled slides found for comparison!")
        return None
    
    # Convert to 0-indexed for comparison
    labeled['gt_idx'] = labeled['ground_truth_label'].map(ORIGINAL_TO_IDX)
    labeled['pred_idx'] = labeled['predicted_label'].map(ORIGINAL_TO_IDX)
    
    y_true = labeled['gt_idx'].values
    y_pred = labeled['pred_idx'].values
    
    # Overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    
    print(f"\n--- Overall Metrics (on {len(labeled)} labeled slides) ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    
    # Per-split metrics
    print("\n--- Metrics by Split ---")
    for split in ['Training', 'Validation', 'Testing']:
        split_df = labeled[labeled['split'] == split]
        if len(split_df) > 0:
            split_acc = accuracy_score(
                split_df['gt_idx'].values, 
                split_df['pred_idx'].values
            )
            split_bal_acc = balanced_accuracy_score(
                split_df['gt_idx'].values, 
                split_df['pred_idx'].values
            )
            print(f"{split:12} ({len(split_df):3} slides): Acc={split_acc:.4f}, Balanced Acc={split_bal_acc:.4f}")
    
    # Classification report
    print("\n--- Classification Report ---")
    print(classification_report(
        y_true, y_pred,
        target_names=CLASS_NAMES,
        labels=[0, 1, 2],
        zero_division=0
    ))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    print("--- Confusion Matrix ---")
    print("              Predicted")
    print("              G3    G4    G5")
    print(f"Actual G3   {cm[0,0]:4d}  {cm[0,1]:4d}  {cm[0,2]:4d}")
    print(f"       G4   {cm[1,0]:4d}  {cm[1,1]:4d}  {cm[1,2]:4d}")
    print(f"       G5   {cm[2,0]:4d}  {cm[2,1]:4d}  {cm[2,2]:4d}")
    
    # Save confusion matrix plot
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    
    # Add text annotations
    for i in range(3):
        for j in range(3):
            text = ax.text(j, i, cm[i, j], ha="center", va="center", 
                          color="white" if cm[i, j] > cm.max()/2 else "black")
    
    plt.colorbar(im)
    plt.tight_layout()
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"\nConfusion matrix saved to: {cm_path}")
    
    # Misclassified slides
    misclassified = labeled[labeled['gt_idx'] != labeled['pred_idx']]
    print(f"\n--- Misclassified Slides ({len(misclassified)}) ---")
    if len(misclassified) > 0:
        misc_summary = misclassified.groupby(['ground_truth_name', 'predicted_name']).size().reset_index(name='count')
        print(misc_summary.to_string(index=False))
        
        # Save misclassified list
        misc_path = os.path.join(output_dir, 'misclassified_slides.csv')
        misclassified[['slide_id', 'ground_truth_name', 'predicted_name', 'confidence', 'split']].to_csv(
            misc_path, index=False
        )
        print(f"Misclassified slides saved to: {misc_path}")
    
    return labeled


def filter_by_confidence(merged_df, output_dir, thresholds=[0.5, 0.6, 0.7, 0.8]):
    """Filter and analyze predictions by confidence threshold"""
    print("\n" + "="*60)
    print("3. CONFIDENCE THRESHOLD ANALYSIS")
    print("="*60)
    
    labeled = merged_df[merged_df['ground_truth_label'].notna()].copy()
    labeled['gt_idx'] = labeled['ground_truth_label'].map(ORIGINAL_TO_IDX)
    labeled['pred_idx'] = labeled['predicted_label'].map(ORIGINAL_TO_IDX)
    
    print("\n--- Accuracy at Different Confidence Thresholds ---")
    print(f"{'Threshold':<12} {'Slides':<10} {'Coverage':<12} {'Accuracy':<12} {'Balanced Acc':<12}")
    print("-" * 58)
    
    results = []
    for thresh in thresholds:
        high_conf = labeled[labeled['confidence'] >= thresh]
        if len(high_conf) > 0:
            acc = accuracy_score(high_conf['gt_idx'], high_conf['pred_idx'])
            bal_acc = balanced_accuracy_score(high_conf['gt_idx'], high_conf['pred_idx'])
            coverage = len(high_conf) / len(labeled) * 100
            print(f"{thresh:<12.1%} {len(high_conf):<10} {coverage:<12.1f}% {acc:<12.4f} {bal_acc:<12.4f}")
            results.append({
                'threshold': thresh,
                'slides': len(high_conf),
                'coverage': coverage,
                'accuracy': acc,
                'balanced_accuracy': bal_acc
            })
    
    # Save high-confidence predictions
    high_conf_df = merged_df[merged_df['confidence'] >= 0.7]
    high_conf_path = os.path.join(output_dir, 'high_confidence_predictions.csv')
    high_conf_df.to_csv(high_conf_path, index=False)
    print(f"\nHigh-confidence (≥70%) predictions saved to: {high_conf_path}")
    print(f"  - {len(high_conf_df)} slides with confidence ≥ 70%")
    
    # Low confidence predictions (uncertain)
    low_conf_df = merged_df[merged_df['confidence'] < 0.5]
    low_conf_path = os.path.join(output_dir, 'low_confidence_predictions.csv')
    low_conf_df.to_csv(low_conf_path, index=False)
    print(f"\nLow-confidence (<50%) predictions saved to: {low_conf_path}")
    print(f"  - {len(low_conf_df)} slides with confidence < 50%")
    
    # Confidence distribution plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(merged_df['confidence'], bins=30, edgecolor='black', alpha=0.7)
    axes[0].axvline(x=0.5, color='r', linestyle='--', label='50% threshold')
    axes[0].axvline(x=0.7, color='g', linestyle='--', label='70% threshold')
    axes[0].set_xlabel('Confidence')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Confidence Distribution')
    axes[0].legend()
    
    # Box plot by predicted class
    conf_by_class = [
        merged_df[merged_df['predicted_name'] == cls]['confidence'].values 
        for cls in CLASS_NAMES
    ]
    axes[1].boxplot(conf_by_class, labels=CLASS_NAMES)
    axes[1].set_xlabel('Predicted Class')
    axes[1].set_ylabel('Confidence')
    axes[1].set_title('Confidence by Predicted Class')
    
    plt.tight_layout()
    conf_plot_path = os.path.join(output_dir, 'confidence_distribution.png')
    plt.savefig(conf_plot_path, dpi=150)
    plt.close()
    print(f"\nConfidence distribution plot saved to: {conf_plot_path}")
    
    return results


def generate_attention_heatmaps(
    predictions_path, 
    checkpoint_path,
    features_dir,
    wsi_dir,
    output_dir,
    num_slides=5,
    slide_ids=None
):
    """Generate attention heatmaps for selected slides"""
    print("\n" + "="*60)
    print("2. ATTENTION HEATMAP GENERATION")
    print("="*60)
    
    import torch
    import torch.nn as nn
    
    # Import model classes
    class GatedAttention(nn.Module):
        def __init__(self, input_dim=1536, hidden_dim=256, dropout=0.25):
            super().__init__()
            self.attention_a = nn.Sequential(
                nn.Linear(input_dim, hidden_dim), nn.Tanh(), nn.Dropout(dropout)
            )
            self.attention_b = nn.Sequential(
                nn.Linear(input_dim, hidden_dim), nn.Sigmoid(), nn.Dropout(dropout)
            )
            self.attention_c = nn.Linear(hidden_dim, 1)

        def forward(self, x):
            a = self.attention_a(x)
            b = self.attention_b(x)
            A = self.attention_c(a * b).squeeze(-1)
            return torch.softmax(A, dim=1)

    class ABMILClassifier(nn.Module):
        def __init__(self, input_feature_dim=1536, hidden_dim=256, num_classes=3, dropout=0.25):
            super().__init__()
            self.feature_projection = nn.Sequential(
                nn.Linear(input_feature_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)
            )
            self.attention = GatedAttention(input_dim=hidden_dim, hidden_dim=128, dropout=dropout)
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, num_classes)
            )

        def forward(self, x, return_attention=False):
            if isinstance(x, dict):
                x = x['features']
            h = self.feature_projection(x)
            A = self.attention(h)
            M = torch.bmm(A.unsqueeze(1), h).squeeze(1)
            logits = self.classifier(M)
            if return_attention:
                return logits, A
            return logits
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = ABMILClassifier()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Load predictions to select slides
    preds_df = pd.read_csv(predictions_path)
    
    if slide_ids is None:
        # Select diverse slides: highest confidence from each class
        selected = []
        for cls in CLASS_NAMES:
            cls_df = preds_df[preds_df['predicted_name'] == cls].nlargest(
                min(num_slides // 3 + 1, len(preds_df[preds_df['predicted_name'] == cls])), 
                'confidence'
            )
            selected.append(cls_df)
        selected_df = pd.concat(selected).head(num_slides)
        slide_ids = selected_df['slide_id'].tolist()
    
    print(f"\nGenerating heatmaps for {len(slide_ids)} slides...")
    
    heatmap_dir = os.path.join(output_dir, 'heatmaps')
    os.makedirs(heatmap_dir, exist_ok=True)
    
    # Gradient colormap: normal (0) -> dominant grade (1), color by class
    normal_color = '#D9D9D9'
    dominant_colors = ['#1F77B4', '#FF7F0E', '#2CA02C']  # G3, G4, G5
    
    for slide_id in tqdm(slide_ids, desc="Generating heatmaps"):
        try:
            # Load features
            h5_path = os.path.join(features_dir, f"{slide_id}.h5")
            if not os.path.exists(h5_path):
                print(f"  Features not found for {slide_id}")
                continue
            
            with h5py.File(h5_path, 'r') as f:
                features = torch.from_numpy(f['features'][:]).float()
                coords = f['coords'][:] if 'coords' in f else None
                patch_size = f['coords'].attrs.get('patch_size_level0', 256) if 'coords' in f else 256
            
            if coords is None:
                print(f"  No coordinates found for {slide_id}")
                continue
            
            # Get attention weights
            with torch.no_grad():
                features_batch = features.unsqueeze(0).to(device)
                logits, attention = model(features_batch, return_attention=True)
                attention = attention.squeeze(0).cpu().numpy()
                probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            
            pred_class = int(logits.argmax(dim=1).item())
            
            # Create heatmap visualization
            fig, ax = plt.subplots(figsize=(14, 12))
            
            # Normalize coordinates
            x_coords = coords[:, 0]
            y_coords = coords[:, 1]
            
            # Normalize attention for visualization
            attn_norm = (attention - attention.min()) / (attention.max() - attention.min() + 1e-8)
            
            # Draw each patch as a square rectangle
            from matplotlib.collections import PatchCollection
            rectangles = []
            for i in range(len(coords)):
                rect = plt.Rectangle(
                    (x_coords[i], y_coords[i]),  # Bottom-left corner
                    patch_size, patch_size,       # Width, height
                )
                rectangles.append(rect)
            
            # Create patch collection with gradient colors (0=normal, 1=dominant grade)
            colors = [normal_color, dominant_colors[pred_class]]
            cmap = LinearSegmentedColormap.from_list('attention', colors)
            pc = PatchCollection(rectangles, cmap=cmap, alpha=0.8, edgecolor='none')
            pc.set_array(attn_norm)
            pc.set_clim(0, 1)
            ax.add_collection(pc)
            
            # Set axis limits
            ax.set_xlim(x_coords.min() - patch_size, x_coords.max() + 2 * patch_size)
            ax.set_ylim(y_coords.max() + 2 * patch_size, y_coords.min() - patch_size)  # Invert y
            ax.set_aspect('equal')
            
            # Add colorbar with labels only at 0 and 1
            cbar = plt.colorbar(pc, ax=ax, shrink=0.8, ticks=[0, 1])
            cbar.set_ticklabels(['Normal', CLASS_NAMES[pred_class]], fontsize=18)
            cbar.ax.tick_params(labelsize=18)
            
            # Increase axis label font sizes
            ax.set_xlabel('X coordinate (pixels)', fontsize=16)
            ax.set_ylabel('Y coordinate (pixels)', fontsize=16)
            ax.tick_params(axis='both', labelsize=14)
            
            # Add patch size info
            ax.text(0.02, 0.02, f'Patch size: {patch_size}×{patch_size} px\nTotal patches: {len(coords)}', 
                    transform=ax.transAxes, fontsize=16, verticalalignment='bottom',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            
            # Save
            save_path = os.path.join(heatmap_dir, f"{slide_id}_attention.png")
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            plt.close()
            print(f"  Saved: {slide_id}")
            
            # Save attention weights as npz
            npz_path = os.path.join(heatmap_dir, f"{slide_id}_attention.npz")
            np.savez(
                npz_path,
                attention=attention,
                coords=coords,
                prediction=pred_class,
                probabilities=probs
            )
            
        except Exception as e:
            print(f"  Error processing {slide_id}: {e}")
    
    print(f"\nHeatmaps saved to: {heatmap_dir}")
    return heatmap_dir


def main(args):
    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and merge data
    print("Loading predictions and labels...")
    merged_df = load_predictions_and_labels(args.predictions, args.labels)
    
    # Save merged data
    merged_path = os.path.join(output_dir, 'predictions_with_labels.csv')
    merged_df.to_csv(merged_path, index=False)
    print(f"Merged predictions saved to: {merged_path}")
    
    # 1. Compare with ground truth
    compare_with_ground_truth(merged_df, output_dir)
    
    # 2. Generate attention heatmaps
    if args.generate_heatmaps:
        generate_attention_heatmaps(
            predictions_path=args.predictions,
            checkpoint_path=args.checkpoint,
            features_dir=args.features_dir,
            wsi_dir=args.wsi_dir,
            output_dir=output_dir,
            num_slides=args.num_heatmaps
        )
    
    # 3. Confidence threshold analysis
    filter_by_confidence(merged_df, output_dir)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"All results saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze ABMIL Predictions")
    
    parser.add_argument('--predictions', type=str, 
                        default='output/inference/predictions.csv',
                        help='Path to predictions CSV')
    parser.add_argument('--labels', type=str,
                        default='/sci-it/projects/cedmav/data/ARPA-H/Prostate_HnE/preprocessing/TCGA-PRAD/labels_MIL/slide_labels.tsv',
                        help='Path to ground truth labels TSV')
    parser.add_argument('--output_dir', type=str, 
                        default='output/analysis',
                        help='Output directory for analysis results')
    
    # Heatmap options
    parser.add_argument('--generate_heatmaps', action='store_true',
                        help='Generate attention heatmaps')
    parser.add_argument('--checkpoint', type=str,
                        default='output/MIL_training/run_20260111_084431/best_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--features_dir', type=str,
                        default='/sci-it/projects/cedmav/data/ARPA-H/Prostate_HnE/preprocessing/TCGA-PRAD/20x_256px_0px_overlap/features_uni_v2',
                        help='Directory containing H5 feature files')
    parser.add_argument('--wsi_dir', type=str,
                        default=None,
                        help='Directory containing WSI files (for overlay heatmaps)')
    parser.add_argument('--num_heatmaps', type=int, default=9,
                        help='Number of heatmaps to generate')
    
    args = parser.parse_args()
    main(args)
