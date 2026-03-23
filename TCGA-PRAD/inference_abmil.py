"""
ABMIL Inference Script for TCGA-PRAD Gleason Grading

Runs inference on H5 feature files using a trained ABMIL model.
Outputs predictions, probabilities, and optionally attention weights for heatmaps.

Usage:
    # Single slide inference
    python inference_abmil.py --checkpoint output/MIL_training/run_XXXX/best_model.pt --input slide.h5

    # Batch inference on a directory
    python inference_abmil.py --checkpoint output/MIL_training/run_XXXX/best_model.pt --input_dir /path/to/features/

    # With attention weights for heatmaps
    python inference_abmil.py --checkpoint output/MIL_training/run_XXXX/best_model.pt --input_dir /path/to/features/ --save_attention
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import h5py
import pandas as pd
from tqdm import tqdm
import json
from glob import glob


# Class names mapping (0-indexed internally)
CLASS_NAMES = ['G3-dominant', 'G4-dominant', 'G5-dominant']
LABEL_TO_ORIGINAL = {0: 1, 1: 2, 2: 3}  # Map back to original labels


class GatedAttention(nn.Module):
    """Gated Attention mechanism for MIL"""
    def __init__(self, input_dim=1536, hidden_dim=256, dropout=0.25):
        super().__init__()
        self.attention_a = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout)
        )
        self.attention_b = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Dropout(dropout)
        )
        self.attention_c = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = self.attention_c(a * b)
        A = A.squeeze(-1)
        A = torch.softmax(A, dim=1)
        return A


class ABMILClassifier(nn.Module):
    """Attention-Based Multiple Instance Learning (ABMIL) model"""
    def __init__(
        self, 
        input_feature_dim=1536,
        hidden_dim=256, 
        num_classes=3,
        dropout=0.25
    ):
        super().__init__()
        
        self.feature_projection = nn.Sequential(
            nn.Linear(input_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.attention = GatedAttention(
            input_dim=hidden_dim, 
            hidden_dim=128, 
            dropout=dropout
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
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


def load_model(checkpoint_path, device, args_path=None):
    """Load trained model from checkpoint"""
    
    # Try to load args from the same directory
    if args_path is None:
        args_path = os.path.join(os.path.dirname(checkpoint_path), 'args.json')
    
    if os.path.exists(args_path):
        with open(args_path, 'r') as f:
            train_args = json.load(f)
        feature_dim = train_args.get('feature_dim', 1536)
        hidden_dim = train_args.get('hidden_dim', 256)
        num_classes = train_args.get('num_classes', 3)
        dropout = train_args.get('dropout', 0.25)
    else:
        print("Warning: args.json not found, using default model parameters")
        feature_dim = 1536
        hidden_dim = 256
        num_classes = 3
        dropout = 0.25
    
    # Initialize model
    model = ABMILClassifier(
        input_feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        dropout=dropout
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    if 'val_auc' in checkpoint:
        print(f"Validation AUC: {checkpoint['val_auc']:.4f}")
    
    return model


def load_features(h5_path, max_patches=None):
    """Load features from H5 file"""
    with h5py.File(h5_path, 'r') as f:
        features = torch.from_numpy(f['features'][:]).float()
        coords = f['coords'][:] if 'coords' in f else None
        
        # Get additional metadata if available
        attrs = {}
        if 'coords' in f and hasattr(f['coords'], 'attrs'):
            attrs = dict(f['coords'].attrs)
    
    if max_patches and features.shape[0] > max_patches:
        features = features[:max_patches]
        if coords is not None:
            coords = coords[:max_patches]
    
    return features, coords, attrs


@torch.no_grad()
def predict_single(model, h5_path, device, max_patches=None, return_attention=False):
    """Run inference on a single slide"""
    features, coords, attrs = load_features(h5_path, max_patches)
    features = features.unsqueeze(0).to(device)  # Add batch dimension
    
    if return_attention:
        logits, attention = model(features, return_attention=True)
        attention = attention.squeeze(0).cpu().numpy()
    else:
        logits = model(features)
        attention = None
    
    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    pred_class = int(logits.argmax(dim=1).item())
    
    result = {
        'slide_id': os.path.basename(h5_path).replace('.h5', ''),
        'predicted_class': pred_class,
        'predicted_label': LABEL_TO_ORIGINAL[pred_class],
        'predicted_name': CLASS_NAMES[pred_class],
        'probabilities': {
            CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))
        },
        'confidence': float(probs[pred_class]),
        'num_patches': features.shape[1]
    }
    
    if return_attention:
        attention_norm = (attention - attention.min()) / (attention.max() - attention.min() + 1e-8)
        attention_binary = (attention_norm >= 0.5).astype(np.uint8)
        result['attention'] = attention
        result['attention_norm'] = attention_norm
        result['attention_binary'] = attention_binary
        result['coords'] = coords
        result['attrs'] = attrs
    
    return result


def predict_batch(model, h5_paths, device, max_patches=None, return_attention=False, output_dir=None):
    """Run inference on multiple slides"""
    results = []
    
    for h5_path in tqdm(h5_paths, desc="Running inference"):
        try:
            result = predict_single(model, h5_path, device, max_patches, return_attention)
            results.append(result)
            
            # Save attention weights if requested
            if return_attention and output_dir:
                attention_path = os.path.join(
                    output_dir, 
                    'attention', 
                    f"{result['slide_id']}_attention.npz"
                )
                os.makedirs(os.path.dirname(attention_path), exist_ok=True)
                np.savez(
                    attention_path,
                    attention=result['attention'],
                    attention_norm=result['attention_norm'],
                    attention_binary=result['attention_binary'],
                    coords=result['coords'],
                    predicted_class=result['predicted_class'],
                    predicted_name=result['predicted_name'],
                    **result['attrs']
                )
                # Remove from result dict to save memory
                del result['attention']
                del result['attention_norm']
                del result['attention_binary']
                del result['coords']
                del result['attrs']
                
        except Exception as e:
            print(f"Error processing {h5_path}: {e}")
            results.append({
                'slide_id': os.path.basename(h5_path).replace('.h5', ''),
                'error': str(e)
            })
    
    return results


def main(args):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(args.checkpoint, device)
    
    # Get input files
    if args.input:
        h5_paths = [args.input]
    elif args.input_dir:
        h5_paths = sorted(glob(os.path.join(args.input_dir, '*.h5')))
        print(f"Found {len(h5_paths)} H5 files in {args.input_dir}")
    else:
        raise ValueError("Must specify either --input or --input_dir")
    
    if len(h5_paths) == 0:
        print("No H5 files found!")
        return
    
    # Create output directory
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Run inference
    results = predict_batch(
        model, 
        h5_paths, 
        device, 
        max_patches=args.max_patches,
        return_attention=args.save_attention,
        output_dir=args.output_dir
    )
    
    # Create results DataFrame
    df_results = pd.DataFrame([
        {
            'slide_id': r['slide_id'],
            'predicted_label': r.get('predicted_label'),
            'predicted_name': r.get('predicted_name'),
            'confidence': r.get('confidence'),
            'prob_G3': r.get('probabilities', {}).get('G3-dominant'),
            'prob_G4': r.get('probabilities', {}).get('G4-dominant'),
            'prob_G5': r.get('probabilities', {}).get('G5-dominant'),
            'num_patches': r.get('num_patches'),
            'error': r.get('error')
        }
        for r in results
    ])
    
    # Print summary
    print("\n" + "="*60)
    print("INFERENCE RESULTS")
    print("="*60)
    
    valid_results = df_results[df_results['error'].isna()]
    if len(valid_results) > 0:
        print(f"\nProcessed {len(valid_results)} slides successfully")
        print(f"\nPrediction distribution:")
        print(valid_results['predicted_name'].value_counts())
        print(f"\nMean confidence: {valid_results['confidence'].mean():.4f}")
    
    if df_results['error'].notna().any():
        print(f"\n{df_results['error'].notna().sum()} slides had errors")
    
    # Save results
    if args.output_dir:
        output_path = os.path.join(args.output_dir, 'predictions.csv')
        df_results.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")
        
        # Save full results as JSON
        json_path = os.path.join(args.output_dir, 'predictions.json')
        with open(json_path, 'w') as f:
            # Remove numpy arrays for JSON serialization
            json_results = []
            for r in results:
                jr = {k: v for k, v in r.items() if k not in ['attention', 'coords', 'attrs']}
                json_results.append(jr)
            json.dump(json_results, f, indent=2)
        print(f"Full results saved to: {json_path}")
        
        if args.save_attention:
            print(f"Attention weights saved to: {os.path.join(args.output_dir, 'attention/')}")
    
    # Print first few results
    print("\nSample predictions:")
    print(df_results.head(10).to_string(index=False))
    
    return df_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ABMIL Inference for Gleason Grading")
    
    # Required
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (best_model.pt)')
    
    # Input (one of these required)
    parser.add_argument('--input', type=str, default=None,
                        help='Path to a single H5 feature file')
    parser.add_argument('--input_dir', type=str, default=None,
                        help='Directory containing H5 feature files')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='output/inference',
                        help='Output directory for predictions')
    
    # Options
    parser.add_argument('--max_patches', type=int, default=None,
                        help='Maximum patches per slide (None = use all)')
    parser.add_argument('--save_attention', action='store_true',
                        help='Save attention weights for heatmap generation')
    
    args = parser.parse_args()
    main(args)
