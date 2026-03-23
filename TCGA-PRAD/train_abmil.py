"""
ABMIL Training Script for TCGA-PRAD Gleason Grading
Uses UNI_v2 features (1536-dim) for multi-class classification

Classes (from labels_MIL):
    1: G3-dominant (Gleason 3)
    2: G4-dominant (Gleason 4)
    3: G5-dominant (Gleason 5)

Default paths:
    Features: /sci-it/projects/cedmav/data/ARPA-H/Prostate_HnE/preprocessing/TCGA-PRAD/20x_256px_0px_overlap/features_uni_v2
    Labels: /sci-it/projects/cedmav/data/ARPA-H/Prostate_HnE/preprocessing/TCGA-PRAD/labels_MIL/slide_labels.tsv

Usage:
    python train_abmil.py --epochs 50
    python train_abmil.py --feats_dir /path/to/features --labels_path /path/to/labels.tsv
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import h5py
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    roc_auc_score, 
    accuracy_score, 
    balanced_accuracy_score,
    classification_report,
    confusion_matrix
)
from tqdm import tqdm
import json
from datetime import datetime

# Set deterministic behavior
SEED = 42

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
        # x: [batch, num_patches, feature_dim]
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = self.attention_c(a * b)  # element-wise multiplication
        A = A.squeeze(-1)  # [batch, num_patches]
        A = torch.softmax(A, dim=1)
        return A


class ABMILClassifier(nn.Module):
    """
    Attention-Based Multiple Instance Learning (ABMIL) model
    for multi-class Gleason grading classification
    """
    def __init__(
        self, 
        input_feature_dim=1536,  # UNI_v2 output dimension
        hidden_dim=256, 
        num_classes=4,
        dropout=0.25
    ):
        super().__init__()
        
        # Feature projection
        self.feature_projection = nn.Sequential(
            nn.Linear(input_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Gated attention
        self.attention = GatedAttention(
            input_dim=hidden_dim, 
            hidden_dim=128, 
            dropout=dropout
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x, return_attention=False):
        """
        Args:
            x: [batch, num_patches, feature_dim] or dict with 'features' key
            return_attention: whether to return attention weights
        Returns:
            logits: [batch, num_classes]
            attention: [batch, num_patches] (optional)
        """
        if isinstance(x, dict):
            x = x['features']
        
        # Project features
        h = self.feature_projection(x)  # [batch, num_patches, hidden_dim]
        
        # Compute attention
        A = self.attention(h)  # [batch, num_patches]
        
        # Weighted average
        M = torch.bmm(A.unsqueeze(1), h).squeeze(1)  # [batch, hidden_dim]
        
        # Classification
        logits = self.classifier(M)  # [batch, num_classes]
        
        if return_attention:
            return logits, A
        return logits


class TCGAPRADDataset(Dataset):
    """Dataset for TCGA-PRAD H5 features"""
    
    # Class mapping: original labels (1, 2, 3) -> 0-indexed (0, 1, 2)
    LABEL_MAP = {1: 0, 2: 1, 3: 2}
    CLASS_NAMES = ['G3-dominant', 'G4-dominant', 'G5-dominant']
    
    def __init__(
        self, 
        feats_path, 
        labels_df, 
        split, 
        max_patches=512,
        subsample_train=True
    ):
        """
        Args:
            feats_path: path to directory containing .h5 feature files
            labels_df: pandas DataFrame with slide_id, label, fold_0 columns
            split: 'Training', 'Validation', or 'Testing'
            max_patches: maximum number of patches to use per slide
            subsample_train: whether to randomly subsample patches during training
        """
        self.feats_path = feats_path
        self.max_patches = max_patches
        self.split = split
        self.subsample_train = subsample_train and (split == 'Training')
        
        # Filter by split
        self.df = labels_df[labels_df["fold_0"] == split].reset_index(drop=True)
        
        # Check which slides have feature files
        available_slides = []
        for _, row in self.df.iterrows():
            h5_path = os.path.join(feats_path, row['slide_id'] + '.h5')
            if os.path.exists(h5_path):
                available_slides.append(row)
        
        self.df = pd.DataFrame(available_slides).reset_index(drop=True)
        print(f"[{split}] Found {len(self.df)} slides with features")
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        slide_id = row['slide_id']
        h5_path = os.path.join(self.feats_path, slide_id + '.h5')
        
        with h5py.File(h5_path, "r") as f:
            features = torch.from_numpy(f["features"][:]).float()
            if "coords" in f:
                coords = f["coords"][:]
            else:
                coords = None

        num_patches = features.shape[0]
        
        # Subsample patches if needed
        if self.subsample_train and num_patches > self.max_patches:
            indices = torch.randperm(num_patches)[:self.max_patches]
            features = features[indices]
        elif not self.subsample_train and num_patches > self.max_patches:
            # For validation/test, use first max_patches (deterministic)
            features = features[:self.max_patches]

        # Map labels: 1->0, 2->1, 3->2 for CrossEntropyLoss
        original_label = row["label"]
        label = torch.tensor(self.LABEL_MAP[original_label], dtype=torch.long)
        
        return {
            'features': features,
            'label': label,
            'slide_id': slide_id,
            'original_label': original_label
        }


def collate_fn(batch):
    """Custom collate function to handle variable-length sequences"""
    features = [item['features'] for item in batch]
    labels = torch.stack([item['label'] for item in batch])
    slide_ids = [item['slide_id'] for item in batch]
    
    # Pad sequences to same length
    max_len = max(f.shape[0] for f in features)
    feature_dim = features[0].shape[1]
    
    padded_features = torch.zeros(len(features), max_len, feature_dim)
    mask = torch.zeros(len(features), max_len, dtype=torch.bool)
    
    for i, f in enumerate(features):
        length = f.shape[0]
        padded_features[i, :length] = f
        mask[i, :length] = True
    
    return {
        'features': padded_features,
        'labels': labels,
        'mask': mask,
        'slide_ids': slide_ids
    }


def train_epoch(model, loader, criterion, optimizer, device, class_weights=None):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        features = batch['features'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        logits = model({'features': features})
        
        loss = criterion(logits, labels)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy, balanced_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes=4):
    """Evaluate the model"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    all_slide_ids = []
    
    for batch in tqdm(loader, desc="Evaluating"):
        features = batch['features'].to(device)
        labels = batch['labels'].to(device)
        
        logits = model({'features': features})
        loss = criterion(logits, labels)
        
        total_loss += loss.item()
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_slide_ids.extend(batch['slide_ids'])
    
    avg_loss = total_loss / len(loader)
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    # Metrics
    accuracy = accuracy_score(all_labels, all_preds)
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    
    # Multi-class AUC (one-vs-rest)
    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
    except ValueError:
        auc = 0.0  # If not all classes present
    
    # Per-class metrics (labels are now 0-indexed: 0, 1, 2)
    report = classification_report(
        all_labels, all_preds, 
        target_names=['G3-dominant', 'G4-dominant', 'G5-dominant'],
        labels=[0, 1, 2],
        output_dict=True,
        zero_division=0
    )
    
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    results = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'auc': auc,
        'report': report,
        'confusion_matrix': conf_matrix.tolist(),
        'predictions': {
            'slide_ids': all_slide_ids,
            'labels': all_labels.tolist(),
            'predictions': all_preds.tolist(),
            'probabilities': all_probs.tolist()
        }
    }
    
    return results


def main(args):
    set_seed(SEED)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Handle resume vs new training
    start_epoch = 1
    best_val_auc = 0.0
    best_epoch = 0
    
    if args.resume:
        # Resume from checkpoint - use the same output directory
        output_dir = os.path.dirname(args.resume)
        print(f"Resuming training from: {args.resume}")
    else:
        # Create new output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(args.output_dir, f"run_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save args
        with open(os.path.join(output_dir, 'args.json'), 'w') as f:
            json.dump(vars(args), f, indent=2)
    
    # Load labels
    labels_path = args.labels_path
    df = pd.read_csv(labels_path, sep='\t')
    print(f"Loaded {len(df)} slides from labels file")
    print(f"Class distribution:\n{df['class_name'].value_counts()}")
    
    # Compute class weights for imbalanced data
    # Labels are 1, 2, 3 in the file, mapped to 0, 1, 2 internally
    class_counts = df['label'].value_counts().sort_index()
    total = len(df)
    # Create weights for 0-indexed classes (0, 1, 2)
    class_weights = torch.tensor(
        [total / (len(class_counts) * class_counts.get(orig_label, 1)) 
         for orig_label in [1, 2, 3]],
        dtype=torch.float32
    ).to(device)
    print(f"Class distribution: {dict(class_counts)}")
    print(f"Class weights (for G3, G4, G5): {class_weights.tolist()}")
    
    # Create datasets
    train_dataset = TCGAPRADDataset(
        args.feats_dir, df, 'Training', 
        max_patches=args.max_patches,
        subsample_train=True
    )
    val_dataset = TCGAPRADDataset(
        args.feats_dir, df, 'Validation',
        max_patches=args.max_patches,
        subsample_train=False
    )
    test_dataset = TCGAPRADDataset(
        args.feats_dir, df, 'Testing',
        max_patches=args.max_patches,
        subsample_train=False
    )
    
    # Create dataloaders
    use_pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=use_pin_memory
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    
    # Initialize model
    model = ABMILClassifier(
        input_feature_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        num_classes=args.num_classes,
        dropout=args.dropout
    ).to(device)
    
    print(f"\nModel architecture:")
    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.epochs, 
        eta_min=args.lr * 0.01
    )
    
    # Load checkpoint if resuming
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {checkpoint['epoch']}, starting at epoch {start_epoch}")
        
        # Try to load best_val_auc from best_model.pt if it exists
        best_model_path = os.path.join(output_dir, 'best_model.pt')
        if os.path.exists(best_model_path):
            best_ckpt = torch.load(best_model_path, map_location=device)
            best_val_auc = best_ckpt.get('val_auc', 0.0)
            best_epoch = best_ckpt.get('epoch', 0)
            print(f"Best model so far: epoch {best_epoch}, AUC: {best_val_auc:.4f}")
        
        # Load existing history if available
        history_path = os.path.join(output_dir, 'history.json')
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                history = json.load(f)
        else:
            history = {'train': [], 'val': []}
    else:
        history = {'train': [], 'val': []}
    
    print("\n" + "="*60)
    print(f"Starting training from epoch {start_epoch}...")
    print("="*60)
    
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 40)
        
        # Train
        train_loss, train_acc, train_bal_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_results = evaluate(model, val_loader, criterion, device, args.num_classes)
        
        # Update scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Log
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | Balanced Acc: {train_bal_acc:.4f}")
        print(f"Val Loss: {val_results['loss']:.4f} | Acc: {val_results['accuracy']:.4f} | "
              f"Balanced Acc: {val_results['balanced_accuracy']:.4f} | AUC: {val_results['auc']:.4f}")
        print(f"LR: {current_lr:.6f}")
        
        history['train'].append({
            'epoch': epoch,
            'loss': train_loss,
            'accuracy': train_acc,
            'balanced_accuracy': train_bal_acc
        })
        history['val'].append({
            'epoch': epoch,
            'loss': val_results['loss'],
            'accuracy': val_results['accuracy'],
            'balanced_accuracy': val_results['balanced_accuracy'],
            'auc': val_results['auc']
        })
        
        # Save best model
        if val_results['auc'] > best_val_auc:
            best_val_auc = val_results['auc']
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': best_val_auc,
            }, os.path.join(output_dir, 'best_model.pt'))
            print(f"  -> New best model saved! (AUC: {best_val_auc:.4f})")
        
        # Save latest checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, os.path.join(output_dir, 'latest_checkpoint.pt'))
    
    # Save training history
    with open(os.path.join(output_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    # Final evaluation on test set
    print("\n" + "="*60)
    print(f"Loading best model from epoch {best_epoch}")
    print("="*60)
    
    checkpoint = torch.load(os.path.join(output_dir, 'best_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_results = evaluate(model, test_loader, criterion, device, args.num_classes)
    
    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    print(f"Accuracy: {test_results['accuracy']:.4f}")
    print(f"Balanced Accuracy: {test_results['balanced_accuracy']:.4f}")
    print(f"AUC (macro): {test_results['auc']:.4f}")
    print("\nClassification Report:")
    print(classification_report(
        test_results['predictions']['labels'],
        test_results['predictions']['predictions'],
        target_names=['G3-dominant', 'G4-dominant', 'G5-dominant'],
        labels=[0, 1, 2],
        zero_division=0
    ))
    print("\nConfusion Matrix:")
    print(np.array(test_results['confusion_matrix']))
    
    # Save test results
    # Remove numpy arrays for JSON serialization
    test_results_save = {k: v for k, v in test_results.items() if k != 'report'}
    test_results_save['classification_report'] = test_results['report']
    with open(os.path.join(output_dir, 'test_results.json'), 'w') as f:
        json.dump(test_results_save, f, indent=2)
    
    print(f"\nResults saved to: {output_dir}")
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ABMIL Training for TCGA-PRAD Gleason Grading")
    
    # Data paths
    parser.add_argument('--feats_dir', type=str, 
                        default='/sci-it/projects/cedmav/data/ARPA-H/Prostate_HnE/preprocessing/TCGA-PRAD/20x_256px_0px_overlap/features_uni_v2',
                        help='Directory containing H5 feature files')
    parser.add_argument('--labels_path', type=str, 
                        default='/sci-it/projects/cedmav/data/ARPA-H/Prostate_HnE/preprocessing/TCGA-PRAD/labels_MIL/slide_labels.tsv',
                        help='Path to labels TSV file')
    parser.add_argument('--output_dir', type=str, default='output/training',
                        help='Output directory for checkpoints and results')
    
    # Model parameters
    parser.add_argument('--feature_dim', type=int, default=1536,
                        help='Input feature dimension (1536 for UNI_v2)')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='Hidden dimension for attention network')
    parser.add_argument('--num_classes', type=int, default=3,
                        help='Number of classes (G3, G4, G5-dominant)')
    parser.add_argument('--dropout', type=float, default=0.25,
                        help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for AdamW')
    parser.add_argument('--max_patches', type=int, default=4096,
                        help='Maximum patches per slide')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of dataloader workers')
    
    # Resume training
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint file to resume training from')
    
    args = parser.parse_args()
    main(args)
