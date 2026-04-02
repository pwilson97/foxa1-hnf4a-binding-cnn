"""
Training script for the motif grammar CNN classifier.

Trains on CPU (dataset is small ~8K sequences), saves best model,
generates training curves and confusion matrix.
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (classification_report, confusion_matrix,
                             balanced_accuracy_score, roc_auc_score)
from sklearn.preprocessing import label_binarize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from dl_data import (load_all_sequences, create_splits, get_dataloaders,
                     compute_class_weights, CATEGORIES, IDX_TO_CAT)
from dl_model import MotifGrammarCNN, SimpleMotifCNN, count_parameters


def train_epoch(model, loader, criterion, optimizer):
    """Train for one epoch. Returns average loss and accuracy."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for x, y in loader:
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion):
    """Evaluate model. Returns loss, accuracy, predictions, true labels."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for x, y in loader:
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)

            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.numpy())
            all_labels.extend(y.numpy())
            all_probs.extend(probs.numpy())

    n = len(all_labels)
    return (total_loss / n, np.sum(np.array(all_preds) == np.array(all_labels)) / n,
            np.array(all_preds), np.array(all_labels), np.array(all_probs))


def plot_training_curves(history, save_path):
    """Plot training and validation loss/accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history['train_loss']) + 1)

    ax1.plot(epochs, history['train_loss'], 'b-', label='Train')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training & Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history['train_acc'], 'b-', label='Train')
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training & Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Add chance level
    ax2.axhline(y=0.25, color='gray', linestyle='--', alpha=0.5, label='Chance (25%)')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved training curves: {save_path}")


def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plot confusion matrix with counts and percentages."""
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CATEGORIES, yticklabels=CATEGORIES, ax=ax1)
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('True')
    ax1.set_title('Confusion Matrix (counts)')

    # Percentages
    sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=CATEGORIES, yticklabels=CATEGORIES, ax=ax2)
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('True')
    ax2.set_title('Confusion Matrix (% of true class)')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved confusion matrix: {save_path}")


def plot_per_class_metrics(report_dict, save_path):
    """Bar chart of precision/recall/F1 per class."""
    cats = CATEGORIES
    precision = [report_dict[c]['precision'] for c in cats]
    recall = [report_dict[c]['recall'] for c in cats]
    f1 = [report_dict[c]['f1-score'] for c in cats]

    x = np.arange(len(cats))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width, precision, width, label='Precision', color='#1f77b4')
    ax.bar(x, recall, width, label='Recall', color='#ff7f0e')
    ax.bar(x + width, f1, width, label='F1-score', color='#2ca02c')

    ax.set_xlabel('Category')
    ax.set_ylabel('Score')
    ax.set_title('Per-Class Classification Metrics (Test Set)')
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved per-class metrics: {save_path}")


def train_model(seq_dir, output_dir, fig_dir, n_epochs=100, batch_size=64,
                lr=1e-3, patience=15, model_type='dilated'):
    """Full training pipeline.

    Args:
        seq_dir: path to directory with FP.fa, HP.fa, CB.fa, both_pioneer.fa
        output_dir: path to save model and results
        fig_dir: path to save figures
        n_epochs: max training epochs
        batch_size: batch size
        lr: learning rate
        patience: early stopping patience
        model_type: 'dilated' for MotifGrammarCNN, 'simple' for SimpleMotifCNN
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    # Load data
    print("=" * 60)
    print("STEP 1: Loading sequences")
    print("=" * 60)
    seqs, labels, headers, cats = load_all_sequences(seq_dir)

    print("\nSTEP 2: Creating train/val/test splits")
    splits = create_splits(seqs, labels, headers)

    print("\nSTEP 3: Creating DataLoaders")
    loaders = get_dataloaders(splits, batch_size=batch_size)

    # Class weights for imbalanced dataset
    class_weights = compute_class_weights(splits['train']['labels'])
    print(f"  Class weights: {class_weights.numpy()}")

    # Model
    print(f"\nSTEP 4: Building model ({model_type})")
    if model_type == 'dilated':
        model = MotifGrammarCNN(n_filters=128, dropout=0.2)
    else:
        model = SimpleMotifCNN(dropout=0.25)
    print(f"  Parameters: {count_parameters(model):,}")

    # Loss with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=7, verbose=True
    )

    # Training loop
    print(f"\nSTEP 5: Training (max {n_epochs} epochs, patience={patience})")
    print("-" * 60)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_loss = float('inf')
    best_epoch = 0
    epochs_no_improve = 0
    best_model_path = os.path.join(output_dir, 'best_model.pt')

    start_time = time.time()

    for epoch in range(1, n_epochs + 1):
        train_loss, train_acc = train_epoch(model, loaders['train'], criterion, optimizer)
        val_loss, val_acc, _, _, _ = evaluate(model, loaders['val'], criterion)

        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}: train_loss={train_loss:.4f} "
                  f"train_acc={train_acc:.3f} | val_loss={val_loss:.4f} "
                  f"val_acc={val_acc:.3f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'model_type': model_type,
                'n_classes': len(CATEGORIES),
                'categories': CATEGORIES,
            }, best_model_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\n  Early stopping at epoch {epoch} "
                      f"(best: epoch {best_epoch}, val_loss={best_val_loss:.4f})")
                break

    elapsed = time.time() - start_time
    print(f"\n  Training complete in {elapsed:.1f}s")
    print(f"  Best epoch: {best_epoch}, val_loss={best_val_loss:.4f}")

    # Load best model for evaluation
    print(f"\nSTEP 6: Evaluating best model on test set")
    checkpoint = torch.load(best_model_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_acc, test_preds, test_labels, test_probs = \
        evaluate(model, loaders['test'], criterion)

    # Balanced accuracy
    bal_acc = balanced_accuracy_score(test_labels, test_preds)

    # Multi-class AUC
    test_labels_bin = label_binarize(test_labels, classes=list(range(len(CATEGORIES))))
    try:
        auc = roc_auc_score(test_labels_bin, test_probs, multi_class='ovr', average='macro')
    except ValueError:
        auc = float('nan')

    print(f"  Test loss: {test_loss:.4f}")
    print(f"  Test accuracy: {test_acc:.3f}")
    print(f"  Balanced accuracy: {bal_acc:.3f}")
    print(f"  Macro AUC: {auc:.3f}")

    report = classification_report(test_labels, test_preds,
                                   target_names=CATEGORIES, output_dict=True)
    print("\n  Classification Report:")
    print(classification_report(test_labels, test_preds, target_names=CATEGORIES))

    # Save results
    print("STEP 7: Saving results and figures")

    results = {
        'model_type': model_type,
        'n_params': count_parameters(model),
        'best_epoch': best_epoch,
        'test_loss': float(test_loss),
        'test_accuracy': float(test_acc),
        'balanced_accuracy': float(bal_acc),
        'macro_auc': float(auc),
        'per_class': {cat: {k: float(v) for k, v in report[cat].items()}
                      for cat in CATEGORIES},
        'training_time_seconds': elapsed,
        'n_epochs_trained': len(history['train_loss']),
    }

    with open(os.path.join(output_dir, 'training_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Save splits info for reproducibility
    split_info = {
        name: {
            'n_samples': len(splits[name]['labels']),
            'headers': splits[name]['headers'],
        }
        for name in ['train', 'val', 'test']
    }
    with open(os.path.join(output_dir, 'split_info.json'), 'w') as f:
        json.dump(split_info, f, indent=2)

    # Save test predictions
    np.savez(os.path.join(output_dir, 'test_predictions.npz'),
             preds=test_preds, labels=test_labels, probs=test_probs,
             headers=np.array(splits['test']['headers']))

    # Plot figures
    plot_training_curves(history,
                         os.path.join(fig_dir, 'dl_training_curves.png'))
    plot_confusion_matrix(test_labels, test_preds,
                          os.path.join(fig_dir, 'dl_confusion_matrix.png'))
    plot_per_class_metrics(report,
                           os.path.join(fig_dir, 'dl_per_class_metrics.png'))

    print("\nDone!")
    return model, splits, history, results


if __name__ == '__main__':
    seq_dir = os.path.join(PROJECT_ROOT, 'results', 'sequences')
    output_dir = os.path.join(PROJECT_ROOT, 'results', 'deep_learning')
    fig_dir = os.path.join(PROJECT_ROOT, 'figures')

    model, splits, history, results = train_model(
        seq_dir=seq_dir,
        output_dir=output_dir,
        fig_dir=fig_dir,
        n_epochs=100,
        batch_size=64,
        lr=1e-3,
        patience=15,
        model_type='dilated',
    )
