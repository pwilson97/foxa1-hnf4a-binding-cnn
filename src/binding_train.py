"""
Train the multi-task binding CNN and evaluate.
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_recall_curve, roc_curve)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from binding_data import (assign_binding_labels, extract_sequences,
                          create_chrom_splits, get_dataloaders,
                          compute_pos_weights)
from binding_model import BindingCNN, count_parameters

TF_NAMES = ['FOXA1', 'HNF4A']


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    total = 0
    for x, y in loader:
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        total += x.size(0)
    return total_loss / total


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    total = 0
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            total += x.size(0)
            all_logits.append(logits.numpy())
            all_labels.append(y.numpy())
    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels)
    probs = 1 / (1 + np.exp(-logits))  # sigmoid
    return total_loss / total, probs, labels


def compute_metrics(probs, labels):
    """Compute per-task AUROC and AUPRC."""
    metrics = {}
    for i, tf in enumerate(TF_NAMES):
        try:
            auroc = roc_auc_score(labels[:, i], probs[:, i])
        except ValueError:
            auroc = float('nan')
        try:
            auprc = average_precision_score(labels[:, i], probs[:, i])
        except ValueError:
            auprc = float('nan')
        metrics[tf] = {'auroc': auroc, 'auprc': auprc}
    return metrics


def plot_training_curves(history, save_path):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4))

    epochs = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('BCE Loss')
    ax1.set_title('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    for i, tf in enumerate(TF_NAMES):
        ax2.plot(epochs, [m[tf]['auroc'] for m in history['val_metrics']],
                 label=f'{tf} (val)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('AUROC')
    ax2.set_title('Validation AUROC')
    ax2.legend()
    ax2.set_ylim(0.5, 1.0)
    ax2.grid(True, alpha=0.3)

    for i, tf in enumerate(TF_NAMES):
        ax3.plot(epochs, [m[tf]['auprc'] for m in history['val_metrics']],
                 label=f'{tf} (val)')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('AUPRC')
    ax3.set_title('Validation AUPRC')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_roc_pr_curves(probs, labels, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    colors = ['#1f77b4', '#ff7f0e']

    for i, (tf, color) in enumerate(zip(TF_NAMES, colors)):
        fpr, tpr, _ = roc_curve(labels[:, i], probs[:, i])
        auroc = roc_auc_score(labels[:, i], probs[:, i])
        ax1.plot(fpr, tpr, color=color, label=f'{tf} (AUROC={auroc:.3f})')

        prec, rec, _ = precision_recall_curve(labels[:, i], probs[:, i])
        auprc = average_precision_score(labels[:, i], probs[:, i])
        ax2.plot(rec, prec, color=color, label=f'{tf} (AUPRC={auprc:.3f})')

    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax1.set_xlabel('FPR')
    ax1.set_ylabel('TPR')
    ax1.set_title('ROC Curves (Test Set)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curves (Test Set)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def train_binding_model(peaks_dir, classified_dir, genome_fasta,
                        output_dir, fig_dir,
                        n_epochs=60, batch_size=128, lr=1e-3, patience=10):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    # --- Data prep ---
    print("=" * 60)
    print("STEP 1: Building labeled dataset")
    print("=" * 60)
    regions, labels, sources = assign_binding_labels(peaks_dir, classified_dir)

    print("\nSTEP 2: Extracting sequences from genome")
    sequences, valid = extract_sequences(regions, genome_fasta)

    # Filter to valid sequences
    valid_idx = np.where(valid)[0]
    regions = [regions[i] for i in valid_idx]
    labels = labels[valid_idx]
    sequences = [sequences[i] for i in valid_idx]
    sources = [sources[i] for i in valid_idx]
    print(f"  {len(regions)} valid sequences after filtering")

    print("\nSTEP 3: Chromosome-based splits")
    splits = create_chrom_splits(regions, labels, sequences)

    print("\nSTEP 4: DataLoaders")
    loaders = get_dataloaders(splits, batch_size=batch_size)
    pos_weights = compute_pos_weights(splits['train']['labels'])
    print(f"  Pos weights: FOXA1={pos_weights[0]:.2f}, HNF4A={pos_weights[1]:.2f}")

    # --- Model ---
    print(f"\nSTEP 5: Building model")
    model = BindingCNN(n_filters=64, dropout=0.25)
    print(f"  Parameters: {count_parameters(model):,}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5)

    # --- Training ---
    print(f"\nSTEP 6: Training (max {n_epochs} epochs, patience={patience})")
    print("-" * 60)

    history = {'train_loss': [], 'val_loss': [], 'val_metrics': []}
    best_val_loss = float('inf')
    best_epoch = 0
    epochs_no_improve = 0
    model_path = os.path.join(output_dir, 'binding_model.pt')

    t0 = time.time()
    for epoch in range(1, n_epochs + 1):
        train_loss = train_epoch(model, loaders['train'], criterion, optimizer)
        val_loss, val_probs, val_labels = evaluate(model, loaders['val'], criterion)
        val_metrics = compute_metrics(val_probs, val_labels)

        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_metrics'].append(val_metrics)

        if epoch % 5 == 0 or epoch == 1:
            f_auc = val_metrics['FOXA1']['auroc']
            h_auc = val_metrics['HNF4A']['auroc']
            print(f"  Epoch {epoch:3d}: loss={train_loss:.4f}/{val_loss:.4f} "
                  f"FOXA1_auroc={f_auc:.3f} HNF4A_auroc={h_auc:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'val_metrics': val_metrics,
            }, model_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\n  Early stopping at epoch {epoch} (best: {best_epoch})")
                break

    elapsed = time.time() - t0
    print(f"  Training: {elapsed:.1f}s")

    # --- Evaluate on test ---
    print(f"\nSTEP 7: Test set evaluation")
    checkpoint = torch.load(model_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_probs, test_labels = evaluate(model, loaders['test'], criterion)
    test_metrics = compute_metrics(test_probs, test_labels)

    for tf in TF_NAMES:
        print(f"  {tf}: AUROC={test_metrics[tf]['auroc']:.3f}, "
              f"AUPRC={test_metrics[tf]['auprc']:.3f}")

    # --- Save ---
    print("\nSTEP 8: Saving results and figures")

    results = {
        'best_epoch': best_epoch,
        'test_loss': float(test_loss),
        'test_metrics': {tf: {k: float(v) for k, v in m.items()}
                         for tf, m in test_metrics.items()},
        'n_train': len(splits['train']['labels']),
        'n_val': len(splits['val']['labels']),
        'n_test': len(splits['test']['labels']),
        'training_time_seconds': elapsed,
    }
    with open(os.path.join(output_dir, 'binding_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    np.savez(os.path.join(output_dir, 'binding_test_predictions.npz'),
             probs=test_probs, labels=test_labels)

    plot_training_curves(history, os.path.join(fig_dir, 'binding_training_curves.png'))
    plot_roc_pr_curves(test_probs, test_labels,
                       os.path.join(fig_dir, 'binding_roc_pr.png'))

    print("\nTraining complete!")
    return model, splits


if __name__ == '__main__':
    peaks_dir = os.path.join(PROJECT_ROOT, 'results', 'peaks')
    classified_dir = os.path.join(PROJECT_ROOT, 'results', 'classified')
    genome_fasta = os.path.join(PROJECT_ROOT, 'data', 'genome', 'hg19.fa')
    output_dir = os.path.join(PROJECT_ROOT, 'results', 'deep_learning', 'binding')
    fig_dir = os.path.join(PROJECT_ROOT, 'figures')

    train_binding_model(peaks_dir, classified_dir, genome_fasta,
                        output_dir, fig_dir)
