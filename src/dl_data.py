"""
Data preparation for deep learning motif grammar classifier.
One-hot encodes 1001bp FASTA sequences, creates stratified train/val/test splits.
"""

import os
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader

# Categories and their labels
CATEGORIES = ['FP', 'HP', 'CB', 'both_pioneer']
CAT_TO_IDX = {c: i for i, c in enumerate(CATEGORIES)}
IDX_TO_CAT = {i: c for c, i in CAT_TO_IDX.items()}

# Nucleotide encoding
NUC_TO_IDX = {'A': 0, 'C': 1, 'G': 2, 'T': 3}


def parse_fasta(fasta_path):
    """Parse a FASTA file. Returns list of (header, sequence) tuples."""
    sequences = []
    header = None
    seq_lines = []

    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if header is not None:
                    sequences.append((header, ''.join(seq_lines).upper()))
                header = line[1:]
                seq_lines = []
            else:
                seq_lines.append(line)
        if header is not None:
            sequences.append((header, ''.join(seq_lines).upper()))

    return sequences


def one_hot_encode(sequence):
    """One-hot encode a DNA sequence. Shape: (4, L). N's become all-zero."""
    L = len(sequence)
    encoded = np.zeros((4, L), dtype=np.float32)
    for i, nuc in enumerate(sequence):
        if nuc in NUC_TO_IDX:
            encoded[NUC_TO_IDX[nuc], i] = 1.0
    return encoded


def load_all_sequences(seq_dir):
    """Load sequences from all four category FASTA files.

    Returns:
        sequences: list of one-hot encoded arrays (4, 1001)
        labels: numpy array of integer labels
        headers: list of FASTA headers
        categories: list of category strings
    """
    all_seqs = []
    all_labels = []
    all_headers = []
    all_cats = []

    for cat in CATEGORIES:
        fasta_path = os.path.join(seq_dir, f'{cat}.fa')
        if not os.path.exists(fasta_path):
            raise FileNotFoundError(f"Missing FASTA: {fasta_path}")

        seqs = parse_fasta(fasta_path)
        print(f"  {cat}: {len(seqs)} sequences")

        for header, seq in seqs:
            encoded = one_hot_encode(seq)
            all_seqs.append(encoded)
            all_labels.append(CAT_TO_IDX[cat])
            all_headers.append(header)
            all_cats.append(cat)

    labels = np.array(all_labels, dtype=np.int64)
    print(f"  Total: {len(all_seqs)} sequences")
    print(f"  Class distribution: {dict(Counter(all_cats))}")

    return all_seqs, labels, all_headers, all_cats


class SequenceDataset(Dataset):
    """PyTorch dataset for one-hot encoded DNA sequences."""

    def __init__(self, sequences, labels):
        self.sequences = sequences  # list of (4, L) arrays
        self.labels = labels  # numpy array of ints

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.sequences[idx])  # (4, 1001)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


def create_splits(sequences, labels, headers, test_size=0.15, val_size=0.15,
                  random_state=42):
    """Stratified train/val/test split.

    Returns dict with 'train', 'val', 'test' keys, each containing
    (sequences, labels, headers).
    """
    seqs = np.array(sequences, dtype=object)
    hdrs = np.array(headers, dtype=object)

    # First split: train+val vs test
    idx = np.arange(len(labels))
    idx_trainval, idx_test = train_test_split(
        idx, test_size=test_size, stratify=labels, random_state=random_state
    )

    # Second split: train vs val (from trainval)
    val_frac = val_size / (1 - test_size)
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=val_frac, stratify=labels[idx_trainval],
        random_state=random_state
    )

    splits = {}
    for name, idxs in [('train', idx_train), ('val', idx_val), ('test', idx_test)]:
        splits[name] = {
            'sequences': [sequences[i] for i in idxs],
            'labels': labels[idxs],
            'headers': [headers[i] for i in idxs],
        }
        counts = Counter(labels[idxs])
        print(f"  {name}: {len(idxs)} sequences — "
              f"{', '.join(f'{IDX_TO_CAT[k]}={v}' for k, v in sorted(counts.items()))}")

    return splits


def get_dataloaders(splits, batch_size=64):
    """Create DataLoaders for train/val/test splits."""
    loaders = {}
    for name in ['train', 'val', 'test']:
        ds = SequenceDataset(splits[name]['sequences'], splits[name]['labels'])
        shuffle = (name == 'train')
        loaders[name] = DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                                   num_workers=0, drop_last=False)
    return loaders


def compute_class_weights(labels):
    """Compute inverse-frequency class weights for imbalanced classes."""
    counts = Counter(labels.tolist())
    total = len(labels)
    n_classes = len(counts)
    weights = torch.zeros(n_classes)
    for cls_idx, count in counts.items():
        weights[cls_idx] = total / (n_classes * count)
    return weights


if __name__ == '__main__':
    # Quick test
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    seq_dir = os.path.join(base, 'results', 'sequences')

    print("Loading sequences...")
    seqs, labels, headers, cats = load_all_sequences(seq_dir)

    print("\nCreating splits...")
    splits = create_splits(seqs, labels, headers)

    print("\nClass weights:", compute_class_weights(labels))

    print("\nSequence shape check:", seqs[0].shape)
    print("Label range:", labels.min(), "-", labels.max())
