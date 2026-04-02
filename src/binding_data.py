"""
Data preparation for binary TF binding prediction.

Constructs multi-label dataset:
  - FOXA1 bound: yes/no
  - HNF4A bound: yes/no

Labels come from overlapping CUT&Tag consensus peaks across conditions.
Negatives from pre-computed unbound control regions.
"""

import os
import numpy as np
import pysam
from collections import Counter
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader

# Standard chromosomes (exclude random, Un, hap)
STANDARD_CHROMS = {f'chr{i}' for i in range(1, 23)} | {'chrX', 'chrY'}

# Nucleotide encoding
NUC_TO_IDX = {'A': 0, 'C': 1, 'G': 2, 'T': 3}


def parse_narrowpeak(path):
    """Parse narrowPeak file. Returns list of (chrom, start, end, summit_offset)."""
    peaks = []
    with open(path) as f:
        for line in f:
            fields = line.strip().split('\t')
            chrom = fields[0]
            if chrom not in STANDARD_CHROMS:
                continue
            start = int(fields[1])
            end = int(fields[2])
            summit_offset = int(fields[9]) if len(fields) > 9 else (end - start) // 2
            peaks.append((chrom, start, end, summit_offset))
    return peaks


def parse_bed(path):
    """Parse BED file. Returns list of (chrom, start, end)."""
    regions = []
    with open(path) as f:
        for line in f:
            fields = line.strip().split('\t')
            chrom = fields[0]
            if chrom not in STANDARD_CHROMS:
                continue
            regions.append((chrom, int(fields[1]), int(fields[2])))
    return regions


def peaks_to_summit_regions(peaks, window=500):
    """Convert peaks to summit-centered regions (summit ± window).

    Returns list of (chrom, start, end) where region = summit ± window.
    """
    regions = []
    for chrom, start, end, summit_offset in peaks:
        summit = start + summit_offset
        reg_start = max(0, summit - window)
        reg_end = summit + window + 1  # 1001bp total
        regions.append((chrom, reg_start, reg_end))
    return regions


def build_region_set(regions):
    """Build a set of (chrom, midpoint // resolution) for fast overlap checking."""
    # Use 200bp bins for overlap detection
    resolution = 200
    region_set = set()
    for chrom, start, end in regions:
        mid = (start + end) // 2
        region_set.add((chrom, mid // resolution))
        # Also add neighbors to handle edge cases
        region_set.add((chrom, (mid - resolution) // resolution))
        region_set.add((chrom, (mid + resolution) // resolution))
    return region_set, resolution


def check_overlap(chrom, start, end, region_set, resolution):
    """Check if a region overlaps with any region in the set."""
    mid = (start + end) // 2
    return (chrom, mid // resolution) in region_set


def assign_binding_labels(peaks_dir, classified_dir, window=500):
    """Build the labeled dataset from peak files.

    Strategy:
    - FOXA1 binding = peak in FOXA1_alone OR FOXA1_HNF4A (FOXA1 ab) condition
    - HNF4A binding = peak in HNF4A_alone OR FOXA1_HNF4A (HNF4A ab) condition
    - Use summit-centered regions (± window)
    - Merge overlapping regions, assign binary labels by overlap

    Returns:
        regions: list of (chrom, start, end)
        labels: (N, 2) array — [foxa1_bound, hnf4a_bound]
        sources: list of source strings
    """
    print("  Loading peak files...")

    # Load all peak sets
    foxa1_alone = parse_narrowpeak(
        os.path.join(peaks_dir, 'FOXA1_alone_FOXA1ab.consensus.narrowPeak'))
    hnf4a_alone = parse_narrowpeak(
        os.path.join(peaks_dir, 'HNF4A_alone_HNF4Aab.consensus.narrowPeak'))
    dual_foxa1 = parse_narrowpeak(
        os.path.join(peaks_dir, 'FOXA1_HNF4A_FOXA1ab.consensus.narrowPeak'))
    dual_hnf4a = parse_narrowpeak(
        os.path.join(peaks_dir, 'FOXA1_HNF4A_HNF4Aab.consensus.narrowPeak'))

    print(f"    FOXA1-alone: {len(foxa1_alone)} peaks")
    print(f"    HNF4A-alone: {len(hnf4a_alone)} peaks")
    print(f"    Dual (FOXA1 ab): {len(dual_foxa1)} peaks")
    print(f"    Dual (HNF4A ab): {len(dual_hnf4a)} peaks")

    # Convert to summit-centered regions
    foxa1_regions = peaks_to_summit_regions(foxa1_alone, window)
    hnf4a_regions = peaks_to_summit_regions(hnf4a_alone, window)
    dual_f_regions = peaks_to_summit_regions(dual_foxa1, window)
    dual_h_regions = peaks_to_summit_regions(dual_hnf4a, window)

    # All FOXA1-bound regions (union of foxa1-alone + dual-foxa1ab)
    all_foxa1 = foxa1_regions + dual_f_regions
    # All HNF4A-bound regions (union of hnf4a-alone + dual-hnf4aab)
    all_hnf4a = hnf4a_regions + dual_h_regions

    print(f"    All FOXA1-bound: {len(all_foxa1)} regions")
    print(f"    All HNF4A-bound: {len(all_hnf4a)} regions")

    # Build overlap lookup sets
    foxa1_set, res = build_region_set(all_foxa1)
    hnf4a_set, _ = build_region_set(all_hnf4a)

    # Collect unique regions from all sources + assign labels
    # Use a dict keyed by (chrom, rounded_midpoint) to deduplicate
    seen = {}

    def add_regions(regions, source_name):
        for chrom, start, end in regions:
            mid = (start + end) // 2
            key = (chrom, mid // 500)  # 500bp dedup resolution
            if key not in seen:
                foxa1_label = 1 if check_overlap(chrom, start, end, foxa1_set, res) else 0
                hnf4a_label = 1 if check_overlap(chrom, start, end, hnf4a_set, res) else 0
                seen[key] = (chrom, start, end, foxa1_label, hnf4a_label, source_name)

    add_regions(foxa1_regions, 'foxa1_alone')
    add_regions(hnf4a_regions, 'hnf4a_alone')
    add_regions(dual_f_regions, 'dual_foxa1')
    add_regions(dual_h_regions, 'dual_hnf4a')

    # Add unbound controls
    unbound_path = os.path.join(classified_dir, 'unbound_control_sites.bed')
    if os.path.exists(unbound_path):
        unbound = parse_bed(unbound_path)
        unbound_regions = []
        for chrom, start, end in unbound:
            mid = (start + end) // 2
            reg_start = max(0, mid - window)
            reg_end = mid + window + 1
            unbound_regions.append((chrom, reg_start, reg_end))
        add_regions(unbound_regions, 'unbound')
        print(f"    Unbound controls: {len(unbound)} regions")

    # Compile
    all_data = list(seen.values())
    regions = [(d[0], d[1], d[2]) for d in all_data]
    labels = np.array([[d[3], d[4]] for d in all_data], dtype=np.float32)
    sources = [d[5] for d in all_data]

    # Summary
    n = len(labels)
    n_f = labels[:, 0].sum()
    n_h = labels[:, 1].sum()
    n_both = (labels.sum(axis=1) == 2).sum()
    n_neither = (labels.sum(axis=1) == 0).sum()
    n_f_only = ((labels[:, 0] == 1) & (labels[:, 1] == 0)).sum()
    n_h_only = ((labels[:, 0] == 0) & (labels[:, 1] == 1)).sum()

    print(f"\n  Label summary ({n} total regions):")
    print(f"    FOXA1+/HNF4A-: {int(n_f_only)}")
    print(f"    FOXA1-/HNF4A+: {int(n_h_only)}")
    print(f"    FOXA1+/HNF4A+: {int(n_both)}")
    print(f"    FOXA1-/HNF4A-: {int(n_neither)}")

    return regions, labels, sources


def extract_sequences(regions, genome_fasta, seq_len=1001):
    """Extract DNA sequences for regions from genome FASTA.

    Returns:
        sequences: list of one-hot encoded (4, seq_len) arrays
        valid_mask: boolean array (True = successfully extracted)
    """
    fasta = pysam.FastaFile(genome_fasta)
    sequences = []
    valid_mask = []

    for i, (chrom, start, end) in enumerate(regions):
        try:
            seq = fasta.fetch(chrom, start, end).upper()
            if len(seq) != seq_len:
                # Pad or trim to exact length
                if len(seq) < seq_len:
                    seq = seq + 'N' * (seq_len - len(seq))
                else:
                    seq = seq[:seq_len]

            encoded = one_hot_encode(seq)
            sequences.append(encoded)
            valid_mask.append(True)
        except (ValueError, KeyError):
            sequences.append(np.zeros((4, seq_len), dtype=np.float32))
            valid_mask.append(False)

        if (i + 1) % 10000 == 0:
            print(f"    Extracted {i + 1}/{len(regions)} sequences")

    fasta.close()
    print(f"    Extracted {sum(valid_mask)}/{len(regions)} valid sequences")
    return sequences, np.array(valid_mask)


def one_hot_encode(sequence):
    """One-hot encode DNA sequence. Shape: (4, L)."""
    L = len(sequence)
    encoded = np.zeros((4, L), dtype=np.float32)
    for i, nuc in enumerate(sequence):
        if nuc in NUC_TO_IDX:
            encoded[NUC_TO_IDX[nuc], i] = 1.0
    return encoded


class BindingDataset(Dataset):
    """PyTorch dataset for multi-label binding prediction."""

    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels  # (N, 2) float32

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.sequences[idx])  # (4, 1001)
        y = torch.from_numpy(self.labels[idx])      # (2,)
        return x, y


def create_chrom_splits(regions, labels, sequences,
                        test_chroms=('chr1', 'chr8', 'chr9'),
                        val_chroms=('chr2', 'chr3')):
    """Chromosome-based train/val/test split (avoids data leakage).

    BPNet used chr1, chr8, chr9 for test — we follow the same convention.
    """
    chroms = np.array([r[0] for r in regions])

    test_mask = np.isin(chroms, test_chroms)
    val_mask = np.isin(chroms, val_chroms)
    train_mask = ~test_mask & ~val_mask

    splits = {}
    for name, mask in [('train', train_mask), ('val', val_mask), ('test', test_mask)]:
        idxs = np.where(mask)[0]
        splits[name] = {
            'sequences': [sequences[i] for i in idxs],
            'labels': labels[idxs],
            'regions': [regions[i] for i in idxs],
        }
        n = len(idxs)
        f_pos = labels[idxs, 0].sum()
        h_pos = labels[idxs, 1].sum()
        print(f"  {name}: {n} regions — FOXA1+: {int(f_pos)} ({f_pos/n:.1%}), "
              f"HNF4A+: {int(h_pos)} ({h_pos/n:.1%})")

    return splits


def get_dataloaders(splits, batch_size=64):
    """Create DataLoaders for train/val/test."""
    loaders = {}
    for name in ['train', 'val', 'test']:
        ds = BindingDataset(splits[name]['sequences'], splits[name]['labels'])
        shuffle = (name == 'train')
        loaders[name] = DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                                   num_workers=0, drop_last=False)
    return loaders


def compute_pos_weights(labels):
    """Compute positive class weights for BCEWithLogitsLoss.

    pos_weight = n_neg / n_pos for each task.
    """
    weights = []
    for col in range(labels.shape[1]):
        n_pos = labels[:, col].sum()
        n_neg = len(labels) - n_pos
        weights.append(n_neg / max(n_pos, 1))
    return torch.tensor(weights, dtype=torch.float32)


if __name__ == '__main__':
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    peaks_dir = os.path.join(base, 'results', 'peaks')
    classified_dir = os.path.join(base, 'results', 'classified')
    genome_fasta = os.path.join(base, 'data', 'genome', 'hg19.fa')

    print("Building labels...")
    regions, labels, sources = assign_binding_labels(peaks_dir, classified_dir)

    print("\nExtracting sequences...")
    sequences, valid = extract_sequences(regions, genome_fasta)

    print("\nChromosome splits...")
    splits = create_chrom_splits(regions, labels, sequences)

    print("\nPos weights:", compute_pos_weights(splits['train']['labels']))
