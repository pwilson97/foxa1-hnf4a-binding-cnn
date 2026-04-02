"""
In silico mutagenesis: implant FOXA1 and HNF4A motifs at controlled spacings
into neutral backgrounds and measure how predicted class probabilities change.

Tests whether the model learned spacing-dependent cooperativity rules.
"""

import os
import sys
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from dl_data import CATEGORIES, CAT_TO_IDX, one_hot_encode
from dl_model import MotifGrammarCNN, SimpleMotifCNN

# Consensus motif sequences (JASPAR)
# FOXA1 (MA0148.1): TGTTTAC core
FOXA1_MOTIF = "AATGTAAACAA"  # 11bp FOXA1 consensus (forkhead)
# HNF4A (MA0114.2): CAAAGTCCA direct repeat
HNF4A_MOTIF = "CAAAGTCCAAAGTTCA"  # 16bp HNF4A consensus (DR1-like)

# Orientation variants
ORIENTATIONS = {
    'tandem_forward': ('+', '+'),       # both on + strand
    'convergent': ('+', '-'),           # facing each other (<<>>)
    'divergent': ('-', '+'),            # facing away (<< >>)
    'tandem_reverse': ('-', '-'),       # both on - strand
}


def rev_comp(seq):
    """Reverse complement a DNA sequence."""
    comp = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
    return ''.join(comp.get(b, 'N') for b in reversed(seq))


def generate_neutral_background(length=1001, gc_content=0.42, seed=None):
    """Generate a neutral background sequence with specified GC content.

    Uses shuffled dinucleotides from real genomic sequence to maintain
    dinucleotide frequencies, but simplified here as random with GC bias.
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()

    probs = [(1 - gc_content) / 2,  # A
             gc_content / 2,         # C
             gc_content / 2,         # G
             (1 - gc_content) / 2]   # T

    nucs = ['A', 'C', 'G', 'T']
    return ''.join(rng.choice(nucs, size=length, p=probs))


def implant_motif_pair(background, motif1, motif2, center, spacing,
                       orientation='tandem_forward'):
    """Implant two motifs at specified spacing into background sequence.

    Args:
        background: background DNA string (1001bp)
        motif1: FOXA1 motif string
        motif2: HNF4A motif string
        center: center position for the pair (default: middle of sequence)
        spacing: distance between motif centers (bp)
        orientation: one of 'tandem_forward', 'convergent', 'divergent', 'tandem_reverse'

    Returns:
        modified sequence string
    """
    seq = list(background)
    L = len(seq)

    strand1, strand2 = ORIENTATIONS[orientation]

    # Position motif1 (FOXA1) at center - spacing/2
    # Position motif2 (HNF4A) at center + spacing/2
    m1 = motif1 if strand1 == '+' else rev_comp(motif1)
    m2 = motif2 if strand2 == '+' else rev_comp(motif2)

    pos1 = center - spacing // 2 - len(m1) // 2
    pos2 = center + spacing // 2 - len(m2) // 2

    # Bounds check
    if pos1 < 0 or pos1 + len(m1) > L or pos2 < 0 or pos2 + len(m2) > L:
        return None
    if pos1 + len(m1) > pos2:  # motifs overlap
        return None

    # Implant
    for i, base in enumerate(m1):
        seq[pos1 + i] = base
    for i, base in enumerate(m2):
        seq[pos2 + i] = base

    return ''.join(seq)


def implant_single_motif(background, motif, center, strand='+'):
    """Implant a single motif at specified position."""
    seq = list(background)
    m = motif if strand == '+' else rev_comp(motif)
    pos = center - len(m) // 2

    if pos < 0 or pos + len(m) > len(seq):
        return None

    for i, base in enumerate(m):
        seq[pos + i] = base
    return ''.join(seq)


def predict_batch(model, sequences):
    """Get model predictions for a batch of DNA sequence strings.

    Returns:
        probs: (N, 4) probability array
    """
    model.eval()
    encoded = [one_hot_encode(s) for s in sequences]
    x = torch.stack([torch.from_numpy(e) for e in encoded])

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).numpy()

    return probs


def spacing_scan(model, spacings=None, n_backgrounds=50, seed=42):
    """Scan FOXA1-HNF4A spacing and measure class probability changes.

    For each spacing, implant the motif pair into multiple random backgrounds
    and average the predicted probabilities.

    Returns:
        results dict with spacings, mean probs, std probs for each orientation
    """
    if spacings is None:
        spacings = list(range(10, 301, 5))

    center = 500  # middle of 1001bp sequence

    results = {}

    for orient_name in ORIENTATIONS:
        print(f"  Scanning spacing for orientation: {orient_name}")
        all_probs = []

        for sp in spacings:
            bg_probs = []
            for bg_idx in range(n_backgrounds):
                bg = generate_neutral_background(seed=seed + bg_idx)
                modified = implant_motif_pair(bg, FOXA1_MOTIF, HNF4A_MOTIF,
                                              center, sp, orient_name)
                if modified is not None:
                    bg_probs.append(modified)

            if bg_probs:
                probs = predict_batch(model, bg_probs)
                all_probs.append(probs.mean(axis=0))
            else:
                all_probs.append(np.full(4, np.nan))

        results[orient_name] = np.array(all_probs)

    return spacings, results


def motif_count_scan(model, motif_type='HNF4A', counts=None,
                     n_backgrounds=50, seed=42):
    """Scan the effect of varying motif counts on predictions.

    Implant 1 to N copies of a motif (distributed across the sequence)
    and measure how class probabilities change.
    """
    if counts is None:
        counts = list(range(1, 8))

    center = 500
    L = 1001

    results = []

    for n_motifs in counts:
        bg_probs = []
        motif = HNF4A_MOTIF if motif_type == 'HNF4A' else FOXA1_MOTIF

        for bg_idx in range(n_backgrounds):
            bg = generate_neutral_background(seed=seed + bg_idx)

            # Distribute motifs evenly across the sequence
            positions = np.linspace(100, 900, n_motifs + 2)[1:-1].astype(int)

            seq = bg
            valid = True
            for pos in positions[:n_motifs]:
                seq = implant_single_motif(seq, motif, int(pos))
                if seq is None:
                    valid = False
                    break

            if valid:
                bg_probs.append(seq)

        if bg_probs:
            probs = predict_batch(model, bg_probs)
            results.append(probs.mean(axis=0))
        else:
            results.append(np.full(4, np.nan))

    return counts, np.array(results)


def combined_scan(model, n_foxa1_range=None, n_hnf4a_range=None,
                  n_backgrounds=30, seed=42):
    """Scan combinations of FOXA1 and HNF4A motif counts.

    Creates a grid of (n_FOXA1, n_HNF4A) and measures CB probability.
    """
    if n_foxa1_range is None:
        n_foxa1_range = list(range(0, 6))
    if n_hnf4a_range is None:
        n_hnf4a_range = list(range(0, 7))

    cb_probs = np.zeros((len(n_foxa1_range), len(n_hnf4a_range)))

    for i, n_f in enumerate(n_foxa1_range):
        for j, n_h in enumerate(n_hnf4a_range):
            seqs = []
            for bg_idx in range(n_backgrounds):
                bg = generate_neutral_background(seed=seed + bg_idx)
                seq = bg

                # Place FOXA1 motifs in first half
                if n_f > 0:
                    f_positions = np.linspace(100, 450, n_f + 2)[1:-1].astype(int)
                    for pos in f_positions[:n_f]:
                        seq = implant_single_motif(seq, FOXA1_MOTIF, int(pos))
                        if seq is None:
                            break

                # Place HNF4A motifs in second half
                if seq is not None and n_h > 0:
                    h_positions = np.linspace(550, 900, n_h + 2)[1:-1].astype(int)
                    for pos in h_positions[:n_h]:
                        seq = implant_single_motif(seq, HNF4A_MOTIF, int(pos))
                        if seq is None:
                            break

                if seq is not None:
                    seqs.append(seq)

            if seqs:
                probs = predict_batch(model, seqs)
                cb_probs[i, j] = probs[:, CAT_TO_IDX['CB']].mean()

    return n_foxa1_range, n_hnf4a_range, cb_probs


def plot_spacing_scan(spacings, results, save_path):
    """Plot class probabilities as a function of motif spacing."""
    palette = {'FP': '#1f77b4', 'HP': '#ff7f0e', 'CB': '#2ca02c',
               'both_pioneer': '#d62728'}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for ax, orient_name in zip(axes.flat, ORIENTATIONS):
        probs = results[orient_name]

        for cat_idx, cat in enumerate(CATEGORIES):
            ax.plot(spacings, probs[:, cat_idx], label=cat,
                    color=palette[cat], linewidth=1.5)

        ax.set_xlabel('FOXA1-HNF4A Spacing (bp)')
        ax.set_ylabel('Predicted Probability')
        ax.set_title(f'Orientation: {orient_name}')
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.axvline(x=148, color='green', linestyle='--', alpha=0.4,
                   label='CB median (148bp)')

    plt.suptitle('In Silico Spacing Scan: Class Probabilities vs Motif Distance',
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved spacing scan: {save_path}")


def plot_motif_count_scan(counts, results_hnf4a, results_foxa1, save_path):
    """Plot class probabilities as a function of motif count."""
    palette = {'FP': '#1f77b4', 'HP': '#ff7f0e', 'CB': '#2ca02c',
               'both_pioneer': '#d62728'}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for cat_idx, cat in enumerate(CATEGORIES):
        ax1.plot(counts, results_hnf4a[:, cat_idx], 'o-',
                 label=cat, color=palette[cat])
    ax1.set_xlabel('Number of HNF4A Motifs')
    ax1.set_ylabel('Predicted Probability')
    ax1.set_title('Effect of HNF4A Motif Count')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    for cat_idx, cat in enumerate(CATEGORIES):
        ax2.plot(counts, results_foxa1[:, cat_idx], 'o-',
                 label=cat, color=palette[cat])
    ax2.set_xlabel('Number of FOXA1 Motifs')
    ax2.set_ylabel('Predicted Probability')
    ax2.set_title('Effect of FOXA1 Motif Count')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.suptitle('In Silico Motif Count Scan', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved motif count scan: {save_path}")


def plot_combined_scan(n_foxa1, n_hnf4a, cb_probs, save_path):
    """Heatmap of CB probability as function of both motif counts."""
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(cb_probs, annot=True, fmt='.2f', cmap='Greens',
                xticklabels=n_hnf4a, yticklabels=n_foxa1, ax=ax,
                vmin=0, vmax=1)
    ax.set_xlabel('Number of HNF4A Motifs')
    ax.set_ylabel('Number of FOXA1 Motifs')
    ax.set_title('Predicted CB Probability\n(FOXA1 × HNF4A count grid)')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved combined scan: {save_path}")


def run_mutagenesis(model_path, output_dir, fig_dir):
    """Full in silico mutagenesis pipeline."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    # Load model
    print("=" * 60)
    print("IN SILICO MUTAGENESIS")
    print("=" * 60)
    print("\n1. Loading model...")
    checkpoint = torch.load(model_path, weights_only=False, map_location='cpu')
    model_type = checkpoint.get('model_type', 'dilated')

    if model_type == 'dilated':
        model = MotifGrammarCNN()
    else:
        model = SimpleMotifCNN()

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 1. Spacing scan
    print("\n2. Running spacing scan...")
    spacings, spacing_results = spacing_scan(
        model, spacings=list(range(10, 301, 5)),
        n_backgrounds=50, seed=42
    )
    plot_spacing_scan(spacings, spacing_results,
                      os.path.join(fig_dir, 'dl_spacing_scan.png'))

    # Save spacing data
    np.savez(os.path.join(output_dir, 'spacing_scan.npz'),
             spacings=spacings,
             **{f'{k}_probs': v for k, v in spacing_results.items()})

    # 2. Motif count scan
    print("\n3. Running motif count scan...")
    counts = list(range(1, 8))
    counts_h, results_h = motif_count_scan(model, 'HNF4A', counts,
                                           n_backgrounds=50)
    counts_f, results_f = motif_count_scan(model, 'FOXA1', counts,
                                           n_backgrounds=50)
    plot_motif_count_scan(counts, results_h, results_f,
                          os.path.join(fig_dir, 'dl_motif_count_scan.png'))

    np.savez(os.path.join(output_dir, 'motif_count_scan.npz'),
             counts=counts,
             hnf4a_probs=results_h,
             foxa1_probs=results_f)

    # 3. Combined FOXA1 × HNF4A grid
    print("\n4. Running combined FOXA1 × HNF4A scan...")
    n_f, n_h, cb_grid = combined_scan(model, n_backgrounds=30)
    plot_combined_scan(n_f, n_h, cb_grid,
                       os.path.join(fig_dir, 'dl_combined_motif_grid.png'))

    np.savez(os.path.join(output_dir, 'combined_scan.npz'),
             n_foxa1=n_f, n_hnf4a=n_h, cb_probs=cb_grid)

    # 4. Baseline: what does the model predict for pure backgrounds?
    print("\n5. Baseline predictions (no motifs)...")
    bg_seqs = [generate_neutral_background(seed=i) for i in range(100)]
    bg_probs = predict_batch(model, bg_seqs)
    print(f"   Neutral background mean probs: "
          f"{dict(zip(CATEGORIES, bg_probs.mean(axis=0).round(3)))}")

    summary = {
        'baseline_probs': dict(zip(CATEGORIES, bg_probs.mean(axis=0).tolist())),
        'foxa1_motif': FOXA1_MOTIF,
        'hnf4a_motif': HNF4A_MOTIF,
        'n_backgrounds': 50,
        'spacing_range': [int(spacings[0]), int(spacings[-1])],
    }
    with open(os.path.join(output_dir, 'mutagenesis_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print("\nIn silico mutagenesis complete!")


if __name__ == '__main__':
    model_path = os.path.join(PROJECT_ROOT, 'results', 'deep_learning', 'best_model.pt')
    output_dir = os.path.join(PROJECT_ROOT, 'results', 'deep_learning')
    fig_dir = os.path.join(PROJECT_ROOT, 'figures')

    run_mutagenesis(model_path, output_dir, fig_dir)
