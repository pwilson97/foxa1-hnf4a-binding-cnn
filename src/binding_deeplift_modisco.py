"""
DeepLIFT + TF-MoDISco on the binding model.

Computes per-base importance scores for each output head separately:
  - DeepLIFT w.r.t. FOXA1 head → what sequence features drive FOXA1 binding?
  - DeepLIFT w.r.t. HNF4A head → what sequence features drive HNF4A binding?

Then runs TF-MoDISco to discover motif patterns from each head's attributions.

The grammar signal: if the FOXA1 head attributes importance to HNF4A motif
positions (or vice versa), that's evidence the model learned cross-TF grammar.
"""

import os
import sys
import json
import time
import numpy as np
import torch
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from captum.attr import DeepLift

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from binding_data import one_hot_encode
from binding_model import BindingCNN

TF_NAMES = ['FOXA1', 'HNF4A']


# ── Data loading ────────────────────────────────────────────────

def load_fasta(path):
    """Load FASTA. Returns list of (header, sequence_str)."""
    entries = []
    header = None
    seq_lines = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if header is not None:
                    entries.append((header, ''.join(seq_lines)))
                header = line[1:]
                seq_lines = []
            else:
                seq_lines.append(line)
    if header is not None:
        entries.append((header, ''.join(seq_lines)))
    return entries


def load_cobound_sequences(fasta_dir, max_seqs=2000):
    """Load co-bound sequences (where both motifs should be present).

    Uses CB + FP + HP FASTA files — all sites where both TFs have peaks
    in the dual condition. Cap at max_seqs for computational feasibility.
    """
    all_entries = []

    for cat in ['CB', 'FP', 'HP']:
        path = os.path.join(fasta_dir, f'{cat}.fa')
        if os.path.exists(path):
            entries = load_fasta(path)
            # Tag with category
            all_entries.extend([(h, s, cat) for h, s in entries])
            print(f"  {cat}: {len(entries)} sequences")

    print(f"  Total: {len(all_entries)} sequences")

    # Subsample if needed
    if len(all_entries) > max_seqs:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(all_entries), max_seqs, replace=False)
        all_entries = [all_entries[i] for i in sorted(idx)]
        print(f"  Subsampled to {max_seqs}")

    headers = [e[0] for e in all_entries]
    sequences = [e[1].upper() for e in all_entries]
    categories = [e[2] for e in all_entries]

    # One-hot encode
    encoded = [one_hot_encode(s) for s in sequences]

    return headers, sequences, encoded, categories


# ── DeepLIFT ────────────────────────────────────────────────────

def compute_deeplift_per_head(model, encoded_seqs, batch_size=64):
    """Compute DeepLIFT attributions for FOXA1 head and HNF4A head separately.

    Returns:
        foxa1_attrs: (N, 4, 1001) attributions for FOXA1 output
        hnf4a_attrs: (N, 4, 1001) attributions for HNF4A output
    """
    model.eval()
    dl = DeepLift(model)
    n = len(encoded_seqs)

    foxa1_attrs_list = []
    hnf4a_attrs_list = []

    for i in range(0, n, batch_size):
        batch = encoded_seqs[i:i + batch_size]
        x = torch.stack([torch.from_numpy(s) for s in batch])
        x.requires_grad = True
        ref = torch.zeros_like(x)

        # Attribution for FOXA1 head (output index 0)
        attrs_f = dl.attribute(x, baselines=ref, target=0)
        foxa1_attrs_list.append(attrs_f.detach().numpy())

        # Attribution for HNF4A head (output index 1)
        x2 = torch.stack([torch.from_numpy(s) for s in batch])
        x2.requires_grad = True
        ref2 = torch.zeros_like(x2)
        attrs_h = dl.attribute(x2, baselines=ref2, target=1)
        hnf4a_attrs_list.append(attrs_h.detach().numpy())

        done = min(i + batch_size, n)
        if (i // batch_size) % 5 == 0:
            print(f"    {done}/{n} sequences")

    foxa1_attrs = np.concatenate(foxa1_attrs_list, axis=0)
    hnf4a_attrs = np.concatenate(hnf4a_attrs_list, axis=0)

    return foxa1_attrs, hnf4a_attrs


# ── TF-MoDISco ──────────────────────────────────────────────────

def setup_memelite_stub():
    """Monkey-patch memelite to avoid numba issues on Python 3.8."""
    import types as _types
    memelite = _types.ModuleType('memelite')
    fimo_mod = _types.ModuleType('memelite.fimo')
    tomtom_mod = _types.ModuleType('memelite.tomtom')
    io_mod = _types.ModuleType('memelite.io')
    io_mod.read_meme = lambda *a, **kw: {}
    fimo_mod.fimo = lambda *a, **kw: None
    tomtom_mod.tomtom = lambda *a, **kw: None
    memelite.fimo = fimo_mod
    memelite.tomtom = tomtom_mod
    memelite.io = io_mod
    sys.modules['memelite'] = memelite
    sys.modules['memelite.fimo'] = fimo_mod
    sys.modules['memelite.tomtom'] = tomtom_mod
    sys.modules['memelite.io'] = io_mod


def run_modisco_on_attrs(attrs, one_hot_seqs, label,
                          max_seqlets_per_metacluster=5000):
    """Run TF-MoDISco on attribution scores.

    Args:
        attrs: (N, 4, L) raw DeepLIFT attributions
        one_hot_seqs: (N, 4, L) one-hot sequences
        label: string label for this run (e.g., 'foxa1_head')
    """
    setup_memelite_stub()
    from modiscolite import tfmodisco

    # Hypothetical contribution scores = attrs * one_hot
    ohe = np.array(one_hot_seqs)
    hyp = attrs * ohe

    # MoDISco expects (N, L, 4)
    hyp_T = np.transpose(hyp, (0, 2, 1))
    ohe_T = np.transpose(ohe, (0, 2, 1))

    print(f"    Running MoDISco for {label}...")
    print(f"    Input: {hyp_T.shape}")

    pos_patterns, neg_patterns = tfmodisco.TFMoDISco(
        hypothetical_contribs=hyp_T,
        one_hot=ohe_T,
        max_seqlets_per_metacluster=max_seqlets_per_metacluster,
        n_leiden_runs=2,
        verbose=True,
    )

    print(f"    Found {len(pos_patterns)} positive, {len(neg_patterns)} negative patterns")
    return pos_patterns, neg_patterns


def save_modisco_results(pos_patterns, neg_patterns, h5_path):
    """Save MoDISco results to HDF5.

    modisco-lite returns SeqletSet objects with .contrib_scores,
    .hypothetical_contribs, .sequence as attributes (not dicts).
    """
    with h5py.File(h5_path, 'w') as f:
        for i, p in enumerate(pos_patterns):
            grp = f.create_group(f'pos_{i}')
            grp.create_dataset('contrib_scores', data=p.contrib_scores)
            grp.create_dataset('hypothetical_contribs',
                               data=p.hypothetical_contribs)
            grp.create_dataset('sequence', data=p.sequence)
            grp.attrs['n_seqlets'] = len(p.seqlets)
        for i, p in enumerate(neg_patterns):
            grp = f.create_group(f'neg_{i}')
            grp.create_dataset('contrib_scores', data=p.contrib_scores)
            grp.create_dataset('hypothetical_contribs',
                               data=p.hypothetical_contribs)
            grp.create_dataset('sequence', data=p.sequence)
            grp.attrs['n_seqlets'] = len(p.seqlets)
    print(f"    Saved: {h5_path}")


# ── Plotting ────────────────────────────────────────────────────

def plot_attribution_profiles(foxa1_attrs, hnf4a_attrs, one_hot_seqs,
                               categories, save_path):
    """Plot average attribution profiles per head, split by category.

    This is the key figure: where in the 1001bp window does each head
    look for signal?
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 8), sharey='row')

    cat_colors = {'FP': '#1b9e77', 'HP': '#d95f02', 'CB': '#7570b3'}
    cat_labels = {'FP': 'FOXA1-Pioneered', 'HP': 'HNF4A-Pioneered', 'CB': 'Co-Bound'}
    cats = ['FP', 'HP', 'CB']

    ohe = np.array(one_hot_seqs)
    categories = np.array(categories)

    for col, cat in enumerate(cats):
        mask = categories == cat
        if mask.sum() == 0:
            continue

        n_cat = mask.sum()

        # FOXA1 head attributions for this category
        f_attrs = foxa1_attrs[mask]
        # Contribution scores = attr * one_hot, sum over nucleotides
        f_contrib = (f_attrs * ohe[mask]).sum(axis=1)  # (n, L)
        f_mean = f_contrib.mean(axis=0)

        # Smooth for visualization
        kernel = np.ones(20) / 20
        f_smooth = np.convolve(f_mean, kernel, mode='same')

        ax = axes[0, col]
        ax.fill_between(range(1001), f_smooth, alpha=0.6, color='#1f77b4')
        ax.axvline(x=500, color='gray', linestyle='--', alpha=0.3)
        ax.set_title(f'{cat_labels[cat]} (n={n_cat})', fontsize=11)
        if col == 0:
            ax.set_ylabel('FOXA1 head\ncontribution', fontsize=10)
        ax.set_xlim(0, 1000)
        ax.grid(True, alpha=0.2)

        # HNF4A head attributions
        h_attrs = hnf4a_attrs[mask]
        h_contrib = (h_attrs * ohe[mask]).sum(axis=1)
        h_mean = h_contrib.mean(axis=0)
        h_smooth = np.convolve(h_mean, kernel, mode='same')

        ax = axes[1, col]
        ax.fill_between(range(1001), h_smooth, alpha=0.6, color='#ff7f0e')
        ax.axvline(x=500, color='gray', linestyle='--', alpha=0.3)
        if col == 0:
            ax.set_ylabel('HNF4A head\ncontribution', fontsize=10)
        ax.set_xlabel('Position in 1001bp window')
        ax.set_xlim(0, 1000)
        ax.grid(True, alpha=0.2)

    plt.suptitle('DeepLIFT Attribution Profiles per Output Head\n'
                 'Where does each head look in the sequence?',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_modisco_cwms(h5_path, head_label, save_path, max_patterns=6):
    """Plot CWM (contribution weight matrix) logos from MoDISco results."""
    with h5py.File(h5_path, 'r') as f:
        pos_keys = sorted([k for k in f.keys() if k.startswith('pos_')])

    n_patterns = min(len(pos_keys), max_patterns)
    if n_patterns == 0:
        print(f"  No positive patterns found for {head_label}")
        return

    fig, axes = plt.subplots(n_patterns, 1, figsize=(12, 2.5 * n_patterns))
    if n_patterns == 1:
        axes = [axes]

    with h5py.File(h5_path, 'r') as f:
        for i, key in enumerate(pos_keys[:max_patterns]):
            cwm = f[key]['contrib_scores'][:]  # (L, 4)
            seq = f[key]['sequence'][:]  # (L, 4)

            ax = axes[i]
            # Plot CWM as simplified logo (height = contribution per position)
            pos_sum = cwm.clip(min=0).sum(axis=1)
            neg_sum = cwm.clip(max=0).sum(axis=1)

            ax.bar(range(len(pos_sum)), pos_sum, color='#2166ac', width=1.0, alpha=0.8)
            ax.bar(range(len(neg_sum)), neg_sum, color='#b2182b', width=1.0, alpha=0.8)
            ax.axhline(y=0, color='black', linewidth=0.5)
            ax.set_title(f'{head_label} pattern {i} ({key}, length={len(pos_sum)})',
                         fontsize=10)
            ax.set_xlim(-0.5, len(pos_sum) - 0.5)

            # Add consensus sequence
            nuc_map = ['A', 'C', 'G', 'T']
            consensus = ''.join(nuc_map[np.argmax(seq[j])] for j in range(len(seq)))
            # Show first/last 30 chars if long
            if len(consensus) > 40:
                label = consensus[:20] + '...' + consensus[-20:]
            else:
                label = consensus
            ax.set_xlabel(label, fontsize=8, fontfamily='monospace')

    plt.suptitle(f'TF-MoDISco Discovered Motifs — {head_label}\n'
                 f'(top {n_patterns} positive patterns)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_cross_head_correlation(foxa1_attrs, hnf4a_attrs, one_hot_seqs,
                                 save_path):
    """Plot correlation between FOXA1-head and HNF4A-head attributions.

    At each position, do both heads agree on what's important?
    High correlation at motif positions = shared grammar.
    """
    ohe = np.array(one_hot_seqs)

    # Per-position contribution scores
    f_contrib = (foxa1_attrs * ohe).sum(axis=1)  # (N, 1001)
    h_contrib = (hnf4a_attrs * ohe).sum(axis=1)  # (N, 1001)

    # Per-position correlation across sequences
    corr = np.zeros(1001)
    for pos in range(1001):
        f_vals = f_contrib[:, pos]
        h_vals = h_contrib[:, pos]
        if f_vals.std() > 0 and h_vals.std() > 0:
            corr[pos] = np.corrcoef(f_vals, h_vals)[0, 1]

    # Smooth
    kernel = np.ones(20) / 20
    corr_smooth = np.convolve(corr, kernel, mode='same')

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(range(1001), corr_smooth, color='purple', linewidth=1.5)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.axvline(x=500, color='gray', linestyle='--', alpha=0.3, label='Summit')
    ax.fill_between(range(1001), corr_smooth, alpha=0.3, color='purple')
    ax.set_xlabel('Position in 1001bp window')
    ax.set_ylabel('Pearson r (FOXA1 head vs HNF4A head)')
    ax.set_title('Cross-Head Attribution Correlation\n'
                 'Where do both heads agree on importance?',
                 fontsize=12, fontweight='bold')
    ax.set_xlim(0, 1000)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ── Runner ──────────────────────────────────────────────────────

def run_deeplift_modisco(model_path, fasta_dir, output_dir, fig_dir,
                          max_seqs=2000, run_modisco=True):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    print("=" * 60)
    print("DeepLIFT + TF-MoDISco ON BINDING MODEL")
    print("=" * 60)

    # 1. Load model
    print("\n1. Loading binding model...")
    checkpoint = torch.load(model_path, weights_only=False, map_location='cpu')
    model = BindingCNN()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 2. Load sequences
    print(f"\n2. Loading sequences (max {max_seqs})...")
    headers, sequences, encoded, categories = load_cobound_sequences(
        fasta_dir, max_seqs=max_seqs)

    # 3. DeepLIFT
    print("\n3. Computing DeepLIFT attributions per head...")
    t0 = time.time()
    foxa1_attrs, hnf4a_attrs = compute_deeplift_per_head(
        model, encoded, batch_size=64)
    dt = time.time() - t0
    print(f"  DeepLIFT: {dt:.1f}s for {len(encoded)} sequences")

    # Save attributions
    attr_path = os.path.join(output_dir, 'binding_deeplift_attrs.npz')
    np.savez_compressed(attr_path,
                        foxa1_attrs=foxa1_attrs,
                        hnf4a_attrs=hnf4a_attrs,
                        categories=np.array(categories),
                        headers=np.array(headers))
    print(f"  Saved: {attr_path}")

    # 4. Attribution profile plots
    print("\n4. Plotting attribution profiles...")
    plot_attribution_profiles(foxa1_attrs, hnf4a_attrs, encoded,
                               categories,
                               os.path.join(fig_dir, 'binding_deeplift_profiles.png'))

    plot_cross_head_correlation(foxa1_attrs, hnf4a_attrs, encoded,
                                os.path.join(fig_dir, 'binding_cross_head_corr.png'))

    # 5. TF-MoDISco (optional — can be slow)
    if run_modisco:
        print("\n5. Running TF-MoDISco...")
        ohe = np.array(encoded)

        # MoDISco for FOXA1 head
        print("\n  5a. FOXA1 head patterns...")
        t0 = time.time()
        f_pos, f_neg = run_modisco_on_attrs(
            foxa1_attrs, encoded, 'FOXA1_head',
            max_seqlets_per_metacluster=5000)
        print(f"  FOXA1 MoDISco: {time.time() - t0:.1f}s")

        f_h5 = os.path.join(output_dir, 'modisco_foxa1_head.h5')
        save_modisco_results(f_pos, f_neg, f_h5)
        plot_modisco_cwms(f_h5, 'FOXA1 Head',
                          os.path.join(fig_dir, 'binding_modisco_foxa1_head.png'))

        # MoDISco for HNF4A head
        print("\n  5b. HNF4A head patterns...")
        t0 = time.time()
        h_pos, h_neg = run_modisco_on_attrs(
            hnf4a_attrs, encoded, 'HNF4A_head',
            max_seqlets_per_metacluster=5000)
        print(f"  HNF4A MoDISco: {time.time() - t0:.1f}s")

        h_h5 = os.path.join(output_dir, 'modisco_hnf4a_head.h5')
        save_modisco_results(h_pos, h_neg, h_h5)
        plot_modisco_cwms(h_h5, 'HNF4A Head',
                          os.path.join(fig_dir, 'binding_modisco_hnf4a_head.png'))
    else:
        print("\n5. Skipping MoDISco (run_modisco=False)")

    print("\nDeepLIFT + MoDISco analysis complete!")


if __name__ == '__main__':
    model_path = os.path.join(PROJECT_ROOT, 'results', 'deep_learning',
                              'binding', 'binding_model.pt')
    fasta_dir = os.path.join(PROJECT_ROOT, 'results', 'sequences')
    output_dir = os.path.join(PROJECT_ROOT, 'results', 'deep_learning', 'binding')
    fig_dir = os.path.join(PROJECT_ROOT, 'figures')

    run_deeplift_modisco(model_path, fasta_dir, output_dir, fig_dir,
                          max_seqs=2000, run_modisco=True)
