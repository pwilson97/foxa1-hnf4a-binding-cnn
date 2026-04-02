"""
Interpretability pipeline: DeepLIFT attribution + TF-MoDISco motif discovery.

Computes per-base importance scores for each sequence using DeepLIFT,
then runs TF-MoDISco to cluster importance patterns into discovered motifs.
"""

import os
import sys
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

from dl_data import (load_all_sequences, create_splits, CATEGORIES,
                     CAT_TO_IDX, IDX_TO_CAT)
from dl_model import MotifGrammarCNN, SimpleMotifCNN


def load_trained_model(model_path):
    """Load a trained model from checkpoint."""
    checkpoint = torch.load(model_path, weights_only=False, map_location='cpu')
    model_type = checkpoint.get('model_type', 'dilated')

    if model_type == 'dilated':
        model = MotifGrammarCNN()
    else:
        model = SimpleMotifCNN()

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint


def compute_deeplift_scores(model, sequences, labels, target_class=None,
                            batch_size=32):
    """Compute DeepLIFT attribution scores for sequences.

    Args:
        model: trained PyTorch model
        sequences: list of (4, L) numpy arrays
        labels: array of integer labels
        target_class: if None, use true label; otherwise compute for this class
        batch_size: batch size for processing

    Returns:
        attributions: numpy array (N, 4, L) of importance scores
    """
    model.eval()
    dl = DeepLift(model)

    all_attrs = []
    n = len(sequences)

    for i in range(0, n, batch_size):
        batch_seqs = sequences[i:i + batch_size]
        batch_labels = labels[i:i + batch_size]

        x = torch.stack([torch.from_numpy(s) for s in batch_seqs])
        x.requires_grad = True

        # Reference: all-zero (equivalent to "no DNA" baseline)
        ref = torch.zeros_like(x)

        # Compute attribution for each sequence's target class
        if target_class is not None:
            targets = torch.full((len(batch_seqs),), target_class, dtype=torch.long)
        else:
            targets = torch.tensor(batch_labels, dtype=torch.long)

        # DeepLIFT for each target class
        attrs = dl.attribute(x, baselines=ref, target=targets)
        all_attrs.append(attrs.detach().numpy())

        if (i // batch_size) % 10 == 0:
            print(f"    DeepLIFT: {i + len(batch_seqs)}/{n} sequences")

    return np.concatenate(all_attrs, axis=0)


def compute_hypothetical_scores(attributions, sequences):
    """Compute hypothetical importance scores (attribution × one-hot).

    This is what TF-MoDISco expects: the importance score at each position
    weighted by whether that nucleotide is actually present.
    """
    # attributions: (N, 4, L), sequences: list of (4, L) arrays
    seqs_arr = np.array(sequences)  # (N, 4, L)
    return attributions * seqs_arr


def run_modisco(hypothetical_scores, one_hot_seqs, output_dir,
                n_seqlets=50000, max_seqlets_per_metacluster=20000):
    """Run TF-MoDISco on importance scores to discover motif patterns.

    Args:
        hypothetical_scores: (N, 4, L) importance × one-hot
        one_hot_seqs: (N, 4, L) one-hot encoded sequences
        output_dir: directory to save results
        n_seqlets: max seqlets to use
    """
    # Monkey-patch memelite to avoid numba issue on Python 3.8
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

    from modiscolite import tfmodisco

    print("    Running TF-MoDISco...")
    print(f"    Input shape: {hypothetical_scores.shape}")

    # TF-MoDISco expects (N, L, 4) — transpose from (N, 4, L)
    hyp_T = np.transpose(hypothetical_scores, (0, 2, 1))  # (N, L, 4)
    ohe_T = np.transpose(one_hot_seqs, (0, 2, 1))  # (N, L, 4)

    pos_patterns, neg_patterns = tfmodisco.TFMoDISco(
        hypothetical_contribs=hyp_T,
        one_hot=ohe_T,
        max_seqlets_per_metacluster=max_seqlets_per_metacluster,
        n_leiden_runs=2,  # faster for small datasets
        verbose=True,
    )

    # Save results
    h5_path = os.path.join(output_dir, 'modisco_results.h5')
    with h5py.File(h5_path, 'w') as f:
        for i, pattern in enumerate(pos_patterns):
            grp = f.create_group(f'pos_pattern_{i}')
            grp.create_dataset('contrib_scores', data=pattern['contrib_scores'])
            grp.create_dataset('hypothetical_contribs', data=pattern['hypothetical_contribs'])
            grp.create_dataset('sequence', data=pattern['sequence'])
            grp.attrs['n_seqlets'] = pattern.get('n_seqlets', len(pattern.get('seqlets_and_alnmts', {}).get('seqlets', [])))

        for i, pattern in enumerate(neg_patterns):
            grp = f.create_group(f'neg_pattern_{i}')
            grp.create_dataset('contrib_scores', data=pattern['contrib_scores'])
            grp.create_dataset('hypothetical_contribs', data=pattern['hypothetical_contribs'])
            grp.create_dataset('sequence', data=pattern['sequence'])

        f.attrs['n_pos_patterns'] = len(pos_patterns)
        f.attrs['n_neg_patterns'] = len(neg_patterns)

    print(f"    Found {len(pos_patterns)} positive and {len(neg_patterns)} negative patterns")
    print(f"    Saved to: {h5_path}")

    return pos_patterns, neg_patterns


def plot_cwm(pattern, title, save_path):
    """Plot a contribution weight matrix (CWM) as a logo-style plot."""
    # pattern['contrib_scores'] is (L, 4)
    cwm = pattern['contrib_scores']  # (L, 4)

    fig, ax = plt.subplots(figsize=(max(6, len(cwm) * 0.3), 2))

    # Simple bar-based logo
    colors = {'A': '#109648', 'C': '#255C99', 'G': '#F7B32B', 'T': '#D62839'}
    nucs = ['A', 'C', 'G', 'T']

    for pos in range(len(cwm)):
        scores = cwm[pos]
        # Sort by absolute value for stacking
        sorted_idx = np.argsort(np.abs(scores))

        y_offset = 0
        for idx in sorted_idx:
            height = scores[idx]
            if abs(height) > 0.001:
                color = colors[nucs[idx]]
                ax.bar(pos, height, bottom=y_offset, width=0.9,
                       color=color, alpha=0.8)
                if height > 0:
                    y_offset += height

    ax.set_xlim(-0.5, len(cwm) - 0.5)
    ax.set_xlabel('Position')
    ax.set_ylabel('Contribution')
    ax.set_title(title)
    ax.axhline(y=0, color='black', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_importance_profile(attributions, labels, save_path):
    """Plot average importance profiles per category."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    for idx, (cat, ax) in enumerate(zip(CATEGORIES, axes.flat)):
        mask = labels == idx
        cat_attrs = attributions[mask]  # (n, 4, L)

        # Sum across nucleotides for total importance
        total_importance = np.abs(cat_attrs).sum(axis=1)  # (n, L)
        mean_profile = total_importance.mean(axis=0)  # (L,)

        # Smooth with rolling mean
        window = 20
        smoothed = np.convolve(mean_profile, np.ones(window) / window, mode='same')

        ax.fill_between(range(len(smoothed)), smoothed, alpha=0.3)
        ax.plot(smoothed, linewidth=1)
        ax.set_title(f'{cat} (n={mask.sum()})')
        ax.set_xlabel('Position (bp from summit)')
        ax.set_ylabel('Mean |importance|')
        ax.axvline(x=500, color='red', linestyle='--', alpha=0.5, label='Summit')
        ax.legend(fontsize=8)

    plt.suptitle('Average DeepLIFT Importance Profiles by Category', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved importance profiles: {save_path}")


def plot_class_specific_importance(attributions, labels, save_path):
    """Plot importance difference: each class vs all others."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    palette = {'FP': '#1f77b4', 'HP': '#ff7f0e', 'CB': '#2ca02c',
               'both_pioneer': '#d62728'}

    for idx, (cat, ax) in enumerate(zip(CATEGORIES, axes.flat)):
        mask = labels == idx
        cat_attrs = np.abs(attributions[mask]).sum(axis=1).mean(axis=0)
        other_attrs = np.abs(attributions[~mask]).sum(axis=1).mean(axis=0)

        diff = cat_attrs - other_attrs

        window = 20
        smoothed = np.convolve(diff, np.ones(window) / window, mode='same')

        ax.fill_between(range(len(smoothed)), smoothed,
                         where=smoothed > 0, alpha=0.3, color=palette[cat])
        ax.fill_between(range(len(smoothed)), smoothed,
                         where=smoothed < 0, alpha=0.3, color='gray')
        ax.plot(smoothed, color=palette[cat], linewidth=1)
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.axvline(x=500, color='red', linestyle='--', alpha=0.5)
        ax.set_title(f'{cat} vs rest')
        ax.set_xlabel('Position (bp from summit)')
        ax.set_ylabel('Importance difference')

    plt.suptitle('Class-Specific Importance (category minus background)', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved class-specific importance: {save_path}")


def run_interpretability(seq_dir, model_path, output_dir, fig_dir,
                         run_modisco_analysis=True):
    """Full interpretability pipeline."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    # Load model
    print("=" * 60)
    print("INTERPRETABILITY PIPELINE")
    print("=" * 60)
    print("\n1. Loading trained model...")
    model, checkpoint = load_trained_model(model_path)
    print(f"   Model type: {checkpoint.get('model_type', 'unknown')}")
    print(f"   Best epoch: {checkpoint['epoch']}")

    # Load all sequences
    print("\n2. Loading sequences...")
    seqs, labels, headers, cats = load_all_sequences(seq_dir)

    # Compute DeepLIFT attributions for all sequences
    print("\n3. Computing DeepLIFT attributions...")
    attributions = compute_deeplift_scores(model, seqs, labels, batch_size=32)
    print(f"   Attribution shape: {attributions.shape}")

    # Save attributions
    attr_path = os.path.join(output_dir, 'deeplift_attributions.npz')
    np.savez_compressed(attr_path,
                        attributions=attributions,
                        labels=labels,
                        headers=np.array(headers))
    print(f"   Saved: {attr_path}")

    # Also compute class-specific attributions (each seq scored for each class)
    print("\n4. Computing class-specific attributions...")
    class_attrs = {}
    for class_idx, cat in enumerate(CATEGORIES):
        print(f"   Computing for class: {cat}")
        class_attrs[cat] = compute_deeplift_scores(
            model, seqs, labels, target_class=class_idx, batch_size=32
        )

    class_attr_path = os.path.join(output_dir, 'deeplift_class_attributions.npz')
    np.savez_compressed(class_attr_path,
                        **{f'{cat}_attrs': class_attrs[cat] for cat in CATEGORIES},
                        labels=labels)
    print(f"   Saved: {class_attr_path}")

    # Plot importance profiles
    print("\n5. Plotting importance profiles...")
    plot_importance_profile(
        attributions, labels,
        os.path.join(fig_dir, 'dl_importance_profiles.png')
    )
    plot_class_specific_importance(
        attributions, labels,
        os.path.join(fig_dir, 'dl_class_specific_importance.png')
    )

    # Run TF-MoDISco
    if run_modisco_analysis:
        print("\n6. Running TF-MoDISco...")
        one_hot_seqs = np.array(seqs)  # (N, 4, L)
        hyp_scores = compute_hypothetical_scores(attributions, seqs)

        try:
            pos_patterns, neg_patterns = run_modisco(
                hyp_scores, one_hot_seqs, output_dir,
                n_seqlets=30000, max_seqlets_per_metacluster=10000
            )

            # Plot discovered motifs
            print("\n7. Plotting discovered motifs...")
            for i, pat in enumerate(pos_patterns[:10]):
                plot_cwm(pat, f'Positive Pattern {i}',
                         os.path.join(fig_dir, f'dl_modisco_pos_{i}.png'))
            for i, pat in enumerate(neg_patterns[:5]):
                plot_cwm(pat, f'Negative Pattern {i}',
                         os.path.join(fig_dir, f'dl_modisco_neg_{i}.png'))

        except Exception as e:
            print(f"   TF-MoDISco failed: {e}")
            print("   Continuing without MoDISco results...")

    print("\nInterpretability pipeline complete!")


if __name__ == '__main__':
    seq_dir = os.path.join(PROJECT_ROOT, 'results', 'sequences')
    model_path = os.path.join(PROJECT_ROOT, 'results', 'deep_learning', 'best_model.pt')
    output_dir = os.path.join(PROJECT_ROOT, 'results', 'deep_learning')
    fig_dir = os.path.join(PROJECT_ROOT, 'figures')

    run_interpretability(
        seq_dir=seq_dir,
        model_path=model_path,
        output_dir=output_dir,
        fig_dir=fig_dir,
        run_modisco_analysis=True,
    )
