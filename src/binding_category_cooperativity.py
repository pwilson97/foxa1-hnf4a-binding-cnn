"""
Category-aware cooperativity analysis using the binding CNN as an oracle.

Uses real FP/HP/CB sequences with known motif positions from FIMO.
Tests: does scrambling one TF's motif change the model's predicted
probability for the *other* TF — and does this depend on category?

Key predictions:
  - FP sites: scrambling FOXA1 should drop P(HNF4A), because FOXA1
    pioneered HNF4A's binding at these sites.
  - HP sites: scrambling HNF4A should drop P(FOXA1).
  - CB sites: scrambling either motif should have less effect, because
    both TFs bind independently.
"""

import os
import sys
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from binding_data import one_hot_encode
from binding_model import BindingCNN

CATEGORIES = ['FP', 'HP', 'CB']
TF_NAMES = ['FOXA1', 'HNF4A']


# ── Data loading ────────────────────────────────────────────────

def load_fasta(path):
    """Load FASTA file. Returns dict of {header: sequence}."""
    seqs = {}
    header = None
    seq_lines = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if header is not None:
                    seqs[header] = ''.join(seq_lines)
                header = line[1:]
                seq_lines = []
            else:
                seq_lines.append(line)
    if header is not None:
        seqs[header] = ''.join(seq_lines)
    return seqs


def load_fimo_hits(parsed_tsv_path):
    """Load FIMO parsed TSV. Returns dict of {peak_id: [(motif_name, start, stop, pvalue, strand)]}."""
    hits = defaultdict(list)
    with open(parsed_tsv_path) as f:
        header = f.readline().strip().split('\t')
        for line in f:
            fields = line.strip().split('\t')
            if len(fields) < len(header):
                continue
            row = dict(zip(header, fields))
            peak_id = row['peak_id']
            motif_name = row['motif_name']
            start = int(row['start'])
            stop = int(row['stop'])
            pvalue = float(row['pvalue'])
            strand = row['strand']
            hits[peak_id].append((motif_name, start, stop, pvalue, strand))
    return hits


def genomic_to_window_pos(genomic_start, genomic_stop, window_start):
    """Convert genomic coordinates to position within 1001bp window."""
    return genomic_start - window_start, genomic_stop - window_start


def get_best_motif_hit(hits_list, motif_name):
    """Get the best (lowest p-value) hit for a given motif."""
    matching = [h for h in hits_list if h[0] == motif_name]
    if not matching:
        return None
    return min(matching, key=lambda h: h[3])  # sort by pvalue


def prepare_category_data(fasta_dir, fimo_dir, categories=CATEGORIES):
    """Load sequences + motif positions for each category.

    Returns dict: {category: list of (sequence_str, foxa1_window_pos, hnf4a_window_pos)}
    where window_pos = (start, stop) within the 1001bp window, or None if no hit.
    """
    data = {}

    for cat in categories:
        fasta_path = os.path.join(fasta_dir, f'{cat}.fa')
        fimo_path = os.path.join(fimo_dir, f'{cat}_parsed.tsv')

        if not os.path.exists(fasta_path) or not os.path.exists(fimo_path):
            print(f"  WARNING: Missing files for {cat}, skipping")
            continue

        sequences = load_fasta(fasta_path)
        fimo_hits = load_fimo_hits(fimo_path)

        entries = []
        n_both = 0
        n_foxa1_only = 0
        n_hnf4a_only = 0
        n_neither = 0

        for header, seq in sequences.items():
            # Parse window coordinates from header (e.g., "chr1:2507099-2508100")
            parts = header.replace(':', '-').split('-')
            chrom = parts[0]
            window_start = int(parts[1])

            # Get FIMO hits for this peak
            peak_hits = fimo_hits.get(header, [])

            # Best FOXA1 and HNF4A hit
            best_foxa1 = get_best_motif_hit(peak_hits, 'FOXA1')
            best_hnf4a = get_best_motif_hit(peak_hits, 'HNF4A')

            foxa1_pos = None
            hnf4a_pos = None

            if best_foxa1:
                s, e = genomic_to_window_pos(best_foxa1[1], best_foxa1[2], window_start)
                if 0 <= s and e <= len(seq):
                    foxa1_pos = (s, e)

            if best_hnf4a:
                s, e = genomic_to_window_pos(best_hnf4a[1], best_hnf4a[2], window_start)
                if 0 <= s and e <= len(seq):
                    hnf4a_pos = (s, e)

            if foxa1_pos and hnf4a_pos:
                n_both += 1
            elif foxa1_pos:
                n_foxa1_only += 1
            elif hnf4a_pos:
                n_hnf4a_only += 1
            else:
                n_neither += 1

            # Only include sequences with both motifs for paired analysis
            if foxa1_pos and hnf4a_pos:
                entries.append({
                    'seq': seq.upper(),
                    'header': header,
                    'foxa1_pos': foxa1_pos,
                    'hnf4a_pos': hnf4a_pos,
                })

        data[cat] = entries
        print(f"  {cat}: {len(sequences)} total, {n_both} with both motifs, "
              f"{n_foxa1_only} FOXA1-only, {n_hnf4a_only} HNF4A-only, "
              f"{n_neither} neither → using {len(entries)} paired")

    return data


# ── Motif manipulation ──────────────────────────────────────────

def scramble_region(seq, start, stop, rng):
    """Scramble (dinucleotide-preserving shuffle) a region in the sequence.

    Uses a simple nucleotide shuffle to destroy the motif while preserving
    local nucleotide composition.
    """
    s = list(seq)
    region = s[start:stop]
    rng.shuffle(region)
    s[start:stop] = region
    return ''.join(s)


def zero_region(encoded, start, stop):
    """Zero out a region in one-hot encoded sequence (equivalent to 'N')."""
    enc = encoded.copy()
    enc[:, start:stop] = 0.0
    return enc


# ── Prediction ──────────────────────────────────────────────────

def predict_batch(model, seqs, batch_size=256):
    """Predict binding probabilities for list of DNA strings."""
    model.eval()
    all_probs = []

    for i in range(0, len(seqs), batch_size):
        batch = seqs[i:i + batch_size]
        encoded = [one_hot_encode(s) for s in batch]
        x = torch.stack([torch.from_numpy(e) for e in encoded])
        with torch.no_grad():
            probs = model.predict_proba(x).numpy()
        all_probs.append(probs)

    return np.concatenate(all_probs, axis=0)


def predict_batch_encoded(model, encoded_list, batch_size=256):
    """Predict from pre-encoded one-hot arrays."""
    model.eval()
    all_probs = []

    for i in range(0, len(encoded_list), batch_size):
        batch = encoded_list[i:i + batch_size]
        x = torch.stack([torch.from_numpy(e) for e in batch])
        with torch.no_grad():
            probs = model.predict_proba(x).numpy()
        all_probs.append(probs)

    return np.concatenate(all_probs, axis=0)


# ── Main analysis ───────────────────────────────────────────────

def motif_masking_analysis(model, category_data, n_scrambles=10, seed=42):
    """For each category, measure the effect of scrambling each TF's motif.

    Returns dict of:
    {
        category: {
            'n_sites': int,
            'original': {'foxa1_prob': array, 'hnf4a_prob': array},
            'scramble_foxa1': {'foxa1_prob': array, 'hnf4a_prob': array},
            'scramble_hnf4a': {'foxa1_prob': array, 'hnf4a_prob': array},
            'scramble_both': {'foxa1_prob': array, 'hnf4a_prob': array},
        }
    }
    """
    rng = np.random.RandomState(seed)
    results = {}

    for cat in CATEGORIES:
        if cat not in category_data:
            continue

        entries = category_data[cat]
        if not entries:
            continue

        print(f"\n  {cat}: processing {len(entries)} sequences...")

        # Original predictions
        orig_seqs = [e['seq'] for e in entries]
        orig_probs = predict_batch(model, orig_seqs)

        # Scramble FOXA1 motif (average over n_scrambles random shuffles)
        scr_foxa1_probs = np.zeros_like(orig_probs)
        for s in range(n_scrambles):
            scrambled = []
            for e in entries:
                scrambled.append(
                    scramble_region(e['seq'], e['foxa1_pos'][0],
                                   e['foxa1_pos'][1], rng))
            scr_foxa1_probs += predict_batch(model, scrambled)
        scr_foxa1_probs /= n_scrambles

        # Scramble HNF4A motif
        scr_hnf4a_probs = np.zeros_like(orig_probs)
        for s in range(n_scrambles):
            scrambled = []
            for e in entries:
                scrambled.append(
                    scramble_region(e['seq'], e['hnf4a_pos'][0],
                                   e['hnf4a_pos'][1], rng))
            scr_hnf4a_probs += predict_batch(model, scrambled)
        scr_hnf4a_probs /= n_scrambles

        # Scramble both motifs
        scr_both_probs = np.zeros_like(orig_probs)
        for s in range(n_scrambles):
            scrambled = []
            for e in entries:
                seq = scramble_region(e['seq'], e['foxa1_pos'][0],
                                     e['foxa1_pos'][1], rng)
                seq = scramble_region(seq, e['hnf4a_pos'][0],
                                     e['hnf4a_pos'][1], rng)
                scrambled.append(seq)
            scr_both_probs += predict_batch(model, scrambled)
        scr_both_probs /= n_scrambles

        results[cat] = {
            'n_sites': len(entries),
            'original': {
                'foxa1_prob': orig_probs[:, 0],
                'hnf4a_prob': orig_probs[:, 1],
            },
            'scramble_foxa1': {
                'foxa1_prob': scr_foxa1_probs[:, 0],
                'hnf4a_prob': scr_foxa1_probs[:, 1],
            },
            'scramble_hnf4a': {
                'foxa1_prob': scr_hnf4a_probs[:, 0],
                'hnf4a_prob': scr_hnf4a_probs[:, 1],
            },
            'scramble_both': {
                'foxa1_prob': scr_both_probs[:, 0],
                'hnf4a_prob': scr_both_probs[:, 1],
            },
        }

    return results


def compute_deltas(results):
    """Compute delta P for each category and scramble condition.

    Returns summary dict with means and SEMs.
    """
    summary = {}

    for cat in CATEGORIES:
        if cat not in results:
            continue

        r = results[cat]
        n = r['n_sites']

        # Delta when scrambling FOXA1
        delta_foxa1_on_foxa1 = r['scramble_foxa1']['foxa1_prob'] - r['original']['foxa1_prob']
        delta_foxa1_on_hnf4a = r['scramble_foxa1']['hnf4a_prob'] - r['original']['hnf4a_prob']

        # Delta when scrambling HNF4A
        delta_hnf4a_on_foxa1 = r['scramble_hnf4a']['foxa1_prob'] - r['original']['foxa1_prob']
        delta_hnf4a_on_hnf4a = r['scramble_hnf4a']['hnf4a_prob'] - r['original']['hnf4a_prob']

        summary[cat] = {
            'n_sites': n,
            # Self-effects (sanity check: scrambling a TF's motif should reduce its own prediction)
            'scr_foxa1_delta_foxa1': {
                'mean': float(delta_foxa1_on_foxa1.mean()),
                'sem': float(delta_foxa1_on_foxa1.std() / np.sqrt(n)),
                'median': float(np.median(delta_foxa1_on_foxa1)),
            },
            'scr_hnf4a_delta_hnf4a': {
                'mean': float(delta_hnf4a_on_hnf4a.mean()),
                'sem': float(delta_hnf4a_on_hnf4a.std() / np.sqrt(n)),
                'median': float(np.median(delta_hnf4a_on_hnf4a)),
            },
            # Cross-effects (the interesting ones!)
            'scr_foxa1_delta_hnf4a': {
                'mean': float(delta_foxa1_on_hnf4a.mean()),
                'sem': float(delta_foxa1_on_hnf4a.std() / np.sqrt(n)),
                'median': float(np.median(delta_foxa1_on_hnf4a)),
            },
            'scr_hnf4a_delta_foxa1': {
                'mean': float(delta_hnf4a_on_foxa1.mean()),
                'sem': float(delta_hnf4a_on_foxa1.std() / np.sqrt(n)),
                'median': float(np.median(delta_hnf4a_on_foxa1)),
            },
            # Mean original probs
            'mean_orig_foxa1': float(r['original']['foxa1_prob'].mean()),
            'mean_orig_hnf4a': float(r['original']['hnf4a_prob'].mean()),
        }

    return summary


# ── Plotting ────────────────────────────────────────────────────

CAT_COLORS = {'FP': '#1b9e77', 'HP': '#d95f02', 'CB': '#7570b3'}
CAT_LABELS = {
    'FP': 'FOXA1-Pioneered',
    'HP': 'HNF4A-Pioneered',
    'CB': 'Co-Bound',
}


def plot_cross_effects(summary, save_path):
    """Main figure: cross-TF effects of motif scrambling by category.

    4-panel figure:
    Top-left:  Scramble FOXA1 → effect on P(HNF4A), by category
    Top-right: Scramble HNF4A → effect on P(FOXA1), by category
    Bottom-left: Self-effects (sanity check)
    Bottom-right: Summary bar chart of cross-effects
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    cats = [c for c in CATEGORIES if c in summary]

    # ── Top left: Scramble FOXA1 → ΔP(HNF4A) ──
    ax = axes[0, 0]
    means = [summary[c]['scr_foxa1_delta_hnf4a']['mean'] for c in cats]
    sems = [summary[c]['scr_foxa1_delta_hnf4a']['sem'] for c in cats]
    colors = [CAT_COLORS[c] for c in cats]
    labels = [f"{CAT_LABELS[c]}\n(n={summary[c]['n_sites']})" for c in cats]

    bars = ax.bar(labels, means, yerr=sems, color=colors, edgecolor='black',
                  linewidth=0.8, capsize=5, alpha=0.85)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_ylabel('ΔP(HNF4A bound)')
    ax.set_title('Scramble FOXA1 motif → effect on HNF4A prediction',
                 fontsize=11, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, m in zip(bars, means):
        va = 'bottom' if m >= 0 else 'top'
        ax.text(bar.get_x() + bar.get_width() / 2, m,
                f'{m:+.3f}', ha='center', va=va, fontsize=9, fontweight='bold')

    # ── Top right: Scramble HNF4A → ΔP(FOXA1) ──
    ax = axes[0, 1]
    means = [summary[c]['scr_hnf4a_delta_foxa1']['mean'] for c in cats]
    sems = [summary[c]['scr_hnf4a_delta_foxa1']['sem'] for c in cats]

    bars = ax.bar(labels, means, yerr=sems, color=colors, edgecolor='black',
                  linewidth=0.8, capsize=5, alpha=0.85)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_ylabel('ΔP(FOXA1 bound)')
    ax.set_title('Scramble HNF4A motif → effect on FOXA1 prediction',
                 fontsize=11, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    for bar, m in zip(bars, means):
        va = 'bottom' if m >= 0 else 'top'
        ax.text(bar.get_x() + bar.get_width() / 2, m,
                f'{m:+.3f}', ha='center', va=va, fontsize=9, fontweight='bold')

    # ── Bottom left: Self-effects (sanity check) ──
    ax = axes[1, 0]
    x = np.arange(len(cats))
    w = 0.35

    # Scramble FOXA1 → ΔP(FOXA1) [self effect]
    self_f = [summary[c]['scr_foxa1_delta_foxa1']['mean'] for c in cats]
    self_f_sem = [summary[c]['scr_foxa1_delta_foxa1']['sem'] for c in cats]
    # Scramble HNF4A → ΔP(HNF4A) [self effect]
    self_h = [summary[c]['scr_hnf4a_delta_hnf4a']['mean'] for c in cats]
    self_h_sem = [summary[c]['scr_hnf4a_delta_hnf4a']['sem'] for c in cats]

    ax.bar(x - w / 2, self_f, w, yerr=self_f_sem, label='Scramble FOXA1 → ΔP(FOXA1)',
           color='#1f77b4', edgecolor='black', linewidth=0.5, capsize=4, alpha=0.8)
    ax.bar(x + w / 2, self_h, w, yerr=self_h_sem, label='Scramble HNF4A → ΔP(HNF4A)',
           color='#ff7f0e', edgecolor='black', linewidth=0.5, capsize=4, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([CAT_LABELS[c] for c in cats])
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_ylabel('ΔP(self binding)')
    ax.set_title('Self-effect (sanity check: scrambling motif\nshould reduce own prediction)',
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    # ── Bottom right: Cross-effects comparison ──
    ax = axes[1, 1]

    # Cross-effects: the key result
    cross_foxa1_helps_hnf4a = [summary[c]['scr_foxa1_delta_hnf4a']['mean'] for c in cats]
    cross_hnf4a_helps_foxa1 = [summary[c]['scr_hnf4a_delta_foxa1']['mean'] for c in cats]
    cross_f_sem = [summary[c]['scr_foxa1_delta_hnf4a']['sem'] for c in cats]
    cross_h_sem = [summary[c]['scr_hnf4a_delta_foxa1']['sem'] for c in cats]

    ax.bar(x - w / 2, cross_foxa1_helps_hnf4a, w, yerr=cross_f_sem,
           label='FOXA1 → HNF4A\n(scr FOXA1, measure HNF4A)',
           color='#ff7f0e', edgecolor='black', linewidth=0.5, capsize=4, alpha=0.8)
    ax.bar(x + w / 2, cross_hnf4a_helps_foxa1, w, yerr=cross_h_sem,
           label='HNF4A → FOXA1\n(scr HNF4A, measure FOXA1)',
           color='#1f77b4', edgecolor='black', linewidth=0.5, capsize=4, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([CAT_LABELS[c] for c in cats])
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_ylabel('ΔP(partner binding)')
    ax.set_title('Cross-TF cooperativity by category\n'
                 '(negative = partner motif helps binding)',
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Category-Aware In Silico Cooperativity\n'
                 'Does the model learn that pioneering TFs enable partner binding?',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_delta_distributions(results, save_path):
    """Violin/box plots showing the distribution of ΔP per site, per category."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    cats = [c for c in CATEGORIES if c in results]

    # Left: Scramble FOXA1 → ΔP(HNF4A) distributions
    ax = axes[0]
    data_list = []
    positions = []
    colors_list = []
    tick_labels = []
    for i, cat in enumerate(cats):
        r = results[cat]
        delta = r['scramble_foxa1']['hnf4a_prob'] - r['original']['hnf4a_prob']
        data_list.append(delta)
        positions.append(i)
        colors_list.append(CAT_COLORS[cat])
        tick_labels.append(f"{CAT_LABELS[cat]}\n(n={r['n_sites']})")

    parts = ax.violinplot(data_list, positions=positions, showmeans=True,
                          showmedians=True, showextrema=False)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors_list[i])
        pc.set_alpha(0.6)
    parts['cmeans'].set_color('red')
    parts['cmedians'].set_color('black')

    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels)
    ax.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
    ax.set_ylabel('ΔP(HNF4A bound)')
    ax.set_title('Effect of scrambling FOXA1 on HNF4A prediction\n'
                 '(red=mean, black=median)', fontsize=10, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Right: Scramble HNF4A → ΔP(FOXA1) distributions
    ax = axes[1]
    data_list = []
    for i, cat in enumerate(cats):
        r = results[cat]
        delta = r['scramble_hnf4a']['foxa1_prob'] - r['original']['foxa1_prob']
        data_list.append(delta)

    parts = ax.violinplot(data_list, positions=positions, showmeans=True,
                          showmedians=True, showextrema=False)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors_list[i])
        pc.set_alpha(0.6)
    parts['cmeans'].set_color('red')
    parts['cmedians'].set_color('black')

    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels)
    ax.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
    ax.set_ylabel('ΔP(FOXA1 bound)')
    ax.set_title('Effect of scrambling HNF4A on FOXA1 prediction\n'
                 '(red=mean, black=median)', fontsize=10, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Per-Site Cooperativity Distributions by Category',
                 fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_original_predictions(results, save_path):
    """Show what the model originally predicts for each category's real sequences."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    cats = [c for c in CATEGORIES if c in results]

    # FOXA1 predictions by category
    ax = axes[0]
    data_list = []
    for cat in cats:
        data_list.append(results[cat]['original']['foxa1_prob'])

    parts = ax.violinplot(data_list, positions=range(len(cats)),
                          showmeans=True, showmedians=True, showextrema=False)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(CAT_COLORS[cats[i]])
        pc.set_alpha(0.6)
    parts['cmeans'].set_color('red')
    parts['cmedians'].set_color('black')

    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels([f"{CAT_LABELS[c]}\n(n={results[c]['n_sites']})" for c in cats])
    ax.axhline(y=0.5, color='gray', linewidth=0.5, linestyle='--')
    ax.set_ylabel('P(FOXA1 bound)')
    ax.set_title('Model predictions for FOXA1 binding\non real category sequences')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(axis='y', alpha=0.3)

    # HNF4A predictions by category
    ax = axes[1]
    data_list = []
    for cat in cats:
        data_list.append(results[cat]['original']['hnf4a_prob'])

    parts = ax.violinplot(data_list, positions=range(len(cats)),
                          showmeans=True, showmedians=True, showextrema=False)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(CAT_COLORS[cats[i]])
        pc.set_alpha(0.6)
    parts['cmeans'].set_color('red')
    parts['cmedians'].set_color('black')

    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels([f"{CAT_LABELS[c]}\n(n={results[c]['n_sites']})" for c in cats])
    ax.axhline(y=0.5, color='gray', linewidth=0.5, linestyle='--')
    ax.set_ylabel('P(HNF4A bound)')
    ax.set_title('Model predictions for HNF4A binding\non real category sequences')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Binding Predictions on Real FP/HP/CB Sequences',
                 fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ── Runner ──────────────────────────────────────────────────────

def run_category_cooperativity(model_path, fasta_dir, fimo_dir,
                                output_dir, fig_dir):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    print("=" * 60)
    print("CATEGORY-AWARE COOPERATIVITY ANALYSIS")
    print("=" * 60)

    # Load model
    print("\n1. Loading trained binding model...")
    checkpoint = torch.load(model_path, weights_only=False, map_location='cpu')
    model = BindingCNN()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"   Best epoch: {checkpoint['epoch']}")

    # Load data
    print("\n2. Loading category sequences + FIMO motif positions...")
    category_data = prepare_category_data(fasta_dir, fimo_dir)

    # Run masking analysis
    print("\n3. Motif scrambling analysis (10 random shuffles each)...")
    results = motif_masking_analysis(model, category_data, n_scrambles=10)

    # Compute summary stats
    print("\n4. Computing summary statistics...")
    summary = compute_deltas(results)

    for cat in CATEGORIES:
        if cat not in summary:
            continue
        s = summary[cat]
        print(f"\n  {CAT_LABELS[cat]} ({s['n_sites']} sites):")
        print(f"    Mean P(FOXA1)={s['mean_orig_foxa1']:.3f}, "
              f"P(HNF4A)={s['mean_orig_hnf4a']:.3f}")
        print(f"    Scramble FOXA1 → ΔP(FOXA1)={s['scr_foxa1_delta_foxa1']['mean']:+.4f}, "
              f"ΔP(HNF4A)={s['scr_foxa1_delta_hnf4a']['mean']:+.4f}")
        print(f"    Scramble HNF4A → ΔP(FOXA1)={s['scr_hnf4a_delta_foxa1']['mean']:+.4f}, "
              f"ΔP(HNF4A)={s['scr_hnf4a_delta_hnf4a']['mean']:+.4f}")

    # Save results
    print("\n5. Saving results and figures...")
    with open(os.path.join(output_dir, 'category_cooperativity.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # Save raw arrays for potential follow-up
    save_dict = {}
    for cat in CATEGORIES:
        if cat not in results:
            continue
        r = results[cat]
        for condition in ['original', 'scramble_foxa1', 'scramble_hnf4a', 'scramble_both']:
            for tf in ['foxa1_prob', 'hnf4a_prob']:
                key = f'{cat}_{condition}_{tf}'
                save_dict[key] = r[condition][tf]
    np.savez(os.path.join(output_dir, 'category_cooperativity_raw.npz'), **save_dict)

    # Plots
    plot_cross_effects(summary,
                       os.path.join(fig_dir, 'binding_category_cooperativity.png'))
    plot_delta_distributions(results,
                             os.path.join(fig_dir, 'binding_category_deltas.png'))
    plot_original_predictions(results,
                              os.path.join(fig_dir, 'binding_category_predictions.png'))

    print("\nCategory-aware cooperativity analysis complete!")
    return results, summary


if __name__ == '__main__':
    model_path = os.path.join(PROJECT_ROOT, 'results', 'deep_learning',
                              'binding', 'binding_model.pt')
    fasta_dir = os.path.join(PROJECT_ROOT, 'results', 'sequences')
    fimo_dir = os.path.join(PROJECT_ROOT, 'results', 'fimo')
    output_dir = os.path.join(PROJECT_ROOT, 'results', 'deep_learning', 'binding')
    fig_dir = os.path.join(PROJECT_ROOT, 'figures')

    run_category_cooperativity(model_path, fasta_dir, fimo_dir,
                                output_dir, fig_dir)
