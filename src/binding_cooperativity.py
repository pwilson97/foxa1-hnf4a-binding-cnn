"""
In silico cooperativity analysis using the binding model as an oracle.

Core question: Does inserting a FOXA1 motif near an HNF4A motif change
the predicted HNF4A binding probability (and vice versa)?

This is the Option C analog of BPNet's Figure 4.
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

from binding_data import one_hot_encode
from binding_model import BindingCNN

# Consensus motifs (JASPAR)
FOXA1_MOTIF = "AATGTAAACAA"      # 11bp forkhead
HNF4A_MOTIF = "CAAAGTCCAAAGTTCA"  # 16bp DR1


def rev_comp(seq):
    comp = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
    return ''.join(comp.get(b, 'N') for b in reversed(seq))


def generate_background(length=1001, gc=0.42, seed=None):
    rng = np.random.RandomState(seed)
    p = [(1 - gc) / 2, gc / 2, gc / 2, (1 - gc) / 2]
    return ''.join(rng.choice(['A', 'C', 'G', 'T'], size=length, p=p))


def implant(seq, motif, pos, strand='+'):
    """Implant motif at position in sequence string."""
    m = motif if strand == '+' else rev_comp(motif)
    s = list(seq)
    if pos < 0 or pos + len(m) > len(s):
        return None
    for i, b in enumerate(m):
        s[pos + i] = b
    return ''.join(s)


def predict(model, seqs):
    """Predict binding probabilities for DNA strings."""
    model.eval()
    encoded = [one_hot_encode(s) for s in seqs]
    x = torch.stack([torch.from_numpy(e) for e in encoded])
    with torch.no_grad():
        probs = model.predict_proba(x).numpy()
    return probs  # (N, 2) — [foxa1_prob, hnf4a_prob]


def cooperativity_spacing_scan(model, n_bg=100, seed=42):
    """Measure how a partner motif changes predicted binding at varying distances.

    For each distance d:
    1. Implant FOXA1 motif alone at center → measure P(FOXA1), P(HNF4A)
    2. Implant FOXA1 + HNF4A at distance d → measure P(FOXA1), P(HNF4A)
    3. The cooperativity = P(HNF4A | both) - P(HNF4A | FOXA1 alone)
       and                  P(FOXA1 | both) - P(FOXA1 | HNF4A alone)

    This directly tests: does the presence of a nearby partner motif
    change the predicted binding of the other TF?
    """
    spacings = list(range(15, 301, 5))
    center = 500

    # Baselines: single motif only
    bg_seqs = [generate_background(seed=seed + i) for i in range(n_bg)]

    # FOXA1 alone
    foxa1_only_seqs = []
    for bg in bg_seqs:
        s = implant(bg, FOXA1_MOTIF, center - len(FOXA1_MOTIF) // 2)
        if s:
            foxa1_only_seqs.append(s)
    foxa1_only_probs = predict(model, foxa1_only_seqs).mean(axis=0)

    # HNF4A alone
    hnf4a_only_seqs = []
    for bg in bg_seqs:
        s = implant(bg, HNF4A_MOTIF, center - len(HNF4A_MOTIF) // 2)
        if s:
            hnf4a_only_seqs.append(s)
    hnf4a_only_probs = predict(model, hnf4a_only_seqs).mean(axis=0)

    # No motif baseline
    bg_probs = predict(model, bg_seqs).mean(axis=0)

    print(f"  Baselines:")
    print(f"    No motif:    P(FOXA1)={bg_probs[0]:.3f}, P(HNF4A)={bg_probs[1]:.3f}")
    print(f"    FOXA1 only:  P(FOXA1)={foxa1_only_probs[0]:.3f}, P(HNF4A)={foxa1_only_probs[1]:.3f}")
    print(f"    HNF4A only:  P(FOXA1)={hnf4a_only_probs[0]:.3f}, P(HNF4A)={hnf4a_only_probs[1]:.3f}")

    # Scan: FOXA1 at center, HNF4A at center + d
    # Measure how HNF4A's presence affects FOXA1 prediction and vice versa
    results = {
        'foxa1_anchor': {'spacings': [], 'foxa1_prob': [], 'hnf4a_prob': []},
        'hnf4a_anchor': {'spacings': [], 'foxa1_prob': [], 'hnf4a_prob': []},
    }

    for d in spacings:
        # Experiment 1: FOXA1 anchored at center, HNF4A at center + d
        pair_seqs = []
        for bg in bg_seqs:
            s = implant(bg, FOXA1_MOTIF, center - len(FOXA1_MOTIF) // 2)
            if s:
                s = implant(s, HNF4A_MOTIF, center + d - len(HNF4A_MOTIF) // 2)
            if s:
                pair_seqs.append(s)

        if pair_seqs:
            probs = predict(model, pair_seqs).mean(axis=0)
            results['foxa1_anchor']['spacings'].append(d)
            results['foxa1_anchor']['foxa1_prob'].append(probs[0])
            results['foxa1_anchor']['hnf4a_prob'].append(probs[1])

        # Experiment 2: HNF4A anchored at center, FOXA1 at center + d
        pair_seqs2 = []
        for bg in bg_seqs:
            s = implant(bg, HNF4A_MOTIF, center - len(HNF4A_MOTIF) // 2)
            if s:
                s = implant(s, FOXA1_MOTIF, center + d - len(FOXA1_MOTIF) // 2)
            if s:
                pair_seqs2.append(s)

        if pair_seqs2:
            probs2 = predict(model, pair_seqs2).mean(axis=0)
            results['hnf4a_anchor']['spacings'].append(d)
            results['hnf4a_anchor']['foxa1_prob'].append(probs2[0])
            results['hnf4a_anchor']['hnf4a_prob'].append(probs2[1])

    # Convert to arrays
    for key in results:
        for subkey in results[key]:
            results[key][subkey] = np.array(results[key][subkey])

    return results, {
        'bg': bg_probs,
        'foxa1_only': foxa1_only_probs,
        'hnf4a_only': hnf4a_only_probs,
    }


def plot_cooperativity(results, baselines, save_path):
    """Plot the cooperativity result — the key figure."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top left: FOXA1 anchored — effect on HNF4A binding
    ax = axes[0, 0]
    sp = results['foxa1_anchor']['spacings']
    hnf4a_probs = results['foxa1_anchor']['hnf4a_prob']
    hnf4a_baseline = baselines['hnf4a_only'][1]
    bg_hnf4a = baselines['bg'][1]

    ax.plot(sp, hnf4a_probs, 'o-', color='#ff7f0e', markersize=3, linewidth=1.5,
            label='P(HNF4A) with FOXA1 nearby')
    ax.axhline(y=hnf4a_baseline, color='#ff7f0e', linestyle='--', alpha=0.5,
               label=f'HNF4A alone ({hnf4a_baseline:.3f})')
    ax.axhline(y=bg_hnf4a, color='gray', linestyle=':', alpha=0.5,
               label=f'No motif ({bg_hnf4a:.3f})')
    ax.set_xlabel('FOXA1–HNF4A distance (bp)')
    ax.set_ylabel('P(HNF4A bound)')
    ax.set_title('Does nearby FOXA1 boost HNF4A binding?')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Top right: FOXA1 anchored — effect on FOXA1 itself
    ax = axes[0, 1]
    foxa1_probs = results['foxa1_anchor']['foxa1_prob']
    foxa1_baseline = baselines['foxa1_only'][0]

    ax.plot(sp, foxa1_probs, 'o-', color='#1f77b4', markersize=3, linewidth=1.5,
            label='P(FOXA1) with HNF4A nearby')
    ax.axhline(y=foxa1_baseline, color='#1f77b4', linestyle='--', alpha=0.5,
               label=f'FOXA1 alone ({foxa1_baseline:.3f})')
    ax.set_xlabel('FOXA1–HNF4A distance (bp)')
    ax.set_ylabel('P(FOXA1 bound)')
    ax.set_title('Does nearby HNF4A boost FOXA1 binding?')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Bottom left: HNF4A anchored — effect on FOXA1
    ax = axes[1, 0]
    sp2 = results['hnf4a_anchor']['spacings']
    foxa1_probs2 = results['hnf4a_anchor']['foxa1_prob']

    ax.plot(sp2, foxa1_probs2, 'o-', color='#1f77b4', markersize=3, linewidth=1.5,
            label='P(FOXA1) with HNF4A nearby')
    ax.axhline(y=baselines['foxa1_only'][0], color='#1f77b4', linestyle='--', alpha=0.5,
               label=f'FOXA1 alone ({baselines["foxa1_only"][0]:.3f})')
    ax.axhline(y=baselines['bg'][0], color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('HNF4A–FOXA1 distance (bp)')
    ax.set_ylabel('P(FOXA1 bound)')
    ax.set_title('HNF4A anchored: effect on FOXA1')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Bottom right: Cooperativity delta
    ax = axes[1, 1]
    delta_hnf4a = results['foxa1_anchor']['hnf4a_prob'] - baselines['hnf4a_only'][1]
    delta_foxa1 = results['hnf4a_anchor']['foxa1_prob'] - baselines['foxa1_only'][0]

    ax.plot(sp, delta_hnf4a, 'o-', color='#ff7f0e', markersize=3, linewidth=1.5,
            label='ΔP(HNF4A) from FOXA1')
    ax.plot(sp2, delta_foxa1, 'o-', color='#1f77b4', markersize=3, linewidth=1.5,
            label='ΔP(FOXA1) from HNF4A')
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.axvline(x=148, color='green', linestyle='--', alpha=0.4, label='CB median (148bp)')
    ax.set_xlabel('Inter-motif distance (bp)')
    ax.set_ylabel('ΔP(binding) vs single motif')
    ax.set_title('Cooperativity: binding boost from partner motif')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle('In Silico Cooperativity Analysis\n'
                 'FOXA1–HNF4A binding interaction as a function of spacing',
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def directionality_test(model, n_bg=100, seed=42):
    """Test if cooperativity is directional (like BPNet's Oct4-Sox2).

    Does FOXA1 help HNF4A more than HNF4A helps FOXA1?
    """
    bg_seqs = [generate_background(seed=seed + i) for i in range(n_bg)]
    center = 500
    distances = [20, 50, 100, 150, 200]

    results = {'distance': [], 'delta_foxa1': [], 'delta_hnf4a': []}

    # Baselines
    f_alone = [implant(bg, FOXA1_MOTIF, center - len(FOXA1_MOTIF) // 2) for bg in bg_seqs]
    h_alone = [implant(bg, HNF4A_MOTIF, center - len(HNF4A_MOTIF) // 2) for bg in bg_seqs]
    f_alone = [s for s in f_alone if s]
    h_alone = [s for s in h_alone if s]
    p_f_alone = predict(model, f_alone).mean(axis=0)
    p_h_alone = predict(model, h_alone).mean(axis=0)

    for d in distances:
        # Both motifs: FOXA1 left, HNF4A right
        pairs = []
        for bg in bg_seqs:
            s = implant(bg, FOXA1_MOTIF, center - d // 2 - len(FOXA1_MOTIF) // 2)
            if s:
                s = implant(s, HNF4A_MOTIF, center + d // 2 - len(HNF4A_MOTIF) // 2)
            if s:
                pairs.append(s)

        if pairs:
            p_both = predict(model, pairs).mean(axis=0)
            results['distance'].append(d)
            results['delta_foxa1'].append(p_both[0] - p_f_alone[0])
            results['delta_hnf4a'].append(p_both[1] - p_h_alone[1])

    return results


def run_cooperativity(model_path, output_dir, fig_dir):
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("COOPERATIVITY ANALYSIS")
    print("=" * 60)

    print("\n1. Loading model...")
    checkpoint = torch.load(model_path, weights_only=False, map_location='cpu')
    model = BindingCNN()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print("\n2. Spacing-dependent cooperativity scan...")
    results, baselines = cooperativity_spacing_scan(model, n_bg=100)
    plot_cooperativity(results, baselines,
                       os.path.join(fig_dir, 'binding_cooperativity.png'))

    # Save raw data
    np.savez(os.path.join(output_dir, 'cooperativity_scan.npz'),
             foxa1_anchor_spacings=results['foxa1_anchor']['spacings'],
             foxa1_anchor_foxa1_prob=results['foxa1_anchor']['foxa1_prob'],
             foxa1_anchor_hnf4a_prob=results['foxa1_anchor']['hnf4a_prob'],
             hnf4a_anchor_spacings=results['hnf4a_anchor']['spacings'],
             hnf4a_anchor_foxa1_prob=results['hnf4a_anchor']['foxa1_prob'],
             hnf4a_anchor_hnf4a_prob=results['hnf4a_anchor']['hnf4a_prob'],
             baselines_bg=baselines['bg'],
             baselines_foxa1_only=baselines['foxa1_only'],
             baselines_hnf4a_only=baselines['hnf4a_only'])

    print("\n3. Directionality test...")
    dir_results = directionality_test(model)
    print(f"  Distance | ΔP(FOXA1) | ΔP(HNF4A)")
    print(f"  ---------|-----------|----------")
    for i, d in enumerate(dir_results['distance']):
        df = dir_results['delta_foxa1'][i]
        dh = dir_results['delta_hnf4a'][i]
        print(f"  {d:5d} bp  | {df:+.4f}   | {dh:+.4f}")

    summary = {
        'baselines': {
            'no_motif': baselines['bg'].tolist(),
            'foxa1_only': baselines['foxa1_only'].tolist(),
            'hnf4a_only': baselines['hnf4a_only'].tolist(),
        },
        'directionality': {
            'distances': dir_results['distance'],
            'delta_foxa1': dir_results['delta_foxa1'],
            'delta_hnf4a': dir_results['delta_hnf4a'],
        },
        'foxa1_motif': FOXA1_MOTIF,
        'hnf4a_motif': HNF4A_MOTIF,
    }
    with open(os.path.join(output_dir, 'cooperativity_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=lambda x: x.tolist()
                  if hasattr(x, 'tolist') else x)

    print("\nCooperativity analysis complete!")


if __name__ == '__main__':
    model_path = os.path.join(PROJECT_ROOT, 'results', 'deep_learning',
                              'binding', 'binding_model.pt')
    output_dir = os.path.join(PROJECT_ROOT, 'results', 'deep_learning', 'binding')
    fig_dir = os.path.join(PROJECT_ROOT, 'figures')

    run_cooperativity(model_path, output_dir, fig_dir)
