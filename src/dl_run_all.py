#!/usr/bin/env python3
"""
Master script: runs the full deep learning motif grammar pipeline.

Usage:
    python dl_run_all.py              # Run everything
    python dl_run_all.py --skip-modisco  # Skip TF-MoDISco (faster)
    python dl_run_all.py --train-only    # Only train, no interpretability
"""

import os
import sys
import argparse
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

SEQ_DIR = os.path.join(PROJECT_ROOT, 'results', 'sequences')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'results', 'deep_learning')
FIG_DIR = os.path.join(PROJECT_ROOT, 'figures')
MODEL_PATH = os.path.join(OUTPUT_DIR, 'best_model.pt')


def main():
    parser = argparse.ArgumentParser(description='Run deep learning motif grammar pipeline')
    parser.add_argument('--skip-modisco', action='store_true',
                        help='Skip TF-MoDISco analysis')
    parser.add_argument('--train-only', action='store_true',
                        help='Only train model, skip interpretability')
    parser.add_argument('--interpret-only', action='store_true',
                        help='Only run interpretability (model must exist)')
    parser.add_argument('--mutate-only', action='store_true',
                        help='Only run in silico mutagenesis (model must exist)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Max training epochs (default: 100)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size (default: 64)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 1e-3)')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience (default: 15)')
    parser.add_argument('--model-type', choices=['dilated', 'simple'],
                        default='dilated',
                        help='Model architecture (default: dilated)')
    args = parser.parse_args()

    total_start = time.time()

    # ---- STEP 1: Training ----
    if not args.interpret_only and not args.mutate_only:
        from dl_train import train_model
        model, splits, history, results = train_model(
            seq_dir=SEQ_DIR,
            output_dir=OUTPUT_DIR,
            fig_dir=FIG_DIR,
            n_epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=args.patience,
            model_type=args.model_type,
        )

    if args.train_only:
        print(f"\nTotal time: {time.time() - total_start:.1f}s")
        return

    # ---- STEP 2: Interpretability ----
    if not args.mutate_only:
        if not os.path.exists(MODEL_PATH):
            print(f"ERROR: No trained model found at {MODEL_PATH}")
            print("Run training first or provide a model.")
            sys.exit(1)

        from dl_interpret import run_interpretability
        run_interpretability(
            seq_dir=SEQ_DIR,
            model_path=MODEL_PATH,
            output_dir=OUTPUT_DIR,
            fig_dir=FIG_DIR,
            run_modisco_analysis=not args.skip_modisco,
        )

    # ---- STEP 3: In silico mutagenesis ----
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: No trained model found at {MODEL_PATH}")
        sys.exit(1)

    from dl_mutagenesis import run_mutagenesis
    run_mutagenesis(
        model_path=MODEL_PATH,
        output_dir=OUTPUT_DIR,
        fig_dir=FIG_DIR,
    )

    elapsed = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"PIPELINE COMPLETE — Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"{'=' * 60}")
    print(f"Results: {OUTPUT_DIR}")
    print(f"Figures: {FIG_DIR}")


if __name__ == '__main__':
    main()
