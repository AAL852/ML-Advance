"""
main_reid.py
------------
Entry point for Q1 — Person Re-Identification.

Runs the full pipeline:
  1. Load and pre-process Market-1501 data
  2. Train and evaluate PCA (non-DL method)
  3. Train and evaluate Siamese CNN (DL method)
  4. Generate comparison table and CMC curve plot

Usage:
    python main_reid.py
    python main_reid.py --data ../Data/Q1 --pairs 3000 --epochs 15
"""

import argparse
from utils_reid import load_data, plot_images
from preprocess_reid import build_splits
from model_pca import run_pca_reid
from model_siamese import run_siamese_reid
from evaluate_reid import summarise


def parse_args():
    parser = argparse.ArgumentParser(description='CAB420 1B Q1 — Person Re-ID')
    parser.add_argument('--data',   default='../Data/Q1', help='Path to Q1 data directory')
    parser.add_argument('--pairs',  default=2000, type=int, help='Training pairs for Siamese network')
    parser.add_argument('--epochs', default=10,   type=int, help='Training epochs for Siamese network')
    parser.add_argument('--output', default='outputs', help='Directory to save results')
    return parser.parse_args()


def main():
    args = parse_args()

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print(f"Loading data from {args.data}...")
    train_X, train_Y, gallery_X, gallery_Y, probe_X, probe_Y = load_data(args.data)
    print(f"  Train: {train_X.shape} | Gallery: {gallery_X.shape} | Probe: {probe_X.shape}")

    plot_images(gallery_X, gallery_Y, title='Sample gallery images (raw)')

    # ── 2. Pre-process ────────────────────────────────────────────────────────
    print("\nPre-processing (grayscale conversion, keep 128×64)...")
    splits = build_splits(train_X, train_Y, gallery_X, gallery_Y, probe_X, probe_Y)
    plot_images(splits['gallery_X'], splits['gallery_Y'],
                title='Sample gallery images (pre-processed)')

    # ── 3. PCA ────────────────────────────────────────────────────────────────
    pca_results = run_pca_reid(splits)

    # ── 4. Siamese CNN ────────────────────────────────────────────────────────
    dl_results = run_siamese_reid(
        splits, n_pairs=args.pairs, epochs=args.epochs
    )

    # ── 5. Evaluation ─────────────────────────────────────────────────────────
    summarise(pca_results, dl_results, save_dir=args.output)


if __name__ == '__main__':
    main()
