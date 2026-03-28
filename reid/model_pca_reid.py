"""
model_pca.py
------------
Non-deep-learning method for Q1 — Person Re-Identification.

PCA is used to learn a compact subspace from the training set,
retaining 95% of the explained variance.  Gallery and probe images
are projected into this subspace, and matching is performed using
L1 distance between projected vectors.

Rationale for PCA over LDA:
  PCA is unsupervised and makes no assumptions about within-class
  distributions, making it a natural first baseline.  LDA would
  require enough per-class samples to estimate within-class scatter
  reliably — with 300 identities and varying numbers of images per
  identity this can be unstable.
"""

import time
import numpy as np
from sklearn.decomposition import PCA

from utils_reid import get_ranked_histogram, ranked_hist_to_cmc


def train_pca(train_vec: np.ndarray, variance_threshold: float = 0.95) -> PCA:
    """
    Fit a PCA model on vectorised training images.

    Parameters
    ----------
    train_vec : np.ndarray
        Vectorised training images, shape (N, D).
    variance_threshold : float
        Fraction of variance to retain (default 0.95).

    Returns
    -------
    sklearn.decomposition.PCA
        Fitted PCA model.
    """
    pca = PCA(n_components=variance_threshold)
    pca.fit(train_vec)
    print(f"  PCA components retained: {pca.n_components_} "
          f"({variance_threshold*100:.0f}% variance)")
    return pca


def run_pca_reid(splits: dict) -> dict:
    """
    Train PCA on the training split and evaluate on gallery/probe.

    Parameters
    ----------
    splits : dict
        Output of preprocess_reid.build_splits().

    Returns
    -------
    dict with keys: cmc, top1, top5, top10, time_s
    """
    print("[PCA] Training...")
    t_start = time.time()

    pca         = train_pca(splits['train_vec'])
    gallery_pca = pca.transform(splits['gallery_vec'])
    probe_pca   = pca.transform(splits['probe_vec'])

    elapsed = time.time() - t_start

    ranked_hist = get_ranked_histogram(
        gallery_pca, splits['gallery_Y'],
        probe_pca,   splits['probe_Y'],
    )
    cmc = ranked_hist_to_cmc(ranked_hist)

    results = {
        'cmc':    cmc,
        'top1':   cmc[0],
        'top5':   cmc[4],
        'top10':  cmc[9],
        'time_s': elapsed,
    }

    print(f"  Top-1:  {results['top1']:.4f}")
    print(f"  Top-5:  {results['top5']:.4f}")
    print(f"  Top-10: {results['top10']:.4f}")
    print(f"  Time:   {elapsed:.2f}s")

    return results
