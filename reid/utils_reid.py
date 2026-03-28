"""
utils_reid.py
-------------
Shared utilities for Q1 — Person Re-Identification.
Covers data loading, image helpers, pair/triplet generation,
and CMC curve computation and plotting.
"""

import os
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf


# ── Data Loading ──────────────────────────────────────────────────────────────
def get_subject_id(filename: str) -> int:
    """Extract the 4-digit subject ID from a Market-1501 filename."""
    return int(os.path.basename(filename)[0:4])


def load_directory(base_path: str):
    """Load all .jpg images in a directory with subject ID labels."""
    files = sorted(glob.glob(os.path.join(base_path, '*.jpg')))
    X = np.array([
        cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) / 255.0
        for f in files
    ])
    Y = np.array([get_subject_id(f) for f in files])
    return X, Y


def load_data(base_path: str):
    """
    Load the Market-1501 split into training, gallery, and probe sets.

    Expected directory layout:
        base_path/
            Training/       ← ~5,933 images, 300 identities
            Testing/
                Gallery/    ← 301 images, one per identity
                Probe/      ← 301 images, one per identity
    """
    train_X,   train_Y   = load_directory(os.path.join(base_path, 'Training'))
    gallery_X, gallery_Y = load_directory(os.path.join(base_path, 'Testing/Gallery'))
    probe_X,   probe_Y   = load_directory(os.path.join(base_path, 'Testing/Probe'))
    return train_X, train_Y, gallery_X, gallery_Y, probe_X, probe_Y


# ── Image Helpers ─────────────────────────────────────────────────────────────
def resize(X: np.ndarray, size: tuple) -> np.ndarray:
    """Resize a batch of images to (height, width)."""
    return tf.image.resize(X, size).numpy()


def convert_to_grayscale(X: np.ndarray) -> np.ndarray:
    """Convert RGB images to single-channel grayscale (keeps channel dim)."""
    return np.mean(X, axis=-1, keepdims=True)


def vectorise(X: np.ndarray) -> np.ndarray:
    """Flatten image arrays into 1D feature vectors."""
    return X.reshape(X.shape[0], -1)


def plot_images(X: np.ndarray, Y: np.ndarray, n: int = 8, title: str = ''):
    """Display n sample images with their subject IDs."""
    fig, axes = plt.subplots(1, n, figsize=(14, 3))
    if title:
        fig.suptitle(title)
    for i, ax in enumerate(axes):
        img = X[i, :, :, 0] if X.shape[-1] == 1 else X[i]
        ax.imshow(img, cmap='gray' if X.shape[-1] == 1 else None)
        ax.set_title(str(Y[i]), fontsize=8)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


# ── Pair / Triplet Generation ─────────────────────────────────────────────────
def get_siamese_pairs(X: np.ndarray, Y: np.ndarray, n_pairs: int):
    """
    Sample n_pairs positive (same identity) and n_pairs negative
    (different identity) pairs for contrastive training.

    Returns ([img_a, img_b], labels) where label=1 for positive, 0 for negative.
    """
    ids    = np.unique(Y)
    a_list, b_list, labels = [], [], []

    for _ in range(n_pairs):
        # Positive pair
        id_ = random.choice(ids)
        idx = np.where(Y == id_)[0]
        if len(idx) >= 2:
            i, j = random.sample(list(idx), 2)
            a_list.append(X[i])
            b_list.append(X[j])
            labels.append(1)

        # Negative pair
        id_a, id_b = random.sample(list(ids), 2)
        i = random.choice(np.where(Y == id_a)[0])
        j = random.choice(np.where(Y == id_b)[0])
        a_list.append(X[i])
        b_list.append(X[j])
        labels.append(0)

    return [np.array(a_list), np.array(b_list)], np.array(labels)


def get_triplet_data(X: np.ndarray, Y: np.ndarray, n_triplets: int):
    """
    Sample n_triplets (anchor, positive, negative) triplets for triplet loss.
    Returns array of shape (n_triplets, 3, H, W, C).
    """
    ids     = np.unique(Y)
    anchors, positives, negatives = [], [], []

    for _ in range(n_triplets):
        id_pos = random.choice(ids)
        idx    = np.where(Y == id_pos)[0]
        if len(idx) >= 2:
            a_idx, p_idx = random.sample(list(idx), 2)
            id_neg = random.choice([i for i in ids if i != id_pos])
            n_idx  = random.choice(np.where(Y == id_neg)[0])
            anchors.append(X[a_idx])
            positives.append(X[p_idx])
            negatives.append(X[n_idx])

    return np.stack([anchors, positives, negatives], axis=1)


# ── CMC Evaluation ────────────────────────────────────────────────────────────
def get_ranked_histogram(
    gallery_feats: np.ndarray, gallery_ids: np.ndarray,
    probe_feats:   np.ndarray, probe_ids:   np.ndarray,
) -> np.ndarray:
    """
    Build a ranked histogram by matching each probe image against the full
    gallery using L1 distance.  ranked_hist[k] = number of probes whose
    correct match appeared at rank k.
    """
    n_gallery   = len(gallery_ids)
    ranked_hist = np.zeros(n_gallery, dtype=int)

    for i in range(len(probe_ids)):
        dists  = np.sum(np.abs(gallery_feats - probe_feats[i]), axis=1)
        ranked = np.argsort(dists)
        rank   = np.where(gallery_ids[ranked] == probe_ids[i])[0]
        if len(rank) > 0:
            ranked_hist[rank[0]] += 1

    return ranked_hist


def ranked_hist_to_cmc(ranked_hist: np.ndarray) -> np.ndarray:
    """Convert a ranked histogram to a Cumulative Match Characteristic curve."""
    return np.cumsum(ranked_hist) / np.sum(ranked_hist)


def plot_cmc(cmc_curves: dict, save_path: str = None):
    """
    Plot one or more CMC curves on the same axis.

    Parameters
    ----------
    cmc_curves : dict
        Mapping of {label: cmc_array}.
    save_path : str, optional
        If provided, save the figure to this path.
    """
    plt.figure(figsize=(8, 5))
    for label, cmc in cmc_curves.items():
        plt.plot(np.arange(1, len(cmc) + 1), cmc, label=label, linewidth=1.5)
    plt.xlabel('Rank')
    plt.ylabel('Cumulative Match Accuracy')
    plt.title('Fig 1: CMC Curves — PCA vs Siamese CNN')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
