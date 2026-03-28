"""
preprocess_reid.py
------------------
Pre-processing pipeline for Q1 — Person Re-Identification.

Both the non-DL (PCA) and DL (Siamese) models use identical
pre-processing for a fair comparison:
  - Images are kept at their original 128×64 resolution.
  - Converted to grayscale to reduce input dimensionality.
  - Pixel values remain in [0, 1] (loaded as float in utils_reid).
"""

import numpy as np
from utils_reid import resize, convert_to_grayscale, vectorise


def preprocess(X: np.ndarray, target_size: tuple = (128, 64)) -> np.ndarray:
    """
    Resize and convert a batch of images to grayscale.

    Parameters
    ----------
    X : np.ndarray
        Raw images of shape (N, H, W, 3).
    target_size : tuple
        (height, width) to resize to. Defaults to (128, 64).

    Returns
    -------
    np.ndarray
        Processed images of shape (N, H, W, 1), values in [0, 1].
    """
    X = resize(X, target_size)
    X = convert_to_grayscale(X)
    return X


def build_splits(train_X, train_Y, gallery_X, gallery_Y, probe_X, probe_Y):
    """
    Apply pre-processing to all three data splits and return both
    image arrays (for the DL method) and vectorised arrays (for PCA).

    Returns
    -------
    dict with keys:
        train_X, train_Y       — pre-processed training images + labels
        gallery_X, gallery_Y   — pre-processed gallery images + labels
        probe_X, probe_Y       — pre-processed probe images + labels
        train_vec              — vectorised training images
        gallery_vec            — vectorised gallery images
        probe_vec              — vectorised probe images
    """
    train_X   = preprocess(train_X)
    gallery_X = preprocess(gallery_X)
    probe_X   = preprocess(probe_X)

    return {
        'train_X':   train_X,   'train_Y':   train_Y,
        'gallery_X': gallery_X, 'gallery_Y': gallery_Y,
        'probe_X':   probe_X,   'probe_Y':   probe_Y,
        'train_vec':   vectorise(train_X),
        'gallery_vec': vectorise(gallery_X),
        'probe_vec':   vectorise(probe_X),
    }
