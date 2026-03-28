"""
model_siamese.py
----------------
Deep learning method for Q1 — Person Re-Identification.

A Siamese network with a shared CNN backbone is trained using
contrastive loss.  The backbone maps each input image to a
64-dimensional embedding; at inference time, gallery and probe
embeddings are extracted and matched using L1 distance.

Architecture:
    Conv(32) → BN → ReLU → MaxPool
    Conv(64) → BN → ReLU → MaxPool
    Conv(64) → BN → ReLU
    GlobalAvgPool → Dense(128, relu) → Dense(64)

Contrastive loss drives same-identity embeddings close together
and pushes different-identity embeddings beyond a margin of 1.
"""

import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, ReLU,
    MaxPooling2D, GlobalAveragePooling2D, Dense, Lambda,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from utils_reid import get_ranked_histogram, ranked_hist_to_cmc, get_siamese_pairs


# ── Loss & Distance ───────────────────────────────────────────────────────────
def euclidean_distance(vectors):
    """Compute Euclidean distance between two embedding vectors."""
    a, b = vectors
    dist = tf.reduce_sum(tf.square(a - b), axis=1, keepdims=True)
    return tf.sqrt(tf.maximum(dist, tf.keras.backend.epsilon()))


def contrastive_loss(y_true, y_pred, margin: float = 1.0):
    """
    Contrastive loss (Hadsell et al., 2006).

    For matched pairs (y=1): minimise distance.
    For non-matched pairs (y=0): penalise distances below margin.
    """
    sq   = tf.square(y_pred)
    mg   = tf.square(tf.maximum(margin - y_pred, 0))
    return tf.reduce_mean(tf.cast(y_true, tf.float32) * sq +
                          (1 - tf.cast(y_true, tf.float32)) * mg)


# ── Architecture ──────────────────────────────────────────────────────────────
def build_embedding_cnn(input_shape: tuple) -> Model:
    """
    Shared CNN backbone that maps an image to a 64-d embedding vector.

    Parameters
    ----------
    input_shape : tuple
        (H, W, C) of each input image.

    Returns
    -------
    keras.Model
    """
    inp = Input(shape=input_shape)

    x = Conv2D(32, (3, 3), padding='same')(inp)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    out = Dense(64)(x)

    return Model(inputs=inp, outputs=out, name='embedding_cnn')


def build_siamese(input_shape: tuple):
    """
    Wrap the shared backbone in a Siamese network for contrastive training.

    Returns
    -------
    (backbone, siamese_model)
    """
    backbone = build_embedding_cnn(input_shape)

    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)
    emb_a   = backbone(input_a)
    emb_b   = backbone(input_b)
    dist    = Lambda(euclidean_distance)([emb_a, emb_b])

    siamese = Model(inputs=[input_a, input_b], outputs=dist, name='siamese')
    siamese.compile(optimizer=Adam(learning_rate=1e-4), loss=contrastive_loss)

    return backbone, siamese


# ── Training & Evaluation ─────────────────────────────────────────────────────
def run_siamese_reid(splits: dict, n_pairs: int = 2000,
                     epochs: int = 10, batch_size: int = 32) -> dict:
    """
    Train a Siamese CNN and evaluate on the gallery/probe test set.

    Parameters
    ----------
    splits : dict
        Output of preprocess_reid.build_splits().
    n_pairs : int
        Number of positive (and negative) pairs to sample per epoch.
    epochs : int
        Training epochs.
    batch_size : int
        Mini-batch size.

    Returns
    -------
    dict with keys: cmc, top1, top5, top10, time_s, backbone
    """
    input_shape = splits['train_X'].shape[1:]   # (H, W, C)
    backbone, siamese = build_siamese(input_shape)
    siamese.summary()

    print("\n[Siamese CNN] Generating training pairs...")
    (pair_a, pair_b), pair_labels = get_siamese_pairs(
        splits['train_X'], splits['train_Y'], n_pairs
    )

    print("[Siamese CNN] Training...")
    t_start = time.time()
    siamese.fit(
        [pair_a, pair_b], pair_labels,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=1,
    )
    elapsed = time.time() - t_start

    # Extract embeddings via the shared backbone
    gallery_emb = backbone.predict(splits['gallery_X'], verbose=0)
    probe_emb   = backbone.predict(splits['probe_X'],   verbose=0)

    ranked_hist = get_ranked_histogram(
        gallery_emb, splits['gallery_Y'],
        probe_emb,   splits['probe_Y'],
    )
    cmc = ranked_hist_to_cmc(ranked_hist)

    results = {
        'cmc':      cmc,
        'top1':     cmc[0],
        'top5':     cmc[4],
        'top10':    cmc[9],
        'time_s':   elapsed,
        'backbone': backbone,
    }

    print(f"  Top-1:  {results['top1']:.4f}")
    print(f"  Top-5:  {results['top5']:.4f}")
    print(f"  Top-10: {results['top10']:.4f}")
    print(f"  Time:   {elapsed:.2f}s")

    return results
