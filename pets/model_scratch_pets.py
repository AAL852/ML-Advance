"""
model_scratch.py
----------------
From-scratch multi-task CNN for Q2 — Multi-Task Learning & Fine Tuning.

Architecture:
  Shared encoder:
      Conv(4) → MaxPool → Conv(8) → MaxPool → Conv(16) → MaxPool

  Classification head:
      GlobalAvgPool → Dense(64, relu) → Dropout(0.3) → Dense(37, softmax)

  Segmentation head (decoder):
      ConvTranspose(16) → ConvTranspose(8) → ConvTranspose(4) → Conv(1, sigmoid)

Design rationale:
  - Progressively doubling filters (4→8→16) captures features from
    coarse to fine while keeping parameter count low for limited compute.
  - Global Average Pooling in the classification head avoids overfitting
    on the small dataset.
  - The segmentation decoder mirrors the encoder depth with transposed
    convolutions to restore spatial resolution for pixel-level prediction.
"""

import keras
from keras.layers import (
    Input, Conv2D, MaxPooling2D, Conv2DTranspose,
    GlobalAveragePooling2D, Dense, Dropout,
)
from keras.models import Model
from keras.optimizers import Adam


def build_scratch_model(image_size: int, n_classes: int) -> Model:
    """
    Build the from-scratch multi-task CNN.

    Parameters
    ----------
    image_size : int
        Input image resolution (assumed square).
    n_classes : int
        Number of classification classes (37 for Oxford-IIIT Pets).

    Returns
    -------
    Compiled keras.Model with two outputs:
        'classification' — (N, n_classes) softmax probabilities
        'segmentation'   — (N, H, W, 1) sigmoid foreground probability
    """
    inp = Input(shape=(image_size, image_size, 3))

    # ── Shared encoder ────────────────────────────────────────────────────────
    x = Conv2D(4,  (3, 3), activation='relu', padding='same')(inp)
    x = MaxPooling2D((2, 2))(x)                                     # H/2

    x = Conv2D(8,  (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)                                     # H/4

    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    enc = MaxPooling2D((2, 2))(x)                                   # H/8

    # ── Classification head ───────────────────────────────────────────────────
    c = GlobalAveragePooling2D()(enc)
    c = Dense(64, activation='relu')(c)
    c = Dropout(0.3)(c)
    cls_out = Dense(n_classes, activation='softmax', name='classification')(c)

    # ── Segmentation head (decoder) ───────────────────────────────────────────
    s = Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same', activation='relu')(enc)
    s = Conv2DTranspose(8,  (3, 3), strides=(2, 2), padding='same', activation='relu')(s)
    s = Conv2DTranspose(4,  (3, 3), strides=(2, 2), padding='same', activation='relu')(s)
    seg_out = Conv2D(1, (1, 1), activation='sigmoid', name='segmentation')(s)

    model = Model(inputs=inp, outputs=[cls_out, seg_out], name='scratch_multitask')
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss={
            'classification': 'sparse_categorical_crossentropy',
            'segmentation':   'binary_crossentropy',
        },
        metrics={
            'classification': 'accuracy',
            'segmentation':   'accuracy',
        },
    )
    return model
