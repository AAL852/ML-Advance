"""
model_mobilenet.py
------------------
Fine-tuned MobileNetV3Small model for Q2 — Multi-Task Learning & Fine Tuning.

Strategy:
  1. Load MobileNetV3Small pre-trained on ImageNet with frozen weights.
  2. Attach a classification head and a segmentation decoder.
  3. Train with backbone frozen (transfer learning phase).
  4. Optionally unfreeze the top N backbone layers and fine-tune at a
     lower learning rate to adapt ImageNet features to the pets domain.

Design rationale:
  - MobileNetV3Small is designed for resource-constrained environments
    with ~2.5M parameters, suitable for limited compute budgets.
  - Pre-trained ImageNet weights provide strong low-level features
    (edges, textures) that transfer well to pet image segmentation.
  - Freezing the backbone initially prevents catastrophic forgetting
    of ImageNet features before the heads are warm.
"""

import keras
from keras.layers import (
    Input, Conv2D, Conv2DTranspose, UpSampling2D,
    GlobalAveragePooling2D, Dense, Dropout,
)
from keras.models import Model
from keras.optimizers import Adam


def build_mobilenet_model(image_size: int, n_classes: int,
                           freeze_backbone: bool = True):
    """
    Build the MobileNetV3Small-based multi-task model.

    Parameters
    ----------
    image_size : int
        Input image resolution (assumed square). MobileNetV3Small was
        pre-trained at 224×224; smaller inputs reduce compute at the
        cost of some feature quality.
    n_classes : int
        Number of classification classes.
    freeze_backbone : bool
        If True, the MobileNetV3Small backbone is frozen (default).
        Set False for the fine-tuning phase.

    Returns
    -------
    (model, backbone) — compiled keras.Model and the backbone layer,
    so the caller can selectively unfreeze layers later.
    """
    backbone = keras.applications.MobileNetV3Small(
        input_shape=(image_size, image_size, 3),
        include_top=False,
        include_preprocessing=False,   # preprocessing handled in data loader
    )
    backbone.trainable = not freeze_backbone

    inp          = Input(shape=(image_size, image_size, 3))
    features     = backbone(inp, training=False)   # (N, h, w, C)

    # ── Classification head ───────────────────────────────────────────────────
    c = GlobalAveragePooling2D()(features)
    c = Dense(128, activation='relu')(c)
    c = Dropout(0.3)(c)
    cls_out = Dense(n_classes, activation='softmax', name='classification')(c)

    # ── Segmentation head ─────────────────────────────────────────────────────
    # Upsample MobileNet output (typically image_size/32 or similar)
    # back to the input resolution through a series of transposed convolutions.
    s = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', activation='relu')(features)
    s = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', activation='relu')(s)
    s = Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same', activation='relu')(s)
    s = UpSampling2D(size=(2, 2))(s)
    seg_out = Conv2D(1, (1, 1), activation='sigmoid', name='segmentation')(s)

    model = Model(inputs=inp, outputs=[cls_out, seg_out], name='mobilenet_multitask')
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss={
            'classification': 'sparse_categorical_crossentropy',
            'segmentation':   'binary_crossentropy',
        },
        metrics={
            'classification': 'accuracy',
            'segmentation':   'accuracy',
        },
    )
    return model, backbone


def unfreeze_top_layers(model, backbone, n_layers: int = 20,
                         learning_rate: float = 1e-5):
    """
    Unfreeze the top n_layers of the backbone for fine-tuning.
    Recompiles the model with a lower learning rate to avoid
    disrupting pre-trained weights.

    Parameters
    ----------
    model : keras.Model
        The full multi-task model.
    backbone : keras.Model
        The MobileNetV3Small backbone (returned by build_mobilenet_model).
    n_layers : int
        Number of top backbone layers to unfreeze.
    learning_rate : float
        Reduced learning rate for fine-tuning phase.
    """
    backbone.trainable = True
    for layer in backbone.layers[:-n_layers]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss={
            'classification': 'sparse_categorical_crossentropy',
            'segmentation':   'binary_crossentropy',
        },
        metrics={
            'classification': 'accuracy',
            'segmentation':   'accuracy',
        },
    )
    trainable = sum(1 for l in backbone.layers if l.trainable)
    print(f"  Fine-tuning: {trainable} backbone layers unfrozen (lr={learning_rate})")
