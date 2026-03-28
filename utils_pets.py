"""
utils_pets.py
-------------
Shared utilities for Q2 — Multi-Task Learning & Fine Tuning.

Handles Oxford-IIIT Pets dataset loading via tensorflow_datasets,
segmentation mask preprocessing, data augmentation, and
evaluation helpers for the dual classification + segmentation task.
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds


# ── Mask Preprocessing ────────────────────────────────────────────────────────
def preprocess_segmentation_mask(mask):
    """
    Convert the 3-class Oxford mask to binary foreground/background.

    Original labels:
        1 = pet edge / outline
        2 = background
        3 = foreground (pet body)

    Output: 0 = background, 1 = foreground (edges merged with foreground).
    """
    mask = tf.cast(mask, tf.float32)
    mask = tf.abs(mask - 2.0)           # edge→1, background→0, foreground→1
    return tf.clip_by_value(mask, 0, 1)


# ── Augmentation ──────────────────────────────────────────────────────────────
def flip_lr(image, outputs):
    """Random horizontal flip applied consistently to image and mask."""
    if tf.random.uniform(()) > 0.5:
        image   = tf.image.flip_left_right(image)
        seg     = tf.image.flip_left_right(outputs[1])
        outputs = (outputs[0], seg)
    return image, outputs


# ── Data Loader ───────────────────────────────────────────────────────────────
def load_oxford_pets(split: str, image_size: int = 128, batch_size: int = 32,
                     shuffle: bool = True, augment: bool = True):
    """
    Load the Oxford-IIIT Pets dataset for simultaneous classification
    and semantic segmentation.

    Parameters
    ----------
    split : str
        'train' or 'test'.
    image_size : int
        Images are resized to (image_size × image_size).
    batch_size : int
        Mini-batch size.
    shuffle : bool
        Shuffle training data (forced False for test split).
    augment : bool
        Apply random horizontal flip (forced False for test split).

    Returns
    -------
    tf.data.Dataset yielding (image, (class_label, seg_mask)).
    """
    if split == 'test':
        shuffle = False
        augment = False

    ds = tfds.load('oxford_iiit_pet', split=split, as_supervised=False)

    def format_sample(sample):
        image = tf.image.resize(sample['image'], [image_size, image_size])
        image = tf.cast(image, tf.float32) / 127.5 - 1.0   # normalise to [-1, 1]
        label = sample['label']
        mask  = tf.image.resize(
            preprocess_segmentation_mask(sample['segmentation_mask']),
            [image_size, image_size],
        )
        return image, (label, mask)

    ds = ds.map(format_sample, num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        ds = ds.map(flip_lr, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(buffer_size=1000)

    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# ── Display Helpers ───────────────────────────────────────────────────────────
def unprocess_image(image: np.ndarray) -> np.ndarray:
    """Reverse [-1, 1] normalisation for display."""
    return np.clip((image + 1.0) * 127.5, 0, 255).astype('uint8')


def plot_samples(ds, n: int = 4, title: str = ''):
    """Display n sample images alongside their segmentation masks."""
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))
    if title:
        fig.suptitle(title)
    for images, (labels, masks) in ds.take(1):
        for i in range(n):
            axes[0, i].imshow(unprocess_image(images[i].numpy()))
            axes[0, i].set_title(f'Class {labels[i].numpy()}', fontsize=9)
            axes[0, i].axis('off')
            axes[1, i].imshow(masks[i, :, :, 0].numpy(), cmap='gray')
            axes[1, i].axis('off')
    axes[0, 0].set_ylabel('Image',  fontsize=9)
    axes[1, 0].set_ylabel('Mask',   fontsize=9)
    plt.tight_layout()
    plt.show()
