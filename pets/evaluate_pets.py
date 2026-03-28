"""
evaluate_pets.py
----------------
Evaluation and visualisation for Q2 — Multi-Task Learning & Fine Tuning.

Computes classification F1 and segmentation IoU for both models,
generates a comparison table, training curve plots, and segmentation
visualisations on test samples.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

from utils_pets import unprocess_image


# ── Per-model Evaluation ──────────────────────────────────────────────────────
def evaluate_model(model, test_ds, model_name: str) -> dict:
    """
    Evaluate a dual-output model on the test set.

    Returns
    -------
    dict with keys: cls_acc, cls_f1, seg_acc, seg_iou
    """
    y_cls_true, y_cls_pred = [], []
    y_seg_true, y_seg_pred = [], []

    for images, (labels, masks) in test_ds:
        cls_preds, seg_preds = model.predict(images, verbose=0)
        y_cls_true.extend(labels.numpy())
        y_cls_pred.extend(np.argmax(cls_preds, axis=1))
        y_seg_true.extend(masks.numpy().flatten())
        y_seg_pred.extend((seg_preds.numpy().flatten() > 0.5).astype(int))

    y_cls_true = np.array(y_cls_true)
    y_cls_pred = np.array(y_cls_pred)
    y_seg_true = np.array(y_seg_true)
    y_seg_pred = np.array(y_seg_pred)

    cls_acc = float(np.mean(y_cls_true == y_cls_pred))
    cls_f1  = f1_score(y_cls_true, y_cls_pred, average='macro', zero_division=0)

    intersection = np.sum((y_seg_pred == 1) & (y_seg_true == 1))
    union        = np.sum((y_seg_pred == 1) | (y_seg_true == 1))
    seg_iou      = float(intersection / (union + 1e-7))
    seg_acc      = float(np.mean(y_seg_true == y_seg_pred))

    print(f"\n{model_name}")
    print(f"  Classification — Accuracy: {cls_acc:.4f} | F1 (macro): {cls_f1:.4f}")
    print(f"  Segmentation   — Pixel Acc: {seg_acc:.4f} | IoU: {seg_iou:.4f}")

    return {
        'cls_acc': cls_acc, 'cls_f1':  cls_f1,
        'seg_acc': seg_acc, 'seg_iou': seg_iou,
    }


# ── Summary Table ─────────────────────────────────────────────────────────────
def print_summary(scratch_res: dict, mobile_res: dict):
    """Print a side-by-side comparison table."""
    df = pd.DataFrame({
        'Model':    ['From-Scratch CNN', 'Fine-Tuned MobileNet'],
        'Cls Acc':  [scratch_res['cls_acc'],  mobile_res['cls_acc']],
        'Cls F1':   [scratch_res['cls_f1'],   mobile_res['cls_f1']],
        'Seg Acc':  [scratch_res['seg_acc'],  mobile_res['seg_acc']],
        'Seg IoU':  [scratch_res['seg_iou'],  mobile_res['seg_iou']],
    })
    print("\nTable 1: Multi-Task Model Comparison")
    print(df.to_string(index=False))


# ── Training Curves ───────────────────────────────────────────────────────────
def plot_training_curves(histories: dict, save_path: str = None):
    """
    Plot classification accuracy and segmentation accuracy training curves
    for one or more models.

    Parameters
    ----------
    histories : dict
        Mapping of {model_name: keras History object}.
    save_path : str, optional
        Path to save the figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for name, h in histories.items():
        h = h.history
        if 'classification_accuracy' in h:
            axes[0].plot(h['classification_accuracy'],     label=f'{name} Train')
            axes[0].plot(h['val_classification_accuracy'], label=f'{name} Val', linestyle='--')
        if 'segmentation_accuracy' in h:
            axes[1].plot(h['segmentation_accuracy'],       label=f'{name} Train')
            axes[1].plot(h['val_segmentation_accuracy'],   label=f'{name} Val', linestyle='--')

    for ax, title in zip(axes, ['Classification Accuracy', 'Segmentation Accuracy']):
        ax.set_xlabel('Epoch')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Fig 2: Training Curves — From-Scratch vs MobileNet')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


# ── Segmentation Visualisation ────────────────────────────────────────────────
def plot_segmentation_samples(scratch_model, mobile_model,
                               test_ds, n: int = 4, save_path: str = None):
    """
    Display n test images alongside predicted segmentation masks
    from both models.
    """
    fig, axes = plt.subplots(3, n, figsize=(3 * n, 9))

    for images, (labels, masks) in test_ds.take(1):
        imgs = images[:n]
        scratch_cls, scratch_seg = scratch_model.predict(imgs, verbose=0)
        mobile_cls,  mobile_seg  = mobile_model.predict(imgs,  verbose=0)

        for i in range(n):
            axes[0, i].imshow(unprocess_image(imgs[i].numpy()))
            axes[0, i].set_title(f'GT class: {labels[i].numpy()}', fontsize=8)
            axes[0, i].axis('off')

            axes[1, i].imshow((scratch_seg[i, :, :, 0] > 0.5).astype(float), cmap='gray')
            axes[1, i].set_title(f'Scratch (pred {np.argmax(scratch_cls[i])})', fontsize=8)
            axes[1, i].axis('off')

            axes[2, i].imshow((mobile_seg[i, :, :, 0] > 0.5).astype(float), cmap='gray')
            axes[2, i].set_title(f'MobileNet (pred {np.argmax(mobile_cls[i])})', fontsize=8)
            axes[2, i].axis('off')

    axes[0, 0].set_ylabel('Input image',    fontsize=8)
    axes[1, 0].set_ylabel('From-scratch',   fontsize=8)
    axes[2, 0].set_ylabel('MobileNet',      fontsize=8)

    fig.suptitle('Fig 3: Segmentation Predictions on Test Samples')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
