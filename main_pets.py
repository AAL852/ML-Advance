"""
main_pets.py
------------
Entry point for Q2 — Multi-Task Learning & Fine Tuning.

Runs the full pipeline:
  1. Load and visualise Oxford-IIIT Pets data
  2. Train the from-scratch multi-task CNN
  3. Train the fine-tuned MobileNetV3Small (frozen → unfrozen)
  4. Evaluate and compare both models
  5. Save plots to outputs/

Usage:
    python main_pets.py
    python main_pets.py --size 128 --batch 32 --epochs 20 --finetune-epochs 10
"""

import argparse
import os

from utils_pets import load_oxford_pets, plot_samples
from model_scratch import build_scratch_model
from model_mobilenet import build_mobilenet_model, unfreeze_top_layers
from evaluate_pets import (
    evaluate_model, print_summary,
    plot_training_curves, plot_segmentation_samples,
)

N_CLASSES = 37


def parse_args():
    parser = argparse.ArgumentParser(description='CAB420 1B Q2 — Multi-Task Learning')
    parser.add_argument('--size',            default=128, type=int, help='Input image size')
    parser.add_argument('--batch',           default=32,  type=int, help='Batch size')
    parser.add_argument('--epochs',          default=20,  type=int, help='Initial training epochs')
    parser.add_argument('--finetune-epochs', default=10,  type=int, help='Fine-tuning epochs (MobileNet)')
    parser.add_argument('--output',          default='outputs', help='Directory to save results')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print(f"Loading Oxford-IIIT Pets (image_size={args.size}, batch={args.batch})...")
    train_ds = load_oxford_pets('train', image_size=args.size,
                                batch_size=args.batch, shuffle=True, augment=True)
    test_ds  = load_oxford_pets('test',  image_size=args.size,
                                batch_size=args.batch, shuffle=False, augment=False)

    plot_samples(train_ds, title='Fig 1: Sample training images and segmentation masks')

    # ── 2. From-scratch CNN ───────────────────────────────────────────────────
    print("\n[From-Scratch] Building and training...")
    scratch_model = build_scratch_model(args.size, N_CLASSES)
    scratch_model.summary()

    history_scratch = scratch_model.fit(
        train_ds, epochs=args.epochs, validation_data=test_ds, verbose=1,
    )

    # ── 3. MobileNetV3Small — transfer learning phase ─────────────────────────
    print("\n[MobileNet] Building with frozen backbone...")
    mobile_model, backbone = build_mobilenet_model(
        args.size, N_CLASSES, freeze_backbone=True
    )
    mobile_model.summary()

    history_mobile = mobile_model.fit(
        train_ds, epochs=args.epochs, validation_data=test_ds, verbose=1,
    )

    # ── 4. MobileNetV3Small — fine-tuning phase ───────────────────────────────
    print("\n[MobileNet] Unfreezing top backbone layers for fine-tuning...")
    unfreeze_top_layers(mobile_model, backbone, n_layers=20, learning_rate=1e-5)

    history_finetune = mobile_model.fit(
        train_ds, epochs=args.finetune_epochs, validation_data=test_ds, verbose=1,
    )

    # ── 5. Evaluation ─────────────────────────────────────────────────────────
    scratch_res = evaluate_model(scratch_model, test_ds, 'From-Scratch CNN')
    mobile_res  = evaluate_model(mobile_model,  test_ds, 'Fine-Tuned MobileNet')
    print_summary(scratch_res, mobile_res)

    plot_training_curves(
        {'From-Scratch': history_scratch, 'MobileNet': history_mobile},
        save_path=os.path.join(args.output, 'q2_training_curves.png'),
    )

    plot_segmentation_samples(
        scratch_model, mobile_model, test_ds,
        save_path=os.path.join(args.output, 'q2_segmentation_samples.png'),
    )


if __name__ == '__main__':
    main()
