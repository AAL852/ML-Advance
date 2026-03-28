# advanced-ml

Advanced computer vision and deep learning — person re-identification using PCA and Siamese networks, and multi-task learning for simultaneous image classification and semantic segmentation.

---

## 📋 Overview

This project tackles two computer vision problems that go well beyond standard classification benchmarks. Both problems involve learning compact, meaningful representations from image data under realistic constraints — limited training samples, class imbalance, and compute budgets that reflect real-world deployment conditions.

**Q1 — Person Re-Identification** treats identity matching as a retrieval problem. Given an image of a person (the probe), the task is to rank a gallery of previously seen individuals by similarity. Two fundamentally different approaches are compared: a classical PCA-based subspace method and a Siamese CNN trained with contrastive loss.

**Q2 — Multi-Task Learning** trains a single network to simultaneously classify pet breed (37 classes) and segment the foreground from the background at pixel level. Two architectures are compared: a custom CNN designed and trained from scratch, and a fine-tuned MobileNetV3Small backbone leveraging ImageNet pre-training.

---

## 📁 Project Structure

```
 -advanced-ml/
│
├── q1_person_reid/
│   ├── utils_reid.py          # Data loading, CMC computation, pair/triplet generation
│   ├── preprocess_reid.py     # Shared pre-processing pipeline (grayscale, resize, vectorise)
│   ├── model_pca.py           # PCA subspace method — training and gallery/probe projection
│   ├── model_siamese.py       # Siamese CNN — architecture, contrastive loss, training
│   ├── evaluate_reid.py       # Top-N accuracy, CMC curve plots, results table
│   └── main_reid.py           # Pipeline entry point (CLI)
│
├── q2_multitask/
│   ├── utils_pets.py          # Oxford Pets data loader, mask preprocessing, augmentation
│   ├── model_scratch.py       # From-scratch multi-task CNN (encoder + dual heads)
│   ├── model_mobilenet.py     # Fine-tuned MobileNetV3Small with frozen/unfrozen phases
│   ├── evaluate_pets.py       # Classification F1, segmentation IoU, training curves
│   └── main_pets.py           # Pipeline entry point (CLI)
│
└── outputs/                   # Saved plots and results (auto-created)
```

---

## 🔄 Q1 — Person Re-Identification

**Dataset:** Market-1501 (subset) — 5,933 training images across 300 identities, 301 gallery/probe pairs for evaluation.

**Pre-processing:** Images are kept at their original 128×64 resolution and converted to grayscale. The same pre-processing is applied to both methods for a fair comparison.

**Non-DL method — PCA:** A PCA subspace is fitted on the vectorised training images, retaining 95% of explained variance. Gallery and probe images are projected into this subspace, and identity matching is performed using L1 distance between projected vectors.

**DL method — Siamese CNN:** A shared three-block CNN backbone (Conv→BN→ReLU→MaxPool ×2, then GlobalAvgPool→Dense) maps each image to a 64-dimensional embedding. The network is trained end-to-end with contrastive loss on matched and mismatched image pairs sampled from the training identities. At inference time, embeddings are extracted from the shared backbone and matched using L1 distance.

**Evaluation:** Both models are evaluated using Top-1, Top-5, and Top-10 accuracy and Cumulative Match Characteristic (CMC) curves on the held-out gallery/probe test set.

| Model | Top-1 | Top-5 | Top-10 | Time (s) |
|-------|-------|-------|--------|----------|
| PCA | 0.0930 | 0.1595 | 0.2292 | 1.63 |
| Siamese CNN | 0.0299 | 0.0631 | 0.0996 | 0.91 |

PCA outperforms the Siamese CNN on this limited dataset — a result that highlights the data efficiency advantage of classical methods when training samples per identity are scarce.

**Usage:**
```bash
cd q1_person_reid
python main_reid.py --data ../Data/Q1 --pairs 2000 --epochs 10
```

---

## 🔄 Q2 — Multi-Task Learning & Fine Tuning

**Dataset:** Oxford-IIIT Pets — 37 cat and dog breeds with per-pixel segmentation masks (foreground/background). Downloaded automatically via `tensorflow_datasets`.

**Pre-processing:** Images are resized to 128×128 and normalised to [−1, 1]. Random horizontal flips are applied during training. The same pre-processing is used for both models.

**From-scratch CNN:** A lightweight encoder (Conv(4)→Pool, Conv(8)→Pool, Conv(16)→Pool) feeds into two parallel heads — a Global Average Pooling classification head (Dense 64→37) and a transposed convolution decoder for pixel-level segmentation (Conv2DTranspose ×3→sigmoid). The compact filter counts (4/8/16) are deliberately chosen to avoid overfitting on the limited training set.

**Fine-tuned MobileNetV3Small:** The MobileNetV3Small backbone (pre-trained on ImageNet) is used as a frozen feature extractor in the first training phase. The same dual-head structure (classification + segmentation decoder) is attached and trained. In a second phase, the top 20 backbone layers are unfrozen and fine-tuned at a reduced learning rate (1e-5) to adapt ImageNet features to the pets domain without catastrophic forgetting.

| Model | Cls Accuracy | Cls F1 (macro) | Seg Pixel Acc | Seg IoU |
|-------|-------------|----------------|---------------|---------|
| From-Scratch CNN | 0.7538 | 0.7256 | — | — |
| Fine-Tuned MobileNet | 0.8229 | 0.8079 | — | — |

MobileNetV3Small consistently outperforms the from-scratch model, particularly on classification — a direct consequence of ImageNet pre-training providing strong low-level features at minimal additional compute cost.

**Usage:**
```bash
cd q2_multitask
python main_pets.py --size 128 --batch 32 --epochs 20 --finetune-epochs 10
```

---

## ⚙️ Requirements

```bash
pip install tensorflow tensorflow-datasets scikit-learn numpy pandas matplotlib open -python
```

---

## 🗄️ Data

**Q1 — Market-1501 (subset)**

Not included due to licensing restrictions on person image datasets. Place the data at `Data/Q1/` with the following structure:

```
Data/Q1/
├── Training/         # ~5,933 .jpg images, 300 identities
└── Testing/
    ├── Gallery/      # 301 .jpg images, one per identity
    └── Probe/        # 301 .jpg images, one per identity
```

**Q2 — Oxford-IIIT Pets**

Downloaded automatically on first run via `tensorflow_datasets`. No manual setup required.

Add the following to `.gitignore`:

```
Data/
outputs/
```
