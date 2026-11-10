# Audit: CoMoFoD Dataset Forgery Detection

**Auditor:** Principal AI Engineer
**Date:** 2026-03-14
**File:** `comofod-dataset-forgery.ipynb` (3.7 MB)

---

## Notebook Overview

A Keras U-Net segmentation notebook for **copy-move forgery detection** on the CoMoFoD dataset. Unlike most other reference notebooks which do classification, this one performs pixel-level localization using a custom U-Net with DCT (Discrete Cosine Transform) preprocessing. Achieves 79.3% Mean IoU on the test set.

| Attribute | Value |
|---|---|
| Cell Count | ~16 (15 code, 0-1 markdown) |
| Model | Custom U-Net (Keras Functional API) |
| Dataset | CoMoFoD (Copy-Move Forgery Detection) |
| Task | **Pixel-level segmentation** |
| Preprocessing | DCT (8×8 block-wise) |
| Image Size | 256×256 (grayscale, 1 channel) |

---

## Dataset Pipeline Review

| Property | Value |
|---|---|
| Dataset | CoMoFoD Small v2 |
| Total images | 10,402 |
| Labels (binary masks) | 200 (`*_B.png`) |
| Masks | 200 (`*_M.png`) |
| Forged/Original images | 9,600 (`*_F_*.png` + `*_O_*.png`) |
| Train | 8,640 (90%) |
| Test | 960 (10%) |
| Split | 90/10 (no validation split) |

**No validation split.** The model trains on 90% and evaluates on 10% at the end. No per-epoch validation loss is monitored — only training loss guides the `ModelCheckpoint`.

**Data loading:** Uses a custom generator class, but `on_epoch_end` is empty (no shuffling between epochs).

---

## Model Architecture Review

Custom U-Net built with Keras Functional API:

```
Input: (256, 256, 1) — single channel grayscale DCT

Encoder:
  Conv2D(16,3)+BN → Conv2D(16,3)+BN → MaxPool2D(2×2)
  Conv2D(32,3)+BN → Conv2D(32,3)+BN → MaxPool2D(2×2)
  Conv2D(64,3)+BN → Conv2D(64,3)+BN → MaxPool2D(2×2)
  Conv2D(128,3)+BN → Conv2D(128,3)+BN → MaxPool2D(2×2)

Bottleneck:
  Conv2D(256,3)+BN → Conv2D(256,3)+BN

Decoder (skip connections via concatenate):
  UpSampling2D → Conv2D(128,2)+BN → concat → Conv2D(128,3)×2
  UpSampling2D → Conv2D(64,2)+BN  → concat → Conv2D(64,3)+BN ×2
  UpSampling2D → Conv2D(32,2)+BN  → concat → Conv2D(32,3)+BN ×2
  UpSampling2D → Conv2D(16,2)+BN  → concat → Conv2D(16,3)+BN ×2 → Conv2D(2,3)+BN

Output: Conv2D(1, 1, sigmoid) — pixel-wise mask
```

**Note:** Uses `UpSampling2D` (bilinear interpolation) instead of `ConvTranspose2d`. Skip connections via `concatenate`, consistent with standard U-Net design.

---

## Training Pipeline Review

| Component | Configuration |
|---|---|
| Optimizer | Adam (lr=1e-4) |
| Loss | binary_crossentropy |
| Epochs | 40 |
| Batch Size | 128 |
| Multi-GPU | `tf.distribute.MirroredStrategy` (GPU:0, GPU:1) |
| Checkpointing | ModelCheckpoint (best by loss → `unet.hdf5`) |
| EarlyStopping | **None** |
| Validation | **None during training** |
| LR Scheduler | **None** |

**Training progression:** Loss decreased from 0.7160 (epoch 1) to 0.3155 (epoch 40). Training accuracy reached 99.0%.

---

## Evaluation Metrics Review

| Metric | Value |
|---|---|
| Test Mean IoU | **79.28%** |
| Training Accuracy | 99.0% (epoch 40) |
| Training Loss | 0.3155 (epoch 40) |

IoU computed manually over 960 test images with threshold=0.5. No precision, recall, F1, or per-class breakdown reported.

---

## Visualization Assessment

Minimal. The notebook includes:
- Training loss curve (decreasing)
- No test prediction visualizations
- No overlay comparisons (predicted vs ground truth masks)
- No failure case analysis

---

## Engineering Quality Assessment

| Criterion | Rating | Notes |
|---|---|---|
| Architecture | **Good** | Standard U-Net with skip connections |
| DCT Preprocessing | **Novel** | Unique approach among reference notebooks |
| Training Setup | **Poor** | No validation, no early stopping, no LR schedule |
| Multi-GPU | **Good** | MirroredStrategy for distributed training |
| Data Loader | **Buggy** | Empty `on_epoch_end` (no shuffle) |
| Code Quality | **Fair** | Functional but sparse documentation |
| Evaluation | **Minimal** | Only IoU, no other metrics |

---

## Strengths

1. **Pixel-level segmentation** — one of only two reference notebooks that perform localization (along with `image-detection-with-mask`)
2. **DCT preprocessing** — novel approach: 8×8 block-wise DCT transforms encode frequency-domain artifacts from JPEG tampering
3. **79.3% IoU** — competitive result for copy-move detection
4. **Multi-GPU training** — properly uses `MirroredStrategy` for distributed training
5. **Different dataset** — CoMoFoD provides diversity beyond CASIA 2.0

---

## Weaknesses

1. **No validation during training** — cannot detect overfitting
2. **No early stopping** — trains for fixed 40 epochs regardless
3. **No LR scheduling** — fixed learning rate throughout
4. **Grayscale-only input** — discards color information that may help distinguish tampering
5. **Empty `on_epoch_end`** — data is never shuffled between epochs
6. **No data augmentation** — spatial transforms could improve generalization
7. **Single metric (IoU)** — no F1, precision, recall, Dice reported
8. **`os.system("cp ...")` for file operations** — non-portable, fragile

---

## Critical Issues

1. **No validation split.** The model trains for 40 epochs with `ModelCheckpoint` monitoring only training loss. There is no way to detect overfitting. The 99% training accuracy with no validation check is a red flag.

2. **Empty `on_epoch_end`.** The data generator never shuffles data between epochs, meaning the model sees the same batch order every epoch. This can cause the optimizer to converge to a poor local minimum along the batch ordering.

3. **IoU calculation may be flawed.** The manual IoU computation divides `mask/255` and compares with threshold=0.5, but for non-forged images (zero masks), the IoU of (0 predicted, 0 ground truth) should be 1.0 (both correctly predict no forgery). If the code handles this incorrectly, the 79.3% may be inflated or deflated.

4. **Missing BatchNormalization on some decoder layers.** The `conv6` block in the decoder omits BN, creating an asymmetry with other decoder stages.

---

## Suggested Improvements

1. Add a validation split (at least 10% of training data) with per-epoch monitoring
2. Add EarlyStopping on validation loss
3. Implement `on_epoch_end` with data shuffling
4. Add data augmentation (flips, rotations, elastic deformation)
5. Experiment with RGB input alongside DCT (multi-channel input)
6. Report F1, precision, recall, and Dice in addition to IoU
7. Add visualization of predicted masks vs ground truth
8. Replace `os.system("cp")` with `shutil.copy`

---

## Roast Section

This notebook has one genuinely interesting idea — DCT preprocessing for copy-move detection — buried inside a training pipeline that would make a machine learning professor weep. No validation set, no early stopping, no learning rate schedule, no data shuffling, no augmentation, and exactly one evaluation metric. The model trains for 40 epochs into the void, achieving 99% training accuracy with no way to know if it's generalizing or just memorizing.

The `on_epoch_end` method is defined and... left empty. The docstring says "Updates indexes after each epoch" but the body says pass. The data generator faithfully serves the same batch ordering every single epoch for 40 epochs. It's like writing a shuffle function that doesn't shuffle — the interface promises randomness, but the implementation delivers determinism.

79.3% IoU is actually respectable for copy-move detection, which makes the lack of validation even more frustrating. Is this result robust? Would it hold on different forgery types? On images with post-processing? We'll never know, because the evaluation amounts to a single IoU number computed once on one test split with one threshold.

The multi-GPU setup (`MirroredStrategy`) is the most sophisticated engineering in the entire notebook, and it's wasted on a training loop that doesn't even track validation loss. It's like putting a Formula 1 engine in a car with no steering wheel.

**Bottom line:** DCT preprocessing for tampering detection is a legitimately good idea. The implementation needs a complete training infrastructure overhaul — validation, early stopping, augmentation, and proper evaluation — before the results can be trusted.
