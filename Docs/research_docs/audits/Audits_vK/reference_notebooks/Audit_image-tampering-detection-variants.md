# Audit: Image Tampering Detection Variants

**Auditor:** Principal AI Engineer
**Date:** 2026-03-14
**Files:**
- `image-tampering-detection.ipynb` (314 KB) — Original, executed
- `image-tampering-detection (1).ipynb` (75 KB) — VGG16 rewrite, crashed
- `image-tampering-detection (2).ipynb` (165 KB) — Copy of original, crashed

---

## Notebook Overview

Three variants of the same tampering detection approach using the CG-1050 dataset. Only the original successfully completed training. The other two are broken copies that crash during execution.

| Variant | Model | Executed? | Best Val Acc | Status |
|---|---|---|---|---|
| Original (314KB) | Custom CNN + InstanceNorm | Yes (40 epochs) | **66.08%** | Plateaus at 64.8% |
| (1) (75KB) | VGG16 Transfer Learning | **No** | N/A | ImportError crash |
| (2) (165KB) | Custom CNN + InstanceNorm | **No** | N/A | NameError crash |

---

## Dataset Pipeline Review

| Property | Value |
|---|---|
| Dataset | CG-1050 |
| Training | 730 original + 730 tampered = 1,460 images |
| Validation | 314 original + 314 tampered = 628 images |
| Image Size | 150×150 (original, variant 2), 224×224 (variant 1) |
| Loading | `keras.ImageDataGenerator` with `flow_from_directory` |
| Augmentation | **None** (original), rotation/shift/zoom (variant 1) |
| Normalization | **None** — raw pixel values (0-255) fed directly |

**Critical:** No data normalization or rescaling (`rescale=1/255.` is missing). Feeding raw 0-255 pixel values to a CNN without normalization causes training instability.

---

## Model Architecture Review

### Original + Variant (2): Custom CNN with InstanceNormalization

| Layer | Output Shape | Params |
|---|---|---|
| Conv2D(32, 3×3, stride=2, relu) | (75, 75, 32) | 896 |
| InstanceNormalization | (75, 75, 32) | 64 |
| MaxPooling2D(2×2) | (37, 37, 32) | 0 |
| Conv2D(64, 3×3, stride=2, **no activation**) | (19, 19, 64) | 18,496 |
| MaxPooling2D(2×2) | (9, 9, 64) | 0 |
| Dropout(0.4) | (9, 9, 64) | 0 |
| Conv2D(128, 3×3, stride=2, **no activation**) | (5, 5, 128) | 73,856 |
| MaxPooling2D(2×2) | (2, 2, 128) | 0 |
| Flatten | 512 | 0 |
| Dense(100, relu) | 100 | 51,300 |
| Dropout(0.4) | 100 | 0 |
| Dense(2, softmax) | 2 | 202 |
| **Total** | | **144,814** |

**Missing activations:** Conv layers 2 and 3 have no activation function, making them linear layers. Two consecutive linear convolutions are mathematically equivalent to a single convolution — the depth adds no representational capacity.

### Variant (1): VGG16 Transfer Learning (Never Executed)

- VGG16 (ImageNet pretrained, frozen base) → GlobalAveragePooling → Dense(512) → BatchNorm → Dropout(0.5) → Dense(256) → BatchNorm → Dropout(0.3) → Dense(1, sigmoid)
- **Crashed** with `ImportError: cannot import name 'VGG16' from 'keras.applications'`

---

## Training Pipeline Review

### Original (40 epochs)

| Component | Configuration |
|---|---|
| Optimizer | Adam (default LR) |
| Loss | categorical_crossentropy |
| Scheduler | ReduceLROnPlateau (patience=2) |
| Early Stopping | Defined but **NOT passed to callbacks** |
| Epochs | 40 |

**Training progression:**
- Epoch 1: train_acc=0.50, val_acc=0.50
- Epoch 13: val_acc=**0.6608** (best)
- Epoch 15+: val_acc stuck at **0.6481** for remaining 25 epochs
- LR collapsed to near-zero, training stalled

---

## Evaluation Metrics Review

| Metric | Original | (1) | (2) |
|---|---|---|---|
| Best Val Accuracy | 66.08% | N/A (crashed) | N/A (crashed) |
| Final Val Accuracy | 64.81% | N/A | N/A |
| Train Accuracy | 66.64% | N/A | N/A |

66% accuracy on a balanced binary dataset is poor — only 16% above random chance.

---

## Visualization Assessment

The original notebook includes:
- Training/validation accuracy curves (showing the plateau)
- Training/validation loss curves
- No confusion matrix, no per-class metrics, no sample predictions

---

## Engineering Quality Assessment

| Criterion | Rating | Notes |
|---|---|---|
| Code Quality | **Poor** | No comments, no markdown cells, no documentation |
| Data Preprocessing | **Critical failure** | No normalization (0-255 fed raw) |
| Architecture Design | **Poor** | Missing activations on 2/3 conv layers |
| Training Setup | **Broken** | EarlyStopping defined but never used |
| Variant Management | **Poor** | Three copies with no version control or tracking |

---

## Strengths

1. **Uses InstanceNormalization** — an interesting choice for small-batch scenarios where BatchNorm statistics are unreliable
2. **ReduceLROnPlateau** correctly configured (patience=2, monitors val_loss)
3. **Variant (1) demonstrates awareness** that transfer learning might improve results (even though the implementation crashed)

---

## Weaknesses

1. **66% accuracy** — poor for binary classification
2. **No data normalization** — pixels fed raw (0-255)
3. **Two broken variants** that were never debugged or re-executed
4. **Missing activations** — Conv layers 2 and 3 are linear, reducing model expressivity
5. **EarlyStopping never triggers** — defined but not passed to callbacks
6. **Tiny dataset** — CG-1050 has only 1,460 training images
7. **No augmentation** in the original — only variant (1) attempted augmentation
8. **No localization** — classification only
9. **No documentation** — zero markdown explanatory cells

---

## Critical Issues

1. **Missing `rescale=1/255.` in ImageDataGenerator.** This is the most impactful bug. Without normalization, the model receives pixel values 100× larger than expected, causing gradient instability and poor convergence. This single fix would likely improve accuracy by 10-20%.

2. **EarlyStopping defined but never used.** The callback is created but omitted from `model.fit(callbacks=[reduce_lr])` — only `reduce_lr` is passed. This means the model trains for all 40 epochs even after convergence stalls.

3. **Conv layers without activation functions.** `Conv2D(64, ...)` and `Conv2D(128, ...)` have no `activation` parameter, making them linear transformations. Only Conv layer 1 has `activation='relu'`.

4. **Variant (1) data loading broken.** `flow_from_directory` is called on leaf directories (`ORIGINAL/`, `TAMPERED/`) which contain images directly — Keras expects subdirectories for each class. Result: "Found 0 images belonging to 0 classes."

5. **Variant (2) missing `import keras`.** The import was accidentally removed, causing `NameError` on `keras.callbacks.EarlyStopping`.

---

## Suggested Improvements

1. Add `rescale=1/255.` to ImageDataGenerator
2. Add activation functions to all Conv layers
3. Pass EarlyStopping to model.fit callbacks
4. Use CASIA 2.0 instead of CG-1050 for a larger, more standardized dataset
5. Add data augmentation (at minimum: horizontal flip, rotation)
6. Fix variant (1) data loading paths and VGG16 import
7. Delete or clearly mark broken variants to avoid confusion

---

## Roast Section

This is a trilogy where only the first installment works, and it works badly. The original notebook achieves 66% accuracy on a balanced binary dataset — which means if you showed it 100 images, it would misclassify 34 of them. A coin flip gets you 50%. This model adds 16 percentage points of value over random chance, but at the cost of defining three Conv layers, two of which forgot to include activation functions. Those linear convolutions are the architectural equivalent of stacking cardboard boxes and calling it load-bearing infrastructure.

The most devastating bug is the simplest: forgetting `rescale=1/255.`. The model is eating pixel values in the range [0, 255] when it expects [0, 1]. This is like feeding a baby elephant-sized meals and wondering why it's not thriving — the scale is wrong by two orders of magnitude.

Variant (1) tried to do the right thing — VGG16 transfer learning with ImageNet weights. But it crashed on the import line and was never debugged. It's like buying a Ferrari, not being able to find the key, and just walking away. The data loading was also broken (pointing at leaf directories instead of class-structured folders), so even if the import worked, there would be no data to train on.

Variant (2) is the most puzzling: someone copied the original, deleted `import keras` from the imports, and apparently never ran it to check. It crashes on the very first callback definition. This is version control by copy-paste with no quality control.

**Bottom line:** Add rescaling, add activations, and use a real dataset. Or better yet, start from a proven architecture (SMP UNet + pretrained encoder) and skip the custom CNN phase entirely.
