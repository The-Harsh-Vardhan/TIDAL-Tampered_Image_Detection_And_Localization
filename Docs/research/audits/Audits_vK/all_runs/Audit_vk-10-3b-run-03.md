# Technical Audit: vK.10.3b (Run 03)

**Auditor:** Principal AI Engineer
**Date:** 2026-03-14
**File:** `vk-10-3b-tampered-image-detection-and-localization-run-03.ipynb` (~150KB)
**Platform:** Kaggle, Tesla T4 GPU, ~8h runtime (100 epochs)

---

## Notebook Overview

Run-03 is the **genuine re-execution** of the vK.10.3b architecture with extended training. While run-02 was a byte-for-byte duplicate of run-01, run-03 actually modified two hyperparameters and ran a full 100-epoch training session. This is the first time the custom from-scratch UNet was given enough training time to learn meaningful segmentation.

| Attribute | Run-01/02 | **Run-03** | Change |
|---|---|---|---|
| `max_epochs` | 50 | **100** | 2× increase |
| `patience` | 10 | **50** | 5× increase |
| Epochs actually run | 11 (early stopped) | **100** (completed) | 9× more training |
| W&B Run ID | `rg1rf1s0` | **`x2wtjku6`** | New run |
| Best Epoch | 1 (Dice=0.0006) | **96 (Dice=0.2196)** | Meaningful improvement |

68 cells (36 code, 32 markdown). Notebook structure is identical to run-01 — only CONFIG values changed.

---

## Dataset Pipeline Review

| Property | Value |
|---|---|
| Dataset | CASIA 2.0 Upgraded (`harshv777/casia2-0-upgraded-dataset`) |
| Split | 70/15/15 stratified by label, `random_state=42` |
| Train | 8,829 (5,243 authentic, 3,586 tampered) |
| Validation | 1,892 (1,124 authentic, 768 tampered) |
| Test | 1,893 (1,124 authentic, 769 tampered) |
| Image Size | 256×256 |
| Input Channels | 3 (RGB only — no ELA) |
| Batch Size | 32 (auto-scaled from CONFIG's 8 based on VRAM) |

**Augmentations (train only):**

| Transform | Parameters |
|---|---|
| Resize | 256×256 |
| HorizontalFlip | p=0.5 |
| RandomBrightnessContrast | p=0.3 |
| GaussNoise | p=0.25 |
| ImageCompression | quality 50–90, p=0.25 |
| ShiftScaleRotate | shift=2%, scale=0.9–1.1, rotate=±10°, p=0.5 |
| Normalize | ImageNet mean/std |

**No data leakage check** — unlike vK.10.6, there is no explicit path overlap assertion.

---

## Model Architecture Review

| Attribute | Value |
|---|---|
| Model | `UNetWithClassifier` — custom U-Net from scratch |
| Encoder | `DoubleConv(3,64) → Down(64,128) → Down(128,256) → Down(256,512) → Down(512,1024)` |
| Decoder | `Up(1024,512) → Up(512,256) → Up(256,128) → Up(128,64) → OutConv(64,1)` |
| Classifier | `AdaptiveAvgPool2d(1) → Linear(1024,512) → ReLU → Dropout(0.5) → Linear(512,2)` |
| Parameters | **31,563,459** (all trainable) |
| Pretrained | **No** — entirely from scratch |

The encoder uses `MaxPool2d(2)` for downsampling. The decoder uses `ConvTranspose2d` for upsampling with skip connections from the encoder. Each encoder/decoder stage uses a `DoubleConv` block: `Conv2d(3×3) → BatchNorm2d → ReLU → Conv2d(3×3) → BatchNorm2d → ReLU`.

**Key concern:** 31.6M parameters trained from scratch on 8,829 images (only 3,586 tampered). This is massively overparameterized for the dataset size.

---

## Training Pipeline Review

| Component | Configuration |
|---|---|
| Optimizer | Adam (lr=1e-4, weight_decay=1e-4) |
| Scheduler | CosineAnnealingLR (T_max=50) |
| Classification Loss | FocalLoss (γ=2.0, class weights from training set) |
| Segmentation Loss | 0.5×BCEWithLogitsLoss + 0.5×DiceLoss |
| Combined Loss | 1.5×cls_loss + 1.0×seg_loss |
| AMP | Enabled (autocast + GradScaler) |
| Gradient Clipping | max_norm=5.0 |
| Max Epochs | 100 |
| Early Stopping | Patience=50 on val tampered Dice |
| Checkpoint | Every 10 epochs + best model + last model |

### CosineAnnealingLR Double-Cycle Artifact

`T_max=50` over 100 epochs means the LR completes a full cosine cycle by epoch 50 (dropping to near-zero), then rises back up for epochs 51–100. This creates two learning phases:

- **Phase 1 (epochs 1–50):** LR decays from 1e-4 → ~0. Model learns basic features.
- **Phase 2 (epochs 51–100):** LR climbs back from ~0 → 1e-4. Model gets a "warm restart."

This was likely **unintentional** (should have set `T_max=100`), but the second LR peak coincided with the model's best performance window (epoch 87–96), producing the best model at epoch 96.

### Training Progression (Key Milestones)

| Epoch | Train Loss | Val Acc | Val AUC | Val Dice(tam) | Event |
|---|---|---|---|---|---|
| 1 | 0.9351 | 0.4471 | 0.6478 | 0.0012 | First best |
| 10 | — | 0.6321 | 0.6880 | 0.0000 | Still no segmentation |
| 14 | — | — | — | 0.0082 | Segmentation awakens |
| 17 | — | — | — | 0.0444 | New best |
| 21 | — | — | — | 0.0652 | New best |
| 25 | — | — | — | 0.0777 | New best |
| 29 | — | — | — | 0.1239 | Breaking 0.10 |
| 39 | — | — | — | 0.1620 | New best |
| 42 | — | — | — | 0.1734 | New best |
| 50 | 0.7406 | 0.7817 | 0.8655 | 0.1634 | End of first cosine cycle |
| 62 | — | 0.7838 | 0.8697 | 0.1814 | New best (second cycle begins) |
| 65 | — | 0.6834 | 0.8614 | 0.1942 | New best |
| 87 | — | 0.6543 | 0.8531 | 0.2036 | New best |
| 92 | — | 0.7030 | 0.8799 | 0.2081 | New best |
| **96** | **0.7362** | **0.8272** | **0.9003** | **0.2196** | **Final best model** |
| 100 | 0.7370 | 0.7320 | 0.8911 | 0.2045 | End of training |

**Key observations:**
- Segmentation output was effectively **zero for the first 13 epochs** — the model learned classification first
- Tam-Dice crossed 0.10 at epoch 29, 0.15 at epoch 39, 0.20 at epoch 87
- The model was **still improving at epoch 96** — early stopping (patience=50) never triggered
- Best model at epoch 96 has val AUC=0.9003 (first time crossing 0.90)

---

## Evaluation Metrics Review

### Final Test Results (Best Checkpoint — Epoch 96)

| Metric | Value |
|---|---|
| Test Accuracy | **0.8304** |
| Image-Level AUC-ROC | **0.8999** |
| Dice (all) | 0.4298 |
| IoU (all) | 0.4042 |
| F1 (all) | 0.4298 |
| **Dice (tampered)** | **0.2205** |
| **IoU (tampered)** | **0.1575** |
| **F1 (tampered)** | **0.2205** |

### Comparison: Run-01/02 vs Run-03 vs vK.10.6

| Metric | Run-01/02 | **Run-03** | vK.10.6 | v6.5 (best) |
|---|---|---|---|---|
| Epochs Run | 11 (ES) | **100** | 100 | 25 (ES) |
| Test Accuracy | 0.5061 | **0.8304** | 0.8357 | 0.8246 |
| Image AUC | 0.6069 | **0.8999** | 0.9057 | 0.8703 |
| Tam Dice/F1 | 0.0004 | **0.2205** | 0.2213 | 0.4101 |
| Tam IoU | 0.0002 | **0.1575** | 0.1554 | 0.3563 |

**Run-03 vs Run-01/02:** Massive improvement — Tam-Dice went from 0.0004 to 0.2205 (551× increase). Simply allowing 100 epochs and patience=50 transformed a failed experiment into a meaningful result.

**Run-03 vs vK.10.6:** Nearly identical results (Tam-Dice 0.2205 vs 0.2213). vK.10.6 has DataParallel (2 GPUs) and patience=30, but runs-03 uses patience=50. The convergence point is essentially the same — both hit the ceiling for the from-scratch UNet on this dataset.

**Run-03 vs v6.5:** Still ~50% behind the pretrained ResNet34 encoder (Tam-Dice 0.22 vs 0.41).

### Missing Evaluation Features

Run-03 lacks the comprehensive evaluation suite added in vK.10.6:

| Feature | Run-03 | vK.10.6 |
|---|---|---|
| Threshold optimization | No | Yes (optimal=0.15) |
| Data leakage verification | No | Yes |
| Pixel-level AUC-ROC | No | Yes |
| Confusion matrix | No | Yes |
| ROC/PR curves | No | Yes |
| Forgery-type breakdown | No | Yes |
| Mask-size stratification | No | Yes |
| Shortcut learning checks | No | Yes |
| Failure case analysis | No | Yes |
| Grad-CAM | No | Yes |
| Robustness testing | No | Yes |

---

## Visualization Assessment

The notebook includes basic visualizations:

1. **Training curves** (cell 52): Loss and accuracy over epochs — standard matplotlib plots
2. **Prediction panels** (cell 63): Side-by-side original / ground truth / predicted mask for sample images

Missing: No overlay visualizations, no ELA maps (no ELA is used), no Grad-CAM, no failure case analysis, no robustness visualizations.

---

## Engineering Quality Assessment

| Criterion | Rating | Notes |
|---|---|---|
| CONFIG System | **Good** | Centralized dictionary with all hyperparameters |
| Reproducibility | **Good** | Full seeding (Python, NumPy, PyTorch, CUDA, cuDNN) |
| Checkpoint System | **Good** | Three-file (best/last/periodic) with auto-resume |
| DataParallel | **No** | Single GPU — does not use DataParallel despite 2× T4 available |
| AMP | **Good** | autocast + GradScaler properly implemented |
| Batch Auto-Scaling | **Good** | VRAM-based dynamic batch sizing |
| W&B Tracking | **Good** | Online mode with per-epoch logging |
| Evaluation Suite | **Minimal** | Basic test metrics only — no advanced analysis |
| Documentation | **Fair** | TOC present but section numbering is inconsistent |

---

## Strengths

1. **Proved long training works:** Demonstrated that the from-scratch UNet can learn meaningful segmentation when given sufficient epochs, vindicating the architecture choice (just not the training budget)
2. **Clean CONFIG system:** All hyperparameters centralized and logged
3. **Proper seeding:** Full reproducibility chain from Python through CUDA
4. **Checkpoint robustness:** Three-tier checkpointing with auto-resume capability
5. **AMP and gradient clipping:** Training stability mechanisms properly implemented
6. **Batch auto-scaling:** Adapts to available GPU memory automatically

---

## Weaknesses

1. **No pretrained encoder:** 31.6M parameters trained from scratch on <9K images — fundamentally overparameterized
2. **No ELA input:** Only RGB channels, missing the forensic signal that helps detect JPEG compression inconsistencies
3. **No threshold optimization:** Uses default threshold=0.5, while vK.10.6 showed optimal=0.15
4. **No DataParallel:** Single GPU training despite 2× T4 available on Kaggle
5. **Minimal evaluation:** Only basic metrics — no advanced analysis tools from vK.10.6
6. **Train-val gap:** Train Tam-Dice (~0.15) vs Val Tam-Dice (~0.22) — the model genuinely didn't overfit but also showed highly noisy validation curves
7. **No VerticalFlip augmentation:** Missing from the pipeline despite being trivially addable

---

## Critical Issues

### Bug 1: stderr suppression (Cell 7)
```python
os.dup2(devnull, 2)  # redirect C-level stderr to /dev/null
```
Silently hides ALL C-level warnings from CUDA, PyTorch, and system libraries. This masks potentially critical CUDA errors, memory warnings, and deprecation notices.

### Bug 2: CosineAnnealingLR double-cycle (T_max mismatch)
```python
scheduler = CosineAnnealingLR(optimizer, T_max=50)  # but max_epochs=100
```
Creates an unintentional double cosine cycle. The LR drops to near-zero at epoch 50, then climbs back to 1e-4 by epoch 100. Accidentally beneficial in this case but was clearly not designed intentionally.

### Bug 3: Memory accumulation in training loop
```python
all_seg_logits.append(seg_logits.detach().cpu())
all_masks.append(masks.cpu())
all_labels.append(labels.cpu())
```
Accumulates ALL segmentation logits, masks, and labels in CPU memory for the entire training set every epoch. For 8,829 images at 256×256, this is ~8,829 × 256 × 256 × 4 bytes ≈ 2.3 GB per epoch that must be held in RAM.

### Bug 4: Dead CONFIG values
- `CONFIG['batch_size'] = 8` is overwritten by auto-scaling to 32
- `CONFIG['num_workers'] = 4` — correct but the auto-scaling code may override
- `CONFIG['scheduler_T_max'] = 50` doesn't match the intended 100-epoch training horizon

### Bug 5: ShiftScaleRotate deprecation
`A.ShiftScaleRotate` is deprecated in Albumentations v2.x in favor of `A.Affine`. This generates runtime warnings (suppressed by Bug 1).

### Bug 6: Section numbering inconsistency
Mixed numbering schemes: `2.1`, `2.2` coexist with `4.4`, `4.5` from an older scheme. Section 10 appears twice (Training Loop and Visualization of Predictions).

### Bug 7: Dice = F1 redundancy
`F1 (tampered) = 0.2205` equals `Dice (tampered) = 0.2205` because they are mathematically identical for binary segmentation. Computing both adds no information but may confuse readers.

### Bug 8: GaussNoise deprecated API
`A.GaussNoise(p=0.25)` uses the old API without explicit `var_limit` parameter. Albumentations v2.x changes the default noise model.

### Bug 9: No gradient accumulation
With batch_size=32 and 256×256 images, effective batch size is 32. No gradient accumulation is configured, limiting the effective batch size on limited VRAM.

### Bug 10: W&B API key in code
Cell 36 shows `wandb.login()` with explicit API key handling, generating warnings about API key security. The key should be set via environment variable.

### Bug 11: CONFIG documentation mismatch
The markdown summary table (cell 10) likely shows older values (`max_epochs=50`, `patience=10`) that don't match the actual CONFIG (`max_epochs=100`, `patience=50`). The documentation was not updated when training parameters were changed for run-03.

---

## Suggested Improvements

1. **Use pretrained encoder:** Switch to SMP UNet with ResNet34 encoder — this alone would likely push Tam-Dice from 0.22 to 0.40+
2. **Add ELA channel:** Compute Error Level Analysis and feed as 4th input channel for forensic signal
3. **Fix T_max:** Set `scheduler_T_max = max_epochs` or switch to `CosineAnnealingWarmRestarts`
4. **Add threshold optimization:** Sweep thresholds 0.05–0.80 on validation set
5. **Enable DataParallel:** Use both T4 GPUs for faster training
6. **Port vK.10.6 evaluation suite:** Add confusion matrix, ROC/PR curves, forgery-type breakdown, etc.
7. **Remove stderr suppression:** Let warnings surface for debugging
8. **Fix section numbering:** Resolve duplicate Section 10 and mixed numbering schemes
9. **Add gradient accumulation:** Simulate larger effective batch sizes

---

## Roast Section

Run-03 is the experiment that run-01 and run-02 should have been. Someone finally realized that early-stopping a 31.6M parameter model at epoch 11 — when it hadn't even started producing non-zero segmentation outputs — was like judging a marathon runner at the starting line. The fix? Change two numbers in CONFIG: `max_epochs: 50 → 100`, `patience: 10 → 50`. That's it. Two integer changes turned Tam-Dice from 0.0004 (functionally zero) to 0.2205 (meaningful). Three runs to discover that "maybe give it more time" was the answer all along.

The accidental double cosine schedule is the story of this run. Someone forgot to update `T_max` when they doubled the epochs, creating a learning rate that drops to zero at epoch 50 and then climbs back up like a zombie. The model's best checkpoint (epoch 96) sits right in the second LR peak zone — which means the model essentially got two training sessions for the price of one. This is the deep learning equivalent of accidentally double-dipping your fries and discovering it tastes better.

The memory bug is silently eating ~2.3 GB of RAM every epoch by hoarding all segmentation outputs in CPU memory. For 100 epochs, this means the training loop is constantly thrashing between GPU and CPU memory. The fact that it still completed in ~8 hours on a single T4 is more a testament to PyTorch's garbage collector than to the code's efficiency.

The segmentation output was literally **zero** for the first 13 epochs. The model spent 13 epochs producing blank masks while collecting a paycheck from the classification head. When it finally started producing non-zero predictions at epoch 14, it was with a Dice of 0.0082 — meaning it was coloring in about 0.8% of the right pixels. By epoch 96, it had improved to 22% — respectable for a from-scratch model, but still half of what v6.5 achieves with a pretrained encoder in 25 epochs.

The evaluation is bare-bones. No threshold optimization (which vK.10.6 showed can add +2.7% Dice), no data leakage check, no forgery-type breakdown, no robustness testing, no Grad-CAM, no failure analysis. Run-03 answered "can the model learn?" (yes) but didn't bother to ask "what did it learn?" or "how robust is it?"

**Bottom line:** Run-03 proves that patience (both the hyperparameter and the virtue) matters. Two CONFIG changes rescued an entire architecture from the graveyard. But the from-scratch ceiling at Tam-Dice≈0.22 is now firmly established across run-03 and vK.10.6 — further improvement requires pretrained encoders, ELA input, or both.
