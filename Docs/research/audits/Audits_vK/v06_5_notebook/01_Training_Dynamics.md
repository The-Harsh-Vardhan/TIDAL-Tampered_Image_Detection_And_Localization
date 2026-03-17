# Audit 6.5 — Part 1: Training Dynamics

## Training Curve Summary

| Epoch | Train Loss | Val Loss | Val F1 | Val IoU | Notes |
|---|---|---|---|---|---|
| 1 | 1.0902 | 1.0520 | 0.4663 | 0.4379 | New best |
| 2 | 0.9852 | 0.9587 | 0.5686 | 0.5424 | New best |
| 3 | 0.9042 | 0.9866 | 0.6456 | 0.6269 | New best (val loss spike) |
| 4 | 0.8473 | 0.8940 | 0.6582 | 0.6326 | New best |
| 5 | 0.8161 | 0.8808 | 0.6621 | 0.6358 | New best |
| 6 | 0.7852 | 0.8379 | 0.6721 | 0.6476 | New best |
| 7 | 0.7550 | 0.9417 | 0.6252 | 0.5979 | Val loss spike, F1 drop |
| 8 | 0.7266 | 0.9184 | 0.6911 | 0.6686 | New best (recovery) |
| 9 | 0.7175 | 0.8238 | 0.6998 | 0.6765 | New best |
| 10 | 0.6956 | 1.0022 | 0.6738 | 0.6589 | Val loss spike (periodic pattern) |
| 11 | 0.6692 | 0.7671 | 0.7140 | 0.6958 | New best |
| 12 | 0.6543 | 0.8505 | 0.6985 | 0.6803 | |
| 13 | 0.6609 | 0.9406 | 0.6788 | 0.6624 | Train loss bump |
| 14 | 0.6316 | 0.8569 | 0.6978 | 0.6799 | |
| **15** | **0.6268** | **0.8107** | **0.7289** | **0.7088** | **Best model saved** |
| 16 | 0.6103 | 0.9078 | 0.6880 | 0.6665 | |
| 17 | 0.5901 | 0.8747 | 0.7103 | 0.6930 | |
| 18 | 0.5807 | 0.9506 | 0.6860 | 0.6686 | |
| 19 | 0.5730 | 0.9460 | 0.6953 | 0.6784 | |
| 20 | 0.5823 | 0.8960 | 0.7014 | 0.6851 | Train loss bump |
| 21 | 0.5446 | 1.2402 | 0.6408 | 0.6306 | Worst val loss |
| 22 | 0.5261 | 1.1266 | 0.6684 | 0.6516 | |
| 23 | 0.5189 | 0.9311 | 0.6891 | 0.6656 | |
| 24 | 0.5062 | 1.1797 | 0.6451 | 0.6344 | |
| 25 | 0.5075 | 1.1975 | 0.6574 | 0.6452 | Early stopping triggered |

---

## Convergence Analysis

### Did the model converge?

**Partially.** The training loss shows a clean, monotonic decrease from 1.09 → 0.51, indicating the optimizer is working correctly and the model is learning representations. However, validation loss tells a different story:

- Val loss decreases from 1.05 to a minimum of ~0.77 at epoch 11
- After epoch 15, val loss becomes increasingly erratic: 0.81 → 0.95 → 1.24 → 1.20
- The final val loss (1.20) is **higher than the initial val loss** (1.05)

This is a textbook **overfitting pattern**: the model memorizes training data while losing generalization.

### Overfitting Assessment

**Overfitting is confirmed and severe after epoch 15.**

Evidence:
1. **Train-val loss divergence:** Train loss at epoch 25 is 0.51 while val loss is 1.20 — a 2.35× gap
2. **Val loss increases:** After epoch 11, val loss trends upward while train loss continues down
3. **Val F1 plateau:** F1 oscillates between 0.64–0.73 from epoch 3–25, never breaking through 0.73
4. **Early stopping triggered:** Correctly stopped at epoch 25 (10 epochs after best at 15)

### Training Instability

**Moderate instability present:**

1. **Periodic val loss spikes:** Epochs 3, 7, 10, 13, 18, 21, 24 show val loss spikes of 0.1–0.4 above the local trend. This cyclical pattern (every 2–4 epochs) suggests:
   - Gradient accumulation may interact poorly with the learning rate
   - No learning rate scheduler is used — a constant LR can cause oscillation as the model approaches a minimum
   - DataParallel across 2 GPUs may introduce batch distribution variance

2. **Small train loss bumps** at epochs 13 and 20 — unusual for a model with monotonically decreasing loss. Could indicate noisy batches or gradient norm clipping events.

### Suspicious Patterns

| Pattern | Status |
|---|---|
| Perfect metrics early | ❌ Not observed (F1 starts at 0.47) |
| Val metrics > train metrics | ⚠️ Val loss is higher than train loss throughout (normal for val having no augmentation and different distribution) — but the **gap** is abnormally large |
| Flat loss curves | ❌ Not observed — train loss decreases steadily |
| Training instability | ⚠️ Moderate — val loss oscillations |
| Overfitting | ✅ Confirmed — severe after epoch 15 |

---

## Root Cause Analysis

The overfitting pattern is most likely caused by:

1. **No learning rate scheduler.** The constant LR=1e-3 for the decoder is too aggressive for later training. A CosineAnnealing or ReduceLROnPlateau scheduler would allow the model to fine-tune in later epochs rather than overshooting.

2. **Weak augmentation.** Only geometric transforms (flip, rotate) are used. No color augmentation (ColorJitter), elastic deformation, or occlusion (CoarseDropout/Cutout). The model can memorize texture patterns from the training set.

3. **No regularization beyond weight decay.** No dropout, no stochastic depth. The 24M-parameter U-Net has ample capacity to overfit a 8829-sample training set.

4. **DataParallel with small batch.** DataParallel splits the effective batch of 4 across 2 GPUs (2 images per GPU), which can lead to noisy BatchNorm statistics per GPU.

---

## Verdict

The training dynamics are **consistent with a legitimate training run** that suffers from overfitting. There is no evidence of bugs, metric manipulation, or data leakage. The early stopping mechanism worked correctly. The primary concern is that the model could have been significantly better with proper LR scheduling and stronger regularization.
