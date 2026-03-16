# Technical Audit: vK.10.3b (Run 02)

**Auditor:** Principal AI Engineer
**Date:** 2026-03-14
**File:** `vk-10-3b-tampered-image-detection-and-localization-run-02.ipynb` (~137KB)

---

## CRITICAL FINDING: Run-02 Is an Exact Duplicate of Run-01

**Run-02 is byte-for-byte identical to run-01.** This is not a second experiment — it is the same notebook file saved under a different name.

### Evidence

| Attribute | Run-01 | Run-02 |
|---|---|---|
| Cell count | 68 | 68 |
| Source code | Identical | Identical |
| Cell outputs | Identical | Identical |
| W&B Run ID | `rg1rf1s0` | `rg1rf1s0` |
| W&B Timestamp | `run-20260313_214414` | `run-20260313_214414` |
| Kernelspec | Identical | Identical |

---

## Metrics (Same as Run-01)

| Metric | Value |
|---|---|
| Test Accuracy | 0.5061 |
| Dice (all) | 0.5781 |
| **Dice (tampered)** | **0.0004** |
| **IoU (tampered)** | **0.0002** |
| **F1 (tampered)** | **0.0004** |
| AUC-ROC | 0.6069 |
| Epochs run | 11 (early stopped) |
| Best epoch | 1 (Dice=0.0006) |

---

## Architecture & Training

Identical to vK.10.3b run-01. See `Audit_vK10.3b_vK10.4_vK10.5.md` for the full audit.

- Custom `UNetWithClassifier`, 31.6M params, no pretrained weights, 256×256
- Adam(lr=1e-4), CosineAnnealingLR(T_max=50), FocalLoss + BCE+Dice
- Batch=32 (auto-scaled), AMP, early stopping patience=10
- **No DataParallel** (single GPU despite 2 available)

---

## Roast

Run-02 is run-01 wearing a fake mustache. Same W&B run ID, same outputs, same timestamps. Someone copied the notebook and gave it a new name. This adds zero experimental value — no different hyperparameters, no different seed, no different data split. It's not even a reproducibility check because it wasn't re-executed. If you want a second run, actually run it again.

---

## Recommendation

Remove run-02 from the experiment record or clearly label it as a duplicate to avoid confusion. If a genuine second run is desired, re-execute with a different seed or modified hyperparameters to add experimental value.
