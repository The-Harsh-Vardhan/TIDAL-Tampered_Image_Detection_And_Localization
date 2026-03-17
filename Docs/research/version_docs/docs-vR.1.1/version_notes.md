# Version Notes: vR.1.1 — Evaluation Fix

| Field | Value |
|-------|-------|
| **Version** | vR.1.1 |
| **Parent** | vR.1.0 (baseline paper reproduction) |
| **Date** | 2026-03-14 |
| **Category** | Evaluation methodology |
| **Status** | Ready for execution |

---

## Change Implemented

Fix the evaluation methodology to produce honest, unbiased metrics.

Five specific changes:

1. **70/15/15 train/val/test split** — Replace the 80/20 train/val split with a proper 3-way split. The validation set is used ONLY for early stopping. The test set is used ONLY for final metric reporting. This eliminates the evaluation bias where model selection and metric reporting used the same data.

2. **Per-class + macro metrics** — Replace `average='weighted'` with per-class reporting for both Authentic and Tampered classes, plus macro-average. The `weighted` average inflates scores toward the majority class (Authentic) and produced numbers that were not comparable to the paper's per-class metrics.

3. **ROC-AUC** — Add Receiver Operating Characteristic curve and Area Under Curve. This is the standard threshold-independent metric for binary classification. The baseline didn't compute it at all.

4. **ELA visualization** — Add a 4x4 grid showing original images and their ELA maps for both authentic and tampered examples. The paper's entire contribution is ELA preprocessing, yet the baseline never visualized what ELA produces.

5. **Model save** — Activate the model save cell (was commented out in baseline). Saves as `.keras` format.

---

## Motivation

The audit report (vr-etasr-run-01-audit.md) identified three MAJOR issues:

> **MAJOR-2:** "No Test Set (Evaluation Bias) — Reporting metrics on the validation set that was used for early stopping produces optimistically biased numbers."

> **MAJOR-3:** "Metric Averaging Mismatch — The weighted average metrics (P=0.9068) are presented alongside paper metrics that are per-class. The actual tampered-class precision is 82.79%, not 90.68%."

These must be fixed before any model improvements can be measured honestly.

---

## What Did NOT Change

Everything not listed above is frozen at baseline values:

- ELA quality: 90
- Image size: 128x128
- CNN architecture: Conv2D(32) -> Conv2D(32) -> MaxPool -> Dropout -> Flatten -> Dense(256) -> Dropout -> Dense(2, softmax)
- Optimizer: Adam(lr=0.0001)
- Loss: categorical_crossentropy
- Batch size: 32
- Early stopping: patience=5 on val_accuracy, restore_best_weights
- Seed: 42

---

## Expected Impact

| Metric | vR.1.0 | Expected vR.1.1 | Reason |
|--------|--------|------------------|--------|
| Test Accuracy | 89.89%* | ~87-89% | Smaller training set (70% vs 80%), unbiased test set |
| Tampered Precision | 0.8279* | Similar | Architecture unchanged |
| Tampered Recall | 0.9483* | Similar | Architecture unchanged |
| ROC-AUC | — | ~0.94-0.96 | New metric; expected to be high given 89% accuracy |

\* vR.1.0 numbers are on the validation set (biased). The expected "drop" is not regression — it's honest measurement.

---

## Experiment Configuration

```
SEED = 42
IMAGE_SIZE = (128, 128)
ELA_QUALITY = 90
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.0001
EARLY_STOP_PATIENCE = 5
TEST_SPLIT = 0.15
VAL_SPLIT = 0.15
```

---

## Evaluation Checklist

After running on Kaggle, verify:

- [ ] ELA visualization shows 4 authentic + 4 tampered examples
- [ ] Split sizes: ~8,830 train / ~1,892 val / ~1,892 test
- [ ] Training completes without errors
- [ ] Test accuracy reported (not validation)
- [ ] Per-class metrics for both Authentic and Tampered
- [ ] Macro-average metrics
- [ ] ROC-AUC computed and plotted
- [ ] Confusion matrix on test set
- [ ] Training curves show loss + accuracy
- [ ] Model saved as .keras file
- [ ] Ablation comparison table printed

---

## Next Version Preview

**vR.1.2** will add data augmentation (horizontal flip, vertical flip, random rotation +-15 degrees) to address the overfitting gap visible in the training curves. All other parameters will remain frozen at vR.1.1 values.
