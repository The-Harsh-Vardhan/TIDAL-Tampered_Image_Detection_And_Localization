# vK.7.5 Run Audit

**Notebook:** `Run vK.7.5/vK.7.5_run_output.ipynb`
**Environment:** Local Windows machine with CUDA GPU
**Epochs:** 2 (effective training loop)

---

## Final Test Metrics

### Prior Experiment Block (BROKEN)
| Metric | Value | Notes |
|--------|-------|-------|
| Test Accuracy | 0.6025 | Trained on TEST set (data leakage) |

### Effective Training Loop
| Metric | Value |
|--------|-------|
| Test Accuracy | 0.5526 |
| Test Dice | 0.5935 |
| Test IoU | 0.5935 |
| Test F1 | 0.5935 |
| Best Val Acc | 0.5439 |
| Best Epoch | 1 (of 2) |

---

## Training Observations

- **Only 2 epochs** — catastrophically undertrained
- Epoch 1: Train Acc 0.4763 (**below random** for binary)
- Epoch 2: Train Acc 0.4640 (getting worse)
- Val Dice = IoU = F1 = 0.5949 both epochs — **suspiciously identical** (Dice and IoU should differ)
- The model is almost certainly predicting all-zeros and getting credit from authentic images with empty GT masks

---

## Critical Bugs

1. **Data leakage in prior experiment block (cell 36)**:
   ```
   TRAIN_CSV = "test_metadata.csv"   ← trains on TEST set
   TEST_CSV  = "val_metadata.csv"    ← evaluates on VAL set
   ```
2. **2 epochs is not training** — the model hasn't learned anything meaningful
3. **Dice = IoU = F1 = 0.5935** is mathematically impossible unless the model predicts constant outputs
4. **Train accuracy below 50%** confirms the model hasn't learned
5. **`IN_COLAB: False`** despite documentation claiming Colab execution

---

## Strengths

1. **Best documentation quality** — structured docstrings (Purpose/Inputs/Returns/Notes), TOC, project objectives table with Fulfilled/Remaining markers
2. **Varied visualization formats** — overlay, side-by-side, grid layout, submission panels
3. **Environment handling** — Colab/local detection, Drive search, Kaggle API fallback
4. **Assignment alignment notes** throughout every section
5. **Combined loss function** — Focal + BCE + Dice (good design, just not trained)
6. **Custom dual-head U-Net** architecture (classification + segmentation from shared bottleneck)

---

## Weaknesses

1. **Only 2 epochs** — effectively untrained
2. **Data leakage** in prior experiment block
3. **No robustness testing**
4. **No explainability**
5. **No threshold optimization** (fixed 0.5)
6. **No forgery-type breakdown**
7. **No mask-size stratification**
8. **No shortcut learning checks**
9. **No data leakage verification**
10. **AUC-ROC not computed**
11. **Model predictions are degenerate** — doesn't actually detect tampering

---

## Verdict

Beautiful documentation wrapped around a model that never learned. The notebook is well-structured but ran only 2 epochs, producing meaningless metrics. The prior experiment block has data leakage. The identical Dice/IoU/F1 values are a red flag for a degenerate model. This is a showcase of good notebook engineering with zero scientific value.

**Score: 30/100** — Great docs, no actual results.
