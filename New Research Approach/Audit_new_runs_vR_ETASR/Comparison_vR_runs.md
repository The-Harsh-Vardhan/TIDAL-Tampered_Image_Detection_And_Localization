# Cross-Run Comparison: vR.ETASR Series (All Runs)

| Field | Value |
|-------|-------|
| **Auditor** | Principal AI Engineer (Automated Deep Audit) |
| **Date** | 2026-03-14 |
| **Scope** | 4 experiment runs: vR.ETASR, vR.0, vR.1, vR.1.1 |
| **Paper** | ETASR_9593 — "Enhanced Image Tampering Detection using ELA and a CNN" |
| **Dataset** | CASIA v2.0 (12,614 images: 7,491 Au + 5,123 Tp) |
| **Architecture** | Constant across all runs: 2×Conv2D(32,5×5) + MaxPool + Flatten + Dense(256) + Dense(2,Softmax) — 29,520,034 params |
| **Verdict** | **All 4 runs converge. Evaluation methodology fixed in vR.1.1. True baseline: 88.38% test accuracy, 0.9601 AUC.** |

---

## 1. Executive Summary

Four runs of the ETASR paper reproduction have been completed. The architecture is identical across all four — the only changes are evaluation methodology and presentation:

| Run | What Changed | Key Result |
|-----|-------------|------------|
| **vR.ETASR** | Original baseline | 89.89% val acc (biased) |
| **vR.1** | + ELA viz, + model save | 89.81% val acc (biased) — effectively identical |
| **vR.0** | + 70/15/15 split, + ROC-AUC | 88.33% test acc (honest) |
| **vR.1.1** | + Per-class/macro metrics, + version tracking | 88.38% test acc (honest) — the definitive baseline |

**Bottom line:** The ETASR CNN achieves **~88.4% accuracy** on a proper held-out test set with **0.9601 ROC-AUC**. The previously reported ~89.9% was inflated by ~1.5pp due to evaluating on the validation set used for model selection.

---

## 2. Run Configuration Comparison

| Parameter | vR.ETASR | vR.0 | vR.1 | vR.1.1 |
|-----------|----------|------|------|--------|
| GPU | P100 | T4×2 | T4×2 | P100 |
| Data split | 80/20 | **70/15/15** | 80/20 | **70/15/15** |
| Test set | ❌ No | ✅ Yes | ❌ No | ✅ Yes |
| Eval set | Val (biased) | **Test (honest)** | Val (biased) | **Test (honest)** |
| Metric type | Weighted | Weighted | Weighted | **Per-class + macro** |
| ROC-AUC | ❌ | ✅ | ❌ | ✅ |
| ELA viz | ❌ | ✅ (3+3) | ✅ (3+3) | ✅ (4+4) |
| Model saved | ❌ | ✅ | ✅ | ✅ |
| Version tracking | ❌ | ❌ | ❌ | ✅ |
| Ablation table | ❌ | ❌ | ❌ | ✅ |
| Cells (code/md) | 18/11 | 20/11 | 19/11 | 19/12 |

### Architecture (Frozen Across All Runs)

```
Input(128, 128, 3)
Conv2D(32, 5×5, valid, ReLU)
Conv2D(32, 5×5, valid, ReLU)
MaxPooling2D(2×2)
Dropout(0.25)
Flatten          ← 29.5M params live here
Dense(256, ReLU)
Dropout(0.5)
Dense(2, Softmax)
Total: 29,520,034 parameters
```

### Training Config (Frozen Across All Runs)

```
Optimizer:      Adam(lr=0.0001)
Loss:           categorical_crossentropy
Batch size:     32
Max epochs:     50
Early stopping: patience=5, val_accuracy, restore_best_weights
Seed:           42
```

---

## 3. Full Metrics Comparison

### Primary Metrics

| Metric | vR.ETASR (val) | vR.1 (val) | vR.0 (test) | vR.1.1 (test) | Paper |
|--------|----------------|------------|-------------|---------------|-------|
| **Accuracy** | 89.89%* | 89.81%* | 88.33% | **88.38%** | 96.21% |
| **ROC-AUC** | — | — | 0.9600 | **0.9601** | — |
| **Macro F1** | 0.8972* | 0.8964* | 0.8799 | **0.8805** | — |

\* Val-set metrics are optimistically biased by 1–2pp.

### Per-Class Metrics

| Metric | vR.ETASR (val) | vR.1 (val) | vR.0 (test) | vR.1.1 (test) |
|--------|----------------|------------|-------------|---------------|
| Au Precision | 0.9607 | 0.9593 | 0.9154 | 0.9170 |
| Au Recall | 0.8652 | 0.8652 | 0.8852 | 0.8843 |
| Au F1 | 0.9104 | 0.9098 | 0.9000 | 0.9004 |
| **Tp Precision** | 0.8279 | 0.8276 | 0.8400 | **0.8393** |
| **Tp Recall** | 0.9483 | 0.9463 | 0.8804 | **0.8830** |
| **Tp F1** | 0.8840 | 0.8830 | 0.8597 | **0.8606** |

### Confusion Matrices

| | vR.ETASR | vR.1 | vR.0 | vR.1.1 |
|---|---|---|---|---|
| **TN** (Au→Au) | 1,296 | 1,296 | 995 | 994 |
| **FP** (Au→Tp) | 202 | 202 | 129 | 130 |
| **FN** (Tp→Au) | 53 | 55 | 92 | 90 |
| **TP** (Tp→Tp) | 972 | 970 | 677 | 679 |
| Total | 2,523 | 2,523 | 1,893 | 1,893 |
| FP rate | 13.5% | 13.5% | 11.5% | 11.6% |
| **FN rate** | **5.2%** | **5.4%** | **12.0%** | **11.7%** |

### Training Summary

| Metric | vR.ETASR | vR.1 | vR.0 | vR.1.1 |
|--------|----------|------|------|--------|
| Epochs trained | 13 | 13 | 13 | 13 |
| Best epoch | 8 | 8 | 8 | 8 |
| Best val_acc | 0.8989 | 0.8981 | 0.8811 | 0.8864 |
| Best val_loss | 0.2473 | 0.2463 | 0.2669 | 0.2662 |
| Instability epoch | 11 | 11 | 12–13 | 12–13 |
| Worst val_acc | 0.8565 | 0.8514 | 0.7955 | 0.7960 |
| Worst val_loss | 0.3589 | 0.3744 | 0.6387 | 0.6360 |
| Model saved | ❌ | ✅ | ✅ | ✅ |

---

## 4. Three Key Findings

### Finding 1: The Architecture Reproduces Reliably

All four runs converge to the same performance level. The best epoch is always epoch 8. The confusion matrix TN/FP/FN/TP values are within ±2 of each other for runs with the same split ratio. This is a **well-controlled experiment** — the seed, data loading, and architecture produce deterministic results.

Evidence:
- vR.ETASR vs vR.1 (same split): accuracy difference is 0.08pp
- vR.0 vs vR.1.1 (same split): accuracy difference is 0.05pp
- All four runs train for exactly 13 epochs with best at epoch 8

### Finding 2: The Honest Baseline is 1.5pp Below the Biased One

The val-based accuracy (~89.9%) is inflated by approximately 1.5pp over the test-based accuracy (~88.4%). This is the expected cost of proper evaluation methodology. The inflation comes from the validation set being used for both model selection (early stopping) and metric reporting.

| Comparison | Val Acc | Test Acc | Bias |
|------------|---------|----------|------|
| vR.ETASR → vR.0 | 89.89% | 88.33% | 1.56pp |
| vR.1 → vR.1.1 | 89.81% | 88.38% | 1.43pp |
| **Average bias** | | | **1.50pp** |

### Finding 3: The FN Rate Regression is Real and Needs Fixing

The most concerning finding is the **doubling of the false negative rate** from ~5% (80/20 split) to ~12% (70/15/15 split). This is NOT a measurement artifact — it reflects the model genuinely being less sensitive to tampered images when trained on 10% less data.

| Split | Training images | FN rate | Missed tampered |
|-------|----------------|---------|-----------------|
| 80/20 | 10,091 | 5.2–5.4% | 53–55 of 1,025 |
| 70/15/15 | 8,829 | 11.7–12.0% | 90–92 of 769 |

Adjusting for test set size: the model misses ~12% of tampered images regardless of how many are in the test set. The 80/20 split's lower FN rate was achieved with more training data and a non-independent evaluation set.

**Fix required:** Data augmentation (vR.1.2) should compensate for the smaller training set and reduce the FN rate.

---

## 5. Training Dynamics Comparison

```
Val Accuracy Across All Runs (Best Epoch = 8 for all)

Epoch  vR.ETASR   vR.1      vR.0      vR.1.1
  1     0.872     0.873     0.833     0.833
  2     0.879     0.877     0.860     0.854
  3     0.882     0.886     0.866     0.870
  4     0.891     0.893     0.876     0.863
  5     0.894     0.893     0.878     0.871
  6     0.897     0.896     0.882     0.877
  7     0.893     0.892     0.876     0.872
  8*    0.899*    0.898*    0.881*    0.886*    ← Best for all
  9     0.880     0.895     0.870     0.864
 10     0.897     0.891     0.853     0.873
 11     0.857     0.851     0.819     0.872
 12     0.892     0.889     0.795     0.821
 13     0.883     0.878     0.796     0.796
```

**Pattern:** The 70/15/15 split runs (vR.0, vR.1.1) show significantly more instability at epochs 12–13. The 80/20 split runs (vR.ETASR, vR.1) have a bad epoch at 11 but recover. The 70/15/15 runs collapse catastrophically and never recover before early stopping fires.

**Root cause:** 1,262 fewer training images (8,829 vs 10,091) leads to more pronounced overfitting. The massive Flatten→Dense(256) layer (29.5M params) is the architectural culprit — it memorizes the training set when there isn't enough data to regularize.

---

## 6. Feature Completeness

| Feature | vR.ETASR | vR.0 | vR.1 | vR.1.1 | Required? |
|---------|----------|------|------|--------|-----------|
| ELA preprocessing | ✅ | ✅ | ✅ | ✅ | ✅ |
| CNN classification | ✅ | ✅ | ✅ | ✅ | ✅ |
| 3-way data split | ❌ | ✅ | ❌ | ✅ | ✅ |
| Per-class metrics | ✅ (buried) | ✅ (buried) | ✅ (buried) | ✅ (headline) | ✅ |
| Macro metrics | ✅ | ✅ | ✅ | ✅ (headline) | ✅ |
| ROC-AUC | ❌ | ✅ | ❌ | ✅ | ✅ |
| ELA visualization | ❌ | ✅ | ✅ | ✅ | ✅ |
| Confusion matrix | ✅ | ✅ | ✅ | ✅ | ✅ |
| Training curves | ✅ | ✅ | ✅ | ✅ | ✅ |
| Model saved | ❌ | ✅ | ✅ | ✅ | ✅ |
| Data augmentation | ❌ | ❌ | ❌ | ❌ | ✅ (next: vR.1.2) |
| Class weights | ❌ | ❌ | ❌ | ❌ | Recommended |
| Localization | ❌ | ❌ | ❌ | ❌ | ✅ (future: vR.2.0) |
| Version tracking | ❌ | ❌ | ❌ | ✅ | Ablation requirement |
| Ablation table | ❌ | ❌ | ❌ | ✅ | Ablation requirement |

---

## 7. Run Lineage

```
vR.ETASR (bare baseline)
│   Split: 80/20 | Val acc: 89.89% | No save, no viz, no AUC
│
├── vR.1 (cosmetic upgrade)
│   Split: 80/20 | Val acc: 89.81% | +Model save, +ELA viz
│   Delta: -0.08pp (noise)
│
├── vR.0 (eval fix, prototype)
│   Split: 70/15/15 | Test acc: 88.33% | +AUC: 0.9600
│   Delta: -1.56pp (honest measurement)
│
└── vR.1.1 (eval fix, ablation series)  ← DEFINITIVE BASELINE
    Split: 70/15/15 | Test acc: 88.38% | +AUC: 0.9601
    Delta: -1.51pp (honest measurement)
    +Per-class headline, +version tracking, +ablation table
```

**vR.0 and vR.1.1 are near-duplicates.** Both implement the 70/15/15 split and ROC-AUC. vR.1.1 additionally fixes the metric headlines (per-class instead of weighted) and adds version tracking. Going forward, **vR.1.1 is the official baseline** for the ablation series.

---

## 8. Gap to Paper Claims

| Metric | Paper | Best Honest (vR.1.1) | Gap |
|--------|-------|----------------------|-----|
| Accuracy | 96.21% | 88.38% | **-7.83pp** |
| Precision | 98.58% | 83.93% (Tp) / 91.70% (Au) | -14.65pp / -6.88pp |
| Recall | 92.36% | 88.30% (Tp) / 88.43% (Au) | -4.06pp / -3.93pp |
| F1 | 95.37% | 86.06% (Tp) / 90.04% (Au) | -9.31pp / -5.33pp |

The 7.83pp accuracy gap remains after evaluation methodology is fixed. Possible explanations:
1. **Data version mismatch** — Paper may have used a different subset or version of CASIA
2. **Preprocessing differences** — Paper's ELA implementation details are underspecified
3. **Unreported data augmentation** — Paper may have used augmentation without documenting it
4. **Evaluation leakage** — Paper may have reported val-set numbers (which would reduce the gap to ~6.3pp)

---

## 9. Verdict and Next Steps

### Per-Run Verdicts

| Run | Score | Verdict |
|-----|-------|---------|
| vR.ETASR | 3/10 | Bare baseline. Architecture works, but evaluation is broken. |
| vR.1 | 3/10 | Cosmetic upgrade. Same broken eval, same results. |
| vR.0 | 4/10 | Eval partially fixed. Still uses weighted headlines. |
| **vR.1.1** | **5/10** | **Definitive baseline. Eval methodology is now sound.** |

### Ablation Study Status

| Version | Status | Test Acc | Macro F1 | ROC-AUC | Verdict |
|---------|--------|----------|----------|---------|---------|
| vR.1.0 | ✅ Done | 89.89%* | 0.8972* | — | Baseline (biased) |
| vR.1.1 | ✅ Done | **88.38%** | **0.8805** | **0.9601** | **Honest baseline** |
| vR.1.2 | ⏳ Next | — | — | — | Augmentation |
| vR.1.3 | Pending | — | — | — | Class weights |
| vR.1.4 | Pending | — | — | — | BatchNorm |
| vR.1.5 | Pending | — | — | — | LR scheduler |
| vR.1.6 | Pending | — | — | — | Deeper CNN |
| vR.1.7 | Pending | — | — | — | GAP replaces Flatten |
| vR.2.0 | Pending | — | — | — | ELA localization |

\* Val-set metrics (biased). Not directly comparable.

### Recommended Next Action

Generate **vR.1.2** notebook with:
- Data augmentation: horizontal flip, vertical flip, random rotation ±15°
- Using Keras `ImageDataGenerator` or `tf.keras.preprocessing.image.ImageDataGenerator`
- All other parameters frozen at vR.1.1 values
- Expected impact: +1–3pp accuracy, reduced overfitting gap, stabilized late-epoch training

---

## Appendix: Full Metric Table

| Metric | vR.ETASR | vR.1 | vR.0 | vR.1.1 | Paper |
|--------|----------|------|------|--------|-------|
| Split | 80/20 | 80/20 | 70/15/15 | 70/15/15 | ? |
| Eval set | Val | Val | Test | Test | ? |
| GPU | P100 | T4×2 | T4×2 | P100 | ? |
| Accuracy | 89.89% | 89.81% | 88.33% | 88.38% | 96.21% |
| Au Precision | 0.9607 | 0.9593 | 0.9154 | 0.9170 | — |
| Au Recall | 0.8652 | 0.8652 | 0.8852 | 0.8843 | — |
| Au F1 | 0.9104 | 0.9098 | 0.9000 | 0.9004 | — |
| Tp Precision | 0.8279 | 0.8276 | 0.8400 | 0.8393 | 98.58% |
| Tp Recall | 0.9483 | 0.9463 | 0.8804 | 0.8830 | 92.36% |
| Tp F1 | 0.8840 | 0.8830 | 0.8597 | 0.8606 | 95.37% |
| Macro Prec | 0.8943 | 0.8935 | 0.8777 | 0.8781 | — |
| Macro Rec | 0.9067 | 0.9057 | 0.8828 | 0.8837 | — |
| Macro F1 | 0.8972 | 0.8964 | 0.8799 | 0.8805 | — |
| ROC-AUC | — | — | 0.9600 | 0.9601 | — |
| TN | 1,296 | 1,296 | 995 | 994 | — |
| FP | 202 | 202 | 129 | 130 | — |
| FN | 53 | 55 | 92 | 90 | — |
| TP | 972 | 970 | 677 | 679 | — |
| FP rate | 13.5% | 13.5% | 11.5% | 11.6% | — |
| FN rate | 5.2% | 5.4% | 12.0% | 11.7% | — |
| Epochs | 13 | 13 | 13 | 13 | — |
| Best epoch | 8 | 8 | 8 | 8 | — |
| Best val_acc | 0.8989 | 0.8981 | 0.8811 | 0.8864 | — |
| Model saved | ❌ | ✅ | ✅ | ✅ | — |
