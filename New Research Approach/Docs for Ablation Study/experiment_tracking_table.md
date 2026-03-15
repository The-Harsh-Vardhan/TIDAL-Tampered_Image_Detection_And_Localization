# Experiment Tracking Table

| Field | Value |
|-------|-------|
| **Date** | 2026-03-15 |
| **Scope** | Complete metrics reference for all experimental runs |
| **Paper** | ETASR_9593 -- "Enhanced Image Tampering Detection using ELA and a CNN" |
| **Versions Covered** | ETASR Track: vR.1.0--vR.1.7 (8 runs) / Pretrained Track: vR.P.0--vR.P.6 (7 planned) |

---

## 1. ETASR Classification Track (vR.1.x)

### Full Metrics Table

| Version | Change | Test Acc | Au P | Au R | Au F1 | Tp P | Tp R | Tp F1 | Macro F1 | ROC-AUC | Epochs (Best) | Total Params | Verdict |
|---------|--------|----------|------|------|-------|------|------|-------|----------|---------|---------------|-------------|---------|
| **vR.1.0** | Paper reproduction baseline | 89.89%* | 0.9138* | 0.8897* | 0.9016* | 0.8279* | 0.9483* | 0.8840* | 0.8972* | -- | 13 (8) | 29,520,034 | Baseline |
| **vR.1.1** | Eval fix (70/15/15 split) | 88.38% | 0.9237 | 0.8577 | 0.8895 | 0.8393 | 0.8830 | 0.8606 | 0.8805 | 0.9601 | 13 (8) | 29,520,034 | Honest baseline |
| **vR.1.2** | Data augmentation (flips+rot) | 85.53% | 0.8829 | 0.8443 | 0.8632 | 0.8145 | 0.8336 | 0.8239 | 0.8505 | 0.9011 | 6 (1) | 29,520,034 | **REJECTED** |
| **vR.1.3** | Class weights (inverse-freq) | 89.17% | 0.9274 | 0.8648 | 0.8950 | 0.8431 | 0.9012 | 0.8712 | 0.8889 | 0.9580 | 14 (9) | 29,520,034 | POSITIVE +0.79pp |
| **vR.1.4** | BatchNormalization | 88.75% | 0.9350 | 0.8559 | 0.8937 | 0.8240 | 0.9194 | 0.8691 | 0.8852 | 0.9536 | 8 (3) | 29,520,290 | NEUTRAL -0.42pp |
| **vR.1.5** | ReduceLROnPlateau scheduler | 88.96% | 0.9349 | 0.8594 | 0.8956 | 0.8279 | 0.9194 | 0.8712 | 0.8873 | 0.9560 | 10 (5) | 29,520,290 | NEUTRAL +0.21pp |
| **vR.1.6** | Deeper CNN (+Conv64+MaxPool) | **90.23%** | **0.9572** | 0.8746 | **0.9140** | 0.8372 | 0.9428 | **0.8869** | **0.9004** | **0.9657** | 18 (13) | 13,826,530 | POSITIVE +1.27pp |
| **vR.1.7** | GlobalAveragePooling2D | 89.17% | 0.9590 | 0.8541 | 0.9035 | 0.8161 | **0.9467** | 0.8766 | 0.8901 | 0.9495 | 10 (5) | 63,970 | NEUTRAL -1.06pp |

*vR.1.0 metrics are validation-set only (no test split). Not directly comparable with vR.1.1+ honest evaluations.*

**Bold** values = best in series for that metric.

---

## 2. Pretrained Localization Track (vR.P.x)

| Version | Change | Encoder | Input | Pixel F1 | IoU | Pixel AUC | Img Acc | Epochs (Best) | Status |
|---------|--------|---------|-------|----------|-----|-----------|---------|---------------|--------|
| vR.P.0 | Baseline: ResNet-34 + UNet | ResNet-34 (frozen) | RGB | -- | -- | -- | -- | -- | Pending |
| vR.P.1 | Dataset fix + GT mask detect | ResNet-34 (frozen) | RGB | -- | -- | -- | -- | -- | Pending |
| vR.P.1.5 | Training speed optimizations | ResNet-34 (frozen) | RGB | -- | -- | -- | -- | -- | Pending |
| vR.P.2 | Gradual unfreeze (layer3+4) | ResNet-34 (partial) | RGB | -- | -- | -- | -- | -- | Pending |
| vR.P.3 | ELA as input (replace RGB) | ResNet-34 (frozen+BN) | ELA | -- | -- | -- | -- | -- | Pending |
| vR.P.4 | 4-channel (RGB + ELA) | ResNet-34 (frozen) | RGB+ELA | -- | -- | -- | -- | -- | Pending |
| vR.P.5 | ResNet-50 encoder | ResNet-50 (frozen) | RGB | -- | -- | -- | -- | -- | Pending |
| vR.P.6 | EfficientNet-B0 encoder | EfficientNet-B0 (frozen) | RGB | -- | -- | -- | -- | -- | Pending |

*All pretrained track results pending Kaggle execution.*

---

## 3. Delta Table (vs vR.1.1 Honest Baseline)

| Version | Change | Acc Delta | Macro F1 Delta | AUC Delta | Assessment |
|---------|--------|-----------|----------------|-----------|------------|
| vR.1.1 | (baseline) | 0.00pp | 0.0000 | 0.0000 | Reference |
| vR.1.2 | Augmentation | **-2.85pp** | -0.0300 | **-0.0590** | Catastrophic regression |
| vR.1.3 | Class weights | **+0.79pp** | +0.0084 | -0.0021 | Accuracy up, AUC flat |
| vR.1.4 | BatchNorm | +0.37pp | +0.0047 | -0.0065 | Marginal, AUC down |
| vR.1.5 | LR Scheduler | +0.58pp | +0.0068 | -0.0041 | Marginal, AUC down |
| vR.1.6 | Deeper CNN | **+1.85pp** | **+0.0199** | **+0.0056** | Best overall improvement |
| vR.1.7 | GAP | +0.79pp | +0.0096 | -0.0106 | Tied vR.1.3 acc, AUC down |

**Key insight:** Only vR.1.6 improved ALL three core metrics (Accuracy, Macro F1, ROC-AUC) from the honest baseline. Every other version traded AUC for accuracy.

---

## 4. Confusion Matrix Summary (Honest-Eval Runs Only)

| Version | TN | FP | FN | TP | FP Rate | FN Rate | Net Correct |
|---------|----|----|----|----|---------|---------|-------------|
| vR.1.1 | 964 | 160 | 90 | 679 | 14.2% | 11.7% | 1,643/1,893 |
| vR.1.2 | 949 | 175 | 99 | 670 | 15.6% | 12.9% | 1,619/1,893 |
| vR.1.3 | 972 | 152 | 76 | 693 | 13.5% | 9.9% | 1,665/1,893 |
| vR.1.4 | 962 | 162 | 62 | 707 | 14.4% | 8.1% | 1,669/1,893 |
| vR.1.5 | 977 | 147 | 62 | 707 | 13.1% | 8.1% | 1,684/1,893 |
| vR.1.6 | **983** | **141** | **44** | **725** | **12.5%** | **5.7%** | **1,708/1,893** |
| vR.1.7 | 960 | 164 | **41** | 728 | 14.6% | **5.3%** | 1,688/1,893 |

**Bold** = best in series for that column.

**FN rate trajectory:** 11.7% → 12.9% → 9.9% → 8.1% → 8.1% → 5.7% → **5.3%**. The model has become progressively better at detecting ALL tampered images.

**FP rate trajectory:** 14.2% → 15.6% → 13.5% → 14.4% → 13.1% → **12.5%** → 14.6%. FP rate improved through vR.1.6 but regressed in vR.1.7 (GAP increases false tampering accusations).

---

## 5. Per-Metric Champions

| Metric | Best Version | Value | Runner-Up | Value |
|--------|-------------|-------|-----------|-------|
| Test Accuracy | **vR.1.6** | 90.23% | vR.1.3/vR.1.7 | 89.17% |
| Au Precision | **vR.1.7** | 0.9590 | vR.1.6 | 0.9572 |
| Au Recall | **vR.1.6** | 0.8746 | vR.1.3 | 0.8648 |
| Au F1 | **vR.1.6** | 0.9140 | vR.1.7 | 0.9035 |
| Tp Precision | **vR.1.3** | 0.8431 | vR.1.1 | 0.8393 |
| Tp Recall | **vR.1.7** | 0.9467 | vR.1.6 | 0.9428 |
| Tp F1 | **vR.1.6** | 0.8869 | vR.1.7 | 0.8766 |
| Macro F1 | **vR.1.6** | 0.9004 | vR.1.7 | 0.8901 |
| ROC-AUC | **vR.1.6** | 0.9657 | vR.1.1 | 0.9601 |
| Lowest FN Rate | **vR.1.7** | 5.3% | vR.1.6 | 5.7% |
| Lowest FP Rate | **vR.1.6** | 12.5% | vR.1.5 | 13.1% |
| Total Params | **vR.1.7** | 63,970 | vR.1.6 | 13,826,530 |

**vR.1.6 dominates** 8 of 12 metrics. **vR.1.7** wins on 3 metrics (Au Precision, Tp Recall, lowest FN rate) — all reflecting its aggressive tampered-detection bias. **vR.1.3** wins Tp Precision.

---

## 6. Verdict Summary

### ETASR Track

| Verdict | Count | Versions |
|---------|-------|----------|
| POSITIVE | 2 | vR.1.3 (+0.79pp), vR.1.6 (+1.27pp) |
| NEUTRAL | 3 | vR.1.4 (-0.42pp), vR.1.5 (+0.21pp), vR.1.7 (-1.06pp) |
| REJECTED | 1 | vR.1.2 (-2.85pp) |
| Baseline | 2 | vR.1.0 (val-only), vR.1.1 (honest baseline) |

### Pretrained Track

All 8 versions pending Kaggle execution.
