# Experiment Tracking Table

| Field | Value |
|-------|-------|
| **Date** | 2026-03-15 |
| **Scope** | Complete metrics reference for all experimental runs |
| **Paper** | ETASR_9593 -- "Enhanced Image Tampering Detection using ELA and a CNN" |
| **Versions Covered** | ETASR Track: vR.1.0--vR.1.7 (8 runs) / Pretrained Track: vR.P.0--vR.P.18 (19 runs, 3 pending) / Standalone: 4 runs |

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

### Full Metrics Table

| Version | Change | Encoder | Input | Pixel F1 | IoU | Pixel AUC | Img Acc | Macro F1 | Img AUC | Epochs (Best) | Trainable | Verdict |
|---------|--------|---------|-------|----------|-----|-----------|---------|----------|---------|---------------|-----------|---------|
| **vR.P.0** | Baseline (divg07, ELA pseudo-masks) | ResNet-34 (frozen) | RGB | 0.3749 | 0.2307 | 0.8486 | 70.63% | 0.6814 | 0.7860 | 24 (17) | 3.15M | Baseline (no GT) |
| **vR.P.1** | Dataset fix + GT masks | ResNet-34 (frozen) | RGB | 0.4546 | 0.2942 | 0.8509 | 70.15% | 0.6867 | 0.7785 | 25 (18) | 3.15M | Proper baseline |
| **vR.P.1.5** | Speed opts (AMP, TF32) | ResNet-34 (frozen) | RGB | 0.4227 | 0.2680 | 0.8560 | 71.05% | 0.7016 | 0.7980 | 23 (16) | 3.15M | NEUTRAL (speed) |
| **vR.P.2** | Gradual unfreeze (L3+L4) | ResNet-34 (partial) | RGB | 0.5117 | 0.3439 | 0.8688 | 69.04% | 0.6673 | 0.7196 | 14 (7) | 23.1M | POSITIVE (pixel) |
| **vR.P.3** | **ELA input (BN unfrozen)** | ResNet-34 (frozen+BN) | **ELA** | **0.6920** | **0.5291** | **0.9528** | **86.79%** | **0.8560** | **0.9502** | 25 (25) | 3.17M | **STRONG POSITIVE** |
| **vR.P.4** | 4ch RGB+ELA (conv1+BN) | ResNet-34 (frozen+conv1+BN) | RGB+ELA | **0.7053** | **0.5447** | 0.9433 | 84.42% | 0.8322 | 0.9229 | 25 (24) | 3.18M | NEUTRAL |
| **vR.P.5** | ResNet-50 encoder | ResNet-50 (frozen) | RGB | 0.5137 | 0.3456 | 0.8828 | 72.00% | 0.7143 | 0.8126 | 25 (19) | 9.01M | POSITIVE |
| **vR.P.6** | EfficientNet-B0 encoder | EffNet-B0 (frozen) | RGB | 0.5217 | 0.3529 | 0.8708 | 70.68% | 0.6950 | 0.7801 | 23 (16) | 2.24M | POSITIVE |
| vR.P.3 r02 | Reproducibility re-run | ResNet-34 (frozen+BN) | ELA | 0.6920 | 0.5291 | 0.9528 | 86.79% | 0.8560 | 0.9502 | 25 (25) | 3.17M | BASELINE (re-run) |
| **vR.P.8** | **Progressive unfreeze** | ResNet-34 (progressive) | **ELA** | **0.6985** | **0.5367** | **0.9541** | **87.59%** | **0.8650** | **0.9578** | 32 (23) | 3.17M→14.1M | NEUTRAL (+0.65pp) |
| vR.P.9 | Focal+Dice loss | ResNet-34 (frozen+BN) | ELA | 0.6923 | 0.5294 | 0.9323 | 87.16% | 0.8606 | 0.9076 | 25 (21) | 3.17M | NEUTRAL (+0.03pp) |
| **vR.P.7** | **Extended training (50ep)** | ResNet-34 (frozen+BN) | **ELA** | **0.7154** | **0.5569** | 0.9504 | 87.37% | 0.8637 | 0.9433 | 46 (36) | 3.17M | **POSITIVE (+2.34pp)** |
| **vR.P.10** | **CBAM attention + Focal** | ResNet-34 (frozen+BN) | **ELA** | **0.7277** | **0.5719** | **0.9573** | 87.32% | 0.8615 | **0.9633** | 25 (24) | 3.18M | **POSITIVE (+3.57pp)** |
| vR.P.10 r02 | Reproducibility re-run | ResNet-34 (frozen+BN) | ELA | 0.7277 | 0.5719 | 0.9573 | 87.32% | 0.8615 | 0.9633 | 25 (24) | 3.18M | BASELINE (re-run) |
| **vR.P.12** | **Augmentation + Focal+Dice** | ResNet-34 (frozen+BN) | **ELA** | 0.6968 | 0.5347 | 0.9502 | **88.48%** | **0.8756** | 0.9427 | 45 (35) | 3.17M | **NEUTRAL (+0.48pp)** |
| **vR.P.14** | **Test-Time Augmentation (TTA)** | ResNet-34 (frozen+BN) | **ELA** | 0.6388* | 0.4693* | 0.9618* | --** | --** | --** | 25 (25) | 3.17M | **NEGATIVE (-5.32pp)** |
| vR.P.16 | DCT spatial feature maps | ResNet-34 (frozen+BN) | **DCT** | *pending* | *pending* | *pending* | *pending* | *pending* | *pending* | — | 3.17M | *pending* |
| vR.P.17 | ELA + DCT fusion (6ch) | ResNet-34 (frozen+conv1+BN) | **ELA+DCT** | *pending* | *pending* | *pending* | *pending* | *pending* | *pending* | — | 3.18M | *pending* |
| vR.P.18 | Compression robustness test | ResNet-34 (frozen+BN) | **ELA** | *eval-only* | *eval-only* | *eval-only* | *eval-only* | *eval-only* | *eval-only* | — (P.3 model) | 3.17M | *measurement* |

*P.14 TTA metrics; without TTA: Pixel F1=0.6919, IoU=0.5290 (identical to P.3). **Image-level metrics unavailable due to code bug (cell 18 crash).*

**Bold** values = best in series for that metric.

---

### Pretrained Delta Table (vs vR.P.1 Proper Baseline)

| Version | Change | Pixel F1 Delta | IoU Delta | Pixel AUC Delta | Img Acc Delta | Assessment |
|---------|--------|----------------|-----------|-----------------|---------------|------------|
| vR.P.1 | (baseline) | 0.0000 | 0.0000 | 0.0000 | 0.00pp | Reference |
| vR.P.1.5 | Speed opts | -0.0319 | -0.0262 | +0.0051 | +0.90pp | AMP noise, not causal |
| vR.P.2 | Gradual unfreeze | +0.0571 | +0.0497 | +0.0179 | -1.11pp | Pixel positive, image negative |
| **vR.P.3** | **ELA input** | **+0.2374** | **+0.2349** | **+0.1019** | **+16.64pp** | **Breakthrough** |
| vR.P.4 | 4ch RGB+ELA | +0.2507 | +0.2505 | +0.0924 | +14.27pp | Best absolute, marginal over P.3 |
| vR.P.5 | ResNet-50 | +0.0591 | +0.0514 | +0.0319 | +1.85pp | Encoder depth helps modestly |
| vR.P.6 | EffNet-B0 | +0.0671 | +0.0587 | +0.0199 | +0.53pp | Best param efficiency |
| vR.P.3 r02 | Reproducibility re-run | +0.2374 | +0.2349 | +0.1019 | +16.64pp | Confirms P.3 result |
| **vR.P.8** | **Progressive unfreeze** | **+0.2439** | **+0.2425** | **+0.1032** | **+17.44pp** | Best overall Pixel F1 |
| vR.P.9 | Focal+Dice loss | +0.2377 | +0.2352 | +0.0814 | +17.01pp | AUC regression |
| **vR.P.12** | **Augmentation + Focal+Dice** | **+0.2422** | **+0.2405** | **+0.0993** | **+18.33pp** | Marginal over P.3; instability |
| **vR.P.14** | **TTA (4 views)** | **+0.1842*** | **+0.1751*** | **+0.1109*** | --** | TTA hurts at threshold=0.5 |

*P.14 TTA metrics. Without TTA: identical to P.3 baseline. **Image-level metrics not available (code crash).*

### Pretrained Confusion Matrix Summary

| Version | TN | FP | FN | TP | FP Rate | FN Rate |
|---------|----|----|----|----|---------|---------|
| vR.P.0 | 831 | 293 | 263 | 506 | 26.1% | 34.2% |
| vR.P.1 | 870 | 254 | 311 | 458 | 22.6% | 40.4% |
| vR.P.1.5 | 836 | 288 | 260 | 509 | 25.6% | 33.8% |
| vR.P.2 | 903 | 221 | 365 | 404 | 19.7% | 47.5% |
| **vR.P.3** | **1,094** | **30** | 220 | 549 | **2.7%** | 28.6% |
| vR.P.4 | 1,052 | 72 | 223 | 546 | 6.4% | 29.0% |
| vR.P.5 | 816 | 308 | 222 | **547** | 27.4% | 28.9% |
| vR.P.6 | 855 | 269 | 286 | 483 | 23.9% | 37.2% |
| vR.P.3 r02 | 1,094 | 30 | 220 | 549 | 2.7% | 28.6% |
| **vR.P.8** | **1,098** | **26** | 209 | 560 | **2.3%** | 27.2% |
| vR.P.9 | 1,088 | 36 | 207 | 562 | 3.2% | 26.9% |
| **vR.P.7** | 1,084 | 40 | 199 | 570 | 3.6% | **25.9%** |
| **vR.P.10** | **1,102** | **22** | 218 | 551 | **2.0%** | 28.3% |
| vR.P.10 r02 | 1,102 | 22 | 218 | 551 | 2.0% | 28.3% |
| **vR.P.12** | 1,095 | 29 | 189 | **580** | 2.6% | **24.6%** |
| vR.P.14* | -- | -- | -- | -- | -- | -- |

*P.14 image-level confusion matrix not available (code bug crashed cell 18).*

### Pretrained Per-Metric Champions

| Metric | Best Version | Value | Runner-Up | Value |
|--------|-------------|-------|-----------|-------|
| Pixel F1 | **vR.P.10** | 0.7277 | vR.P.7 | 0.7154 |
| Pixel IoU | **vR.P.10** | 0.5719 | vR.P.7 | 0.5569 |
| Pixel AUC | **vR.P.10** | 0.9573 | vR.P.8 | 0.9541 |
| Pixel Precision | **vR.P.8** | 0.8857 | vR.P.10 | 0.8611 |
| Pixel Recall | **vR.P.10** | 0.6300 | vR.P.7 | 0.6245 |
| Image Accuracy | **vR.P.12** | 88.48% | vR.P.8 | 87.59% |
| Image Macro F1 | **vR.P.12** | 0.8756 | vR.P.8 | 0.8650 |
| Image ROC-AUC | **vR.P.10** | 0.9633 | vR.P.8 | 0.9578 |
| Lowest FP Rate | **vR.P.10** | 2.0% | vR.P.8 | 2.3% |
| Lowest FN Rate | **vR.P.12** | 24.6% | vR.P.7 | 25.9% |
| Param Efficiency | **vR.P.6** | 2.24M trainable | vR.P.3 | 3.17M |

**vR.P.10 dominates** 5 of 11 metrics (Pixel F1, IoU, AUC, Pixel Recall, FP rate). **vR.P.12** wins 3 (Image Acc, Image F1, FN rate). **vR.P.8** wins Pixel Precision. **vR.P.7** is runner-up in multiple categories. **vR.P.6** wins param efficiency.

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

| Verdict | Count | Versions |
|---------|-------|----------|
| STRONG POSITIVE | 1 | vR.P.3 (+23.74pp from P.1) |
| POSITIVE | 5 | vR.P.2 (+5.71pp), vR.P.5 (+9.10pp), vR.P.6 (+6.71pp), vR.P.7 (+2.34pp), vR.P.10 (+3.57pp) |
| NEUTRAL | 5 | vR.P.1.5 (speed only), vR.P.4 (+1.33pp from P.3), vR.P.8 (+0.65pp from P.3), vR.P.9 (+0.03pp from P.3), vR.P.12 (+0.48pp from P.3) |
| NEGATIVE | 1 | vR.P.14 (-5.32pp TTA, code bug) |
| Baseline | 3 | vR.P.0 (no GT), vR.P.1 (proper baseline), vR.P.3 r02 (reproducibility) |

---

## 7. Standalone Research Paper Architecture Runs

### Metrics Table

| Run | Architecture | Params | Dataset | Test Acc | F1 | Test Loss | Epochs | Localization | Verdict |
|-----|-------------|--------|---------|----------|-----|-----------|--------|-------------|---------|
| Paper CNN (divg07) | 2×Conv32(5x5) + Dense(150) | 24.2M | divg07 (standard) | 90.33% | 0.9006 | 0.6185 | 40 (no ES) | NO | Not assignment-viable |
| Paper CNN (sagnik) | 2×Conv32(5x5) + Dense(150) | 24.2M | sagnik (**LEAKED**) | 100.00% | 1.0000 | 0.0000 | 40 (no ES) | NO | **INVALID — DATA LEAK** |
| **Deeper CNN (divg07)** | **3×Conv(64/128/256) + BN + Dense(512)** | **38.2M** | **divg07** | **90.76%** | **0.9082** | **0.2178** | 7 (ES) | NO | Best classification |
| **Deeper CNN (sagnik)** | **3×Conv(64/128/256) + BN + Dense(512)** | **38.3M** | **sagnik (LEAKED)** | **99.95%** | 0.9995 | 0.0071 | 6 (ES) | NO | **INVALID — DATA LEAK** |

### Paper Claims vs Reproduction

| Claim (Nagm et al. 2024) | Reproduction Result | Gap |
|--------------------------|--------------------|-----|
| Training Accuracy: 99.05% | 98.57% (divg07) | -0.48pp |
| **Testing Accuracy: 94.14%** | **90.33%** (divg07) | **-3.81pp** |
| Precision: 94.1% | 90.31% | -3.79pp |
| Recall: 94.07% | 90.10% | -3.97pp |

**Note:** All standalone CNN runs are classification-only — they cannot produce localization masks and do not satisfy the assignment requirement.
