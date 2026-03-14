# Cross-Run Comparison & Roast: All ETASR Ablation Runs

| Field | Value |
|-------|-------|
| **Auditor** | Principal AI Engineer (Automated Deep Audit) |
| **Date** | 2026-03-15 |
| **Scope** | 7 experiment runs: vR.ETASR, vR.0, vR.1, vR.1.1, vR.1.2, vR.1.3, vR.1.4 |
| **Paper** | ETASR_9593 — "Enhanced Image Tampering Detection using ELA and a CNN" |
| **Dataset** | CASIA v2.0 (12,614 images: 7,491 Au + 5,123 Tp) |
| **Architecture** | 2×Conv2D(32,5×5) + MaxPool + Flatten + Dense(256) + Dense(2,Softmax) — 29.52M params |
| **Paper Claim** | Acc=96.21%, Prec=98.58%, Rec=92.36%, F1=95.37% |

---

## 1. Executive Summary

Seven runs of the ETASR CNN. Two years of the paper's reported accuracy have been spent. The best honest result is **89.17%** — a 7.04pp gap that no ablation has seriously closed. Here's the one-line story of each:

| Run | One-Line Verdict |
|-----|------------------|
| **vR.ETASR** | A prototype that evaluates on its own training validation set and calls it a result. |
| **vR.0** | Fixed the split, broke the metrics. Weighted averages hide the tampered detection problem. |
| **vR.1** | A cosmetic rebrand of vR.ETASR with prettier visualizations and the same broken evaluation. |
| **vR.1.1** | The first honest result. Also the first time anyone noticed the model is 7.83pp below the paper. |
| **vR.1.2** | Augmentation destroyed the model. Best epoch was epoch 1. REJECTED. |
| **vR.1.3** | Class weights squeezed out +0.79pp. The val collapse at epoch 12 says the real problem is untouched. |
| **vR.1.4** | BatchNorm caused the worst epoch-1 spike in history (val_loss=16.13) then converged to the same numbers. |

---

## 2. Configuration Comparison

| Config | vR.ETASR | vR.0 | vR.1 | vR.1.1 | vR.1.2 | vR.1.3 | vR.1.4 |
|--------|----------|------|------|--------|--------|--------|--------|
| Data Split | 80/20 | 70/15/15 | 80/20 | 70/15/15 | 70/15/15 | 70/15/15 | 70/15/15 |
| Eval Set | Val ⚠️ | Test ✅ | Val ⚠️ | Test ✅ | Test ✅ | Test ✅ | Test ✅ |
| Metrics | Weighted ⚠️ | Weighted ⚠️ | Weighted ⚠️ | Per-class ✅ | Per-class ✅ | Per-class ✅ | Per-class ✅ |
| ROC-AUC | No ❌ | Yes ✅ | No ❌ | Yes ✅ | Yes ✅ | Yes ✅ | Yes ✅ |
| ELA Viz | No ❌ | No ❌ | No ❌ | Yes ✅ | Yes ✅ | Yes ✅ | Yes ✅ |
| Model Saved | No ❌ | No ❌ | No ❌ | Yes ✅ | Yes ✅ | Yes ✅ | Yes ✅ |
| Class Weights | None | None | None | None | None | **Yes** | **Yes** |
| BatchNorm | None | None | None | None | None | None | **Yes** |
| Augmentation | None | None | None | None | **Yes** | None | None |
| Seed | 42 | 42 | 42 | 42 | 42 | 42 | 42 |
| LR | 0.0001 | 0.0001 | 0.0001 | 0.0001 | 0.0001 | 0.0001 | 0.0001 |
| Batch Size | 32 | 32 | 32 | 32 | 32 | 32 | 32 |

**Reading this table:** Everything before vR.1.1 has at least one ⚠️ or ❌ in the evaluation methodology. Numbers from those runs are not trustworthy for comparison purposes.

---

## 3. Full Metrics Comparison

### Headline Numbers

| Metric | vR.ETASR | vR.0 | vR.1 | vR.1.1 | vR.1.2 | vR.1.3 | vR.1.4 |
|--------|----------|------|------|--------|--------|--------|--------|
| **Test Acc** | 89.89%* | 88.33% | 89.81%* | 88.38% | 85.53% | **89.17%** | 88.75% |
| **Macro F1** | 0.8972* | 0.8799 | 0.8964* | 0.8805 | 0.8505 | **0.8889** | 0.8852 |
| **ROC-AUC** | — | 0.9600 | — | **0.9601** | 0.9011 | 0.9580 | 0.9536 |
| Epochs | 13 (8) | 13 (8) | 13 (8) | 13 (8) | 6 (1) | 14 (9) | 8 (3) |

\* Val-set metrics (biased). Not directly comparable.

### Per-Class Metrics (Honest Evaluation Only: vR.1.1+)

| Metric | vR.1.1 | vR.1.2 | vR.1.3 | vR.1.4 |
|--------|--------|--------|--------|--------|
| Au Precision | 0.9170 | 0.8843 | **0.9290** | **0.9401** |
| Au Recall | 0.8843 | 0.8701 | 0.8852 | 0.8657 |
| Au F1 | 0.9004 | 0.8771 | **0.9066** | 0.9013 |
| Tp Precision | 0.8393 | 0.8145 | 0.8431 | 0.8240 |
| Tp Recall | 0.8830 | 0.8336 | 0.9012 | **0.9194** |
| Tp F1 | 0.8606 | 0.8239 | **0.8712** | 0.8691 |

### Confusion Matrix (Honest Evaluation Only)

| | vR.1.1 | vR.1.2 | vR.1.3 | vR.1.4 |
|-|--------|--------|--------|--------|
| TN (Au correct) | 994 | 978 | 995 | 973 |
| FP (Au→Tp) | 130 | 146 | 129 | 151 |
| FN (Tp→Au) | 90 | 128 | 76 | 62 |
| TP (Tp correct) | 679 | 641 | 693 | 707 |
| **FP Rate** | 11.6% | 13.0% | 11.5% | **13.4%** |
| **FN Rate** | 11.7% | 16.6% | 9.9% | **8.1%** |

**Trend:** The ablation is trading FP for FN. vR.1.4 has the best FN rate (8.1%) but the worst FP rate (13.4%). The model is increasingly biased toward calling things "tampered."

---

## 4. The Roast — Honest Assessment of Every Run

### vR.ETASR — "The Prototype That Believed Its Own Lies" (2/10)

A minimum viable prototype that does not meet basic ML evaluation standards. It evaluates on the same validation set used for model selection — this is Evaluation Methodology 101 failure. The reported 89.89% accuracy is inflated because the model has indirectly been optimized for this exact set via early stopping on `val_accuracy`. There is no ROC-AUC. There is no per-class breakdown. The model weights are not saved. The last code cell didn't even execute. This notebook exists as a historical artifact: the moment someone wired up ELA + CNN and said "good enough" before understanding what proper evaluation means.

**What it taught us:** The ELA + CNN pipeline can reach ~90% on the validation set. This is the ceiling, not the floor.

---

### vR.0 — "Fixed One Problem, Ignored Three Others" (3/10)

vR.0 is the strange middle child. It correctly implements the 70/15/15 split (the first to do so) and adds ROC-AUC computation, but then reports all metrics as **weighted averages** — which inflate the results toward the majority class (authentic). The 88.33% accuracy looks like a regression from vR.ETASR's 89.89%, but in reality it's the first honest number in the series. Nobody celebrated this. Nobody even noticed the FN rate tripled from vR.ETASR's biased 5.2% to an honest 12.0%. This version proves that when you stop cheating, the model looks worse. That's the point.

**What it taught us:** The honest accuracy is ~88%, not ~90%.

---

### vR.1 — "A Fresh Coat of Paint on a Condemned Building" (2/10)

vR.1 adds ELA visualizations, sample prediction grids, and a table of contents. It looks professional. It is, functionally, identical to vR.ETASR: same 80/20 split, same val-set evaluation, same lack of ROC-AUC, same inflated numbers. The 89.81% accuracy is as meaningless as vR.ETASR's 89.89% — the 0.08pp difference is noise on a biased evaluation set. This version exists because the project decided to branch the ablation numbering from "vR.1" rather than "vR.0", creating a naming fork that confused the lineage. It's a cosmetic upgrade: the architectural equivalent of hanging curtains in a house with no foundation.

**What it taught us:** Visualization doesn't fix methodology.

---

### vR.1.1 — "The Moment of Honesty" (5/10)

The most important version in the series, and the only one where the evaluation methodology is correct. vR.1.1 implements everything that should have been there from the start: proper 70/15/15 train/val/test split, per-class precision/recall/F1, ROC-AUC, ELA visualization, and model saving. The result? **88.38%** — a 7.83pp gap from the paper's claimed 96.21%.

Let's be blunt about what this number means. A model with 29.5 million parameters, trained on 8,829 images, using ELA features at 128×128 resolution, with a fixed learning rate of 0.0001 and no regularization beyond dropout — this model hits a hard wall at ~88%. The val-loss collapse at epochs 12-13 (val_acc drops from 0.8864 to 0.7960) reveals the fundamental problem: the Flatten→Dense(256) layer has 29.49M parameters that memorize the training set and then catastrophically fail on validation data.

Score: 5/10 because it finally does evaluation right. But the result is sobering.

**What it taught us:** The true baseline is 88.38%, not 89.89%. The paper gap is 7.83pp.

---

### vR.1.2 — "How to Destroy a Model in One Easy Step" (1/10)

The hypothesis was sound: data augmentation (horizontal flip, vertical flip, rotation ±15°) should reduce overfitting and improve generalization. Instead, **every single metric regressed**:

- Accuracy: 85.53% (−2.85pp)
- Macro F1: 0.8505 (−0.0300)
- ROC-AUC: 0.9011 (−0.0590 — the largest AUC drop in the series)
- Best epoch: **1** (the model peaked at epoch 1 and got worse for 5 more epochs)

Why did augmentation fail so catastrophically? Because this architecture cannot handle spatial transforms. The Flatten→Dense layer converts a 60×60×32 feature map into a 115,200-dimensional vector and passes it through a dense layer. It memorizes the exact pixel positions of features. When you flip or rotate the image, those pixel positions change, and the dense layer's memorized weights become anti-correlated with the new patterns. Augmentation exposes the model's total lack of spatial invariance.

The learning rate (0.0001) is also too low for augmented training — the model needs to rapidly update its weights to learn from the augmented distribution, but can barely adjust before early stopping kills it.

Score: 1/10. Correctly REJECTED. But the failure is informative: it proves the Flatten→Dense architecture is fundamentally incompatible with augmentation.

**What it taught us:** You cannot augment your way out of an architecture problem.

---

### vR.1.3 — "A Band-Aid on a Bullet Wound" (5/10)

Class weights (Au=0.8420, Tp=1.2310) produced the best test accuracy in the series: **89.17%**. Tampered recall improved from 0.8830 to 0.9012 — 14 more tampered images correctly detected. The FN rate dropped from 11.7% to 9.9%. Macro F1 rose to 0.8889. This is a genuine, if modest, improvement. Verdict: POSITIVE.

Now the roast. Class weights shifted the decision boundary — they told the model "missing a tampered image costs 1.23× more than a false alarm." The model responded by calling more things tampered. This is threshold tuning, not feature learning. The evidence: **ROC-AUC regressed from 0.9601 to 0.9580.** ROC-AUC is threshold-independent. If the model learned to better separate the classes, AUC would rise. It fell. The model's discriminative ability did not improve; it just moved where it draws the line.

And the catastrophic val collapse at epochs 12-14? Still there. Val_acc drops from 0.8901 to 0.8039 (−8.62pp). Class weights do not address the 29.5M-param overfitting bomb. They never could.

Score: 5/10. Real improvement, but the underlying pathology is untreated.

**What it taught us:** Class weights improve recall but don't improve the model.

---

### vR.1.4 — "The Cure Was Worse Than the Disease (But the Patient Survived)" (5/10)

BatchNormalization after each Conv2D was supposed to stabilize training — smooth the loss landscape, prevent the epoch 12 collapse, enable longer convergence. Instead, it produced the **worst single-epoch catastrophe in the entire series**: val_loss = **16.13** at epoch 1 (val_acc = 0.4059). That's worse than random. The BN running statistics were uninitialized, and the first batch through an untrained network with BN creates extreme activation distributions.

The model recovered by epoch 3 (val_loss = 0.275, val_acc = 0.886), then early stopping fired at epoch 8 because nothing improved after that. Total training: 8 epochs. Compare to vR.1.3's 14 epochs. BN made the model converge faster, but to essentially the same place: 88.75% (−0.42pp from vR.1.3). Macro F1: 0.8852 (−0.0037). ROC-AUC: 0.9536 (−0.0044). All within noise. Verdict: NEUTRAL.

The silver lining: tampered recall of **0.9194** is the best in the entire series. BN + class weights together are pushing the model toward correctly identifying tampered images. The price: FP rate of 13.4% — the worst in the honest-evaluation era. The model is increasingly trigger-happy.

Score: 5/10. It didn't hurt, and the tampered recall is useful, but the epoch 1 catastrophe and shortened training are concerning.

**What it taught us:** BN needs a learning rate scheduler to handle the warmup instability.

---

## 5. Ablation Verdict Table

All deltas are computed from **vR.1.1** (the honest baseline). Only runs with proper test-set evaluation (vR.1.1+) are included.

| Version | Change | Test Acc | Δ Acc | Macro F1 | Δ F1 | ROC-AUC | Δ AUC | Verdict |
|---------|--------|----------|-------|----------|------|---------|-------|---------|
| **vR.1.1** | Eval fix (baseline) | 88.38% | — | 0.8805 | — | 0.9601 | — | **BASELINE** |
| **vR.1.2** | Augmentation | 85.53% | −2.85pp | 0.8505 | −0.0300 | 0.9011 | −0.0590 | **REJECTED** ❌ |
| **vR.1.3** | Class weights | 89.17% | +0.79pp | 0.8889 | +0.0084 | 0.9580 | −0.0021 | **POSITIVE** ✅ |
| **vR.1.4** | BatchNorm | 88.75% | +0.37pp | 0.8852 | +0.0047 | 0.9536 | −0.0065 | **NEUTRAL** |

**Running total improvement from baseline:** +0.37pp accuracy, +0.0047 Macro F1, −0.0065 AUC.

The AUC has regressed in every single ablation. Not one change has improved the model's threshold-independent discrimination. The accuracy gains are almost entirely from threshold/boundary shifting (class weights) and noise.

---

## 6. Run Lineage

```
vR.ETASR (bare prototype, 80/20 val eval)
│
├── vR.1 (cosmetic, 80/20 val eval)
│
├── vR.0 (70/15/15, weighted metrics)
│
└── vR.1.1 ← DEFINITIVE HONEST BASELINE (88.38%)
    │
    ├── vR.1.2 (augmentation) ← REJECTED (85.53%)
    │
    └── vR.1.3 (class weights) ← POSITIVE (89.17%)
        │
        └── vR.1.4 (BatchNorm) ← NEUTRAL (88.75%)
            │
            └── vR.1.5 (LR scheduler) ← PENDING
                │
                └── vR.1.6 (deeper CNN) ← PLANNED
                    │
                    └── vR.1.7 (GAP) ← PLANNED
```

**Key:** vR.1.2 is a dead branch. The lineage continues from vR.1.1 → vR.1.3 → vR.1.4 → ...

---

## 7. Training Dynamics Across All Runs

### Val Accuracy at Key Epochs

| Epoch | vR.ETASR* | vR.0 | vR.1* | vR.1.1 | vR.1.2 | vR.1.3 | vR.1.4 |
|-------|-----------|------|-------|--------|--------|--------|--------|
| 1 | 0.7336 | 0.7381 | 0.7395 | 0.8399 | **0.8663** | 0.8399 | **0.4059** |
| 3 | 0.8616 | 0.8570 | 0.8615 | 0.8695 | 0.8531 | 0.8695 | **0.8858** |
| 5 | 0.8777 | 0.8764 | 0.8805 | 0.8679 | 0.8510 | 0.8710 | 0.8832 |
| 8 | **0.8989** | 0.8807 | **0.8981** | **0.8864** | — | 0.8890 | 0.8811 |
| 9 | 0.8701 | 0.8710 | 0.8690 | 0.8858 | — | **0.8901** | — |
| 12 | 0.8736 | — | 0.8756 | **0.7960** | — | **0.8399** | — |
| 13 | 0.8713 | — | 0.8673 | **0.7934** | — | **0.8266** | — |

\* Val-set evaluation (biased). "—" means training had already stopped.

### The Instability Pattern

**Every run with proper training length (≥12 epochs) shows catastrophic val collapse:**

| Run | Collapse Epoch | Val Acc Drop | Val Loss Spike |
|-----|---------------|--------------|---------------|
| vR.ETASR | 9 | 0.899→0.870 (−2.9pp) | Moderate |
| vR.1 | 9 | 0.898→0.869 (−2.9pp) | Moderate |
| vR.1.1 | 12 | 0.886→0.796 (−9.0pp) | Severe |
| vR.1.3 | 12 | 0.890→0.804 (−8.6pp) | Severe |
| vR.1.4 | 1 | (BN warmup, different cause) | Extreme (16.13) |

**Root cause:** The Flatten→Dense(256) layer creates a 115,200→256 bottleneck with 29.49M parameters. These parameters memorize training patterns. When the training loss drops below ~0.16, the memorized features overfit and become anti-correlated with validation data, causing the collapse.

**Evidence:** The collapse always occurs when train_acc exceeds ~0.93 and train_loss drops below ~0.16. The model has memorized the training set and begins actively misclassifying validation samples.

---

## 8. The Unsolved Problems

### Problem 1: The 7.04pp Paper Gap

Best honest result: **89.17%**. Paper claims: **96.21%**. Gap: **7.04pp**.

After 4 ablation experiments (eval fix, augmentation, class weights, BatchNorm), the gap has closed by exactly **0.79pp** — all from class weights. The other three ablations contributed zero net accuracy improvement. At this rate, closing the remaining 6.25pp gap would require ~8 more equally successful ablations, each contributing +0.79pp. This is unlikely.

The gap is probably not closable with this architecture. The paper's claimed 96.21% may be:
- On a different dataset split (not reproducible without their exact split)
- On a subset of CASIA v2.0 (not all 12,614 images)
- Using a different ELA implementation (OpenCV vs PIL)
- Simply overstated

### Problem 2: No Localization

Seven runs. Zero pixels of localization. The assignment requires pixel-level tampering masks. This CNN produces a single 2-class softmax output. It cannot localize. This is why Track 2 (pretrained UNet, vR.P.x) exists.

### Problem 3: The 29.5M Parameter Bottleneck

```
Flatten (60×60×32 = 115,200) → Dense(256) → 29,491,456 parameters
```

This single layer contains **99.9%** of all model parameters. It memorizes pixel-exact spatial patterns from the 60×60×32 feature maps. This causes:
- Training instability (val collapse)
- Incompatibility with augmentation (vR.1.2 failure)
- Massive overfitting (train_acc=0.93 vs val_acc=0.89)
- Extremely high memory usage for a trivial model

**Fix:** GlobalAveragePooling2D (vR.1.7) would reduce this to 32→256 = 8,192 parameters. A 3,600× reduction.

### Problem 4: Declining ROC-AUC

| Run | ROC-AUC | Δ from vR.1.1 |
|-----|---------|---------------|
| vR.1.1 | **0.9601** | — |
| vR.1.2 | 0.9011 | −0.0590 |
| vR.1.3 | 0.9580 | −0.0021 |
| vR.1.4 | 0.9536 | −0.0065 |

No ablation has improved AUC. The model's threshold-independent discriminatory power peaked at the honest baseline and has regressed with every change. All accuracy improvements come from threshold shifting (class weights), not better feature learning.

### Problem 5: Training Instability

Every run collapses between epochs 9-14. The model cannot train longer than ~12 epochs without catastrophic validation failure. This limits the optimization horizon: the model plateaus at epoch 8-9 and then self-destructs. ReduceLROnPlateau (vR.1.5) may help by reducing the learning rate at the plateau, allowing gentler convergence beyond epoch 9 without triggering the collapse.

---

## 9. Feature Completeness Matrix

| Feature | vR.ETASR | vR.0 | vR.1 | vR.1.1 | vR.1.2 | vR.1.3 | vR.1.4 |
|---------|----------|------|------|--------|--------|--------|--------|
| ELA preprocessing | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Proper test set | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ |
| Per-class metrics | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ |
| ROC-AUC | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ |
| Confusion matrix | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Training curves | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| ELA visualization | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ |
| Sample predictions | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Model saved | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ |
| Class weights | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ |
| BatchNorm | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| LR scheduler | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Localization | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Pixel-level masks | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |

**Across 7 runs:** 0/7 have localization. 0/7 have LR scheduling. 4/7 have proper test evaluation. The feature completeness is improving, but the assignment's core requirement (localization) remains unaddressed in Track 1.

---

## 10. Appendix — Master Metric Reference

### All Metrics, All Runs

| Metric | vR.ETASR* | vR.0 | vR.1* | vR.1.1 | vR.1.2 | vR.1.3 | vR.1.4 |
|--------|-----------|------|-------|--------|--------|--------|--------|
| Accuracy | 89.89% | 88.33% | 89.81% | 88.38% | 85.53% | 89.17% | 88.75% |
| Au Prec | 0.9607 | — | 0.9555 | 0.9170 | 0.8843 | 0.9290 | 0.9401 |
| Au Rec | 0.8652 | — | 0.8660 | 0.8843 | 0.8701 | 0.8852 | 0.8657 |
| Au F1 | 0.9104 | — | 0.9086 | 0.9004 | 0.8771 | 0.9066 | 0.9013 |
| Tp Prec | 0.8279 | — | 0.8316 | 0.8393 | 0.8145 | 0.8431 | 0.8240 |
| Tp Rec | 0.9483 | — | 0.9457 | 0.8830 | 0.8336 | 0.9012 | 0.9194 |
| Tp F1 | 0.8840 | — | 0.8850 | 0.8606 | 0.8239 | 0.8712 | 0.8691 |
| Macro F1 | 0.8972 | 0.8799 | 0.8964 | 0.8805 | 0.8505 | 0.8889 | 0.8852 |
| ROC-AUC | — | 0.9600 | — | 0.9601 | 0.9011 | 0.9580 | 0.9536 |
| TN | 1296 | — | 1302 | 994 | 978 | 995 | 973 |
| FP | 202 | — | 201 | 130 | 146 | 129 | 151 |
| FN | 53 | — | 56 | 90 | 128 | 76 | 62 |
| TP | 972 | — | 964 | 679 | 641 | 693 | 707 |
| FP Rate | 13.5% | — | 13.4% | 11.6% | 13.0% | 11.5% | 13.4% |
| FN Rate | 5.2% | — | 5.5% | 11.7% | 16.6% | 9.9% | 8.1% |
| Epochs | 13 (8) | 13 (8) | 13 (8) | 13 (8) | 6 (1) | 14 (9) | 8 (3) |
| Params | 29,520,034 | 29,520,034 | 29,520,034 | 29,520,034 | 29,520,034 | 29,520,034 | 29,520,290 |

\* Val-set metrics. "—" = not computed or not applicable.

**vR.0 note:** Per-class metrics not available in standard format; only weighted averages and macro F1 were reported.

---

## 11. Final Verdict

After 7 runs, the ETASR CNN reproduction has demonstrated:

1. **The paper's results are not reproducible.** Best honest accuracy: 89.17% vs claimed 96.21%.
2. **The architecture is fundamentally limited.** The 29.5M-param Flatten→Dense bottleneck causes instability, overfitting, and augmentation incompatibility.
3. **Training tricks have diminishing returns.** Class weights helped (+0.79pp). BatchNorm was neutral. The model is near its capacity ceiling.
4. **No localization exists.** Track 2 (pretrained UNet) is addressing this.

**Remaining experiments expected to help:**
- **vR.1.5 (ReduceLROnPlateau):** Should fix training instability by reducing LR at plateau. May push past 89%.
- **vR.1.6 (deeper CNN):** Adding a 3rd Conv2D layer should improve feature extraction.
- **vR.1.7 (GlobalAveragePooling):** Replacing Flatten→Dense should eliminate the 29.5M bottleneck and the root cause of all instability.

The ETASR track is a documentation exercise in systematic ablation methodology. The actual assignment submission depends on Track 2.
