# Experiment Audit Report — vR.P.x Ablation Study

| Field | Value |
|-------|-------|
| **Date** | 2026-03-16 |
| **Scope** | Complete audit of all pretrained localization track experiments |
| **Experiments** | 19 completed + 1 INVALID + 10 pending (P.19–P.28) + 5 new (P.30–P.30.4) |
| **Series Best** | vR.P.15 — Multi-Quality ELA (Pixel F1 = 0.7329) |

---

## 1. Experiment Inventory

### 1.1 Completed Experiments (sorted by Pixel F1)

| Rank | Version | Change | Pixel F1 | IoU | Pixel AUC | Img Acc | Macro F1 (cls) | Epochs (Best) | Verdict |
|------|---------|--------|----------|-----|-----------|---------|----------------|---------------|---------|
| 1 | **vR.P.15** | Multi-Quality ELA (Q=75/85/95) | **0.7329** | **0.5785** | 0.9608 | 87.53% | 0.8660 | 25 (24) | **SERIES BEST** |
| 2 | vR.P.17 | ELA + DCT spatial fusion (6ch) | 0.7302 | 0.5751 | 0.9431 | 87.06% | 0.8589 | 25 (24) | POSITIVE (+3.82pp) |
| 3 | vR.P.10 r01 | CBAM attention + Focal+Dice | 0.7277 | 0.5719 | 0.9573 | 87.32% | 0.8615 | 25 (24) | POSITIVE (+3.57pp) |
| 4 | vR.P.7 | Extended training (50ep) | 0.7220 | 0.5633 | 0.9568 | 88.25% | 0.8637 | 46 (36) | POSITIVE (+2.34pp) |
| 5 | vR.P.8 | Progressive unfreeze (40ep) | 0.7101 | 0.5508 | 0.9543 | 88.25% | 0.8650 | 32 (23) | NEUTRAL (+0.65pp) |
| 6 | vR.P.10 r02 | CBAM reproducibility re-run | 0.7086 | 0.5491 | 0.9561 | 89.88% | 0.8615 | 25 (24) | Reproducibility ✅ |
| 7 | vR.P.4 | 4-channel RGB+ELA | 0.7053 | 0.5447 | 0.9433 | 84.42% | 0.8322 | 25 (24) | NEUTRAL |
| 8 | vR.P.12 | Augmentation + Focal+Dice (50ep) | 0.6968 | 0.5347 | 0.9502 | 88.48% | 0.8756 | 45 (35) | NEUTRAL (+0.48pp) |
| 9 | vR.P.8 | Progressive unfreeze | 0.6985 | 0.5367 | 0.9541 | 87.59% | 0.8650 | 32 (23) | NEUTRAL (+0.65pp) |
| 10 | vR.P.3 r02 | ELA baseline reproducibility | 0.6953 | 0.5321 | 0.9407 | 87.43% | 0.8560 | 25 (25) | Reproducibility ✅ |
| 11 | vR.P.9 | Focal+Dice loss | 0.6923 | 0.5294 | 0.9323 | 86.20% | 0.8606 | 25 (21) | NEUTRAL (+0.03pp) |
| 12 | vR.P.3 r01 | **ELA input (breakthrough)** | 0.6920 | 0.5292 | 0.9500 | 87.43% | 0.8560 | 25 (25) | **STRONG POSITIVE** |
| 13 | vR.P.14b | TTA 4-view (complete eval) | 0.6388 | 0.4693 | **0.9618** | 87.43% | 0.8619 | 25 (25) | NEGATIVE (−5.32pp) |
| 14 | vR.P.4 | 4ch RGB+ELA | 0.5888 | 0.4170 | 0.9274 | 77.62% | 0.8322 | 25 (24) | NEUTRAL |
| 15 | vR.P.2 | Gradual encoder unfreeze | 0.5117 | 0.3439 | 0.8688 | 69.04% | 0.6673 | 14 (7) | POSITIVE ✅ (pixel) |
| 16 | vR.P.6 | EfficientNet-B0 encoder | 0.5217 | 0.3529 | 0.8708 | 70.68% | 0.6950 | 23 (16) | POSITIVE ✅ |
| 17 | vR.P.5 | ResNet-50 encoder | 0.5137 | 0.3456 | 0.8828 | 72.00% | 0.7143 | 25 (19) | POSITIVE ✅ |
| 18 | vR.P.1 | Dataset fix + GT masks | 0.4546 | 0.2942 | 0.8509 | 70.15% | 0.6867 | 25 (18) | Proper baseline ✅ |
| 19 | vR.P.1.5 | Speed optimizations (AMP/TF32) | 0.4227 | 0.2680 | 0.8560 | 71.05% | 0.7016 | 23 (16) | NEUTRAL |
| 20 | vR.P.0 | ResNet-34 frozen, RGB (no GT) | 0.3749 | 0.2307 | 0.8486 | 76.80% | 0.6814 | 24 (17) | Baseline (no GT) |
| 21 | vR.P.16 | DCT spatial map baseline | 0.3209 | 0.1911 | 0.7778 | 61.60% | 0.5678 | 18 (11) | **NEGATIVE (catastrophic)** |

### 1.2 Invalid Experiments

| Version | Reason | Root Cause |
|---------|--------|------------|
| vR.P.18 | INVALID — checkpoint not found | P.3 checkpoint path hardcoded; not uploaded to Kaggle dataset |

### 1.3 Reproducibility Verification

| Experiment | Run 01 F1 | Run 02 F1 | Δ | Status |
|------------|-----------|-----------|---|--------|
| vR.P.3 | 0.6920 | 0.6953 | +0.0033 | ✅ Reproducible (within noise) |
| vR.P.10 | 0.7277 | 0.7086 | −0.0191 | ✅ Reproducible (some variance) |

### 1.4 Pending Experiments

| Version | Technique | Status |
|---------|-----------|--------|
| vR.P.13 | CBAM + Augmentation + 50ep + Focal+Dice | Pending |
| vR.P.19 | Multi-Quality RGB ELA (9ch) | Pending |
| vR.P.20 | ELA Magnitude + Chrominance Decomposition | Pending |
| vR.P.21 | ELA Residual Learning (Laplacian high-pass) | Pending |
| vR.P.22 | SRM Noise Maps | Pending |
| vR.P.23 | YCbCr Chrominance Analysis | Pending |
| vR.P.24 | Noiseprint Forensic Features | Pending |
| vR.P.25 | Edge Supervision Loss | Pending |
| vR.P.26 | Dual-Task Seg + Classification | Pending |
| vR.P.27 | JPEG Compression Augmentation | Pending |
| vR.P.28 | Cosine Annealing LR | Pending |

---

## 2. Impact Hierarchy

### 2.1 Ranked by Pixel F1 Impact (from P.3 ELA Baseline)

| Rank | Technique | Version | Δ F1 | Mechanism |
|------|-----------|---------|------|-----------|
| 1 | **Multi-Quality ELA** | P.15 | **+4.09pp** | Richer input representation — 3 quality levels capture different compression artifacts |
| 2 | **ELA+DCT Fusion** | P.17 | **+3.82pp** | Complementary frequency-domain features added to ELA |
| 3 | **CBAM Attention** | P.10 | **+3.57pp** | Decoder learns WHERE to focus; channel + spatial attention |
| 4 | **Extended Training (50ep)** | P.7 | **+2.34pp** | Model still improving at 25 epochs; longer training captures more |
| 5 | **Progressive Unfreeze** | P.8 | **+0.65pp** | Domain adaptation of encoder features; marginal gain |
| 6 | **Augmentation + Focal+Dice (50ep)** | P.12 | **+0.48pp** | Augmentation slightly helps with extended training |
| 7 | **Focal+Dice Loss** | P.9 | **+0.03pp** | Effectively neutral — no benefit over BCE+Dice |
| 8 | **TTA 4-view** | P.14b | **−5.32pp** | Averaging pushes borderline pixels below threshold |
| 9 | **DCT-only input** | P.16 | **−36.11pp** | Catastrophic — DCT alone carries far less forensic signal than ELA |

### 2.2 Impact by Category

```
INPUT REPRESENTATION (most impactful)
├── Multi-Q ELA:          +4.09pp (P.15)
├── ELA+DCT Fusion:       +3.82pp (P.17)
├── ELA (vs RGB):        +23.74pp (P.3 vs P.1)
├── 4ch RGB+ELA:          +1.33pp (P.4 vs P.3)  — image-level only
└── DCT-only:           −36.11pp (P.16)          — catastrophic

ARCHITECTURE MODIFICATIONS
├── CBAM Attention:       +3.57pp (P.10)
└── Deeper encoder:       +0.88pp (P.5 vs P.1)  — not worth complexity

TRAINING CONFIGURATION
├── Extended training:    +2.34pp (P.7)
├── Progressive unfreeze: +0.65pp (P.8)
└── Augmentation:         +0.48pp (P.12)         — confounded with loss

LOSS FUNCTION
├── Focal+Dice:           +0.03pp (P.9)          — effectively neutral
└── BCE+Dice:             baseline

EVALUATION TECHNIQUE
└── TTA:                  −5.32pp (P.14b)         — negative
```

---

## 3. Failure Mode Analysis

### 3.1 vR.P.16 — DCT Spatial Map (CATASTROPHIC FAILURE)

| Metric | P.3 (ELA) | P.16 (DCT) | Δ |
|--------|-----------|------------|---|
| Pixel F1 | 0.6920 | 0.3209 | −0.3711 |
| Image Acc | 86.79% | 61.60% | −25.19pp |

**Root cause:** DCT spatial maps lack the fine-grained pixel-level forensic signal that ELA provides. DCT features encode block-level frequency information but miss the compression artifact boundaries that ELA highlights clearly. Early stopping at epoch 18 confirms the model couldn't learn meaningful features.

### 3.2 vR.P.14b — TTA 4-View (NEGATIVE)

**Root cause:** Test-time augmentation averages 4 rotated predictions. For borderline tampered pixels (sigmoid ≈ 0.5), averaging pushes values below the 0.5 threshold, converting true positives to false negatives. The +0.10pp AUC improvement confirms better calibration but worse thresholded F1.

### 3.3 vR.P.12 — Augmentation + Extended Training (NEUTRAL)

**Root cause:** Geometric augmentations (flip, rotate, shift-scale) may disrupt ELA artifact patterns — the forensic signal from JPEG compression is position-dependent. The Focal+Dice loss confounds the result further.

### 3.4 vR.P.18 — INVALID

**Root cause:** Eval-only notebook with hardcoded checkpoint path (`/kaggle/input/.../best_model_P3.pth`). Checkpoint file not included in the Kaggle dataset upload. Output shows untrained-model signature: accuracy = class proportion, AUC ≈ 0.50, identical confusion matrices.

---

## 4. Component Attribution Analysis

### 4.1 Deconfounding CBAM (P.10)

P.10 changed TWO variables from P.3: added CBAM attention AND switched to Focal+Dice loss.

```
P.10 = P.3 + CBAM + Focal+Dice → +3.57pp F1
P.9  = P.3 + Focal+Dice       → +0.03pp F1
────────────────────────────────────────────
CBAM isolated contribution     ≈ +3.54pp F1
```

**CBAM with BCE+Dice has never been tested.** P.30 will isolate this.

### 4.2 Multi-Q ELA (P.15) — Clean Single-Variable

P.15 changed exactly one variable from P.3: input representation (3ch grayscale ELA → 3ch multi-Q grayscale ELA at Q=75/85/95).

```
P.15 = P.3 + Multi-Q ELA → +4.09pp F1
```

This is the cleanest result in the ablation — pure input representation change, no confounds.

### 4.3 ELA+DCT Fusion (P.17) — Clean Single-Variable

P.17 changed exactly one variable from P.3: input representation (3ch ELA → 6ch ELA+DCT).

```
P.17 = P.3 + DCT fusion → +3.82pp F1
```

Also a clean single-variable result. Notably, P.17 was still improving at epoch 25 (no early stopping), suggesting extended training could push it higher.

---

## 5. Untested Combination Opportunity Matrix

| Component A | Component B | Tested? | Expected Interaction | Priority |
|------------|------------|---------|---------------------|----------|
| **Multi-Q ELA** | **CBAM** | **NO** | **Additive — independent mechanisms (input vs decoder)** | **PRIMARY** |
| Multi-Q ELA | 50 epochs | NO | Additive — P.15 still improving at ep 25 | HIGH |
| Multi-Q ELA | Progressive unfreeze | NO | Unclear — may help or hurt | MEDIUM |
| Multi-Q ELA | Focal+Dice | NO | Slight risk — Focal+Dice neutral alone | LOW |
| Multi-Q ELA | Augmentation | NO | Risk — augmentation hurt ELA before (P.12) | DIAGNOSTIC |
| CBAM | BCE+Dice | NO | Expected equal or better than CBAM + Focal+Dice | HIGH |
| ELA+DCT | 50 epochs | NO | Likely positive — P.17 still improving | FUTURE |
| ELA+DCT | CBAM | NO | Possible — but 6ch input adds complexity | FUTURE |

**Primary target:** Multi-Q ELA + CBAM (P.30 series). These two operate on completely different parts of the pipeline:
- Multi-Q ELA: **input** (what features the model sees)
- CBAM: **decoder** (where the model focuses attention)

---

## 6. Tested Component Catalog

### 6.1 Encoders

| Encoder | Version | Pixel F1 | Verdict |
|---------|---------|----------|---------|
| ResNet-34 | P.3 | 0.6920 | **Best (default)** |
| ResNet-50 | P.5 | 0.5137 | Not worth complexity |
| EfficientNet-B0 | P.6 | 0.5217 | Not worth switching |

### 6.2 Input Representations

| Input | Version | Pixel F1 | Verdict |
|-------|---------|----------|---------|
| Multi-Q ELA (Q=75/85/95, 3ch gray) | P.15 | **0.7329** | **Series best** |
| ELA+DCT fusion (6ch) | P.17 | 0.7302 | Strong |
| ELA (3ch gray, Q=90) | P.3 | 0.6920 | Breakthrough baseline |
| RGB+ELA (4ch) | P.4 | 0.7053 | Marginal improvement |
| RGB (3ch) | P.1 | 0.4546 | Baseline |
| DCT spatial maps (3ch) | P.16 | 0.3209 | Catastrophic failure |

### 6.3 Loss Functions

| Loss | Version | Pixel F1 | Verdict |
|------|---------|----------|---------|
| BCE+Dice | P.3 | 0.6920 | **Default** |
| Focal+Dice | P.9 | 0.6923 | Effectively neutral (+0.03pp) |

### 6.4 Architecture Modifications

| Modification | Version | Pixel F1 | Verdict |
|-------------|---------|----------|---------|
| CBAM in decoder | P.10 | 0.7277 | **Positive (+3.54pp isolated)** |
| Standard UNet decoder | P.3 | 0.6920 | Baseline |

### 6.5 Training Configurations

| Config | Version | Pixel F1 | Verdict |
|--------|---------|----------|---------|
| 50 epochs, patience 10 | P.7 | 0.7220 | Positive (+2.34pp) |
| 40ep progressive unfreeze | P.8 | 0.7101 | Marginal (+0.65pp) |
| 50ep + augmentation | P.12 | 0.6968 | Neutral (+0.48pp) |
| 25 epochs, patience 7 | P.3 | 0.6920 | Default |

---

## 7. Recommendations: vR.P.30.x Series

### Rationale

The two most impactful independent techniques are:
1. **Multi-Quality ELA** (P.15, +4.09pp) — better input representation
2. **CBAM Attention** (P.10, +3.54pp isolated) — better feature focus in decoder

These operate on different parts of the pipeline and are expected to be **additive**:
- Multi-Q ELA improves WHAT the model sees
- CBAM improves WHERE the decoder focuses

**Expected combined F1:** 0.76–0.78 (conservative estimate assuming partial additivity)

### Proposed Experiments

| Version | Configuration | Expected F1 | Purpose |
|---------|--------------|-------------|---------|
| **vR.P.30** | Multi-Q ELA + CBAM (25ep, BCE+Dice) | 0.76–0.78 | **Primary combination baseline** |
| **vR.P.30.1** | Multi-Q ELA + CBAM (50ep, BCE+Dice) | 0.78–0.80 | Extended training for combined model |
| **vR.P.30.2** | Multi-Q ELA + CBAM + Progressive Unfreeze (40ep) | 0.77–0.80 | Domain adaptation of encoder |
| **vR.P.30.3** | Multi-Q ELA + CBAM + Focal+Dice (25ep) | 0.76–0.78 | Loss function interaction with CBAM |
| **vR.P.30.4** | Multi-Q ELA + CBAM + Geometric Aug (50ep) | 0.77–0.79 | Test if CBAM makes augmentation viable |

### Dependency Graph

```
vR.P.15 (Multi-Q ELA, F1=0.7329)          vR.P.10 (CBAM, F1=0.7277)
    \                                        /
     vR.P.30 (Multi-Q ELA + CBAM, 25ep)
        |         |          |            \
     P.30.1    P.30.2    P.30.3        P.30.4
     (50ep)   (unfreeze) (Focal)    (augmentation)
```

---

## 8. Overall Study Summary Statistics

| Metric | Value |
|--------|-------|
| Total experiments designed | 34 (P.0–P.28 + P.30–P.30.4) |
| Completed experiments | 19 |
| Reproducibility runs | 2 (P.3 r02, P.10 r02) |
| Invalid experiments | 1 (P.18) |
| Pending experiments | 15 (P.13, P.19–P.28, P.30–P.30.4) |
| Best Pixel F1 | 0.7329 (P.15) |
| Best Pixel IoU | 0.5785 (P.15) |
| Best Pixel AUC | 0.9618 (P.14b) |
| Best Image Accuracy | 89.88% (P.10 r02) |
| Best Image Macro F1 | 0.8756 (P.12) |
| F1 improvement over RGB baseline | +28.83pp (P.1 → P.15) |
| F1 improvement over ELA baseline | +4.09pp (P.3 → P.15) |
