# Paper Reproduction Results — Nagm et al. 2024

| Field | Value |
|-------|-------|
| **Date** | 2026-03-15 |
| **Paper** | "Enhanced Image Tampering Detection using ELA and a CNN" (Nagm et al. 2024, PeerJ Computer Science) |
| **Scope** | Reproduction of the paper's CNN architecture and comparison with the pretrained ablation track |
| **Runs** | 3 standalone notebooks (2 datasets, 2 architectures) |

---

## 1. Paper Architecture

### Original Specification (Nagm et al. 2024)

The paper describes a simple CNN for binary image classification (authentic vs tampered) using Error Level Analysis (ELA) as preprocessing:

```
Input: 150×150×3 (ELA at Q=90, scaled to [0,1])
    |
Conv2D(32, 5×5, ReLU, valid)       Params: 2,432
    |
Conv2D(32, 5×5, ReLU, valid)       Params: 25,632
    |
MaxPooling2D(2×2)
    |
Flatten (161,312)                   Params: 0
    |
Dense(150, ReLU)                    Params: 24,196,950
    |
Dense(2, sigmoid)                   Params: 302
    |
Total: 24,225,316 params
```

**Critical observation:** 99.88% of parameters reside in the Flatten→Dense(150) connection. The model essentially memorizes flattened feature maps rather than learning spatial hierarchies.

### Deeper Variant (Our Extension)

We also tested a deeper 3-block architecture with modern practices:

```
Input: 150×150×3 (ELA at Q=90, scaled to [0,1])
    |
Conv2D(64, 3×3, ReLU) + BatchNorm + MaxPool(2×2)     Params: ~2,048
    |
Conv2D(128, 3×3, ReLU) + BatchNorm + MaxPool(2×2)    Params: ~74,368
    |
Conv2D(256, 3×3, ReLU) + BatchNorm + MaxPool(2×2)    Params: ~296,192
    |
Dropout(0.5) + Flatten (73,984)
    |
Dense(512, ReLU) + Dropout(0.5)                       Params: 37,880,320
    |
Dense(2, sigmoid)                                      Params: 1,026
    |
Total: 38,253,954 params
```

---

## 2. Reproduction Results

### Three Runs Executed

| Run | Architecture | Dataset | Test Acc | F1 | Test Loss | Epochs | Early Stopping |
|-----|-------------|---------|----------|-----|-----------|--------|----------------|
| Paper CNN (divg07) | 2×Conv32 | divg07 (standard) | 90.33% | 0.9006 | 0.6185 | 40 | NO |
| Paper CNN (sagnik) | 2×Conv32 | sagnik | 100.00% | 1.0000 | 0.0000 | 40 | NO |
| **Deeper CNN (divg07)** | **3×Conv+BN** | **divg07** | **90.76%** | **0.9082** | **0.2178** | **7** | **YES (patience=5)** |

### Paper Claims vs Reproduction

| Metric | Paper Claim | Our Reproduction | Gap |
|--------|------------|-----------------|-----|
| Training Accuracy | 99.05% | 98.57% | -0.48pp |
| **Testing Accuracy** | **94.14%** | **90.33%** | **-3.81pp** |
| Precision | 94.1% | 90.31% | -3.79pp |
| Recall | 94.07% | 90.10% | -3.97pp |

**The paper's results were not reproduced.** The most likely explanation is dataset filtering: the paper specifies "JPEG images only" (9,501 images) while our reproduction used all image formats (12,614 images, +3,113). TIFF and BMP images produce less meaningful ELA features, potentially diluting accuracy.

---

## 3. Critical Finding: Sagnik Dataset Data Leak

The Sagnik dataset run achieved **100% test accuracy** — a scientifically invalid result caused by a data leak.

**Evidence:**
1. X data range: [0.0, 0.76] (vs standard [0.0, 1.0] for divg07)
2. Dataset path contains "MASK" references
3. Perfect accuracy from the first epoch
4. Training and test loss converge to zero

**Mechanism:** The dataset likely loads ground truth mask images (binary black/white) as the "tampered" class inputs instead of actual tampered photographs. Since masks have fundamentally different pixel distributions than photographs, classification becomes trivial.

**Action:** This result must not be cited, used, or referenced as a valid finding. It serves only as a cautionary example about dataset validation.

---

## 4. Key Insights from Reproduction

### 4.1 Early Stopping is Critical

| Run | Early Stopping | Test Loss | Test Acc | Overfitting |
|-----|---------------|-----------|----------|-------------|
| Paper CNN | NO | 0.6185 | 90.33% | Severe (train 98.57%) |
| Deeper CNN | YES (patience=5) | **0.2178** | **90.76%** | Minimal |

The paper architecture trained for 40 epochs without early stopping, producing severe overfitting (test loss 3x worse than optimal). The deeper CNN with early stopping achieved better accuracy in just 7 epochs.

### 4.2 BatchNorm + Deeper Conv Blocks Help Marginally

The deeper architecture (+BatchNorm, +3rd conv block, +Dropout) improved accuracy by only 0.43pp (90.33% → 90.76%). Most of this improvement likely comes from early stopping and BatchNorm stabilization rather than the additional convolutional depth. The Dense bottleneck (99%+ of parameters) remains the fundamental limitation.

### 4.3 ELA Preprocessing is Confirmed Effective

All runs confirm that ELA preprocessing enables effective forgery detection. The 90%+ classification accuracy demonstrates that JPEG recompression artifacts contain strong forensic signal. This confirms the foundation of the pretrained track's ELA input strategy (P.3+).

---

## 5. Comparison with Pretrained Ablation Track

### Head-to-Head: Best Classification vs Best Localization

| Metric | Deeper CNN (Best Class.) | UNet P.8 (Best Localization) | Winner |
|--------|-------------------------|------------------------------|--------|
| Image Accuracy | **90.76%** | 87.59% | Deeper CNN (+3.17pp) |
| Image F1 | **0.9082** | 0.8650 | Deeper CNN |
| Tampered Recall | **96.27%** | 72.82% | Deeper CNN |
| FP Rate | 2.6% | **2.3%** | UNet |
| Pixel F1 | N/A | **0.6985** | UNet (only option) |
| Pixel IoU | N/A | **0.5367** | UNet (only option) |
| Pixel AUC | N/A | **0.9541** | UNet (only option) |
| Localization mask | NO | **384×384 binary** | UNet (only option) |

### Why Classification Accuracy is Irrelevant

The assignment explicitly requires **pixel-level localization masks** — "predict tampered regions." The paper CNN achieves higher image classification accuracy (+3.17pp) but produces zero spatial information. This is the fundamental reason the pretrained ablation track (vR.P.x) is the only viable path to assignment completion.

The +3.17pp accuracy gap is the cost of localization. The UNet must make pixel-level predictions for every 384×384 position, a much harder task than binary image classification.

---

## 6. Architectural Comparison

| Dimension | Paper CNN | Deeper CNN | ETASR vR.1.6 | UNet P.8 |
|-----------|-----------|-----------|-------------|----------|
| Task | Classification | Classification | Classification | **Segmentation** |
| Framework | TensorFlow/Keras | TensorFlow/Keras | TensorFlow/Keras | PyTorch + SMP |
| Input | ELA 150×150 | ELA 150×150 | ELA 128×128 | ELA 384×384 |
| Architecture | 2×Conv32 + Dense | 3×Conv+BN + Dense | 3×Conv+BN + Dense/GAP | UNet + ResNet-34 |
| Total params | 24.2M | 38.3M | 13.8M | 24.4M |
| Trainable | 24.2M (100%) | 38.3M (100%) | 13.8M (100%) | 3.17M (13%) |
| Dense bottleneck | 99.88% | 99.0% | 99.66% | **None** |
| Output | Binary label | Binary label | Binary label | **384×384 mask** |
| Training time | ~35 min | ~8 min | ~15 min | ~45 min |
| Model size | 92.4 MB | 145.9 MB | ~55 MB | 282.8 MB |

**Key insight:** The UNet is the only architecture that eliminates the Flatten→Dense bottleneck, replacing it with a convolutional decoder that preserves spatial information at every resolution. This structural difference is what enables pixel-level output.

---

## 7. What the Paper Runs Taught Us

### Validated Findings

1. **ELA preprocessing works** — 90%+ accuracy confirms JPEG compression artifacts are strong forensic signals
2. **Early stopping is essential** — saves training time AND improves generalization
3. **BatchNorm helps classification CNNs** — stabilizes training without downsides
4. **Deeper conv blocks provide marginal improvement** — 3 blocks > 2 blocks, but the Dense bottleneck limits the ceiling

### Actionable Insights for the Pretrained Track

| Paper Finding | Application to UNet |
|--------------|---------------------|
| ELA is effective at 150×150 | Already incorporated (P.3+, at 384×384 for more detail) |
| Early stopping prevents overfitting | Already used (patience=7 in pretrained track) |
| BatchNorm stabilizes training | P.3's BN unfreeze strategy already exploits this |
| Deeper features help marginally | Consider deeper decoder blocks in future experiments |
| 96.27% tampered recall achievable | UNet's 72.82% recall has significant room for improvement |

### Hybrid Ideas for Future Work

| Idea | Source | Application |
|------|--------|-------------|
| Deeper decoder blocks | 3-block CNN's success | Add more conv layers per UNet decoder block |
| More aggressive BN | CNN's BN contribution | Already have BN unfreeze; consider more BN layers in decoder |
| Lower patience for early stopping | Deeper CNN converged in 7 epochs | Test patience=5 instead of patience=7 |
| JPEG-only dataset filtering | Paper methodology | Filter to JPEG-only to test if non-JPEG images dilute ELA quality |

---

## 8. Scores and Verdicts

| Run | Score | Verdict | Localization |
|-----|-------|---------|-------------|
| Paper CNN (divg07) | 56/100 | Not assignment-viable — classification only | NO |
| Paper CNN (sagnik) | 28/100 | **INVALID — DATA LEAK** | NO |
| Deeper CNN (divg07) | 66/100 | Best classification, not assignment-viable | NO |
| **Deeper CNN (sagnik)** | **34/100** | **INVALID — DATA LEAK** (99.95% accuracy) | NO |

### Recommendation

**CONTINUE with the pretrained ablation track (vR.P.x).** The paper CNN runs confirm that ELA preprocessing is effective (already incorporated since P.3) but demonstrate that classification-only architectures cannot satisfy the assignment's localization requirement. The pretrained UNet is the only viable path to assignment completion.

---

## 9. Update: Post-Audit Findings (2026-03-15)

### New Standalone Run: ELA-CNN-Forgery-sagnik

A deeper CNN architecture (3×Conv with BN + Dense(512)) trained on the Sagnik dataset achieved 99.95% accuracy. This independently confirms the data leak found in the paper-architecture Sagnik run (100%). The Sagnik dataset is **permanently invalidated** — both architectures achieve near-perfect accuracy, which is physically impossible for real forgery detection.

### Pretrained Track Progress Since Paper Comparison

Since the standalone comparison, the pretrained track has advanced significantly:

| Metric | At Time of Comparison (P.8 best) | Current Best | Improvement |
|--------|----------------------------------|-------------|-------------|
| Pixel F1 | 0.6985 (P.8) | **0.7277** (P.10) | +2.92pp |
| Image Accuracy | 87.59% (P.8) | **88.48%** (P.12) | +0.89pp |
| Series length | P.0--P.9 (11 runs) | P.0--P.14 (16 runs) | +5 runs |

**Key developments:**
- **P.10 (CBAM):** Attention mechanism in decoder achieved series-best Pixel F1 (0.7277, +3.57pp from P.3)
- **P.7 (extended training):** 50 epochs confirmed P.3 was undertrained (best epoch 36, Pixel F1=0.7154)
- **P.12 (augmentation):** Best image accuracy (88.48%) but marginal pixel-level improvement
- **P.14 (TTA):** Negative result — TTA hurt Pixel F1 by 5.32pp at threshold=0.5
- **P.10 Run-02:** Perfect reproducibility confirmed (bit-identical metrics)
