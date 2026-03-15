# Approach Comparison: Ablation Study vs Research Paper Architecture

| Field | Value |
|-------|-------|
| **Auditor** | Principal AI Engineer (Automated Deep Audit) |
| **Date** | 2026-03-15 |
| **Scope** | Cross-approach comparison — Pretrained UNet ablation (vR.P.x) vs Research Paper CNN (Nagm et al. 2024) |

---

## 1. Executive Summary

Two fundamentally different approaches have been tested for image tampering detection:

| Dimension | Pretrained UNet Ablation (vR.P.x) | Research Paper CNN |
|-----------|---------------------------------------|--------------------------|
| **Task** | Pixel-level localization + classification | Image-level classification only |
| **Architecture** | UNet + ResNet-34 encoder (SMP) | Custom CNN (2-3 conv blocks + Dense) |
| **Framework** | PyTorch + SMP | TensorFlow/Keras |
| **Best Image Acc** | 87.59% (P.8) | 90.76% (Deeper CNN) |
| **Localization** | Pixel F1: 0.6985, IoU: 0.5367 | **NOT AVAILABLE** |
| **Assignment Fit** | Full compliance (masks + metrics) | Classification only — fails localization requirement |

**Bottom line:** The paper CNN achieves +3.2pp better image classification, but this advantage is irrelevant because the assignment requires pixel-level localization masks. The pretrained UNet track is the only viable path to assignment completion.

---

## 2. Approach Details

### Approach A: Pretrained UNet Ablation (Track 2 — vR.P.x)

**Architecture:** UNet encoder-decoder with pretrained ResNet-34 encoder and trainable decoder. The encoder provides multi-scale features via skip connections; the decoder upsamples to produce a 384x384 binary pixel mask.

**Key variants tested:**
| Version | Input | Encoder | Loss | Pixel F1 | Image Acc |
|---------|-------|---------|------|----------|-----------|
| P.1 | RGB | ResNet-34 (frozen) | BCE+Dice | 0.4546 | 70.15% |
| P.2 | RGB | ResNet-34 (gradual unfreeze) | BCE+Dice | 0.5117 | 69.04% |
| **P.3** | **ELA** | ResNet-34 (frozen+BN) | BCE+Dice | **0.6920** | **86.79%** |
| P.4 | RGB+ELA (4ch) | ResNet-34 (frozen+conv1+BN) | BCE+Dice | 0.7053 | 84.42% |
| P.5 | RGB | ResNet-50 (frozen) | BCE+Dice | 0.5137 | 72.00% |
| P.6 | RGB | EffNet-B0 (frozen) | BCE+Dice | 0.5217 | 70.68% |
| **P.8** | **ELA** | ResNet-34 (progressive unfreeze) | BCE+Dice | **0.6985** | **87.59%** |
| P.9 | ELA | ResNet-34 (frozen+BN) | Focal+Dice | 0.6923 | 87.16% |

**Strengths:**
- Produces pixel-level localization masks (assignment requirement)
- ELA input dramatically improved all metrics (+23.7pp Pixel F1)
- Skip connections preserve spatial detail for boundary reconstruction
- Pretrained encoder provides strong feature extraction with minimal training
- 13% trainable parameters (3.17M of 24.4M) — efficient fine-tuning

**Weaknesses:**
- Image accuracy (~87%) is lower than classification-only models (~91%)
- Pixel recall limited (~58%) — misses many tampered pixels
- Training is slower and requires more memory (384x384 pixel masks)
- More complex pipeline (GT mask loading, ELA normalization, SMP framework)

### Approach B: Research Paper CNN (Nagm et al. 2024)

**Architecture:** Simple CNN classifier — 2-3 convolution layers with MaxPool, Flatten, Dense output. Takes ELA preprocessed images at 150x150, outputs binary classification.

**Variants tested:**
| Run | Architecture | Params | Dataset | Test Acc | F1 |
|-----|-------------|--------|---------|----------|----|
| Paper-divg07 | 2×Conv32(5x5) + Dense(150) | 24.2M | divg07 (standard) | 90.33% | 0.9006 |
| Paper-sagnik | 2×Conv32(5x5) + Dense(150) | 24.2M | sagnik (LEAKED) | 100.00% | 1.0000 |
| **Deeper-divg07** | **3×Conv(64/128/256) + BN + Dense(512)** | **38.2M** | **divg07** | **90.76%** | **0.9082** |

**Strengths:**
- Higher image-level accuracy (90.76% vs 87.59%)
- Simpler pipeline (no GT masks needed, no decoder, no skip connections)
- Faster training (7 epochs, ~8 min vs 25+ epochs, ~45 min)
- ELA preprocessing proven effective for forgery detection

**Weaknesses:**
- **NO LOCALIZATION** — cannot produce pixel masks (fatal for assignment)
- Massive parameter bottleneck (99% in Flatten→Dense)
- Sagnik dataset run revealed data leak risk (100% accuracy on mask images)
- Overfitting without early stopping (paper arch: train 98.6% vs test 90.3%)
- Paper's claimed 94.14% not reproduced (best: 90.33%)

---

## 3. Head-to-Head Comparison

### 3.1 Performance Matrix

| Metric | Best UNet (P.8) | Best Paper CNN (Deeper) | Winner |
|--------|-----------------|------------------------|--------|
| Image Accuracy | 87.59% | **90.76%** | Paper CNN (+3.17pp) |
| Image F1 (Macro) | 0.8650 | **0.9082** | Paper CNN |
| Tampered Recall (Image) | 72.82% | **96.27%** | Paper CNN |
| FP Rate | **2.3%** | 2.6% | UNet |
| Pixel F1 | **0.6985** | N/A | UNet (only option) |
| Pixel IoU | **0.5367** | N/A | UNet (only option) |
| Pixel AUC | **0.9541** | N/A | UNet (only option) |
| Localization Output | **384×384 binary mask** | Nothing | UNet (only option) |

### 3.2 Model Complexity

| Dimension | UNet (P.8) | Paper CNN (Deeper) | Paper CNN (Original) |
|-----------|-----------|--------------------|--------------------|
| Total params | 24.4M | 38.3M | 24.2M |
| Trainable params | 3.17M (13%) | 38.3M (100%) | 24.2M (100%) |
| Input resolution | 384×384 | 150×150 | 150×150 |
| Dense bottleneck | None (conv decoder) | 37.9M (99%) | 24.2M (99.9%) |
| Model file size | 282.8 MB | 145.9 MB | 92.4 MB |
| Training time | ~45 min (25 epochs) | ~8 min (7 epochs) | ~35 min (40 epochs) |
| Framework | PyTorch + SMP | TensorFlow/Keras | TensorFlow/Keras |

### 3.3 Training Stability

| Metric | UNet (P.3/P.8/P.9) | Paper CNN (Original) | Paper CNN (Deeper) |
|--------|--------------------|--------------------|-------------------|
| Convergence | Steady, LR scheduling helpful | Overfits badly after epoch 8 | Fast (7 epochs, ES) |
| Val loss trajectory | Volatile but improving | Monotonically increasing | Peaked at epoch 5 |
| Early stopping | Patience=7 (effective) | Not used (fatal) | Patience=2 (effective) |
| Overfitting risk | Low (13% trainable) | Severe (100% trainable) | Moderate (100% trainable) |
| BN epoch-1 spike | Not observed (pretrained) | N/A | N/A |

### 3.4 Generalization Ability

| Test | UNet (ELA) | Paper CNN (divg07) | Paper CNN (sagnik) |
|------|-----------|--------------------|--------------------|
| Standard CASIA2 (divg07) | 87.59% Acc + masks | 90.33-90.76% Acc | N/A |
| Sagnik dataset | Not tested | N/A | **100% — DATA LEAK** |
| Cross-dataset robustness | Unknown | Unknown | Invalid |
| JPEG vs non-JPEG | Handles all formats | Handles all (paper says JPEG-only) | N/A |

---

## 4. Assignment Alignment

| Requirement | UNet (P.x) | Paper CNN | Assessment |
|-------------|-----------|-----------|------------|
| Pixel-level mask prediction | **YES** | **NO** | UNet is the only viable path |
| Train/val/test split | YES (70/15/15) | Mixed (70/15/15 and 80/10/10) | Both adequate |
| Standard metrics (F1, IoU, AUC) | YES — pixel + image level | F1 only (image level) | UNet more comprehensive |
| Visual results (Original/GT/Pred/Overlay) | YES — all 4 | NO — no mask output | UNet required |
| Model weights | YES (.pth) | YES (.h5) | Both save |
| Architecture explanation | YES (detailed) | Minimal | UNet better documented |
| Robustness testing (bonus) | Not yet tested | Not tested | Neither |

**The assignment explicitly requires "predict tampered regions" (pixel masks).** The paper CNN cannot do this. This single requirement eliminates the paper CNN as a submission candidate.

---

## 5. Research Paper Claims vs. Reproduction

| Claim (Nagm et al. 2024) | Reproduction Result | Gap |
|--------------------------|--------------------|----|
| Training Accuracy: 99.05% | 98.57% (divg07) | -0.48pp |
| **Testing Accuracy: 94.14%** | **90.33%** (divg07) | **-3.81pp** |
| Precision: 94.1% | 90.31% | -3.79pp |
| Recall: 94.07% | 90.10% | -3.97pp |
| Dataset: CASIA 2.0, JPEG only (9,501 images) | CASIA 2.0, ALL formats (12,614) | +3,113 images |

**The paper's results were not reproduced.** The most likely explanation is the dataset filtering: the paper used only JPEG images (9,501), while the reproduction used all formats (12,614). TIFF and BMP images may produce less meaningful ELA features, diluting accuracy. A reproduction using only JPEG images would be needed to confirm.

---

## 6. Recommendation: Continue Pretrained Ablation

### Verdict: **CONTINUE with pretrained ablation study (Track 2 — vR.P.x)**

### Rationale

1. **Assignment compliance:** Only the UNet track satisfies the core requirement of pixel-level localization masks. This is non-negotiable.

2. **Performance trajectory:** The pretrained track shows consistent improvement:
   - P.1 → P.3: +23.74pp Pixel F1 (ELA input breakthrough)
   - P.3 → P.8: +0.65pp Pixel F1 (gradual unfreeze)
   - Multiple untested experiments remain (P.7, P.10, P.11, P.12)

3. **ELA insight already incorporated:** The paper's primary contribution (ELA preprocessing) is already the foundation of the best UNet models (P.3+).

4. **Hybrid potential:** Insights from the paper CNN runs can inform UNet improvements:
   - Deeper decoder blocks (3-block CNN success suggests more decoder features help)
   - BN in decoder (the deeper CNN's BatchNorm stabilized training)
   - Early stopping validation (patience=2 was sufficient for fast convergence)

### What NOT to do

- **Do NOT switch to paper CNN** — it cannot produce localization masks
- **Do NOT cite the Sagnik dataset result** — it is a data leak, scientifically invalid
- **Do NOT abandon ELA input** — it remains the single most impactful change across all experiments

### Hybrid Ideas for Future Work

| Idea | Source | Application to UNet |
|------|--------|---------------------|
| Deeper decoder blocks | 3-block CNN success | Add more conv layers per decoder block |
| BatchNorm in decoder | CNN stability improvement | SMP UNet decoder already has BN |
| Aggressive early stopping | CNN converged in 7 epochs | Consider patience=5 for efficiency |
| 150x150 input (ablation) | Paper resolution | Test whether 150x150 ELA produces similar Pixel F1 (faster training) |
| JPEG-only filtering | Paper methodology | Test if filtering to JPEG-only images improves UNet metrics |

---

## 7. Future Experiments

Based on the evidence from all runs, the following experiments are proposed for the pretrained ablation track:

| # | Experiment | Motivation | Expected Impact | Difficulty | Priority |
|---|-----------|-----------|-----------------|------------|----------|
| 1 | **vR.P.7** — Extended training (50 epochs) | P.3 was still improving at epoch 25; P.8 peaked at epoch 23 of 32 | +2-5pp Pixel F1 | LOW | HIGH |
| 2 | **vR.P.10** — CBAM attention in decoder | Decoder lacks feature selection mechanism; P.6's EffNet SE attention helped | +1-3pp Pixel F1 | MEDIUM | HIGH |
| 3 | **vR.P.11** — Higher resolution (512×512) | Finer boundary detail; current 384×384 may lose thin edges | +2-4pp Pixel F1 | MEDIUM | MEDIUM |
| 4 | **vR.P.12** — ELA + data augmentation | Augmentation destroyed ETASR CNN (spatial memorization), but UNet's conv decoder should handle it | Unknown | LOW | MEDIUM |
| 5 | **vR.P.13** — EfficientNet-B0 + ELA | P.6 (EffNet+RGB) showed best param efficiency; combine with ELA input | +1-3pp Pixel F1 | MEDIUM | LOW |
| 6 | **vR.2.0** — ETASR localization via thresholding | Use CNN confidence as rough localization map; bridge classification and localization tracks | Qualitative only | HIGH | LOW |
| 7 | **vR.P.14** — JPEG-only dataset filtering | Reproduce paper's dataset filtering; test if excluding non-JPEG improves ELA-based UNet | ±1-2pp Pixel F1 | LOW | LOW |

### Priority Rationale

**HIGH priority:** P.7 (low effort, P.3 clearly undercooked at 25 epochs) and P.10 (attention modules are proven in computer vision, minimal parameter overhead). These have already been developed as notebooks.

**MEDIUM priority:** P.11 (resolution helps boundary detection but doubles memory) and P.12 (augmentation is risky given ETASR failure, but UNet architecture differs fundamentally).

**LOW priority:** P.13/P.14/vR.2.0 are exploratory and unlikely to provide breakthrough improvements.

---

## 8. Appendix: All Runs Ranked by Image Accuracy

| Rank | Run | Track | Image Acc | Pixel F1 | Localization |
|------|-----|-------|-----------|----------|-------------|
| 1 | Deeper CNN (divg07) | Standalone | **90.76%** | N/A | NO |
| 2 | ETASR vR.1.6 | ETASR | **90.23%** | N/A | NO |
| 3 | Paper CNN (divg07) | Standalone | 90.33% | N/A | NO |
| 4 | ETASR vR.1.3/1.7 | ETASR | 89.17% | N/A | NO |
| 5 | **vR.P.8** | **Pretrained** | **87.59%** | **0.6985** | **YES** |
| 6 | **vR.P.9** | **Pretrained** | **87.16%** | **0.6923** | **YES** |
| 7 | **vR.P.3** | **Pretrained** | **86.79%** | **0.6920** | **YES** |
| 8 | vR.P.4 | Pretrained | 84.42% | 0.7053 | YES |

**Key insight:** The top-3 by image accuracy are all classification-only models that cannot localize. The pretrained UNet trade-off is ~3pp lower image accuracy in exchange for pixel-level localization capability — a trade-off that the assignment requires.

---

## 9. Score Summary: All Audited Runs

### New Runs (this audit batch)

| Run | Arch | Data | Method | Eval | Docs | Alignment | **Total** |
|-----|------|------|--------|------|------|-----------|-----------|
| vR.P.3 Run-02 | 13 | 14 | 15 | 18 | 10 | 12 | **82/100** |
| **vR.P.8** | **14** | 14 | 14 | 18 | 12 | 12 | **84/100** |
| vR.P.9 | 13 | 14 | 12 | 17 | 11 | 11 | **78/100** |
| Paper CNN (divg07) | 10 | 12 | 10 | 12 | 8 | 4 | **56/100** |
| Paper CNN (sagnik) | 10 | 3 | 3 | 5 | 5 | 2 | **28/100** |
| Deeper CNN (divg07) | 12 | 12 | 14 | 13 | 10 | 5 | **66/100** |

### Combined Leaderboard (scores from this audit)

| Rank | Run | Score | Track |
|------|-----|-------|-------|
| 1 | vR.P.8 | 84/100 | Pretrained |
| 2 | vR.P.3 Run-02 | 82/100 | Pretrained |
| 3 | vR.P.9 | 78/100 | Pretrained |
| 4 | Deeper CNN (divg07) | 66/100 | Standalone |
| 5 | Paper CNN (divg07) | 56/100 | Standalone |
| 6 | Paper CNN (sagnik) | 28/100 | Standalone (**DATA LEAK**) |
