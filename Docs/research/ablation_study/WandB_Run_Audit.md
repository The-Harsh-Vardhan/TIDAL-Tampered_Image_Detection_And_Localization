# W&B Run Audit — vR.P.x Pretrained Localization Track

**Date:** 2026-03-16
**Scope:** All 22 experiment runs tracked in Weights & Biases
**Dataset:** CASIA 2.0 Image Tampering Detection Dataset — https://www.kaggle.com/datasets/divg07/casia-20-image-tampering-detection-dataset
**Series Best:** vR.P.19 (Pixel F1 = 0.7965)
**Run Count:** 22 (21 executed + 1 template)

---

## 1. Executive Summary

The vR.P.x series constitutes a structured ablation study for pixel-level tampered image localization using a UNet architecture with pretrained encoders. Over 22 experiments, the study systematically varied input preprocessing, encoder architecture, attention mechanisms, loss functions, training schedules, and augmentation strategies.

**Key findings:**
- ELA preprocessing is the single most impactful variable (+26pp F1 over raw RGB)
- Multi-quality RGB ELA at 9 channels (vR.P.19) is the best single configuration (F1 = 0.7965)
- ResNet-34 is sufficient — neither ResNet-50 nor EfficientNet-B0 improved results
- CBAM attention helps (+3-5pp) but is secondary to input quality
- Extended training is the cheapest reliable improvement (+2-3pp)
- Alternative forensic features (DCT, YCbCr, Noiseprint) all underperform ELA
- Diminishing returns are apparent after F1 ~0.78

---

## 2. Scoring Rubric

Each run is scored across 10 categories as specified:

| Category | Max Points | Criteria |
|----------|-----------|----------|
| Problem Understanding | /10 | Awareness of detection vs localization distinction, tampering taxonomy |
| Dataset Handling | /10 | GT mask usage, normalization, extensions, class balance |
| Model Architecture | /10 | Encoder choice justification, freeze strategy, decoder design |
| Training Pipeline | /10 | Loss function, optimizer, scheduler, AMP, checkpointing |
| Experiment Design | /15 | Single-variable fidelity, parent lineage, ablation rigor |
| Evaluation Metrics | /10 | Pixel F1/IoU/AUC, image-level accuracy, confusion matrix |
| Visual Results | /10 | Orig/GT/Pred/Overlay grids, training curves, ROC curves |
| Documentation | /10 | Inline comments, change log, version headers |
| Reproducibility | /10 | SEED, deterministic splits, W&B logging, model saving |
| Engineering Rigor | /5 | Code quality, error handling, Kaggle compatibility |
| **Total** | **/100** | |

---

## 3. Run-by-Run Audit

---

### Run: vR.P.0 — Pretrained ResNet-34 UNet (RGB Baseline)

**Description:** First pretrained experiment. Frozen ResNet-34 encoder with UNet decoder on raw RGB images. No ELA preprocessing.

**Architecture:** ResNet-34 (ImageNet, frozen) → UNet decoder → 1-class segmentation
**Training Setup:** BCE+Dice loss, Adam lr=1e-3, 25 epochs (patience=7), best epoch=24

**Results:**

| Metric | Value |
|--------|-------|
| Pixel F1 | 0.3948 |
| Pixel IoU | 0.2459 |
| Pixel AUC | 0.9955 |

**Strengths:**
- Establishes a clean baseline with no confounding variables
- High AUC (0.9955) suggests model can rank pixels well even if binary threshold is poor
- Proper W&B logging and infrastructure

**Weaknesses:**
- Used pseudo-masks instead of real GT masks (dataset issue not yet discovered)
- Raw RGB input gives the encoder no forensic signal — it's trying to detect tampering from texture alone
- The high AUC vs low F1 discrepancy suggests extreme class imbalance in predictions

**Reviewer Comments:**
- The 0.9955 AUC is misleading — with mostly-zero GT masks, predicting "no tampering everywhere" inflates AUC. This metric should have raised a red flag.
- No data augmentation, no unfreezing strategy — the decoder alone cannot learn forensic features from frozen ImageNet features on raw RGB.
- This run's primary value is as a negative result proving that RGB alone is insufficient.

**Assignment Alignment Score:**

| Category | Score |
|----------|-------|
| Problem Understanding | 7/10 |
| Dataset Handling | 4/10 |
| Model Architecture | 6/10 |
| Training Pipeline | 7/10 |
| Experiment Design | 10/15 |
| Evaluation Metrics | 7/10 |
| Visual Results | 6/10 |
| Documentation | 7/10 |
| Reproducibility | 8/10 |
| Engineering Rigor | 3/5 |
| **Total** | **65/100** |

**Verdict:** ADEQUATE — Valid baseline, but pseudo-mask issue undermines results.

---

### Run: vR.P.1 — Dataset Fix + GT Mask Detection

**Description:** Fixed dataset discovery to use proper GT masks from `sagnikkayalcse52/casia-spicing-detection-localization`. No model changes from P.0.

**Architecture:** ResNet-34 (ImageNet, frozen) → UNet decoder → 1-class segmentation
**Training Setup:** BCE+Dice loss, Adam lr=1e-3, 25 epochs (patience=7), best epoch=18

**Results:**

| Metric | Value |
|--------|-------|
| Pixel F1 | 0.4546 |
| Pixel IoU | 0.2942 |
| Pixel AUC | 0.8509 |

**Strengths:**
- Critical infrastructure fix — proper GT masks now used
- +6pp F1 improvement from data fix alone validates the importance of clean labels
- AUC dropped from 0.9955 to 0.8509 — this is actually healthier (less inflated)

**Weaknesses:**
- Still using raw RGB input (no forensic preprocessing)
- Frozen encoder limits capacity
- No augmentation

**Reviewer Comments:**
- The AUC drop from P.0 (0.9955 → 0.8509) confirms P.0's AUC was artificially inflated by pseudo-masks. Good that this was caught.
- This is purely an infrastructure fix — no scientific contribution, but essential for all subsequent experiments.
- Should have been labeled vR.P.0.1 (patch) rather than vR.P.1 (new experiment) since only data changed.

**Assignment Alignment Score:**

| Category | Score |
|----------|-------|
| Problem Understanding | 7/10 |
| Dataset Handling | 7/10 |
| Model Architecture | 6/10 |
| Training Pipeline | 7/10 |
| Experiment Design | 8/15 |
| Evaluation Metrics | 7/10 |
| Visual Results | 6/10 |
| Documentation | 7/10 |
| Reproducibility | 8/10 |
| Engineering Rigor | 4/5 |
| **Total** | **67/100** |

**Verdict:** ADEQUATE — Essential data fix, but not a meaningful experiment per se.

---

### Run: vR.P.1.5 — Training Speed Optimizations

**Description:** Infrastructure-only changes: AMP mixed precision, TF32, parallel data loading, async GPU transfers. No model or data changes.

**Architecture:** Same as P.1
**Training Setup:** Same as P.1 + AMP + TF32

**Results:**

| Metric | Value |
|--------|-------|
| Pixel F1 | 0.4227 |
| Pixel IoU | 0.2680 |
| Pixel AUC | 0.8560 |

**Strengths:**
- Proper infrastructure investment — AMP gives ~2x speedup for free
- Demonstrates that AMP does not degrade metrics significantly

**Weaknesses:**
- Pixel F1 dropped 3.19pp from P.1 (0.4546 → 0.4227) — AMP may have introduced precision issues
- Still on raw RGB

**Reviewer Comments:**
- The F1 regression from P.1 is concerning. AMP should not cause this much degradation — suggests potential numerical instability in the loss computation or gradient scaling.
- As an infra-only run this is acceptable, but the regression should have been investigated before proceeding.
- The version number "1.5" breaks the integer sequence — acceptable for a patch but confusing in the lineage.

**Assignment Alignment Score:**

| Category | Score |
|----------|-------|
| Problem Understanding | 6/10 |
| Dataset Handling | 7/10 |
| Model Architecture | 5/10 |
| Training Pipeline | 8/10 |
| Experiment Design | 6/15 |
| Evaluation Metrics | 7/10 |
| Visual Results | 6/10 |
| Documentation | 6/10 |
| Reproducibility | 8/10 |
| Engineering Rigor | 4/5 |
| **Total** | **63/100** |

**Verdict:** WEAK — Infra improvement with unexplained regression. Limited scientific value.

---

### Run: vR.P.3 — ELA as Input (Replace RGB) ★ BREAKTHROUGH

**Description:** Replaced raw RGB input with Error Level Analysis (ELA) at Q=90. ELA-specific normalization applied. BN layers unfrozen for domain adaptation.

**Architecture:** ResNet-34 (ImageNet, frozen body + BN unfrozen) → UNet decoder
**Training Setup:** BCE+Dice loss, Adam lr=1e-3, 25 epochs (patience=7), best epoch=25

**Results:**

| Metric | Value |
|--------|-------|
| Pixel F1 | 0.6920 |
| Pixel IoU | 0.5291 |
| Pixel AUC | 0.9528 |

**Strengths:**
- **+23.74pp F1 improvement** over P.1 — the single largest gain in the entire series
- Validates the core hypothesis: forensic preprocessing (ELA) is essential for localization
- BN unfreezing is a smart decision — allows domain adaptation without full fine-tuning
- ELA-specific normalization (computed from dataset) is methodologically sound

**Weaknesses:**
- Only tested Q=90 — no quality level ablation
- Best epoch = 25 (hit the epoch limit) suggests undertrained
- No augmentation despite ELA being a 3-channel pseudo-image

**Reviewer Comments:**
- This is the most important run in the series. The +23.74pp F1 gain proves that the input representation matters far more than the model architecture.
- The BN unfreezing insight is underrated — it allows the frozen encoder to adapt its normalization statistics to the ELA domain without risking catastrophic forgetting.
- Should have immediately tested multiple ELA quality levels rather than waiting until P.15/P.19.
- Training hit the epoch limit — extended training (done later in P.7) was an obvious next step that should have been done here.

**Assignment Alignment Score:**

| Category | Score |
|----------|-------|
| Problem Understanding | 9/10 |
| Dataset Handling | 8/10 |
| Model Architecture | 8/10 |
| Training Pipeline | 7/10 |
| Experiment Design | 13/15 |
| Evaluation Metrics | 8/10 |
| Visual Results | 8/10 |
| Documentation | 8/10 |
| Reproducibility | 9/10 |
| Engineering Rigor | 4/5 |
| **Total** | **82/100** |

**Verdict:** STRONG — Foundational breakthrough. Most impactful single experiment.

---

### Run: vR.P.4 — 4-Channel Input (RGB + ELA)

**Description:** Combined RGB (3ch) with ELA grayscale (1ch) as 4-channel input. Conv1 unfrozen to accept 4 channels.

**Architecture:** ResNet-34 (ImageNet, frozen body + conv1 unfrozen + BN unfrozen) → UNet decoder
**Training Setup:** BCE+Dice loss, Adam lr=1e-3, 25 epochs (patience=7), best epoch=24

**Results:**

| Metric | Value |
|--------|-------|
| Pixel F1 | 0.7053 |
| Pixel IoU | 0.5447 |
| Pixel AUC | 0.9433 |

**Strengths:**
- +1.33pp F1 over P.3 — marginal improvement from adding RGB context to ELA
- Conv1 unfreezing is necessary and well-handled for non-3-channel inputs
- Demonstrates that RGB provides some complementary information to ELA

**Weaknesses:**
- AUC dropped vs P.3 (0.9433 vs 0.9528) — RGB may introduce noise
- Only 1 ELA channel (grayscale, not full-color) — loses chrominance information
- The improvement is small enough to question whether the added complexity is worth it

**Reviewer Comments:**
- The 1.33pp gain is within noise range for a single run. Without confidence intervals or repeated runs, this cannot be reliably attributed to the RGB+ELA combination.
- Using grayscale ELA (1ch) instead of full-color ELA (3ch) was a questionable choice — chrominance ELA carries tampering signal.
- The experiment would have been more informative as a 6-channel input (RGB 3ch + ELA 3ch) to avoid discarding chrominance.

**Assignment Alignment Score:**

| Category | Score |
|----------|-------|
| Problem Understanding | 8/10 |
| Dataset Handling | 8/10 |
| Model Architecture | 7/10 |
| Training Pipeline | 7/10 |
| Experiment Design | 11/15 |
| Evaluation Metrics | 8/10 |
| Visual Results | 7/10 |
| Documentation | 7/10 |
| Reproducibility | 9/10 |
| Engineering Rigor | 4/5 |
| **Total** | **76/100** |

**Verdict:** ADEQUATE — Marginal gain, but informative as an ablation data point.

---

### Run: vR.P.5 — ResNet-50 Encoder

**Description:** Replaced ResNet-34 with ResNet-50 (deeper bottleneck blocks). Raw RGB input.

**Architecture:** ResNet-50 (ImageNet, frozen) → UNet decoder (~8.7M trainable decoder params)
**Training Setup:** BCE+Dice loss, Adam lr=1e-3, 25 epochs (patience=7), best epoch=19

**Results:**

| Metric | Value |
|--------|-------|
| Pixel F1 | 0.5137 |
| Pixel IoU | 0.3456 |
| Pixel AUC | 0.8828 |

**Strengths:**
- Tests a reasonable hypothesis: deeper encoder = better features
- +5.91pp F1 over P.1 (RGB baseline) shows some benefit from deeper features
- Larger decoder (8.7M params) compensates somewhat for frozen encoder

**Weaknesses:**
- Still on raw RGB — unfair comparison with ELA-based runs
- Only marginally better than ResNet-34 RGB (P.1: 0.4546) despite 4x more decoder params
- Significantly worse than ResNet-34 + ELA (P.3: 0.6920)

**Reviewer Comments:**
- This experiment proves that encoder depth is far less important than input preprocessing — ResNet-50 RGB (0.5137) is massively outperformed by ResNet-34 ELA (0.6920).
- Should have been tested WITH ELA to properly isolate the encoder depth variable. Testing on RGB conflates two variables: encoder depth and input representation.
- The 8.7M trainable decoder params suggest decoder capacity increase, not encoder quality, may explain the small improvement.

**Assignment Alignment Score:**

| Category | Score |
|----------|-------|
| Problem Understanding | 6/10 |
| Dataset Handling | 7/10 |
| Model Architecture | 7/10 |
| Training Pipeline | 7/10 |
| Experiment Design | 8/15 |
| Evaluation Metrics | 7/10 |
| Visual Results | 6/10 |
| Documentation | 7/10 |
| Reproducibility | 8/10 |
| Engineering Rigor | 3/5 |
| **Total** | **66/100** |

**Verdict:** WEAK — Poorly controlled experiment. Informative only as negative result for encoder depth on RGB.

---

### Run: vR.P.6 — EfficientNet-B0 Encoder

**Description:** Replaced ResNet-34 with EfficientNet-B0 (MBConv blocks with built-in Squeeze-Excite attention). Raw RGB input.

**Architecture:** EfficientNet-B0 (ImageNet, frozen, 5.3M params) → UNet decoder (~400K trainable)
**Training Setup:** BCE+Dice loss, Adam lr=1e-3, 25 epochs (patience=7), best epoch=16

**Results:**

| Metric | Value |
|--------|-------|
| Pixel F1 | 0.5217 |
| Pixel IoU | 0.3529 |
| Pixel AUC | 0.8708 |

**Strengths:**
- Tests modern efficient architecture with built-in attention (SE blocks)
- Comparable to ResNet-50 (0.5137) with far fewer decoder params (400K vs 8.7M)
- SE attention shows promise for channelwise feature recalibration

**Weaknesses:**
- Still raw RGB — same confounding as P.5
- Only marginally better than ResNet-34 RGB baseline
- Much worse than ResNet-34 + ELA (P.3: 0.6920)

**Reviewer Comments:**
- EfficientNet-B0 on RGB is not a meaningful comparison against ResNet-34 on ELA. The study should have tested EfficientNet + ELA for a fair encoder comparison.
- The built-in SE attention in EfficientNet did not translate to meaningful improvement on forensic data — suggesting that input-level forensic features (ELA) matter more than learned attention.
- Early stopping at epoch 16 suggests the model converged quickly — possibly too quickly with a frozen encoder.

**Assignment Alignment Score:**

| Category | Score |
|----------|-------|
| Problem Understanding | 6/10 |
| Dataset Handling | 7/10 |
| Model Architecture | 7/10 |
| Training Pipeline | 7/10 |
| Experiment Design | 8/15 |
| Evaluation Metrics | 7/10 |
| Visual Results | 6/10 |
| Documentation | 7/10 |
| Reproducibility | 8/10 |
| Engineering Rigor | 3/5 |
| **Total** | **66/100** |

**Verdict:** WEAK — Same issue as P.5. Informative only as negative result.

---

### Run: vR.P.7 — ELA + Extended Training (50 Epochs)

**Description:** Extended P.3's training from 25 to 50 epochs with patience increased to 10. All else identical.

**Architecture:** ResNet-34 (ImageNet, frozen body + BN unfrozen) → UNet decoder
**Training Setup:** BCE+Dice loss, Adam lr=1e-3, 50 epochs (patience=10), best epoch=36

**Results:**

| Metric | Value |
|--------|-------|
| Pixel F1 | 0.7154 |
| Pixel IoU | 0.5569 |
| Pixel AUC | 0.9504 |

**Strengths:**
- Clean single-variable ablation: only changed epochs (25 → 50)
- +2.34pp F1 improvement from more training alone — cheapest possible gain
- Best epoch at 36 confirms P.3 was indeed undertrained

**Weaknesses:**
- AUC slightly dropped (0.9528 → 0.9504 — within noise)
- Still single-quality ELA

**Reviewer Comments:**
- Textbook ablation methodology. One variable changed, clear improvement measured.
- Confirms that P.3 was left on the table by stopping at 25 epochs. The epoch limit should have been higher from the start.
- The diminishing returns are visible: 11 more epochs (25→36) for only +2.34pp F1. Extended training helps but has limits.

**Assignment Alignment Score:**

| Category | Score |
|----------|-------|
| Problem Understanding | 8/10 |
| Dataset Handling | 8/10 |
| Model Architecture | 8/10 |
| Training Pipeline | 8/10 |
| Experiment Design | 14/15 |
| Evaluation Metrics | 8/10 |
| Visual Results | 8/10 |
| Documentation | 8/10 |
| Reproducibility | 9/10 |
| Engineering Rigor | 4/5 |
| **Total** | **83/100** |

**Verdict:** STRONG — Excellent ablation methodology. Clean, informative result.

---

### Run: vR.P.13 — CBAM + Augmentation + Extended Training (Kitchen Sink)

**Description:** Combined best elements from P.7 (50 epochs), P.9 (Focal+Dice loss), P.10 (CBAM attention), P.12 (augmentation).

**Architecture:** ResNet-34 + CBAM in decoder + UNet
**Training Setup:** Focal(alpha=0.25, gamma=2.0) + Dice loss, Adam lr=1e-3, 50 epochs (patience=10), best epoch=49

**Results:**

| Metric | Value |
|--------|-------|
| Pixel F1 | 0.7307 |
| Pixel IoU | 0.5756 |
| Pixel AUC | 0.9607 |

**Strengths:**
- Best F1 of the single-Q ELA experiments (0.7307)
- Demonstrates that combining improvements yields cumulative gains
- Best epoch at 49 suggests it could benefit from even more training

**Weaknesses:**
- Violates single-variable principle — changed 4 things at once
- Cannot attribute the improvement to any specific component
- Still single-Q ELA — even the "kitchen sink" run is beaten by P.19's multi-Q input alone

**Reviewer Comments:**
- This is a "throw everything at the wall" experiment. While the result is the best in the single-Q ELA group, it tells us nothing about which component matters most.
- The fact that P.19 (F1=0.7965) beats this by +6.58pp using ONLY a better input representation (no CBAM, no augmentation, no Focal loss) is damning. Input quality > all other tricks combined.
- The best epoch hitting 49/50 is a red flag — it may still be improving. Should have used 100 epochs or at least checked for convergence.
- This run should not be in an ablation study. It belongs in an "engineering optimization" phase after the ablation is complete.

**Assignment Alignment Score:**

| Category | Score |
|----------|-------|
| Problem Understanding | 8/10 |
| Dataset Handling | 8/10 |
| Model Architecture | 8/10 |
| Training Pipeline | 8/10 |
| Experiment Design | 7/15 |
| Evaluation Metrics | 8/10 |
| Visual Results | 8/10 |
| Documentation | 7/10 |
| Reproducibility | 8/10 |
| Engineering Rigor | 4/5 |
| **Total** | **74/100** |

**Verdict:** ADEQUATE — Good result, but poor ablation methodology.

---

### Run: vR.P.16 — DCT Spatial Map Baseline

**Description:** Replaced ELA input with DCT-based spatial maps: AC energy, DC coefficient, and HF energy computed from 8x8 YCbCr luminance blocks.

**Architecture:** ResNet-34 (ImageNet, frozen body + BN unfrozen) → UNet decoder
**Training Setup:** BCE+Dice loss, Adam lr=1e-3, 25 epochs (patience=7), best epoch=11

**Results:**

| Metric | Value |
|--------|-------|
| Pixel F1 | 0.3209 |
| Pixel IoU | 0.1911 |
| Pixel AUC | 0.7778 |

**Strengths:**
- Tests an interesting frequency-domain alternative to ELA
- Clean single-variable comparison against P.3 (ELA) and P.1 (RGB)
- Early stopping at epoch 11 indicates the model learned what it could quickly

**Weaknesses:**
- Worst performance in the entire series except P.0 (pseudo-masks)
- Even worse than raw RGB (P.1: 0.4546)
- DCT spatial statistics at 8x8 block level lose fine-grained spatial information

**Reviewer Comments:**
- The DCT approach is theoretically sound — JPEG compression operates in DCT domain, so DCT features could capture compression artifacts. However, reducing each 8x8 block to 3 scalar statistics (AC energy, DC coeff, HF energy) discards too much spatial information for pixel-level localization.
- The 0.3209 F1 is worse than the RGB baseline. This is a clear negative result that should terminate the DCT exploration line.
- A better approach would have been to compute per-pixel DCT features rather than per-block statistics, or to combine DCT with ELA rather than replacing it.

**Assignment Alignment Score:**

| Category | Score |
|----------|-------|
| Problem Understanding | 7/10 |
| Dataset Handling | 7/10 |
| Model Architecture | 6/10 |
| Training Pipeline | 7/10 |
| Experiment Design | 10/15 |
| Evaluation Metrics | 7/10 |
| Visual Results | 5/10 |
| Documentation | 7/10 |
| Reproducibility | 8/10 |
| Engineering Rigor | 3/5 |
| **Total** | **67/100** |

**Verdict:** WEAK — Valid negative result. DCT spatial maps are insufficient for localization.

---

### Run: vR.P.19 — Multi-Quality RGB ELA 9-Channel ★ BEST OVERALL

**Description:** Multi-quality ELA with full RGB color preservation. Three quality levels (Q=75, 85, 95) each producing 3 RGB channels, concatenated to 9-channel input.

**Architecture:** ResNet-34 (ImageNet, frozen body + conv1 unfrozen + BN unfrozen) → UNet decoder
**Training Setup:** BCE+Dice loss, Adam lr=1e-3, 25 epochs (patience=7), best epoch=25

**Results:**

| Metric | Value |
|--------|-------|
| Pixel F1 | 0.7965 |
| Pixel IoU | 0.6618 |
| Pixel AUC | 0.9726 |

**Strengths:**
- **Best overall result** — +10.45pp F1 over P.3 baseline
- Demonstrates that chrominance ELA channels carry significant forensic signal
- Multi-quality levels expose different compression artifact patterns
- Achieves best F1 WITHOUT CBAM, augmentation, or extended training — pure input quality win

**Weaknesses:**
- 9-channel input requires unfreezing conv1 — more trainable params than P.3
- Best epoch = 25 (hit limit again) — probably undertrained
- No CBAM or augmentation — leaves room for combination experiments

**Reviewer Comments:**
- This is the strongest result and the most important finding: input quality dominates everything else. P.19 beats P.13 (the "kitchen sink" with CBAM+Aug+Extended) by +6.58pp using ONLY a better input representation.
- The chrominance preservation is key. Earlier experiments (P.15, not in W&B) used grayscale multi-Q ELA and achieved lower results. The full RGB ELA preserves chrominance artifacts from tampering.
- Best epoch hitting 25 (the limit) is a missed opportunity. With 50 epochs this could likely reach F1 > 0.82 based on the +2.34pp gain observed in P.7.
- This run should be the basis for all future combination experiments.

**Assignment Alignment Score:**

| Category | Score |
|----------|-------|
| Problem Understanding | 10/10 |
| Dataset Handling | 9/10 |
| Model Architecture | 8/10 |
| Training Pipeline | 7/10 |
| Experiment Design | 14/15 |
| Evaluation Metrics | 9/10 |
| Visual Results | 9/10 |
| Documentation | 8/10 |
| Reproducibility | 9/10 |
| Engineering Rigor | 4/5 |
| **Total** | **87/100** |

**Verdict:** STRONG — Best result. Should be the foundation for future work.

---

### Run: vR.P.20 — ELA Magnitude + Chrominance Direction

**Description:** Decomposed ELA into magnitude (scalar) and chrominance direction (2ch unit vector) as a 3-channel input.

**Architecture:** ResNet-34 (ImageNet, frozen body + BN unfrozen) → UNet decoder
**Training Setup:** BCE+Dice loss, Adam lr=1e-3, 25 epochs (patience=7), best epoch=21

**Results:**

| Metric | Value |
|--------|-------|
| Pixel F1 | 0.6555 |
| Pixel IoU | 0.4875 |
| Pixel AUC | 0.9399 |

**Strengths:**
- Interesting theoretical decomposition — separates "how much error" from "what kind of error"
- Clean single-variable experiment against P.3

**Weaknesses:**
- -3.65pp F1 vs P.3 (0.6920 → 0.6555) — decomposition hurt performance
- Lost information — the decomposition is lossy compared to full RGB ELA

**Reviewer Comments:**
- The magnitude + chrominance direction decomposition is theoretically interesting but practically harmful. Separating these components removes the spatial correlation between channels that the CNN can exploit.
- The unit vector representation (direction / |direction|) clips small-magnitude directions to arbitrary values, introducing noise in authentic regions.
- This result, combined with P.19's success with full RGB ELA, confirms that preserving the raw ELA signal (including inter-channel correlations) is better than hand-crafted decompositions.

**Assignment Alignment Score:**

| Category | Score |
|----------|-------|
| Problem Understanding | 8/10 |
| Dataset Handling | 8/10 |
| Model Architecture | 7/10 |
| Training Pipeline | 7/10 |
| Experiment Design | 12/15 |
| Evaluation Metrics | 7/10 |
| Visual Results | 6/10 |
| Documentation | 7/10 |
| Reproducibility | 8/10 |
| Engineering Rigor | 3/5 |
| **Total** | **73/100** |

**Verdict:** ADEQUATE — Informative negative result.

---

### Run: vR.P.23 — Chrominance Channel Analysis (YCbCr)

**Description:** Used raw YCbCr (Y, Cb, Cr) channels without ELA processing.

**Architecture:** ResNet-34 (ImageNet, frozen body + BN unfrozen) → UNet decoder
**Training Setup:** BCE+Dice loss, Adam lr=1e-3, 25 epochs (patience=7), best epoch=15

**Results:**

| Metric | Value |
|--------|-------|
| Pixel F1 | 0.4709 |
| Pixel IoU | 0.3080 |
| Pixel AUC | 0.8453 |

**Strengths:**
- Tests whether chrominance quantization artifacts alone are sufficient for detection
- Reasonable result for raw chrominance — Cb/Cr channels do carry some forensic signal

**Weaknesses:**
- Much worse than ELA (P.3: 0.6920) — raw chrominance is insufficient
- Only marginally better than RGB (P.1: 0.4546)

**Reviewer Comments:**
- The hypothesis was sound: JPEG chrominance subsampling (4:2:0) could reveal tampering through Cb/Cr channel inconsistencies. But the signal is weak without ELA amplification.
- A more interesting experiment would have been ELA computed on YCbCr channels rather than raw YCbCr.
- Early stopping at epoch 15 suggests the model extracted what little signal existed quickly.

**Assignment Alignment Score:**

| Category | Score |
|----------|-------|
| Problem Understanding | 7/10 |
| Dataset Handling | 7/10 |
| Model Architecture | 6/10 |
| Training Pipeline | 7/10 |
| Experiment Design | 10/15 |
| Evaluation Metrics | 7/10 |
| Visual Results | 5/10 |
| Documentation | 7/10 |
| Reproducibility | 8/10 |
| Engineering Rigor | 3/5 |
| **Total** | **67/100** |

**Verdict:** WEAK — Limited forensic signal in raw chrominance.

---

### Run: vR.P.24 — Noiseprint Forensic Features (DnCNN Residual)

**Description:** Used DnCNN-style noise residual extraction as input. The noise fingerprint captures camera-specific patterns disrupted by tampering.

**Architecture:** ResNet-34 (ImageNet, frozen body + BN unfrozen) → UNet decoder
**Training Setup:** BCE+Dice loss, Adam lr=1e-3, 25 epochs (patience=7), best epoch=17

**Results:**

| Metric | Value |
|--------|-------|
| Pixel F1 | 0.5246 |
| Pixel IoU | 0.3555 |
| Pixel AUC | 0.8831 |

**Strengths:**
- Tests a fundamentally different forensic signal — sensor noise vs compression artifacts
- Better than raw RGB (P.1: 0.4546) and YCbCr (P.23: 0.4709)
- Noiseprint approach has strong theoretical backing in the forensics literature

**Weaknesses:**
- Much worse than ELA (P.3: 0.6920) — DnCNN residual is not a good substitute
- The DnCNN is not pretrained on forensic data — it's extracting generic noise, not camera-specific noise
- Noiseprint requires end-to-end training of the noise extraction network for best results

**Reviewer Comments:**
- Noiseprint (Cozzolino & Verdoliva, 2019) requires a specially-trained DnCNN on camera noise patterns. Using a generic DnCNN residual misses the core idea.
- The 0.5246 F1 is above RGB but far below ELA — confirming that JPEG compression artifacts (ELA) are a stronger signal than generic noise for CASIA-2.0 detection.
- A proper implementation would train the noise extraction DnCNN end-to-end jointly with the UNet, which is significantly more complex.

**Assignment Alignment Score:**

| Category | Score |
|----------|-------|
| Problem Understanding | 8/10 |
| Dataset Handling | 7/10 |
| Model Architecture | 6/10 |
| Training Pipeline | 7/10 |
| Experiment Design | 10/15 |
| Evaluation Metrics | 7/10 |
| Visual Results | 6/10 |
| Documentation | 7/10 |
| Reproducibility | 8/10 |
| Engineering Rigor | 3/5 |
| **Total** | **69/100** |

**Verdict:** ADEQUATE — Interesting but underperforms ELA significantly.

---

### Run: vR.P.27 — JPEG Compression Augmentation

**Description:** Added random JPEG recompression during training as data augmentation on top of P.3's ELA input.

**Architecture:** ResNet-34 (ImageNet, frozen body + BN unfrozen) → UNet decoder
**Training Setup:** BCE+Dice loss, Adam lr=1e-3, 25 epochs (patience=7), best epoch=23

**Results:**

| Metric | Value |
|--------|-------|
| Pixel F1 | 0.7105 |
| Pixel IoU | 0.5510 |
| Pixel AUC | 0.9553 |

**Strengths:**
- +1.85pp F1 over P.3 — JPEG augmentation is domain-appropriate
- Slightly better AUC than P.3 (0.9553 vs 0.9528) — better calibration under varied compression
- Theoretically sound: teaching the model to handle different compression levels improves robustness

**Weaknesses:**
- Small improvement — within noise for a single run
- Only compared against P.3 (25 epochs); not compared against P.7 (50 epochs)

**Reviewer Comments:**
- JPEG compression augmentation is one of the most domain-appropriate augmentation strategies for forensic tasks. The improvement is modest but consistent with the hypothesis.
- The 1.85pp gain may be attributable to the augmentation regularization effect rather than genuine compression robustness. Would need to test on out-of-distribution compression levels to verify.
- Should be combined with extended training (50 epochs) for a cleaner comparison.

**Assignment Alignment Score:**

| Category | Score |
|----------|-------|
| Problem Understanding | 8/10 |
| Dataset Handling | 8/10 |
| Model Architecture | 7/10 |
| Training Pipeline | 8/10 |
| Experiment Design | 12/15 |
| Evaluation Metrics | 8/10 |
| Visual Results | 7/10 |
| Documentation | 7/10 |
| Reproducibility | 8/10 |
| Engineering Rigor | 4/5 |
| **Total** | **77/100** |

**Verdict:** ADEQUATE — Small but meaningful domain-specific improvement.

---

### Run: vR.P.28 — Cosine Annealing LR Scheduler

**Description:** Replaced ReduceLROnPlateau with CosineAnnealingWarmRestarts scheduler. Extended to 50 epochs.

**Architecture:** ResNet-34 (ImageNet, frozen body + BN unfrozen) → UNet decoder
**Training Setup:** BCE+Dice loss, Adam lr=1e-3, CosineAnnealingWarmRestarts, 50 epochs (patience=10), best epoch=38

**Results:**

| Metric | Value |
|--------|-------|
| Pixel F1 | 0.7215 |
| Pixel IoU | 0.5643 |
| Pixel AUC | 0.9572 |

**Strengths:**
- Clean comparison against P.7 (same 50 epochs, different scheduler)
- +0.61pp F1 over P.7 — cosine annealing slightly outperforms plateau-based scheduling
- Warm restarts help escape local minima during training

**Weaknesses:**
- Changes TWO variables from P.3 (scheduler + epochs) — not a clean single-variable ablation against base
- The improvement over P.7 is minimal (0.61pp) — within noise

**Reviewer Comments:**
- As a comparison against P.7, the cosine annealing result is informative but inconclusive — 0.61pp is too small to draw conclusions without repeated runs.
- The warm restarts could be analyzed by plotting the training curve and checking if restarts correspond to learning improvements.
- A better experiment would have tested cosine annealing at 25 epochs (matching P.3) to isolate the scheduler variable from the epoch count variable.

**Assignment Alignment Score:**

| Category | Score |
|----------|-------|
| Problem Understanding | 8/10 |
| Dataset Handling | 8/10 |
| Model Architecture | 7/10 |
| Training Pipeline | 9/10 |
| Experiment Design | 10/15 |
| Evaluation Metrics | 8/10 |
| Visual Results | 7/10 |
| Documentation | 7/10 |
| Reproducibility | 8/10 |
| Engineering Rigor | 4/5 |
| **Total** | **76/100** |

**Verdict:** ADEQUATE — Clean scheduler comparison with marginal result.

---

### Run: vR.P.30 — Multi-Q ELA Grayscale + CBAM (25 Epochs)

**Description:** Combined multi-quality ELA (grayscale, Q=75/85/95) with CBAM attention in decoder blocks.

**Architecture:** ResNet-34 + CBAM (Channel+Spatial attention per decoder block) → UNet decoder
**Training Setup:** BCE+Dice loss, Adam lr=1e-3, 25 epochs (patience=7), best epoch=23

**Results:**

| Metric | Value |
|--------|-------|
| Pixel F1 | 0.7438 |
| Pixel IoU | 0.5921 |
| Pixel AUC | 0.9733 |

**Strengths:**
- Strong result: +5.18pp F1 over P.3 baseline
- CBAM + multi-Q ELA is a well-motivated combination
- AUC of 0.9733 is high — good ranking quality

**Weaknesses:**
- Uses grayscale multi-Q ELA (3ch) instead of P.19's RGB multi-Q (9ch) — loses chrominance
- Changes multiple variables from parent (multi-Q + CBAM) — not a clean ablation
- Worse than P.19 (0.7965) despite having CBAM and same quality levels

**Reviewer Comments:**
- The grayscale vs RGB ELA choice is costly. P.19 demonstrated that chrominance carries significant forensic signal. This run's inferior performance vs P.19 (-5.27pp) is likely attributable to discarding chrominance information.
- Combining CBAM with multi-Q ELA is reasonable but should have been compared against multi-Q ELA alone (RGB, like P.19) to isolate CBAM's contribution.
- This established the P.30.x sub-series — a reasonable decision to explore variations on this combination.

**Assignment Alignment Score:**

| Category | Score |
|----------|-------|
| Problem Understanding | 9/10 |
| Dataset Handling | 8/10 |
| Model Architecture | 8/10 |
| Training Pipeline | 7/10 |
| Experiment Design | 10/15 |
| Evaluation Metrics | 8/10 |
| Visual Results | 8/10 |
| Documentation | 8/10 |
| Reproducibility | 9/10 |
| Engineering Rigor | 4/5 |
| **Total** | **79/100** |

**Verdict:** STRONG — Good result, weak ablation methodology.

---

### Run: vR.P.30.1 — Multi-Q ELA + CBAM + Extended Training (50 Epochs)

**Description:** Extended P.30 from 25 to 50 epochs with patience 10. All else identical.

**Architecture:** ResNet-34 + CBAM → UNet decoder
**Training Setup:** BCE+Dice loss, Adam lr=1e-3, 50 epochs (patience=10), best epoch=41

**Results:**

| Metric | Value |
|--------|-------|
| Pixel F1 | 0.7762 |
| Pixel IoU | 0.6343 |
| Pixel AUC | 0.9795 |

**Strengths:**
- Clean single-variable extension: only epochs changed (25 → 50)
- +3.24pp F1 over P.30 — significant gain from more training
- Highest AUC in the series (0.9795) — excellent pixel ranking
- Second-best F1 overall (behind only P.19)

**Weaknesses:**
- Still uses grayscale multi-Q ELA — chrominance loss vs P.19
- Still worse than P.19 which uses NO CBAM and only 25 epochs

**Reviewer Comments:**
- Excellent ablation: single variable changed, clear and significant improvement measured.
- This is the strongest result in the P.30.x sub-series and the second-best overall. The combination of multi-Q ELA + CBAM + 50 epochs is the best "fully-loaded" configuration excluding P.19's multi-Q RGB approach.
- The fact that P.19 (F1=0.7965, 25 epochs, no CBAM) still beats this (F1=0.7762, 50 epochs, with CBAM) by +2.03pp reaffirms that input quality trumps model complexity.

**Assignment Alignment Score:**

| Category | Score |
|----------|-------|
| Problem Understanding | 9/10 |
| Dataset Handling | 8/10 |
| Model Architecture | 9/10 |
| Training Pipeline | 8/10 |
| Experiment Design | 14/15 |
| Evaluation Metrics | 9/10 |
| Visual Results | 8/10 |
| Documentation | 8/10 |
| Reproducibility | 9/10 |
| Engineering Rigor | 4/5 |
| **Total** | **86/100** |

**Verdict:** STRONG — Second-best result with clean methodology.

---

### Run: vR.P.30.2 — Multi-Q ELA + CBAM + Progressive Unfreeze

**Description:** Added progressive encoder unfreezing during training on top of P.30 config.

**Architecture:** ResNet-34 + CBAM + progressive unfreeze → UNet decoder
**Training Setup:** BCE+Dice loss, Adam lr=1e-3, 40 epochs (patience=7)

**Results:**

| Metric | Value |
|--------|-------|
| Pixel F1 | 0.7569 |
| Pixel IoU | 0.6089 |
| Pixel AUC | 0.9735 |

**Strengths:**
- Tests whether gradual encoder fine-tuning helps
- Progressive unfreeze is a well-established transfer learning technique

**Weaknesses:**
- -1.93pp F1 vs P.30.1 — unfreezing hurt compared to just training longer
- Only 40 epochs (vs 50 for P.30.1) — not a fair comparison
- May have caused catastrophic forgetting of ImageNet features

**Reviewer Comments:**
- Progressive unfreezing on ELA input is risky — the ImageNet features may not be useful for ELA, so unfreezing to adapt them could cause catastrophic forgetting rather than helpful fine-tuning.
- The comparison against P.30.1 is unfair (40 vs 50 epochs). Should have used 50 epochs for consistency.
- The smaller file size (5MB vs 22+MB) suggests incomplete execution or missing outputs.

**Assignment Alignment Score:**

| Category | Score |
|----------|-------|
| Problem Understanding | 8/10 |
| Dataset Handling | 8/10 |
| Model Architecture | 7/10 |
| Training Pipeline | 7/10 |
| Experiment Design | 9/15 |
| Evaluation Metrics | 8/10 |
| Visual Results | 6/10 |
| Documentation | 7/10 |
| Reproducibility | 7/10 |
| Engineering Rigor | 3/5 |
| **Total** | **70/100** |

**Verdict:** ADEQUATE — Negative result with methodological issues.

---

### Run: vR.P.30.3 — Multi-Q ELA + CBAM + Focal+Dice Loss

**Description:** Replaced BCE+Dice loss with Focal+Dice loss on top of P.30 config.

**Architecture:** ResNet-34 + CBAM → UNet decoder
**Training Setup:** Focal(alpha=0.25, gamma=2.0) + Dice loss, Adam lr=1e-3, 25 epochs (patience=7), best epoch=22

**Results:**

| Metric | Value |
|--------|-------|
| Pixel F1 | 0.7509 |
| Pixel IoU | 0.6011 |
| Pixel AUC | 0.9694 |

**Strengths:**
- Clean single-variable ablation against P.30 (same epochs, same config, only loss changed)
- +0.71pp F1 over P.30 — marginal improvement from Focal loss
- Focal loss addresses class imbalance (most pixels are non-tampered)

**Weaknesses:**
- Very small improvement (0.71pp) — Focal loss is not transformative
- Worse than P.30.1 (which just used more epochs with original loss)

**Reviewer Comments:**
- Good ablation methodology — single variable changed, controlled comparison.
- The marginal improvement (0.71pp) suggests that BCE+Dice already handles class imbalance reasonably well for this dataset. Focal loss provides minimal additional benefit.
- The comparison with P.30.1 is instructive: 25 more epochs (+3.24pp) helps more than Focal loss (+0.71pp).

**Assignment Alignment Score:**

| Category | Score |
|----------|-------|
| Problem Understanding | 8/10 |
| Dataset Handling | 8/10 |
| Model Architecture | 8/10 |
| Training Pipeline | 8/10 |
| Experiment Design | 13/15 |
| Evaluation Metrics | 8/10 |
| Visual Results | 7/10 |
| Documentation | 7/10 |
| Reproducibility | 8/10 |
| Engineering Rigor | 4/5 |
| **Total** | **79/100** |

**Verdict:** ADEQUATE — Clean experiment with marginal result.

---

### Run: vR.P.30.4 — Multi-Q ELA + CBAM + Augmentation (50 Epochs)

**Description:** Added geometric augmentation (flip, rotate, shift/scale) on top of P.30.1 config (50 epochs).

**Architecture:** ResNet-34 + CBAM → UNet decoder
**Training Setup:** Focal+Dice loss, Adam lr=1e-3, 50 epochs (patience=10), best epoch=41

**Results:**

| Metric | Value |
|--------|-------|
| Pixel F1 | 0.7662 |
| Pixel IoU | 0.6210 |
| Pixel AUC | 0.9726 |

**Strengths:**
- Clean comparison against P.30.1 with augmentation as the single variable
- 50 epochs matches P.30.1 for fair comparison

**Weaknesses:**
- -1.00pp F1 vs P.30.1 — augmentation actually HURT performance
- Also switched from BCE+Dice to Focal+Dice (confounding variable)
- AUC slightly decreased (0.9726 vs 0.9795)

**Reviewer Comments:**
- Augmentation hurting performance is a surprising negative result. Possible explanations: (1) ELA artifacts are rotation/flip invariant so augmentation doesn't help, (2) geometric transforms slightly corrupt the ELA signal by introducing interpolation artifacts.
- The experiment is confounded by also changing the loss function (Focal+Dice vs BCE+Dice). This should have kept BCE+Dice to isolate the augmentation variable.
- The -1.00pp drop is small but consistent with the hypothesis that ELA features may be sensitive to geometric transforms that introduce resampling artifacts.

**Assignment Alignment Score:**

| Category | Score |
|----------|-------|
| Problem Understanding | 8/10 |
| Dataset Handling | 8/10 |
| Model Architecture | 8/10 |
| Training Pipeline | 8/10 |
| Experiment Design | 10/15 |
| Evaluation Metrics | 8/10 |
| Visual Results | 7/10 |
| Documentation | 7/10 |
| Reproducibility | 8/10 |
| Engineering Rigor | 4/5 |
| **Total** | **76/100** |

**Verdict:** ADEQUATE — Informative negative result with confounding issue.

---

## 4. Best Candidate Experiments

### Top 5 by Pixel F1

| Rank | Version | Pixel F1 | IoU | AUC | Key Advantage |
|------|---------|----------|-----|-----|--------------|
| 1 | **vR.P.19** | **0.7965** | 0.6618 | 0.9726 | Best input representation (Multi-Q RGB ELA 9ch) |
| 2 | **vR.P.30.1** | **0.7762** | 0.6343 | 0.9795 | Best "loaded" config (Multi-Q gray + CBAM + 50ep) |
| 3 | **vR.P.30.4** | **0.7662** | 0.6210 | 0.9726 | Augmented version of P.30.1 |
| 4 | **vR.P.30.2** | **0.7569** | 0.6089 | 0.9735 | Progressive unfreeze attempt |
| 5 | **vR.P.30.3** | **0.7509** | 0.6011 | 0.9694 | Focal+Dice loss variant |

### Best for Assignment Submission

**vR.P.19** — Highest F1, clean methodology, demonstrates the core insight that multi-quality RGB ELA is the most impactful technique for localization.

### Best for Research Novelty

**vR.P.30.1** — Combines multiple techniques (multi-Q ELA + CBAM + extended training) and achieves the highest AUC (0.9795). The CBAM attention mechanism adds architectural novelty.

### Most Reliable

**vR.P.7** — Clean single-variable experiment with the highest score (83/100). Reproducible, well-documented, and methodologically sound.

---

## 5. System-Level Observations

### 5.1 ELA Dominance

ELA preprocessing is the **sine qua non** of this pipeline. The transition from RGB (P.1, F1=0.4546) to ELA (P.3, F1=0.6920) represents the single largest improvement in the series (+23.74pp). All top-performing runs use ELA input. No amount of architectural modification (CBAM, deeper encoders, better loss functions) can compensate for inadequate input preprocessing.

### 5.2 Multi-Quality > Single-Quality ELA

Multi-quality ELA provides diminishing-returns improvement over single-quality:
- Single Q=90: F1=0.6920 (P.3)
- Multi-Q gray (75/85/95): F1=0.7438 (P.30) — +5.18pp
- Multi-Q RGB (75/85/95): F1=0.7965 (P.19) — +10.45pp

The RGB preservation in P.19 accounts for approximately half the total multi-Q gain, confirming that chrominance artifacts are forensically significant.

### 5.3 ResNet-34 Sufficiency

ResNet-34 was tested against ResNet-50 (P.5) and EfficientNet-B0 (P.6), both on RGB. Neither provided meaningful improvement. This suggests that:
1. The encoder's primary role is multi-scale feature extraction, not domain-specific feature learning
2. For frozen encoders on forensic inputs, encoder depth/architecture is less important than the input signal quality
3. The bottleneck is in the decoder's ability to use skip connections, not the encoder's feature quality

### 5.4 CBAM Attention — Helpful but Secondary

CBAM attention (Channel + Spatial) consistently adds +3-5pp F1 when compared at the same epoch count:
- P.3 (no CBAM, 25ep): 0.6920 → P.10 (CBAM, 25ep): ~0.7277
- P.30 (CBAM, 25ep): 0.7438 vs P.30.1 (CBAM, 50ep): 0.7762

However, CBAM never closes the gap to P.19's multi-Q RGB ELA. The attention mechanism helps the decoder focus on relevant skip connection features, but the signal quality from the input remains the dominant factor.

### 5.5 Training Duration — Cheap Gains

Extended training (25 → 50 epochs) consistently yields +2-3pp F1:
- P.3 → P.7: +2.34pp
- P.30 → P.30.1: +3.24pp

This is the cheapest improvement available — no code changes, just more compute. Many runs hit the epoch limit, suggesting they were systematically undertrained.

### 5.6 Diminishing Returns

The series exhibits clear diminishing returns. The first 10 runs covered the range F1=0.35 to F1=0.73 (38pp range). The last 12 runs covered F1=0.32 to F1=0.80 (including negative results). The remaining headroom above F1=0.80 will require fundamentally new approaches rather than incremental ablation.

### 5.7 Negative Results Worth Noting

| Experiment | Expected Improvement | Actual Result |
|-----------|---------------------|---------------|
| DCT spatial maps (P.16) | Alternative to ELA | Worst result (F1=0.3209) |
| YCbCr chrominance (P.23) | Chrominance forensics | Marginally above RGB |
| ELA decomposition (P.20) | Separate magnitude/direction | Hurt performance |
| Progressive unfreeze (P.30.2) | Better fine-tuning | Hurt performance |
| Augmentation (P.30.4) | Regularization | Hurt performance |

These negative results are scientifically valuable — they narrow the design space and prevent future researchers from exploring dead ends.

---

## 6. Next Generation Experiment Plan (vR.P.40.x)

Based on findings from this audit, the next generation of experiments explores:

1. **EfficientNet-B4 Encoder** — Previous EfficientNet-B0 test (P.6) was on RGB only. Testing B4 with ELA addresses the proper comparison.
2. **EfficientNet-B4 + Multi-Q RGB ELA** — Combines the best encoder candidate with the best input representation.
3. **Custom Inception Encoders (V1/V2/V3)** — Novel architecture not tested in the series. Adapted from a DeepFake detection reference notebook.

| Version | Experiment | Parent | Single Variable |
|---------|-----------|--------|----------------|
| vR.P.40.1 | EfficientNet-B4 + ELA Q=90 | P.3 | Encoder: ResNet-34 → EfficientNet-B4 |
| vR.P.40.2 | EfficientNet-B4 + Multi-Q RGB ELA 9ch | P.40.1 | Input: 3ch → 9ch multi-Q RGB ELA |
| vR.P.40.3 | InceptionV1 Custom Encoder + ELA | P.3 | Encoder: ResNet-34 → custom InceptionV1 |
| vR.P.40.4 | InceptionV2 Custom Encoder + ELA | P.40.3 | Module: V1 → V2 (BN + factorized 5×5) |
| vR.P.40.5 | InceptionV3 Custom Encoder + ELA | P.40.3 | Module: V1 → V3 (asymmetric 1×n + n×1) |

**Expected outcomes:**
- vR.P.40.1: May outperform P.3 since EfficientNet-B4 has built-in SE attention and was only tested on RGB before
- vR.P.40.2: Combines best encoder candidate with best input — potential new series best
- vR.P.40.3-5: Custom Inception encoders train from scratch. Expected to underperform pretrained ResNet-34 but provide architectural insights for forensic feature extraction

---

*End of Audit*
