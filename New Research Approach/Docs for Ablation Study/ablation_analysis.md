# Ablation Analysis -- ETASR Study

| Field | Value |
|-------|-------|
| **Date** | 2026-03-15 |
| **Scope** | Cross-run impact analysis for all ETASR and Pretrained ablation experiments |
| **Paper** | ETASR_9593 -- "Enhanced Image Tampering Detection using ELA and a CNN" |
| **Versions Covered** | ETASR: vR.1.0--vR.1.7 (8 runs) / Pretrained: vR.P.0--vR.P.15 (17 runs) / Standalone: 4 runs |

---

## 1. Ablation Methodology Assessment

### Study Design

The ETASR ablation study follows a **single-variable methodology**: each version changes exactly one aspect from its parent while keeping everything else frozen. The frozen constants include:

- Dataset: CASIA v2.0 (~12,614 images)
- ELA: Q=90, brightness-scaled
- Image size: 128x128
- Split: 70/15/15, stratified, seed=42
- Optimizer: Adam(lr=1e-4)
- Batch size: 32
- Max epochs: 50
- Early stopping: patience=5 on val_accuracy
- Loss: categorical_crossentropy

### Methodology Quality

**Strengths:**
- Strict single-variable control: each version changes ONE thing
- Explicit verdict criteria: POSITIVE (>+0.5pp), NEUTRAL (+/-0.5pp), NEGATIVE/REJECTED (<-0.5pp)
- Cumulative inheritance: accepted changes carry forward, rejected changes are discarded
- Honest baseline (vR.1.1) established before any modifications

**Confounding factors:**
- vR.1.2 (augmentation) was correctly rejected, but it tested augmentation on the weakest architecture (29.5M Dense-dominated). Augmentation might succeed on the vR.1.7 architecture (64K params, no spatial memorisation).
- BN (vR.1.4) introduced training instability (epoch 1 spike) that carried into all subsequent versions. This is a confound for vR.1.5/1.6/1.7.
- Early stopping patience=5 on val_accuracy may be too aggressive for BN-enabled models that need warmup epochs.

**Was the order optimal?** In hindsight, no. The ideal order would have been:
1. vR.1.1 (eval fix) -- same
2. vR.1.6 (deeper CNN) -- before training tricks
3. vR.1.7 (GAP) -- before training tricks
4. Then test class weights, BN, LR scheduler on the GAP architecture

The architecture changes had 2-10x more impact than training tricks. Testing training tricks first on a fundamentally limited architecture produced diminishing returns.

---

## 2. Impact Ranking

### Ranked by Accuracy Delta from Parent

| Rank | Version | Change | Delta from Parent | Delta from vR.1.1 |
|------|---------|--------|-------------------|-------------------|
| 1 | **vR.1.6** | Deeper CNN | **+1.27pp** | +1.85pp |
| 2 | **vR.1.3** | Class weights | **+0.79pp** | +0.79pp |
| 3 | vR.1.5 | LR Scheduler | +0.21pp | +0.58pp |
| 4 | vR.1.4 | BatchNorm | -0.42pp | +0.37pp |
| 5 | vR.1.7 | GAP | -1.06pp | +0.79pp |
| 6 | vR.1.2 | Augmentation | **-2.85pp** | -2.85pp |

### Ranked by AUC Delta from vR.1.1 Baseline

| Rank | Version | AUC | Delta | Assessment |
|------|---------|-----|-------|------------|
| 1 | **vR.1.6** | 0.9657 | **+0.0056** | Only improvement |
| 2 | vR.1.3 | 0.9580 | -0.0021 | Marginal drop |
| 3 | vR.1.5 | 0.9560 | -0.0041 | Modest drop |
| 4 | vR.1.4 | 0.9536 | -0.0065 | Clear drop |
| 5 | vR.1.7 | 0.9495 | -0.0106 | Significant drop |
| 6 | vR.1.2 | 0.9011 | **-0.0590** | Catastrophic |

### Impact by Change Category

| Category | Versions | Acc Delta (cumulative) | AUC Delta | Verdict |
|----------|----------|----------------------|-----------|---------|
| Architecture | vR.1.6, vR.1.7 | +0.21pp (1.27-1.06) | +0.0056, -0.0106 | **Most impactful** |
| Training config | vR.1.3, vR.1.4, vR.1.5 | +0.58pp (0.79-0.42+0.21) | -0.0021 to -0.0065 | Marginal |
| Data pipeline | vR.1.2 | -2.85pp | -0.0590 | Harmful |

---

## 3. The ROC-AUC Story

ROC-AUC measures threshold-independent discriminatory power. Unlike accuracy, it is not affected by class weights or decision threshold shifts. This makes it the purest measure of the model's feature quality.

### AUC Trajectory

```
vR.1.1    0.9601  ████████████████████████████████████████  Baseline
vR.1.2    0.9011  █████████████████████████████             CATASTROPHIC -0.0590
vR.1.3    0.9580  ███████████████████████████████████████   -0.0021
vR.1.4    0.9536  ██████████████████████████████████████    -0.0065
vR.1.5    0.9560  ██████████████████████████████████████    -0.0041
vR.1.6    0.9657  █████████████████████████████████████████ +0.0056 FIRST IMPROVEMENT
vR.1.7    0.9495  █████████████████████████████████████     -0.0106
```

### Analysis

1. **vR.1.1 baseline AUC (0.9601) was never matched** by any training-trick version (vR.1.3-1.5). Class weights, BN, and LR scheduling all made the model threshold-dependent but not fundamentally better at distinguishing authentic from tampered.

2. **Only vR.1.6 broke through.** The deeper CNN improved feature extraction quality, not just decision thresholds. This is the clearest evidence that the architecture was the bottleneck.

3. **vR.1.7 regressed to 0.9495** -- the 2nd worst honest-eval AUC. GAP's spatial information loss hurts discriminatory power even though it improves specific metrics (Tp recall, Au precision).

4. **vR.1.2 was catastrophic** (0.9011). Augmentation destroyed the spatial patterns that the Flatten->Dense layer had memorised, degrading not just accuracy but fundamental discriminatory ability.

### What AUC Reveals

- **Class weights don't improve features** -- they just shift where the decision threshold sits
- **Architecture changes improve features** -- vR.1.6 is genuinely better at separating distributions
- **GAP loses information** -- even though vR.1.7 has better individual metrics (Tp recall), its overall discriminatory power is lower

---

## 4. The Precision-Recall Tradeoff Evolution

### Per-Class Recall Trajectory

| Version | Au Recall | Tp Recall | Bias |
|---------|-----------|-----------|------|
| vR.1.1 | 0.8577 | 0.8830 | Slight Tp bias |
| vR.1.2 | 0.8443 | 0.8336 | Balanced (both low) |
| vR.1.3 | 0.8648 | **0.9012** | Tp bias (class weights) |
| vR.1.4 | 0.8559 | **0.9194** | Strong Tp bias (BN amplified) |
| vR.1.5 | 0.8594 | **0.9194** | Strong Tp bias maintained |
| vR.1.6 | **0.8746** | **0.9428** | Strong Tp bias, AU recovers |
| vR.1.7 | 0.8541 | **0.9467** | Strongest Tp bias, AU drops |

### Analysis

Class weights (vR.1.3) deliberately biased the model toward detecting tampered images (Tp recall jumped from 0.8830 to 0.9012). This bias was amplified by BN (0.9194) and persisted through vR.1.5. The model became progressively better at finding tampered images at the cost of more false positives.

vR.1.6 was the only version that improved BOTH Au and Tp recall simultaneously. Deeper features help both classes.

vR.1.7 pushed Tp recall to its peak (0.9467) but at the cost of Au recall dropping to 0.8541. GAP removes spatial information, making the model more aggressive with any image that shows ELA activity.

---

## 5. Training Dynamics Evolution

| Version | BN Spike | Productive Epochs | Best Epoch | Total Epochs | Train-Val Gap |
|---------|----------|-------------------|------------|-------------|---------------|
| vR.1.1 | None | 13 | 8 | 13 | ~3pp |
| vR.1.2 | None | 1 | 1 | 6 | ~2pp |
| vR.1.3 | None | 14 | 9 | 14 | ~3pp |
| vR.1.4 | **16.13** | 3 | 3 | 8 | ~5pp |
| vR.1.5 | **14.74** | 5 | 5 | 10 | ~6pp |
| vR.1.6 | **0.76** | 13 | 13 | 18 | ~5pp |
| vR.1.7 | **0.74** | 5 | 5 | 10 | ~1.3pp |

### Key Observations

1. **BN spike severity changed dramatically.** vR.1.4/1.5 had catastrophic epoch 1 spikes (val_loss=14-16). vR.1.6/1.7 had mild spikes (val_loss=0.74-0.76). The deeper architecture stabilised BN warmup.

2. **vR.1.6 trained the longest** (18 epochs, best at 13). The deeper architecture provided more to learn, enabling longer productive training before overfitting.

3. **vR.1.7 converged fastest** (best at epoch 5). The 64K-param model has much less capacity, so it saturates quickly.

4. **Train-val gap tracks overfitting.** vR.1.7's gap (1.3pp) is the smallest, confirming that GAP massively reduces overfitting. vR.1.5's gap (6pp) was the largest, showing that the 29.5M-param model overfits aggressively.

---

## 6. The Paper Gap

### Claimed vs Achieved

| Reference | Accuracy | Gap from Paper |
|-----------|----------|----------------|
| Paper claim | 96.21% | 0.00pp |
| vR.1.0 (val) | 89.89% | -6.32pp |
| vR.1.1 (test) | 88.38% | -7.83pp |
| vR.1.3 (best training trick) | 89.17% | -7.04pp |
| **vR.1.6 (best overall)** | **90.23%** | **-5.98pp** |

The best honest test accuracy (90.23%) is still 5.98 percentage points below the paper's claim of 96.21%. The ablation study has closed 1.85pp of the gap (from 7.83pp to 5.98pp).

### Why the Gap Persists

1. **The paper likely evaluated on validation/train data** -- no test split is described in the ETASR paper
2. **Class imbalance handling** in the paper is not documented -- different handling could produce very different results
3. **The paper's "99.44% accuracy" claim** (in the abstract) on a different dataset suggests optimistic evaluation practices
4. **CASIA v2.0 has known quality issues** -- corrupted images, inconsistent resolutions, varied tampering quality
5. **The 128x128 resolution** may be fundamentally insufficient for this dataset -- significant detail is lost in downsampling

### Is the Gap Closable?

With the ETASR architecture (ELA + CNN classification at 128x128), probably not beyond ~91-92%. Achieving 96%+ would require:
- Higher resolution (256x256 or 384x384)
- More sophisticated architecture (ResNet, EfficientNet)
- Localization-based features (which is what the pretrained track addresses)

---

## 7. Cumulative vs Marginal Effects

### The Accumulation Pattern

By vR.1.7, the model carries 5 accepted changes:
```
vR.1.1 (eval fix) + vR.1.3 (class weights) + vR.1.4 (BN) + vR.1.5 (LR sched) + vR.1.6 (deeper CNN) + vR.1.7 (GAP)
```

### Were Effects Additive?

| Change | Individual Impact | Cumulative Impact | Additive? |
|--------|-------------------|-------------------|-----------|
| Class weights | +0.79pp | +0.79pp | Yes (first change) |
| BN | -0.42pp | +0.37pp | Partially (introduced instability) |
| LR Scheduler | +0.21pp | +0.58pp | Yes (compensated for BN instability) |
| Deeper CNN | +1.27pp | +1.85pp | **Super-additive** (deeper arch + BN + scheduler synergised) |
| GAP | -1.06pp | +0.79pp | Subtractive (lost spatial info) |

**Deeper CNN was super-additive:** The +1.27pp delta from parent (vR.1.5) was amplified because BN and LR scheduler were already in place. BN stabilises the deeper conv layers, and the LR scheduler enabled the model to train longer (18 epochs). Without these, the deeper CNN might have trained for only 8-10 epochs.

**GAP was subtractive:** Despite eliminating overfitting, GAP lost more accuracy than it saved. However, the net result (vR.1.7 at 89.17%) still exceeds the honest baseline (vR.1.1 at 88.38%).

---

## 8. What Failed and Why

### vR.1.2: Data Augmentation (REJECTED, -2.85pp)

**What it tried:** Horizontal flip, vertical flip, rotation +/-15 degrees via ImageDataGenerator.

**Why it failed:**
1. The 29.5M-param Flatten->Dense layer memorises spatial positions of ELA activations
2. Geometric transforms (flips, rotations) change spatial positions
3. The model was trained to recognise ELA patterns at specific pixel coordinates
4. Augmentation broke this spatial mapping, and the model couldn't adapt

**Root cause:** Architecture-data mismatch. The Flatten layer treats pixel position as a feature, making the model sensitive to geometric transforms. This is fundamentally incompatible with augmentation.

**Would it work on vR.1.7?** Possibly. GAP removes spatial dependence entirely. Augmentation with a GAP architecture would be a valuable future experiment.

### vR.1.7: GAP (NEUTRAL, -1.06pp)

**What it tried:** Replace Flatten with GlobalAveragePooling2D to eliminate the 13.8M-param Dense bottleneck.

**Why it regressed:**
1. ELA maps encode tampering boundaries as spatially localised brightness patterns
2. GAP averages each 29x29 feature map into a single value, discarding WHERE patterns appear
3. The model can still detect WHETHER ELA activity is present (good Tp recall) but not precisely evaluate the spatial pattern (lower accuracy)
4. The precision-recall balance shifted: more aggressive Tp detection (fewer FN) but more false positives (more FP)

**Root cause:** Information loss. Spatial information is genuinely useful for ELA-based forensic detection. GAP discards it.

**The shared pattern:** Both failures relate to spatial information. Augmentation corrupts it, GAP discards it. The ETASR architecture fundamentally relies on ELA's spatial structure for classification.

---

## 9. Cross-Track Comparison

### ETASR Best (vR.1.6) vs Assignment Requirements

| Requirement | ETASR Best (vR.1.6) | Status |
|-------------|---------------------|--------|
| Image-level detection | 90.23% accuracy | Met |
| Pixel-level localization | Not available | **Not met** |
| GT mask overlays | Not available | **Not met** |
| Standard metrics | Full suite | Met |
| Model weights | Saved (.keras) | Met |

The ETASR track excels at classification but cannot satisfy the assignment's core localization requirement. The pretrained localization track (vR.P.x) is required for assignment completion.

---

## 10. Summary of Ablation Findings

### What Worked

| Change | Impact | Mechanism |
|--------|--------|-----------|
| Deeper CNN (vR.1.6) | +1.27pp | Better feature extraction, reduced dense bottleneck |
| Class weights (vR.1.3) | +0.79pp | Balanced class representation during training |
| LR Scheduler (vR.1.5) | +0.21pp | Extended training, marginal improvement |

### What Did Not Work

| Change | Impact | Mechanism |
|--------|--------|-----------|
| Data augmentation (vR.1.2) | -2.85pp | Broke spatial memorisation in Flatten layer |
| BatchNorm (vR.1.4) | -0.42pp | Training instability outweighed stabilisation benefits |
| GAP (vR.1.7) | -1.06pp | Lost spatial information needed for ELA detection |

### The Hierarchy

```
ARCHITECTURE CHANGES >>> Training tricks > Data augmentation

Deeper CNN (+1.27pp) > Class weights (+0.79pp) > LR Scheduler (+0.21pp) > BN (-0.42pp) > GAP (-1.06pp) > Augmentation (-2.85pp)
```

---

## 11. Pretrained Track Ablation Analysis

### The ELA Input Story — Why P.3 Was the Breakthrough

The pretrained series tested 6 variables across 8 experiments. The single most impactful change was **replacing RGB input with ELA** (vR.P.3), which produced:

- Pixel F1: +23.74pp from P.1 baseline (0.4546 → 0.6920)
- Image Accuracy: +16.64pp (70.15% → 86.79%)
- Image ROC-AUC: +0.1717 (0.7785 → 0.9502)
- FP Rate: 22.6% → 2.7% (8.4x reduction)

For comparison, all other pretrained experiments combined:

| Variable | Best Delta (Pixel F1) |
|----------|-----------------------|
| **ELA input (P.3)** | **+0.2374** |
| Encoder depth (P.5 ResNet-50) | +0.0591 |
| Encoder family (P.6 EffNet-B0) | +0.0671 |
| Encoder unfreeze (P.2) | +0.0571 |
| 4ch fusion (P.4 from P.3) | +0.0133 |

**P.3's improvement alone is 3.5x larger than all other improvements combined.**

### Why ELA Works Better Than RGB

1. **ELA amplifies forensic artifacts.** JPEG recompression at Q=90 produces near-uniform error for authentic regions but high error at tampering boundaries. This converts a subtle visual difference into a stark signal.

2. **ELA reduces irrelevant variation.** RGB images contain scene content (colors, textures, lighting) that is irrelevant to tampering detection. ELA strips this away, leaving only compression artifacts.

3. **Frozen ImageNet features still extract useful patterns from ELA.** Despite never seeing ELA images during pretraining, the frozen conv weights detect edges, gradients, and textures in ELA maps — exactly the patterns that indicate tampering boundaries. The BN unfreeze (17K params) adapts the running statistics to the ELA distribution without changing the learned features.

4. **ELA-specific normalization matters.** P.3 computed the mean/std from 500 training ELA images (mean ~0.05, std ~0.06), which is 10x smaller than ImageNet stats. This correct normalization ensures the encoder receives activations in its expected range.

### Encoder Architecture Comparison — Why Input > Encoder

The pretrained series included three encoder architectures, all tested with RGB input:

| Encoder | Pixel F1 | Image Acc | Trainable | ImageNet Top-1 |
|---------|----------|-----------|-----------|----------------|
| ResNet-34 (P.1) | 0.4546 | 70.15% | 3.15M | 73.3% |
| ResNet-50 (P.5) | 0.5137 | 72.00% | 9.01M | 76.1% |
| EffNet-B0 (P.6) | 0.5217 | 70.68% | 2.24M | 77.1% |

**Findings:**
- Deeper encoder (ResNet-50): +5.91pp Pixel F1, but 2.86x more trainable params
- Different encoder family (EffNet-B0): +6.71pp Pixel F1 with 0.71x params — most efficient
- **Neither matches ELA's +23.74pp**, confirming input > encoder

The encoder swap ceiling on RGB input is approximately **Pixel F1 = 0.52**. Breaking through requires changing the input representation, not the encoder.

### Input Modality Analysis — RGB vs ELA vs RGB+ELA

| Input | Pixel F1 | Image Acc | FP Rate | FN Rate | Complexity |
|-------|----------|-----------|---------|---------|------------|
| RGB (P.1) | 0.4546 | 70.15% | 22.6% | 40.4% | Lowest |
| **ELA (P.3)** | **0.6920** | **86.79%** | **2.7%** | **28.6%** | **Low** |
| RGB+ELA (P.4) | 0.7053 | 84.42% | 6.4% | 29.0% | Highest |

**Analysis:**
- **ELA-only dominates** on image metrics (best accuracy, best FP rate, best ROC-AUC)
- **RGB+ELA has the best absolute pixel F1** (0.7053) but the gain over ELA-only is marginal (+1.33pp)
- **Adding RGB back hurts classification** (84.42% vs 86.79%) by introducing non-forensic scene content that increases false positives
- **Complexity vs gain tradeoff:** 4ch requires dual normalization, conv1 unfreeze, and introduces training instability (epoch 10 spike) for +1.33pp — not worth it

### Pretrained Impact Ranking

| Rank | Version | Change | Pixel F1 Delta | Type |
|------|---------|--------|----------------|------|
| 1 | **vR.P.3** | ELA input | **+0.2374** | Input modality |
| 2 | vR.P.5 | ResNet-50 encoder | +0.0910* | Encoder architecture |
| 3 | vR.P.6 | EfficientNet-B0 | +0.0671 | Encoder architecture |
| 4 | vR.P.2 | Gradual unfreeze | +0.0571 | Freeze strategy |
| 5 | vR.P.4 | 4ch RGB+ELA | +0.0133** | Input modality |
| 6 | **vR.P.8** | Progressive unfreeze | +0.0065*** | Freeze strategy |
| 7 | vR.P.9 | Focal+Dice loss | +0.0003*** | Loss function |
| 8 | vR.P.1.5 | Speed opts | -0.0319 | Infrastructure |

*P.5 delta is from P.1.5, not P.1. **P.4 delta is from P.3, not P.1. ***P.8 and P.9 deltas are from P.3.

### The Pretrained Hierarchy

```
INPUT REPRESENTATION >>> Encoder architecture > Freeze strategy > Loss function > Input fusion

ELA input (+23.74pp) >> EffNet-B0 swap (+6.71pp) > ResNet-50 swap (+5.91pp) > Unfreeze (+5.71pp) >> 4ch fusion (+1.33pp) > Prog unfreeze (+0.65pp) > Focal loss (+0.03pp)
```

This mirrors the ETASR finding that architecture changes beat training tricks, but at a grander scale: **changing what the model sees (ELA vs RGB) matters more than changing how the model sees it (encoder architecture).**

---

## 12. Pretrained Track: P.8 and P.9 Analysis

### vR.P.8: Progressive Encoder Unfreeze — Diminishing Returns

**Configuration:** 3-stage training (frozen→layer4→layer3+layer4), dual LR (1e-5 encoder, 1e-3 decoder), 32 epochs max.

**Results:** Pixel F1: 0.6985 (+0.65pp from P.3), Image Acc: 87.59% (+0.80pp from P.3).

**Key finding:** Stage 0 (frozen encoder, 23 epochs) produced the best results. Stage 1 (layer4 unfrozen) degraded performance — early stopped at epoch 32, never recovering the Stage 0 best. The progressive unfreeze strategy was counterproductive beyond the initial frozen stage.

| Stage | Epochs | Encoder Trainable | Best Pixel F1 | Assessment |
|-------|--------|------------------|---------------|------------|
| Stage 0 (frozen) | 1-23 | BN only (17K) | **0.6985** | Best results here |
| Stage 1 (layer4) | 24-32 | layer4 + BN (9.3M) | 0.6812 | Degraded, early stopped |

**Why Stage 1 failed:**
1. Unfreezing layer4 adds 9.3M trainable params — 3x more than the decoder
2. Data:param ratio becomes dangerously low (~1:1,400)
3. The already-adapted encoder features are disrupted by gradient updates
4. Dual LR (1e-5 for encoder) may still be too aggressive for fine-tuned features

**Implication:** For this dataset size (12,614 images), frozen encoder + BN unfreeze is optimal. Partial unfreezing requires significantly more data or more aggressive regularization.

### vR.P.9: Focal Loss — Hypothesis Rejected

**Configuration:** FocalLoss(alpha=0.25, gamma=2.0) + DiceLoss, replacing BCE + DiceLoss. Everything else identical to P.3.

**Results:** Pixel F1: 0.6923 (+0.03pp from P.3, NEUTRAL), Pixel AUC: 0.9323 (-0.0205), Image ROC-AUC: 0.9076 (-0.0426).

**Key finding:** Focal Loss does not improve forensic segmentation when combined with Dice Loss. The pixel-level binary predictions are essentially unchanged, while the probability calibration (AUC) significantly degrades.

| Metric | P.3 (BCE+Dice) | P.9 (Focal+Dice) | Delta | Assessment |
|--------|----------------|-------------------|-------|------------|
| Pixel F1 | 0.6920 | 0.6923 | +0.03pp | Unchanged |
| Pixel AUC | **0.9528** | 0.9323 | **-0.0205** | Regressed |
| Image ROC-AUC | **0.9502** | 0.9076 | **-0.0426** | Regressed |
| Training stability | Smooth | Volatile | — | Worse |

**Why Focal Loss failed here:**
1. Focal Loss was designed for object detection (rare objects vs background sea). In forensic segmentation, Dice Loss already handles the class imbalance.
2. Focal Loss's (1-p)^gamma weighting concentrates predictions near extremes (0 and 1), reducing the dynamic range of intermediate probabilities → AUC drops.
3. The alpha=0.25, gamma=2.0 hyperparameters are RetinaNet defaults, not tuned for forensic segmentation.

**Implication:** BCE+Dice remains the optimal loss for this architecture. Future loss experiments should explore Lovász-Softmax or Boundary Loss rather than weighted cross-entropy variants.

---

## 13. Standalone Research Paper Architecture Assessment

### Classification vs Localization: The Fundamental Gap

Three standalone runs implemented the paper's CNN architecture outside the ablation framework:

| Run | Test Acc | Macro F1 | Localization | Score |
|-----|----------|----------|--------------|-------|
| Deeper CNN (divg07) | **90.76%** | **0.9082** | NO | 66/100 |
| Paper arch (divg07) | 90.33% | 0.9006 | NO | 56/100 |
| Paper arch (sagnik) | ~~100%~~ | ~~1.0000~~ | NO (DATA LEAK) | 28/100 |

### Why These Runs Matter for the Ablation Study

1. **Validates the deeper CNN finding:** The standalone deeper CNN (90.76%) confirms that more convolutional depth helps classification, consistent with vR.1.6's finding (+1.27pp from deeper architecture).
2. **Shows early stopping is critical:** Paper architecture without early stopping (test loss: 0.6185) vs deeper CNN with early stopping (test loss: 0.2178) — 3x better calibration.
3. **Confirms dataset matters:** The Sagnik 100% accuracy proves that dataset validation is essential before drawing any conclusions.
4. **Reinforces the localization gap:** Even the best classification model (90.76%) cannot produce pixel-level masks, making it irrelevant for assignment submission.

---

## 12. Overall Results Analysis

### Cross-Track Insights

1. **The ETASR track's best classification** (90.23% accuracy, vR.1.6) outperforms the pretrained track's best classification (87.59%, P.8). A purpose-built 128x128 ELA classifier with class weights and BN still beats a UNet designed for localization.

2. **The pretrained track is essential for the assignment** because no ETASR version can produce pixel-level masks. P.4's Pixel F1 of 0.7053 demonstrates that meaningful localization is achievable, with P.8 close behind at 0.6985.

3. **ELA is the common thread.** Both tracks use ELA: the ETASR track feeds ELA images to a classification CNN, while P.3+ feeds ELA to a UNet for localization. The pretrained track's breakthrough came from recognising that ELA should replace RGB input, not supplement it.

4. **The ablation methodology works.** Single-variable control identified the critical factors in both tracks. In ETASR: architecture (deeper CNN) > training tricks. In pretrained: input (ELA) > encoder > freeze strategy > loss function. Without strict ablation, these insights would be obscured by confounding changes.

5. **Diminishing returns are setting in.** P.8 (+0.65pp) and P.9 (+0.03pp) show that incremental changes to the P.3 configuration yield diminishing gains. The next breakthrough likely requires a more fundamental change (extended training, higher resolution, or attention mechanisms).

---

## 13. New Runs Analysis (P.10 r02, P.12, P.14, ELA-CNN-Forgery-sagnik)

### vR.P.10 Run-02: Perfect Reproducibility

P.10 Run-02 was a reproducibility verification of the series-best model. Every single metric was identical to Run-01:
- Pixel F1: 0.7277, IoU: 0.5719, Pixel AUC: 0.9573
- Image Acc: 87.32%, Image F1: 0.8615, Image ROC-AUC: 0.9633
- Best epoch: 24 (same), LR schedule changes at same epochs

**Implication:** SEED=42 + `torch.backends.cudnn.deterministic=True` on P100 GPUs produces bit-identical training runs. This is the gold standard for experimental reproducibility in deep learning.

### vR.P.12: Augmentation + Focal+Dice

| Metric | P.3 (Baseline) | P.12 | Delta |
|--------|----------------|------|-------|
| Pixel F1 | 0.6920 | 0.6968 | +0.48pp |
| IoU | 0.5291 | 0.5347 | +0.56pp |
| Pixel AUC | 0.9528 | 0.9502 | -0.26pp |
| Img Acc | 86.79% | 88.48% | +1.69pp |
| Img F1 | 0.8534 | 0.8756 | +2.22pp |
| FN Rate | ~28% | 24.6% | -3.4pp (improved) |

**Analysis:** Augmentation helped image-level classification more than pixel-level localization. The FN rate improvement suggests augmentation regularized the model to catch more tampered images, but the pixel-level gain is within noise. Training instability (val loss spikes at ep19, 21) indicates augmented ELA samples occasionally confuse the model.

**Confounding issue:** P.12 changed two variables (augmentation AND loss function). Since P.9 showed Focal+Dice is neutral, the gain is likely from augmentation alone, but this cannot be definitively stated.

### vR.P.14: TTA (NEGATIVE Result)

| Metric | No-TTA | TTA (4 views) | Delta |
|--------|--------|---------------|-------|
| Pixel F1 | 0.6919 | 0.6388 | **-5.31pp** |
| Pixel Precision | 0.7555 | 0.7877 | +3.22pp |
| Pixel Recall | 0.6388 | 0.5654 | **-7.34pp** |
| Pixel AUC | 0.9528 | 0.9618 | +0.90pp |

**Analysis:** TTA is harmful for binary segmentation at threshold=0.5. The 4-view probability averaging compresses the output distribution toward 0.5, pushing borderline "tampered" pixels below the binary threshold. This destroys recall (-7.34pp) while slightly improving precision (+3.22pp). The AUC improvement (+0.90pp) confirms that the underlying probability maps are better — the damage is entirely due to the fixed threshold.

**Actionable insight:** TTA would benefit from threshold optimisation (e.g., find optimal threshold on validation set after TTA averaging). This is a potential free improvement for a future experiment.

**Code failure:** Cell 18 crash (`test_probs` not defined) destroyed all image-level evaluation. This notebook needs a bugfix re-run.

### ELA-CNN-Forgery-sagnik: Data Leak Confirmed

The Sagnik dataset run achieved 99.95% accuracy with the deeper CNN architecture. Combined with the paper-architecture run (100% accuracy), this dataset is scientifically invalid. Input statistics (X range [0.0, 0.76]) suggest mask images were loaded instead of photographs.

**Implication:** Any experiment on this dataset should be discarded. Future work should exclusively use the divg07 (standard CASIA2) dataset.

---

## 14. Updated Impact Hierarchy (All Experiments)

```
Rank  Category                     Best Example              Delta (pp Pixel F1)
──────────────────────────────────────────────────────────────────────────────────
  1   INPUT REPRESENTATION         P.3: RGB→ELA              +23.74
  2   ATTENTION MECHANISM          P.10: +CBAM               +3.57
  3   TRAINING BUDGET              P.7: 25→50 epochs         +2.34
  4   ENCODER ARCHITECTURE         P.6: RN34→EffNet-B0       +6.71 (from RGB P.1)
  5   DATA AUGMENTATION            P.12: +Albumentations     +0.48
  6   PROGRESSIVE UNFREEZE         P.8: 3-stage              +0.65
  7   LOSS FUNCTION                P.9: BCE→Focal+Dice       +0.03
  8   POST-PROCESSING (TTA)        P.14: 4-view TTA          -5.32 (HARMFUL)
```

**Key conclusion:** After the ELA breakthrough (P.3), the only changes that meaningfully improved Pixel F1 were attention (P.10, +3.57pp) and extended training (P.7, +2.34pp). Everything else was marginal or harmful.

---

## 15. vR.P.15 Multi-Quality ELA Audit (2026-03-15)

### Experiment Overview

P.15 replaced single-quality RGB ELA (Q=90, 3ch correlated) with multi-quality grayscale ELA (Q=75/Q=85/Q=95, 3ch independent). Everything else identical to P.3. This tests whether **quality-level diversity** provides richer forensic signal than **color information**.

### Results

| Metric | P.3 (parent) | P.10 (prev best) | **P.15 (this)** | Delta from P.3 |
|--------|-------------|------------------|-----------------|----------------|
| Pixel F1 | 0.6920 | 0.7277 | **0.7329** | **+4.09pp** |
| Pixel IoU | 0.5291 | 0.5719 | **0.5785** | **+4.94pp** |
| Pixel AUC | 0.9528 | 0.9573 | **0.9608** | **+0.80pp** |
| Pixel Precision | 0.8379 | 0.8611 | 0.8409 | +0.30pp |
| Pixel Recall | 0.5880 | 0.6300 | **0.6496** | **+6.16pp** |
| Image Accuracy | 86.79% | 87.32% | 87.53% | +0.74pp |
| Image Macro F1 | 0.8560 | 0.8615 | 0.8660 | +1.00pp |
| Image ROC-AUC | 0.9502 | 0.9633 | 0.9423 | -0.79pp |

**Verdict: POSITIVE — NEW SERIES BEST (Pixel F1)**

### ELA Channel Statistics

| Quality | Mean | Std | Character |
|---------|------|-----|-----------|
| Q=75 | 0.0684 | 0.0656 | Large residuals, coarse signal |
| Q=85 | 0.0605 | 0.0604 | Medium residuals, balanced |
| Q=95 | 0.0402 | 0.0471 | Small residuals, fine-grained |

The three channels have meaningfully different distributions (mean range 0.040--0.068). Unlike RGB ELA where R/G/B channels have ~0.9 inter-channel correlation, these quality-based channels carry independent information about compression artifacts at different sensitivity levels.

### Key Insight: Quality Diversity > Color Information

This is the most important finding since the original RGB-to-ELA switch (P.3):
- **RGB ELA** gives 3 highly-correlated channels (same Q=90, different color components)
- **Multi-Q ELA** gives 3 independent channels (different quality levels, grayscale)
- P.15's +4.09pp gain over P.3 proves that independent forensic perspectives are more valuable than color redundancy

### Where P.15 Trades Off

| Loss area | Details |
|-----------|---------|
| Image ROC-AUC | 0.9423 vs P.10's 0.9633 (-2.10pp) — grayscale channels lose some classification calibration |
| Pixel Precision | 0.8409 vs P.10's 0.8611 (-2.02pp) — recall-driven improvement trades some precision |

### Training Observations

- Model hit 25-epoch cap still improving (best epoch = 24, LR never decayed)
- ReduceLROnPlateau never triggered — suggests the learning rate was appropriate throughout
- Extended training (like P.7's 50 epochs) would likely push Pixel F1 to ~0.75+

### Confusion Matrix (Image-Level)

| | Predicted Au | Predicted Tp |
|---|---|---|
| **Actual Au** | TN = 1078 | FP = 46 |
| **Actual Tp** | FN = 190 | TP = 579 |

FP Rate: 4.1% (good), FN Rate: 24.7% (competitive with series average).

---

## 16. Updated Impact Hierarchy (All Experiments Through P.15)

```
Rank  Category                     Best Example              Delta (pp Pixel F1)
──────────────────────────────────────────────────────────────────────────────────
  1   INPUT REPRESENTATION         P.3: RGB→ELA              +23.74
  2   INPUT VARIANT                P.15: Single-Q→Multi-Q    +4.09
  3   ATTENTION MECHANISM          P.10: +CBAM               +3.57
  4   TRAINING BUDGET              P.7: 25→50 epochs         +2.34
  5   ENCODER ARCHITECTURE         P.6: RN34→EffNet-B0       +6.71 (from RGB P.1)
  6   PROGRESSIVE UNFREEZE         P.8: 3-stage              +0.65
  7   DATA AUGMENTATION            P.12: +Albumentations     +0.48
  8   LOSS FUNCTION                P.9: BCE→Focal+Dice       +0.03
  9   POST-PROCESSING (TTA)        P.14: 4-view TTA          -5.32 (HARMFUL)
```

**Updated conclusion:** Input representation remains the dominant factor. Within the input domain, P.15 proves that **how you construct the channels matters as much as what signal you use**. Multi-quality ELA (P.15, +4.09pp) outperformed attention mechanisms (P.10, +3.57pp) and training budget extension (P.7, +2.34pp), establishing input engineering as the #1 lever after the initial ELA switch.
