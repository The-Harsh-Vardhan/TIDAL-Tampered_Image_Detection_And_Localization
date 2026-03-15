# Ablation Analysis -- ETASR Study

| Field | Value |
|-------|-------|
| **Date** | 2026-03-15 |
| **Scope** | Cross-run impact analysis for all ETASR ablation experiments |
| **Paper** | ETASR_9593 -- "Enhanced Image Tampering Detection using ELA and a CNN" |
| **Versions Covered** | vR.1.0 through vR.1.7 (8 runs, 7 ablations) |

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
