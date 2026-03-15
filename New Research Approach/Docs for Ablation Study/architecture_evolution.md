# Architecture Evolution -- ETASR Ablation Study

| Field | Value |
|-------|-------|
| **Date** | 2026-03-15 |
| **Scope** | Structural progression of the ETASR CNN and Pretrained UNet across all ablation versions |
| **Paper** | ETASR_9593 -- "Enhanced Image Tampering Detection using ELA and a CNN" |
| **Versions Covered** | ETASR: vR.1.0--vR.1.7 / Pretrained: vR.P.0--vR.P.15 / Standalone: 3 runs |

---

## 1. Architecture Timeline

```
Phase 1: Paper-Faithful                Phase 2: Training         Phase 3: Architecture
(vR.1.0 -- vR.1.3)                     Stabilization             Restructuring
                                        (vR.1.4 -- vR.1.5)       (vR.1.6 -- vR.1.7)

Conv2D(32,5x5)                         Conv2D(32,5x5)            Conv2D(32,5x5)
    |                                       |                         |
Conv2D(32,5x5)                          +BatchNorm+               +BatchNorm+
    |                                   Conv2D(32,5x5)            Conv2D(32,5x5)
MaxPool(2x2)                            +BatchNorm+               +BatchNorm+
    |                                   MaxPool(2x2)              MaxPool(2x2)
Dropout(0.25)                               |                         |
    |                                   Dropout(0.25)             +Conv2D(64,3x3)+ (vR.1.6)
Flatten (115,200)                           |                     +MaxPool(2x2)+  (vR.1.6)
    |                                   Flatten (115,200)             |
Dense(256)                                  |                     Dropout(0.25)
    |                                   Dense(256)                    |
Dropout(0.5)                                |                     +GAP (64)+      (vR.1.7)
    |                                   Dropout(0.5)              or Flatten(53,824) (vR.1.6)
Dense(2, softmax)                           |                         |
                                        Dense(2, softmax)         Dense(256)
                                                                      |
~29.5M params                           ~29.5M params             Dropout(0.5)
99.9% in Dense                          99.9% in Dense                |
                                                                  Dense(2, softmax)

                                                                  vR.1.6: ~13.8M (53% reduction)
                                                                  vR.1.7: ~64K (99.8% reduction)
```

---

## 2. Phase 1: Paper-Faithful Architecture (vR.1.0 -- vR.1.3)

### Versions: vR.1.0 (baseline), vR.1.1 (eval fix), vR.1.2 (augmentation), vR.1.3 (class weights)

The model architecture is **identical** across all four versions. Changes are limited to evaluation methodology (vR.1.1), data pipeline (vR.1.2), and training configuration (vR.1.3).

```
Input: 128x128x3 (ELA map, /255.0 normalized)
    |
Conv2D(32, 5x5, ReLU, valid)       32 filters, 5x5 kernel
    |                               Output: 124x124x32
    |                               Params: 3*5*5*32 + 32 = 2,432
    |
Conv2D(32, 5x5, ReLU, valid)       32 filters, 5x5 kernel
    |                               Output: 120x120x32
    |                               Params: 32*5*5*32 + 32 = 25,632
    |
MaxPooling2D(2x2)                  Output: 60x60x32
    |
Dropout(0.25)
    |
Flatten()                           Output: 60*60*32 = 115,200
    |
Dense(256, ReLU)                    Params: 115,200*256 + 256 = 29,491,456
    |
Dropout(0.5)
    |
Dense(2, softmax)                   Params: 256*2 + 2 = 514

Total: 29,520,034 trainable parameters
```

### The Fundamental Problem

**99.9% of all parameters** live in a single layer: `Flatten(115,200) -> Dense(256)` = 29,491,456 params. This layer effectively memorizes the spatial positions of ELA activations rather than learning generalizable features. The two Conv2D layers contribute only 28,064 params (0.1%).

This extreme parameter imbalance explains:
- Why vR.1.2 (augmentation) failed: geometric transforms change spatial positions, breaking the memorised mapping
- Why vR.1.3 (class weights) improved recall but not AUC: it shifted the decision threshold, not the underlying feature quality

---

## 3. Phase 2: Training Stabilization (vR.1.4 -- vR.1.5)

### vR.1.4: BatchNormalization

Added `BatchNormalization()` after each Conv2D layer. This introduced 256 additional parameters (128 gamma + 128 beta per BN layer, total 256 trainable + 256 non-trainable for running mean/var).

```
Conv2D(32, 5x5, ReLU)    Params: 2,432
BatchNormalization()      Params: 64 trainable (32 gamma + 32 beta) + 64 non-trainable
Conv2D(32, 5x5, ReLU)    Params: 25,632
BatchNormalization()      Params: 64 trainable + 64 non-trainable
MaxPooling2D(2x2)
...rest unchanged...

Total: 29,520,290 (29,520,162 trainable, 128 non-trainable)
```

**Impact:** BN introduced **epoch 1 catastrophe** (val_loss spiked to 16.13 in vR.1.4) because BN's running statistics are uninitialised at start. Training shortened from 14 to 8 productive epochs. Test accuracy dropped 0.42pp.

### vR.1.5: ReduceLROnPlateau

No architecture change. Added `ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)` callback.

**Impact:** Extended training by 2 epochs (10 vs 8). Marginal +0.21pp accuracy improvement. The scheduler could not fix the underlying overfitting from 29.5M params.

### Phase 2 Lesson

Training tricks (BN, LR scheduling) produced negligible net improvement (+0.21pp from two modifications). The architecture was the bottleneck, not the training procedure.

---

## 4. Phase 3: Architecture Restructuring (vR.1.6 -- vR.1.7)

### vR.1.6: Deeper CNN

Added a third convolutional layer (`Conv2D(64, 3x3, ReLU)`) and second `MaxPooling2D(2x2)` between the existing convolutions and the Flatten layer.

```
Conv2D(32, 5x5, ReLU, valid)       Output: 124x124x32     Params: 2,432
BatchNormalization()                                        Params: 64+64
Conv2D(32, 5x5, ReLU, valid)       Output: 120x120x32     Params: 25,632
BatchNormalization()                                        Params: 64+64
MaxPooling2D(2x2)                   Output: 60x60x32
Conv2D(64, 3x3, ReLU, valid)       Output: 58x58x64       Params: 18,496  [NEW]
MaxPooling2D(2x2)                   Output: 29x29x64                       [NEW]
Dropout(0.25)
Flatten()                           Output: 29*29*64 = 53,824
Dense(256, ReLU)                    Params: 53,824*256+256 = 13,779,200
Dropout(0.5)
Dense(2, softmax)                   Params: 514

Total: 13,826,530 (13,826,402 trainable, 128 non-trainable)
```

**Key effect:** The extra Conv+Pool reduced the spatial dimensions from 60x60 to 29x29 before Flatten, cutting the Flatten->Dense connection from 115,200 to 53,824 inputs (-53.3%). This reduced total parameters by 53.2% while simultaneously improving feature extraction depth.

**Result:** Best test accuracy (90.23%), best Macro F1 (0.9004), best ROC-AUC (0.9657). First run to exceed 90% and first to improve AUC above baseline.

### vR.1.7: GlobalAveragePooling2D

Replaced `Flatten()` with `GlobalAveragePooling2D()`. This averages each of the 64 feature maps (29x29) into a single value, producing a 64-dimensional vector instead of flattening into 53,824.

```
...same as vR.1.6 through MaxPooling2D(2x2)...
Dropout(0.25)
GlobalAveragePooling2D()            Output: 64             Params: 0      [CHANGED]
Dense(256, ReLU)                    Params: 64*256+256 = 16,640
Dropout(0.5)
Dense(2, softmax)                   Params: 514

Total: 63,970 (63,842 trainable, 128 non-trainable)
```

**Key effect:** The Dense(256) layer dropped from 13,779,200 params (Flatten) to 16,640 params (GAP) -- a 99.9% reduction in this single layer. Total model: 13.8M -> 64K.

**Trade-off:** GAP destroys spatial information. The 29x29 spatial map is averaged into a single value per channel, so the model can no longer distinguish WHERE features fire, only WHETHER they fire. For ELA-based detection, where tampering boundaries occupy specific spatial positions, this is a meaningful loss.

**Result:** Accuracy dropped 1.06pp (90.23% -> 89.17%), but the model achieved best-in-series Tp recall (0.9467) and Au precision (0.9590) with massively reduced overfitting (train-val gap: 1.3pp vs vR.1.6's 5.0pp).

---

## 5. Layer-by-Layer Comparison

| Layer | Phase 1 (1.0-1.3) | Phase 2 (1.4-1.5) | vR.1.6 | vR.1.7 |
|-------|-------------------|-------------------|--------|--------|
| Conv2D(32,5x5) | 2,432 | 2,432 | 2,432 | 2,432 |
| BatchNorm | -- | 64+64 | 64+64 | 64+64 |
| Conv2D(32,5x5) | 25,632 | 25,632 | 25,632 | 25,632 |
| BatchNorm | -- | 64+64 | 64+64 | 64+64 |
| MaxPool(2x2) | 0 | 0 | 0 | 0 |
| Conv2D(64,3x3) | -- | -- | 18,496 | 18,496 |
| MaxPool(2x2) | -- | -- | 0 | 0 |
| Dropout(0.25) | 0 | 0 | 0 | 0 |
| **Flatten/GAP** | **0** | **0** | **0** | **0** |
| **Dense(256)** | **29,491,456** | **29,491,456** | **13,779,200** | **16,640** |
| Dropout(0.5) | 0 | 0 | 0 | 0 |
| Dense(2) | 514 | 514 | 514 | 514 |
| **Total** | **29,520,034** | **29,520,290** | **13,826,530** | **63,970** |

---

## 6. Parameter Distribution Analysis

| Version | Conv Params | BN Params | Dense(256) Params | Dense(2) Params | Dense% of Total |
|---------|-------------|-----------|-------------------|-----------------|-----------------|
| vR.1.0-1.3 | 28,064 | 0 | 29,491,456 | 514 | **99.90%** |
| vR.1.4-1.5 | 28,064 | 256 | 29,491,456 | 514 | **99.90%** |
| vR.1.6 | 46,560 | 256 | 13,779,200 | 514 | **99.66%** |
| vR.1.7 | 46,560 | 256 | 16,640 | 514 | **26.88%** |

**The inflection point:** vR.1.7 is the only version where convolutional parameters (46,560 = 72.78%) exceed dense parameters (17,154 = 26.88%). In all prior versions, Dense(256) dominated the parameter budget by 99%+.

---

## 7. The Flatten->Dense Bottleneck Story

### Discovery (vR.1.1 -- vR.1.3)

The baseline architecture had 29.5M parameters, but only two conv layers extracting features. The Dense(256) layer, receiving 115,200 flattened spatial inputs, contained 99.9% of all parameters. This configuration:

1. **Memorizes spatial positions** rather than learning invariant features
2. **Breaks with augmentation** (vR.1.2 REJECTED -2.85pp) because flips/rotations change pixel positions
3. **Limits AUC improvement** -- class weights (vR.1.3) improved recall but AUC barely changed, meaning the underlying feature discrimination was not improved

### Mitigation Attempts (vR.1.4 -- vR.1.5)

BatchNorm and LR scheduling were indirect approaches. They stabilised training and extended it slightly, but could not fundamentally address the 29.5M parameter bottleneck. Combined net improvement: +0.58pp accuracy, -0.0041 AUC.

### Direct Attack (vR.1.6)

Adding a 3rd Conv2D + MaxPool reduced the spatial dimensions before Flatten, cutting the Dense(256) input from 115,200 to 53,824. This:
- Forced the model to extract features through convolution rather than memorising positions
- Reduced total params by 53% while improving all metrics
- Produced the longest productive training (18 epochs) -- more features = more to learn

### Elimination (vR.1.7)

GlobalAveragePooling2D replaced Flatten entirely, reducing Dense(256) from 13.8M to 16.6K params. This:
- Eliminated spatial memorisation completely (GAP outputs channel statistics only)
- Reduced overfitting (train-val gap: 1.3pp vs 5.0pp)
- Preserved most accuracy (89.17% vs 90.23%)
- Proved that convolutional features alone contain meaningful forensic signal

### The Conclusion

The optimal architecture lives between vR.1.6 (too many dense params, overfitting) and vR.1.7 (too aggressive pooling, lost spatial info). The ideal next step would be to combine GAP with more convolutional layers or channels, giving the model enough representational power without the Flatten->Dense bottleneck.

---

## 8. Key Architectural Insights

### 1. Parameter location matters more than parameter count

vR.1.7 (64K params) outperforms vR.1.0-1.3 (29.5M params) on Tp recall and Au precision. Parameters in conv layers (feature extraction) are more valuable than parameters in Dense layers (memorisation).

### 2. Spatial information matters for ELA detection

vR.1.6 (Flatten) outperforms vR.1.7 (GAP) by 1.06pp despite having 216x more parameters. ELA maps encode tampering boundaries as spatially localised brightness patterns. GAP averages these away, losing valuable forensic signal.

### 3. Feature extraction depth is the single most impactful change

Of all 7 ablations, vR.1.6 (+1.27pp from parent, +1.85pp from baseline) had the largest positive impact. Training configuration changes (class weights +0.79pp, BN -0.42pp, LR scheduler +0.21pp) combined for only +0.58pp.

### 4. The overfitting-capacity tradeoff

| Version | Train-Val Gap (final) | Test Accuracy |
|---------|----------------------|---------------|
| vR.1.3 | ~3pp | 89.17% |
| vR.1.5 | ~6pp | 88.96% |
| vR.1.6 | ~5pp | 90.23% |
| vR.1.7 | ~1.3pp | 89.17% |

vR.1.6 slightly overfits but achieves the highest accuracy. vR.1.7 barely overfits but has lower accuracy. The right amount of overfitting is "some" -- the model needs enough capacity to learn complex patterns, but not so much that it memorises training data.

---

## 9. Architecture Comparison with Paper

| Aspect | ETASR Paper | vR.1.0 (Repro) | vR.1.7 (Final) | Divergence |
|--------|-------------|-----------------|-----------------|------------|
| Conv layers | 2 | 2 | 3 | +1 conv layer |
| Conv filters | 32/32 | 32/32 | 32/32/64 | +64 filters |
| BatchNorm | No | No | Yes (2 layers) | Added |
| Pooling | MaxPool | MaxPool | MaxPool + GAP | Changed classifier head |
| Dense input | Flatten | Flatten | GAP | Changed |
| Classifier | Dense(256)+Dense(2) | Dense(256)+Dense(2) | Dense(256)+Dense(2) | Unchanged |
| LR Schedule | None | None | ReduceLROnPlateau | Added |
| Class weights | None | None | Inverse-freq | Added |
| Claimed accuracy | 96.21% | 89.89% (val) | 89.17% (test) | -7.04pp gap |

The final architecture (vR.1.7) diverges from the paper in 5+ ways, all intentional and documented as part of the ablation study. The paper's claimed 96.21% accuracy remains unreproducible -- the best honest test accuracy achieved is 90.23% (vR.1.6).

---

## 10. Pretrained Track Architecture Evolution

### Base Architecture: UNet + ResNet-34

All pretrained experiments use the **UNet** decoder from Segmentation Models PyTorch (SMP) with pretrained ImageNet encoders. The key structural elements:

```
Input (384x384x3 or 4ch)
    |
    v
ENCODER (pretrained, frozen or partially unfrozen)
    |-- Stage 1: 192x192 → [skip 1]
    |-- Stage 2:  96x96  → [skip 2]
    |-- Stage 3:  48x48  → [skip 3]
    |-- Stage 4:  24x24  → [skip 4]
    |-- Bottleneck: 12x12
    v
DECODER (trainable, ~500K-9M params)
    |-- Up1: 12→24  + skip 4
    |-- Up2: 24→48  + skip 3
    |-- Up3: 48→96  + skip 2
    |-- Up4: 96→192 + skip 1
    v
Segmentation Head (1x1 conv → 1 channel)
    v
Sigmoid → 384x384 binary mask
```

### Encoder Comparison

| Property | ResNet-34 | ResNet-50 | EfficientNet-B0 |
|----------|-----------|-----------|-----------------|
| Block type | BasicBlock (3x3+3x3) | Bottleneck (1x1+3x3+1x1) | MBConv (expand+depthwise+SE+project) |
| Encoder params | 21.3M | 23.5M | 4.0M |
| ImageNet top-1 | 73.3% | 76.1% | 77.1% |
| Skip channels | [64, 64, 128, 256, 512] | [64, 256, 512, 1024, 2048] | [16, 24, 40, 112, 320] |
| Decoder params | ~3.15M | ~9.01M | ~2.24M |
| Total trainable | 3.15M (decoder) | 9.01M (decoder) | 2.24M (decoder) |
| Data:param ratio | 1:357 | 1:1,021 | 1:254 |
| Attention | None | None | SE (per block) |
| Best Pixel F1 (RGB) | 0.4546 (P.1) | 0.5137 (P.5) | 0.5217 (P.6) |

**Key observation:** EfficientNet-B0 achieves the best RGB pixel F1 (0.5217) with the fewest trainable parameters (2.24M). ResNet-50's wider skip connections create a 3x larger decoder that doesn't proportionally improve results.

### Input Modality Comparison

| Input | Channels | Normalization | Best Version | Pixel F1 | Img Acc |
|-------|----------|---------------|-------------|----------|---------|
| **ELA** | 3 | ELA-specific (computed) | **vR.P.3** | **0.6920** | **86.79%** |
| RGB+ELA | 4 | Dual (ImageNet + ELA) | vR.P.4 | 0.7053 | 84.42% |
| RGB | 3 | ImageNet | vR.P.6 | 0.5217 | 72.00% |

**ELA is the single biggest improvement in either track.** Switching from RGB to ELA input produced +23.74pp Pixel F1 (vs P.1 baseline) — 3x larger than any encoder swap or training trick.

### Freeze Strategy Evolution

```
P.0/P.1/P.1.5:  All encoder FROZEN (conv + BN)
                 Decoder only trainable (~3.15M)
                 → Pixel F1: 0.37--0.45

P.2:             Encoder layer3+layer4 UNFROZEN (differential LR)
                 23M trainable, severe overfitting
                 → Pixel F1: 0.51

P.3:             Encoder FROZEN body + BN UNFROZEN
                 3.17M trainable, best results
                 → Pixel F1: 0.69

P.4:             Encoder FROZEN body + conv1 UNFROZEN + BN UNFROZEN
                 3.18M trainable, marginal over P.3
                 → Pixel F1: 0.71

P.5/P.6:        All encoder FROZEN (like P.1)
                 Decoder only (9M / 2.24M)
                 → Pixel F1: 0.51 / 0.52
```

**The BN unfreeze strategy (P.3) is the sweet spot:** only 17K extra trainable params for domain adaptation, but they allow the encoder's batch normalization to adapt running statistics to the ELA distribution. This is the key enabler of P.3's breakthrough result.

### Extended Freeze Strategy (P.8, P.9)

```
P.8:             3-STAGE PROGRESSIVE UNFREEZE
                 Stage 0: Frozen body + BN unfrozen (epochs 1-23)
                 Stage 1: layer4 unfrozen + BN (epochs 24-32, early stopped)
                 Stage 2: layer3+layer4 unfrozen (never reached)
                 Dual LR: 1e-5 encoder, 1e-3 decoder
                 Trainable: 3.17M → 12.5M → 18.7M (by stage)
                 → Pixel F1: 0.6985 (best at Stage 0, epoch 23)

P.9:             Same freeze as P.3 (frozen body + BN unfrozen)
                 Loss change: FocalLoss(alpha=0.25, gamma=2.0) + DiceLoss
                 Trainable: 3.17M (unchanged)
                 → Pixel F1: 0.6923
```

**Key finding from P.8:** Progressive unfreezing was counterproductive. The best results occurred during Stage 0 (frozen encoder), and unfreezing layer4 in Stage 1 caused performance to degrade. This confirms that for this dataset size (~12.6K images), the BN-only unfreeze strategy is optimal.

**Key finding from P.9:** Changing the loss function (Focal vs BCE) while keeping the freeze strategy constant had negligible impact on pixel-level metrics but degraded probability calibration (AUC). Loss function is a low-impact variable.

---

## 11. Standalone CNN Architecture Comparison

### Paper Architecture vs Deeper Variant

Three standalone runs tested the original paper CNN architecture and a deeper variant, outside the main ablation framework. These provide classification-only baselines.

```
Paper Architecture (divg07 + sagnik):       Deeper Architecture (divg07):

Input: ELA 150x150, [0,1]                   Input: ELA 150x150, [0,1]
    |                                            |
Conv2D(32, 5x5, ReLU)   2,432 params        Conv2D(64, 3x3, ReLU)   1,792 params
    |                                        BatchNorm
Conv2D(32, 5x5, ReLU)   25,632 params       MaxPool(2x2)
    |                                            |
MaxPool(2x2)                                 Conv2D(128, 3x3, ReLU)  73,856 params
    |                                        BatchNorm
Flatten (161,312)                            MaxPool(2x2)
    |                                            |
Dense(150, ReLU)    **24,196,950** params    Conv2D(256, 3x3, ReLU)  295,168 params
    |                                        BatchNorm
Dense(2, sigmoid)   302 params               MaxPool(2x2)
                                                 |
Total: 24,225,316                            Dropout(0.5)
99.88% in Dense bottleneck                   Flatten (73,984)
                                                 |
                                             Dense(512, ReLU)  **37,880,320** params
                                             Dropout(0.5)
                                             Dense(2, sigmoid)  1,026 params

                                             Total: 38,253,954
                                             99.0% in Dense bottleneck
```

### Comparison with ETASR Ablation Architecture

| Property | ETASR (128x128) | Paper Standalone (150x150) | Deeper Standalone (150x150) |
|----------|-----------------|---------------------------|----------------------------|
| Conv blocks | 2 (32+32) → 3 (32+32+64) | 2 (32+32) | 3 (64+128+256) |
| Kernel size | 5×5 / 3×3 | 5×5 | 3×3 |
| BatchNorm | Added in vR.1.4+ | No | Yes |
| Dense input | 115,200 → 53,824 → GAP(64) | 161,312 | 73,984 |
| Dense bottleneck | 29.5M → 13.8M → 16.6K | 24.2M | 37.9M |
| Early stopping | Yes (patience=5) | No (40 fixed epochs) | Yes (patience=5) |
| Best test acc | 90.23% (vR.1.6) | 90.33% | **90.76%** |
| Test loss | ~0.35 | 0.6185 (severely overfit) | 0.2178 (well calibrated) |

**Key insight:** The deeper standalone CNN (90.76%) marginally outperforms the ETASR deepest (90.23%), but at the cost of 38.3M vs 13.8M parameters. Early stopping is the primary driver of the standalone's lower test loss, not the deeper architecture itself.

---

## 12. Pretrained Track Architecture Evolution (P.10--P.14)

### CBAM Attention Module (vR.P.10)

```
Decoder Block (per stage):
    upsampled features + skip connection
        |
    [Standard UNet Conv Block]
        |
    [Channel Attention]  ← SEBlock variant (avg+max pool → shared MLP → sigmoid)
        |
    [Spatial Attention]  ← 7×7 conv on channel-wise avg+max → sigmoid
        |
    Output features (refined)

5 CBAM blocks injected into all decoder stages.
Trainable param increase: +14K (3.17M → 3.18M)
Impact: +3.57pp Pixel F1 (0.6920 → 0.7277) — SERIES BEST
```

CBAM's success confirms that decoder-side attention is highly effective for forensic segmentation. The attention maps learn to weight ELA-sensitive spatial regions (boundaries, high-frequency areas) while suppressing noise.

### Data Augmentation Pipeline (vR.P.12)

```
Albumentations Compose:
    HorizontalFlip(p=0.5)
    VerticalFlip(p=0.3)
    RandomRotate90(p=0.5)
    ShiftScaleRotate(shift=0.05, scale=0.1, rotate=15, p=0.3)
    GaussianBlur(blur_limit=(3,5), p=0.1)
    RandomBrightnessContrast(brightness=0.1, contrast=0.1, p=0.2)

Applied jointly to image+mask via Albumentations' dual transform.
Architecture unchanged from P.3 baseline.
Loss: Focal+Dice (confounding variable with augmentation).
Impact: +0.48pp Pixel F1 (marginal), +1.69pp Image Accuracy
Training instability: val loss spikes at ep 19 and 21.
```

### Test-Time Augmentation (vR.P.14)

```
TTA Pipeline (inference only):
    Original → predict → prob map
    HFlip   → predict → inverse flip → prob map
    VFlip   → predict → inverse flip → prob map
    HVFlip  → predict → inverse flip → prob map

    Final = mean(4 prob maps)
    Binary mask = final > 0.5

Architecture/training: identical to P.3
Impact: -5.32pp Pixel F1 (HARMFUL at threshold=0.5)
Root cause: Averaging pushes borderline probabilities below 0.5
AUC improved +0.9pp → suggests threshold recalibration could help
```

### Combined Best-of (vR.P.13, pending)

```
Combines:
    CBAM attention (from P.10) → best architecture
    Albumentations (from P.12) → augmentation pipeline
    50 epochs (from P.7)       → extended training budget
    Focal+Dice loss            → from P.10/P.12

Expected: 0.74-0.76 Pixel F1
Risk: Too many changes = hard to attribute improvement
```

### Multi-Quality ELA Input (vR.P.15) — NEW SERIES BEST

```
Input change:
    Before: ELA at Q=90 → RGB image (3 channels, correlated ~0.9)
    After:  ELA at Q=75, Q=85, Q=95 → 3 grayscale images (3 channels, independent)

Each quality level captures:
    Q=75: Strong artifacts, large errors, coarse signal (mean=0.0684)
    Q=85: Medium artifacts, balanced fidelity (mean=0.0605)
    Q=95: Subtle artifacts, fine-grained differences (mean=0.0402)

Architecture: unchanged from P.3 (standard UNet + ResNet-34, frozen+BN)
conv1: Pretrained RGB weights reused (grayscale channels map to R/G/B slots)

Result: Pixel F1 = 0.7329 (+4.09pp from P.3, +0.52pp from P.10)
Verdict: POSITIVE — NEW SERIES BEST
Key insight: Quality-level diversity > color information
```

---

## 13. Architecture Summary Table (Full Series)

| Version | Encoder | Decoder Mods | Input | Freeze | Loss | Pixel F1 |
|---------|---------|-------------|-------|--------|------|----------|
| P.0 | ResNet-34 | Standard | RGB | All frozen | BCE+Dice | 0.3749 |
| P.1 | ResNet-34 | Standard | RGB | All frozen | BCE+Dice | 0.4546 |
| P.1.5 | ResNet-34 | Standard | RGB | All frozen | BCE+Dice | 0.4227 |
| P.2 | ResNet-34 | Standard | RGB | L3+L4 unfrozen | BCE+Dice | 0.5117 |
| **P.3** | ResNet-34 | Standard | **ELA** | Frozen+BN | BCE+Dice | **0.6920** |
| P.4 | ResNet-34 | Standard | RGB+ELA (4ch) | Frozen+BN+conv1 | BCE+Dice | 0.7053 |
| P.5 | ResNet-50 | Standard | RGB | All frozen | BCE+Dice | 0.5137 |
| P.6 | EffNet-B0 | Standard | RGB | All frozen | BCE+Dice | 0.5217 |
| P.7 | ResNet-34 | Standard | ELA | Frozen+BN | BCE+Dice | 0.7154 |
| P.8 | ResNet-34 | Standard | ELA | Progressive | BCE+Dice | 0.6985 |
| P.9 | ResNet-34 | Standard | ELA | Frozen+BN | Focal+Dice | 0.6923 |
| **P.10** | ResNet-34 | **+CBAM** | **ELA** | Frozen+BN | Focal+Dice | **0.7277** |
| P.12 | ResNet-34 | Standard | ELA | Frozen+BN | Focal+Dice | 0.6968 |
| P.14 | ResNet-34 | Standard+TTA | ELA | Frozen+BN | BCE+Dice | 0.6388* |
| **P.15** | ResNet-34 | Standard | **Multi-Q ELA** | Frozen+BN | BCE+Dice | **0.7329** |

