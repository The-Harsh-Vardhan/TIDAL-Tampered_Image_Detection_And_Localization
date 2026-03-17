# Experiment Description — vR.P.10: ELA + Attention Modules (CBAM)

| Field | Value |
|-------|-------|
| **Version** | vR.P.10 |
| **Track** | Pretrained Localization (Track 2) |
| **Parent** | vR.P.3 (ELA as input, frozen body + BN unfrozen) |
| **Change** | Focal+Dice loss (from P.9) + CBAM attention in UNet decoder |
| **Encoder** | ResNet-34 (ImageNet, frozen body, BatchNorm unfrozen) |
| **Input** | ELA 384×384×3 (RGB ELA map, Q=90) |

---

## 1. Motivation

### Why the UNet Decoder Needs Attention

Previous experiments established that ELA input (P.3) and Focal+Dice loss (P.9) significantly improve tampering localization. However, the UNet decoder still uses vanilla convolutions — every spatial location and channel receives equal computational attention. This is suboptimal for tampering detection because:

1. **Forensic artifacts are subtle.** Tampered regions in ELA maps appear as faint brightness differences or compression inconsistencies. Standard convolutions process all features uniformly, diluting these subtle forensic signals among irrelevant activations.

2. **Not all decoder channels are equally informative.** In a 256-channel feature map, some channels encode boundary information (critical for localization), while others encode texture or context features (less important). Channel attention can amplify the forensic-relevant channels.

3. **Tampered regions are spatially sparse.** In CASIA v2.0, tampering covers 5–30% of the image. Spatial attention can help the network focus on the regions where manipulation artifacts are strongest, rather than wasting capacity on uniform background.

### How Attention Modules Address This

| Attention Type | Mechanism | What It Focuses On |
|---------------|-----------|-------------------|
| **Channel Attention (SE)** | Global average + max pooling → FC → sigmoid → channel scaling | Which feature channels contain forensic information |
| **Spatial Attention** | Channel-wise avg/max pooling → conv → sigmoid → spatial scaling | Where in the image manipulation artifacts appear |
| **CBAM** | Channel attention followed by spatial attention (sequential) | Both which features AND where to focus |

Attention modules are **lightweight** (< 0.05% of total parameters) — they guide the existing decoder features without adding significant complexity.

### Forensic Relevance

Image tampering artifacts often manifest as:
- **Compression inconsistencies** — different JPEG quality levels between original and pasted regions
- **Noise pattern mismatches** — spliced regions carry noise characteristics of the source image
- **Boundary artifacts** — imperfect blending at splice boundaries creates subtle edge patterns

ELA (Error Level Analysis) highlights these artifacts as brightness differences. However, the artifacts are often:
- Low-amplitude (only a few intensity levels different from authentic regions)
- Spatially localized (concentrated at boundaries)
- Channel-specific (different RGB channels may show different artifact patterns)

Attention modules can learn to amplify these specific patterns in the decoder's feature maps.

---

## 2. Attention Module Design

### CBAM (Convolutional Block Attention Module) — Primary

CBAM applies attention in two sequential stages:

**Stage 1 — Channel Attention:**
```
Input(B,C,H,W)
  → AdaptiveAvgPool2d(1) + AdaptiveMaxPool2d(1)   # Global context
  → SharedMLP(C → C//16 → C)                       # Channel interaction
  → Sigmoid                                         # Channel weights
  → Element-wise multiply with input                # Recalibrate channels
```

**Stage 2 — Spatial Attention:**
```
ChannelAttentionOutput(B,C,H,W)
  → MeanPool(dim=1) + MaxPool(dim=1)               # Channel compression
  → Conv2d(2 → 1, kernel=7×7, padding=3)           # Spatial context
  → Sigmoid                                         # Spatial weights
  → Element-wise multiply with input                # Focus on locations
```

### SE (Squeeze-and-Excitation) — Alternative

SE applies channel attention only:

```
Input(B,C,H,W)
  → AdaptiveAvgPool2d(1)                            # Global squeeze
  → FC(C → C//16) → ReLU → FC(C//16 → C)          # Excitation
  → Sigmoid                                         # Channel weights
  → Element-wise multiply with input                # Recalibrate
```

### CBAM vs SE Comparison

| Aspect | SE | CBAM |
|--------|-----|------|
| Attention type | Channel only | Channel + spatial (sequential) |
| Pooling | Average only | Average + max (both stages) |
| Parameters per block (256ch) | ~8.2K | ~8.3K |
| Spatial awareness | None | Yes (7×7 conv) |
| Overhead | Minimal | Minimal (+1 conv layer) |

**Default: CBAM** — The spatial component specifically targets WHERE tampering artifacts appear, which is directly relevant for localization tasks.

---

## 3. What Changed from vR.P.3

| Aspect | vR.P.3 | vR.P.10 (This Version) |
|--------|--------|------------------------|
| **Loss function** | SoftBCEWithLogitsLoss + DiceLoss | **FocalLoss(alpha=0.25, gamma=2.0) + DiceLoss** |
| **Hard pixel focus** | No (all pixels weighted equally) | **Yes (focal modulation, gamma=2.0)** |
| **Decoder attention** | None (nn.Identity in all blocks) | **CBAM (channel + spatial) in all 5 decoder blocks** |
| **Attention parameters** | 0 | **~11.2K (0.05% of total model)** |
| **NUM_WORKERS** | 2 | **4** |
| **DataLoader** | No prefetch_factor | **prefetch_factor=2** |

---

## 4. What DID NOT Change (Frozen)

- Architecture base: UNet + ResNet-34 (SMP)
- Input: ELA (Q=90, brightness-scaled)
- Normalization: ELA-specific mean/std (computed from training set)
- Encoder state: Frozen body + BN unfrozen
- Optimizer: Adam, single LR=1e-3, weight_decay=1e-5
- LR Scheduler: ReduceLROnPlateau (factor=0.5, patience=3)
- Early stopping: patience=7, monitor=val_loss
- Batch size: 16
- Max epochs: 25
- Seed: 42
- Data split: 70/15/15 (stratified)
- Dataset: sagnikkayalcse52/casia-spicing-detection-localization
- GT mask handling (same 3-tier fallback)
- AMP + TF32 enabled
- Evaluation: pixel-level + image-level metrics

---

## 5. Experiment Lineage

```
vR.P.0 (baseline)
  └→ P.1 (dataset fix)
       └→ P.1.5 (speed optimizations)
            └→ P.2 (gradual unfreeze, RGB)
                 └→ P.3 (ELA input, frozen + BN)
                      ├→ P.4 (RGB + ELA 4-channel)
                      ├→ P.7 (ELA + extended training)
                      ├→ P.8 (ELA + progressive unfreeze)
                      ├→ P.9 (Focal + Dice loss)
                      └→ P.10 (Focal+Dice + CBAM attention)  ← THIS
            ├→ P.5 (ResNet-50 encoder)
            └→ P.6 (EfficientNet-B0 encoder)
```

Flat lineage: P.0 → P.1 → P.1.5 → P.2 → P.3 → P.10

vR.P.10 tests whether **attention mechanisms can improve the decoder's ability to focus on subtle manipulation artifacts**. It combines the Focal+Dice loss from P.9 (to also focus optimization on hard boundary pixels) with CBAM attention (to focus feature representations on relevant channels and locations).

---

## 6. Risk Assessment

**Overall Risk: LOW-MODERATE**

| Risk | Severity | Likelihood | Mitigation |
|------|----------|------------|------------|
| Attention adds too few parameters to make a difference | LOW | MODERATE | CBAM is proven in segmentation literature; even small attention modules can significantly improve boundary quality |
| Attention causes training instability | LOW | LOW | CBAM uses sigmoid activations bounded in [0,1]; gradients are well-behaved |
| Two changes (loss + attention) conflate effects | MODERATE | CERTAIN | Intentional design choice — P.9 already tested loss alone; comparing P.10 vs P.9 isolates attention effect |
| SMP decoder block internal structure changes across versions | LOW | LOW | Injection targets `attention2` attribute which is a stable part of SMP's API |
| Attention overfits on small dataset | LOW | LOW | Only ~11K additional parameters; attention learns weighting, not features |

---

## 7. Hypothesis

**H0 (null):** Adding CBAM attention to the UNet decoder does not improve Pixel F1 by more than 2pp compared to vR.P.3.

**H1 (alternative):** CBAM attention improves Pixel F1 by ≥ 2pp by helping the decoder focus on forensically relevant channel and spatial features.

**Evidence supporting H1:**
- TransU2-Net (P21 reference) showed +14.2% F-measure from attention mechanisms
- CBAM has been shown to improve segmentation boundaries in medical imaging (similar sparse-target setting)
- Decoder features currently have no mechanism to prioritize forensic channels
- ELA maps have spatially concentrated artifacts — spatial attention is directly relevant

**Evidence supporting H0:**
- The dataset is small (8,829 training images) — attention may not have enough training signal
- P.3's encoder already extracts good features (Pixel F1 = 0.6920 without attention)
- CBAM adds < 0.05% parameters — may be too lightweight to make a meaningful difference
- Attention modules are most beneficial in deeper networks; ResNet-34 may be too shallow
