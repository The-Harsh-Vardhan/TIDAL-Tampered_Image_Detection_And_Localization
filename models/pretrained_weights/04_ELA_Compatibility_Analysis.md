# ELA + Pretrained Models: Compatibility Analysis

---

## 1. The Core Question

ImageNet pretrained models learn features from natural RGB images (dogs, cars, landscapes). ELA maps are synthetic images showing compression artifacts. **Do pretrained features transfer to ELA inputs?**

---

## 2. What ImageNet Features Actually Learn (Layer by Layer)

| Layer Depth | What It Learns | Useful for ELA? |
|-------------|---------------|-----------------|
| **Conv1-Conv2** (very early) | Edges, corners, color gradients, Gabor-like filters | **Yes** — ELA maps have sharp edges at tampered boundaries |
| **Block 1-2** (early) | Textures, repeated patterns, simple shapes | **Partially** — ELA has block-based texture from JPEG grid |
| **Block 3** (middle) | Parts of objects, complex textures | **Uncertain** — ELA doesn't have "objects" |
| **Block 4** (deep) | Whole objects, semantic categories | **No** — ELA has no semantic content |
| **Final layers** (very deep) | ImageNet class discrimination | **No** — completely task-specific |

### Key Insight

The first 30-40% of a pretrained network learns **universal visual features** (edges, textures, gradients) that are useful across domains. The remaining 60-70% learns **ImageNet-specific features** that don't transfer to ELA.

**Implication:** Freezing the full encoder and only training a classifier head means you're using both useful (early) and useless (deep) features. Gradual unfreezing of deep layers lets them adapt to the ELA domain while keeping useful early features intact.

---

## 3. Empirical Evidence

### 3.1 From This Project

| Experiment | Input | Encoder | Pretrained? | Result |
|------------|-------|---------|-------------|--------|
| v6.5 | RGB (not ELA) | ResNet-34 | ImageNet | Tam-F1 = **0.41** (project best) |
| vK.11-12 | RGB + ELA (4ch) | ResNet-34 | ImageNet | Tam-F1 = ~0.13 (catastrophic) |
| vR.1.1 | ELA only | 2-layer CNN | From scratch | Test Acc = 88.38% |

**Observations:**
- RGB + pretrained (v6.5) dramatically outperformed ELA + from scratch (vR.1.1) for localization
- Adding ELA as a 4th channel (vK.11-12) **destroyed** pretrained features when combined with other changes
- ELA + from scratch works for classification but cannot localize

### 3.2 From Literature

From the Research Paper Analysis Report:

- **ETASR paper (P7):** Claims ELA + CNN outperforms VGG16/ResNet101 on CASIA v2 (96.21% vs 90.32%). But this was with from-scratch training, not transfer learning.
- **P9 Survey:** Notes that "EfficientNetV2B0 with transfer learning achieves superior generalization." The input was RGB, not ELA.
- **P17 ME-Net:** Uses RGB + noise residual (SRM), not ELA. Pretrained ConvNeXt + ResNet-50 achieve F1=0.905.

**No paper in the surveyed set uses ELA + pretrained encoder.** The ELA + pretrained combination is largely untested in the literature.

---

## 4. Three Input Strategies Analyzed

### Strategy A: RGB Only → Pretrained Encoder

```
Raw Image → RGB 384×384 → ResNet-34 (ImageNet) → Classifier/Decoder
```

| Aspect | Assessment |
|--------|------------|
| Feature transfer quality | **Excellent** — input matches pretraining distribution |
| What the model detects | Lighting inconsistencies, noise patterns, edge artifacts, blending artifacts, color discrepancies |
| Weakness | Cannot detect JPEG compression-level artifacts (ELA's strength) |
| Project evidence | v6.5 achieved Tam-F1 = 0.41 |
| Risk | Very low |

**Recommendation: Start here.** This is the safest, most proven approach.

### Strategy B: ELA Only → Pretrained Encoder

```
Raw Image → ELA (Q=90) → Resize 384×384 → ResNet-34 (ImageNet) → Classifier/Decoder
```

| Aspect | Assessment |
|--------|------------|
| Feature transfer quality | **Partial** — early layers (edges, textures) transfer; deep layers (objects, semantics) don't |
| What the model detects | JPEG compression inconsistencies, block boundary artifacts, re-save artifacts |
| Weakness | Domain mismatch — ELA statistics differ from ImageNet |
| Project evidence | Untested combination |
| Risk | Medium |

**Key concern:** ELA brightness scaling (`255/max_diff`) creates images with very different intensity distributions than ImageNet. The pretrained BatchNorm statistics (mean, variance per channel) will be wrong initially. With a frozen encoder, these wrong statistics cannot adapt.

**Mitigation:** When using frozen encoder + ELA input, add a small adapter layer (1×1 conv) before the encoder to learn a domain-appropriate transform, OR unfreeze the first BatchNorm layers.

### Strategy C: RGB + ELA (4 Channels) → Modified Pretrained Encoder

```
Raw Image → [RGB, ELA] → 4-channel input → Modified ResNet-34 → Classifier/Decoder
```

| Aspect | Assessment |
|--------|------------|
| Feature transfer quality | **Good for RGB channels, uncertain for ELA channel** |
| What the model detects | Both visual artifacts (RGB) and compression artifacts (ELA) |
| Weakness | First conv layer must be modified (breaks pretrained weights for that layer) |
| Project evidence | vK.11-12 tried this and failed (but with 5 other changes simultaneously) |
| Risk | Medium-High |

**Implementation approaches for the 4th channel:**

```python
# Option 1: Average RGB weights to initialize 4th channel
weights = encoder.conv1.weight.data  # Shape: [64, 3, 7, 7]
new_weights = torch.cat([weights, weights.mean(dim=1, keepdim=True)], dim=1)  # [64, 4, 7, 7]
encoder.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
encoder.conv1.weight.data = new_weights

# Option 2: Zero-initialize 4th channel (ELA channel starts ignored)
new_weights = torch.cat([weights, torch.zeros(64, 1, 7, 7)], dim=1)

# Option 3: SMP handles this automatically
model = smp.Unet(encoder_name='resnet34', encoder_weights='imagenet', in_channels=4)
# SMP averages the 3 RGB channel weights to create the 4th channel weight
```

---

## 5. BatchNorm and Domain Shift

### The Hidden Problem

Pretrained BatchNorm layers store running mean and variance statistics from ImageNet. When you freeze the encoder and feed ELA images:

```
ImageNet statistics:  mean ≈ [0.485, 0.456, 0.406], std ≈ [0.229, 0.224, 0.225]
ELA statistics:       mean ≈ [0.05-0.15, 0.05-0.15, 0.05-0.15], std ≈ [0.08-0.12, ...]
```

The normalization is wrong by a factor of 3-4x. The model receives inputs that are already "standardized" with the wrong statistics, producing features that are numerically distorted.

### Solutions

| Solution | Complexity | Effectiveness |
|----------|-----------|---------------|
| **Input normalization** — Apply ImageNet-style normalization to ELA before feeding | Simple | Partial fix |
| **Freeze all except BN** — Let BN adapt while keeping conv weights frozen | Medium | Good |
| **Full fine-tune with low LR** — Unfreeze everything with lr=1e-5 | Medium | Best |
| **Use RGB instead** — Avoid the problem entirely | Simplest | Full fix |

### Recommended Input Preprocessing

```python
# For RGB input (Strategy A) — standard ImageNet normalization
transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# For ELA input (Strategy B) — compute dataset-specific statistics first
# Run a pass over the training set to compute ELA mean/std, then normalize
ela_mean = compute_channel_mean(train_ela_images)
ela_std = compute_channel_std(train_ela_images)
transform = transforms.Normalize(mean=ela_mean, std=ela_std)
```

---

## 6. ELA-Specific Augmentation Considerations

The ETASR ablation study showed that **standard augmentation (flip, rotation) failed on ELA images** (vR.1.2: -2.85pp regression). However, this failure was with a 2-layer CNN, not a pretrained encoder.

### Why Augmentation Might Work With Pretrained Models

| Factor | ETASR CNN | Pretrained ResNet |
|--------|-----------|-------------------|
| Parameters | 29.5M (all trainable) | ~500K (frozen backbone) |
| Spatial invariance | None (Flatten layer) | Built-in (ResidualBlocks, GAP) |
| LR sensitivity | High (all params at same LR) | Low (only decoder trains) |
| Feature robustness | Pixel-exact memorization | Hierarchical, rotation-tolerant |

Pretrained encoders already encode rotation/flip invariance from ImageNet training. Augmenting ELA images may not confuse the frozen backbone the way it confused the ETASR CNN.

**Recommendation:** Re-test augmentation in the pretrained track (vR.P.x) — it may work where it failed in the ETASR track.

---

## 7. Practical Recommendations

### For Classification (Comparison with ETASR)

```
Priority:
1. RGB input + ResNet-34 frozen + GAP + Dense head     → vR.P.0
2. RGB input + ResNet-34 unfreeze last 2 blocks         → vR.P.1
3. ELA input + ResNet-34 frozen (BN unfrozen)           → vR.P.2
4. RGB+ELA (4ch) + ResNet-34 frozen                     → vR.P.3
```

### For Localization (Assignment Requirement)

```
Priority:
1. RGB input + ResNet-34 UNet, frozen encoder           → vR.P.0-seg
2. RGB input + ResNet-34 UNet, gradual unfreeze         → vR.P.1-seg
3. RGB+ELA + ResNet-34 UNet                             → vR.P.3-seg
```

### What NOT to Do

Based on project history lessons:

1. **Don't change 5 things at once** — vK.11-12 combined ELA + edge loss + classification head + focal loss + resolution change and got catastrophic failure
2. **Don't unfreeze everything at once** — Use gradual unfreezing with differential LR
3. **Don't use ELA as the only input on Day 1** — Start with RGB where transfer is guaranteed
4. **Don't skip the architecture freeze** — 8,829 images cannot constrain 22M unfrozen params
5. **Don't assume augmentation will fail** — It failed on ETASR CNN but may work with pretrained

---

## 8. Summary Decision Table

| Question | Answer | Confidence |
|----------|--------|------------|
| Use pretrained models? | **Yes** | Very high |
| First encoder? | **ResNet-34** | Very high |
| First input? | **RGB** | High |
| Freeze encoder? | **Yes, then gradual unfreeze** | High |
| Test ELA input? | **Yes, as ablation after RGB baseline** | Medium |
| Test 4-channel? | **Yes, but after both RGB and ELA baselines** | Medium |
| Will augmentation work? | **Probably (with pretrained)** | Medium |
| Framework? | **PyTorch + SMP** | High |
