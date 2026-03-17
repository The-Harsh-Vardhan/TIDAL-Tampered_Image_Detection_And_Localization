# Experiment Description: vR.P.2 -- Gradual Encoder Unfreeze

---

## Version Info

| Field | Value |
|-------|-------|
| Version | vR.P.2 |
| Track | Pretrained Localization (Track 2) |
| Parent | **vR.P.1** (dataset fix + GT masks) |
| Change | Gradual unfreeze: unfreeze layer3 + layer4 of ResNet-34 encoder with differential LR |
| Category | Training Strategy / Transfer Learning |

---

## Change Description

Selectively unfreeze the last 2 encoder blocks of the ResNet-34 backbone and train them with a low learning rate:

```python
# Step 1: Freeze all encoder params (same as vR.P.0/P.1)
for param in model.encoder.parameters():
    param.requires_grad = False

# Step 2: Unfreeze last 2 blocks (NEW in vR.P.2)
for child in list(model.encoder.children())[-2:]:  # [layer3, layer4]
    for param in child.parameters():
        param.requires_grad = True

# Step 3: Differential LR optimizer (NEW in vR.P.2)
optimizer = Adam([
    {'params': [p for p in model.encoder.parameters() if p.requires_grad], 'lr': 1e-5},
    {'params': list(model.decoder.parameters()) + list(model.segmentation_head.parameters()), 'lr': 1e-3},
], weight_decay=1e-5)
```

---

## Motivation

### Why This Change

The frozen encoder in vR.P.0/P.1 relies entirely on ImageNet features. While ImageNet features are excellent for natural image understanding, they were not trained to detect tampering artifacts:

- **ImageNet objective:** Classify 1000 object categories from natural photos
- **Forensic objective:** Detect subtle pixel-level manipulation artifacts (splicing boundaries, cloning patterns, compression inconsistencies)

The higher-level encoder features (layer3, layer4) are the most task-specific. Allowing them to adapt with a conservative learning rate lets the model develop forensic-specific representations while preserving the general visual features in the lower layers.

### Why Layer3 + Layer4 (Not All Layers)

ResNet-34 has a hierarchical feature structure:
- **conv1, layer1, layer2:** General features (edges, textures, colors) -- transfer universally
- **layer3:** Mid-level features (object parts, spatial patterns) -- partially task-specific
- **layer4:** High-level features (semantic features, abstract patterns) -- highly task-specific

Unfreezing only the top 2 blocks:
1. Preserves the robust low-level feature extraction
2. Allows domain-specific adaptation where it matters most
3. Limits overfitting risk compared to full unfreezing

### Why Differential LR (100x Ratio)

| Approach | Risk | Justification |
|----------|------|---------------|
| Same LR for all | HIGH -- encoder forgets ImageNet features | Catastrophic forgetting |
| 10x ratio (1e-4 / 1e-3) | MEDIUM -- aggressive for small dataset | v6.5 used this |
| **100x ratio (1e-5 / 1e-3)** | **LOW -- conservative adaptation** | **Chosen: small dataset needs caution** |

---

## What Changes

| Component | vR.P.1 (Parent) | vR.P.2 (This Version) |
|-----------|-----------------|----------------------|
| Encoder layer3 | Frozen | **Unfrozen (lr=1e-5)** |
| Encoder layer4 | Frozen | **Unfrozen (lr=1e-5)** |
| Optimizer | Adam(lr=1e-3, decoder only) | **Adam with 2 param groups** |
| Trainable params | ~500K | **~5M** |
| Data:param ratio | 1:57 | **1:570** |

### What Does NOT Change (Frozen)

- Dataset: sagnikkayalcse52/casia-spicing-detection-localization (GT masks)
- Dataset discovery: fixed find_dataset() (from vR.P.1)
- Input: RGB 384x384, ImageNet normalization
- Architecture: UNet + ResNet-34
- Loss: BCEDiceLoss
- Scheduler: ReduceLROnPlateau (factor=0.5, patience=3)
- Early stopping: patience=7, monitor=val_loss
- Batch size: 16
- Max epochs: 25
- Seed: 42
- Data split: 70/15/15 (stratified)
- Evaluation: pixel-level + image-level metrics

---

## Cumulative Changes from vR.P.0

1. **vR.P.1:** Fixed dataset discovery (prefer IMAGE over MASK), GT mask auto-detection, new dataset (sagnikkayalcse52)
2. **vR.P.2:** Gradual unfreeze (layer3+layer4, differential LR 1e-5/1e-3) -- this version
