# Implementation Plan: vR.P.2 — Gradual Encoder Unfreeze

---

## Notebook Structure

**File:** `vR.P.2 Image Detection and Localisation.ipynb`
**Cells:** 28 (18 code, 10 markdown) — same structure as vR.P.1
**Parent:** vR.P.1 (Dataset fix + GT mask auto-detection)

### Cell Modification Summary

| Cell | Type | Section | Change from vR.P.1 |
|------|------|---------|---------------------|
| 0 | Markdown | Title | **MODIFIED** — Version=vR.P.2, change=Gradual Unfreeze |
| 1 | Markdown | Change log | **MODIFIED** — New changelog entry, comparison table |
| 2 | Code | Setup | **MODIFIED** — `VERSION='vR.P.2'`, `CHANGE` updated, add `ENCODER_LR=1e-5` |
| 3 | Markdown | Dataset | Unchanged |
| 4 | Code | Dataset | Unchanged |
| 5 | Code | Dataset | Unchanged |
| 6 | Code | Dataset | Unchanged |
| 7 | Markdown | Data prep | **MODIFIED** — ELA reference: vR.P.3 (shifted from vR.P.2) |
| 8 | Code | Data prep | Unchanged |
| 9 | Code | Data prep | Unchanged |
| 10 | Code | Data prep | Unchanged |
| 11 | Markdown | Architecture | **MODIFIED** — Partially unfrozen diagram, FROZEN/UNFROZEN labels |
| 12 | Code | Architecture | **KEY CHANGE** — Freeze-then-selectively-unfreeze pattern |
| 13 | Markdown | Training | **MODIFIED** — Differential LR config table |
| 14 | Code | Training | **KEY CHANGE** — 2-group optimizer with differential LR |
| 15 | Code | Training | **MODIFIED** — Print both LRs, track decoder LR in history |
| 16-25 | Mixed | Evaluation+Viz | Unchanged (all 10 cells) |
| 26 | Markdown | Discussion | **MODIFIED** — Gradual unfreeze rationale, next=vR.P.3 |
| 27 | Code | Save | Unchanged (filename auto-derived from VERSION) |

**Total: 10 cells modified, 18 cells unchanged.**

---

## Key Implementation Details

### 1. Encoder Freeze-then-Unfreeze (Cell 12)

```python
model = smp.Unet(encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS,
                  in_channels=IN_CHANNELS, classes=NUM_CLASSES, activation=None)

# Step 1: Freeze ALL encoder parameters
for param in model.encoder.parameters():
    param.requires_grad = False

# Step 2: Selectively unfreeze last 2 blocks (layer3 + layer4)
for block in list(model.encoder.children())[-2:]:
    for param in block.parameters():
        param.requires_grad = True

model = model.to(DEVICE)
```

**Why `list(model.encoder.children())[-2:]`:**

SMP's ResNet-34 encoder exposes children in this order:
```
[0] conv1        — 64ch, 7×7, stride 2
[1] bn1          — BatchNorm for conv1
[2] relu         — ReLU activation
[3] maxpool      — 3×3, stride 2
[4] layer1       — 3 BasicBlock, 64ch
[5] layer2       — 4 BasicBlock, 128ch
[6] layer3       — 6 BasicBlock, 256ch  ← UNFROZEN
[7] layer4       — 3 BasicBlock, 512ch  ← UNFROZEN
```

`children()[-2:]` selects indices [6] and [7] = layer3 + layer4.

### 2. Differential Learning Rate Optimizer (Cell 14)

```python
encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
decoder_params = list(model.decoder.parameters()) + list(model.segmentation_head.parameters())

optimizer = optim.Adam([
    {'params': encoder_params, 'lr': ENCODER_LR, 'weight_decay': 1e-5},
    {'params': decoder_params, 'lr': LEARNING_RATE, 'weight_decay': 1e-5},
])
```

**Critical:** `model.segmentation_head` must be included in the decoder param group. It is a separate SMP module (Conv2d head that outputs the final 1-channel mask) and is NOT part of `model.decoder`.

### 3. Config Additions (Cell 2)

```python
ENCODER_LR = 1e-5   # NEW: 100x lower than decoder LR
```

All other config values remain identical to vR.P.1.

### 4. Training Loop Modifications (Cell 15)

- Print both learning rates at epoch start
- Track `optimizer.param_groups[1]['lr']` (decoder LR) in history for scheduler
- ReduceLROnPlateau adjusts both param groups proportionally

### 5. Parameter Count Verification

```
Frozen (conv1 + bn1 + layer1 + layer2):  ~2.8M
Unfrozen encoder (layer3 + layer4):       ~4.5M
Decoder + segmentation head:              ~0.5M
─────────────────────────────────────────
Total trainable:                          ~5.0M
Total frozen:                             ~2.8M
Total parameters:                         ~7.8M  (UNet decoder adds to ResNet-34's 21.3M)
```

---

## Validation Checklist

Before submitting to Kaggle, verify:

- [ ] `VERSION = 'vR.P.2'` in config cell
- [ ] `ENCODER_LR = 1e-5` exists in config
- [ ] Cell 12: `param.requires_grad = False` for all encoder, then `True` for last 2 children
- [ ] Cell 14: optimizer has exactly 2 param groups
- [ ] Cell 14: encoder group uses `ENCODER_LR` (1e-5)
- [ ] Cell 14: decoder group uses `LEARNING_RATE` (1e-3)
- [ ] Cell 14: `model.segmentation_head` in decoder group (NOT in encoder group)
- [ ] Cell 15: prints both LRs per epoch
- [ ] All 28 cells present and in correct order
- [ ] Dataset cells (3-6) unchanged from vR.P.1
- [ ] Evaluation cells (16-25) unchanged from vR.P.1
- [ ] Kaggle metadata: `sagnikkayalcse52/casia-spicing-detection-localization`, GPU enabled, internet enabled
- [ ] Seed = 42 set for all random sources

---

## Dependencies

Same as vR.P.1:

```
torch >= 1.10          (pre-installed on Kaggle)
torchvision            (pre-installed on Kaggle)
segmentation-models-pytorch >= 0.3.0  (pip install in notebook)
numpy, matplotlib, seaborn, Pillow, scikit-learn, tqdm  (pre-installed)
```

---

## Runtime Estimate

| Stage | Time (T4 GPU) |
|-------|---------------|
| SMP pip install | ~30 sec |
| Dataset discovery + path collection | ~5 sec |
| Model build + freeze/unfreeze | ~2 sec |
| Training (25 epochs × ~60 sec/epoch) | ~20-25 min |
| Test evaluation | ~2 min |
| Visualizations | ~1 min |
| **Total** | **~25-30 min** |

Training is ~30% slower than vR.P.1 due to backward pass through layer3+layer4 (5M trainable vs 500K).
