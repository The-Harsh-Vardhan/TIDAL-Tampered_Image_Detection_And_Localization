# Experiment Description — vR.P.11: Higher Resolution (512x512)

| Field | Value |
|-------|-------|
| **Version** | vR.P.11 |
| **Track** | Pretrained Localization (Track 2) |
| **Parent** | vR.P.3 (ELA as input, frozen body + BN unfrozen) |
| **Change** | Higher resolution (384->512) + Focal+Dice loss + gradient accumulation |
| **Encoder** | ResNet-34 (ImageNet, frozen body, BatchNorm unfrozen) |
| **Input** | ELA 512x512x3 (RGB ELA map, Q=90) |

---

## 1. Motivation

### Resolution as an Untapped Variable

All pretrained experiments (vR.P.0-P.9) use **384x384 resolution**. This was inherited from early experiments as a balance between detail and compute cost. However, tampering artifacts are inherently spatial phenomena:

- **Copy-paste boundaries** create narrow transition zones (1-5 pixels wide at full resolution)
- **Compression seams** between tampered and authentic regions produce subtle ELA differences
- **Small tampered regions** (< 5% of image) may occupy only a handful of pixels at 384x384

At 384x384, fine boundary details are downsampled away. A tampered edge that spans 3 pixels in the original image may collapse to 1-2 pixels, making it indistinguishable from noise. Higher resolution preserves these critical boundary pixels.

### Why 512x512

| Resolution | Pixels | Relative | VRAM (batch=16, FP16) | Notes |
|-----------|--------|----------|-----------------------|-------|
| 384x384 | 147,456 | 1.0x | ~6 GB | Current (P.3) |
| **512x512** | **262,144** | **1.78x** | **~10 GB** | **This experiment** |
| 768x768 | 589,824 | 4.0x | ~22 GB | Exceeds T4 16GB |

512x512 is the largest resolution that fits in T4 16GB VRAM with batch_size=8 and AMP. It provides 1.78x more pixels than 384x384, giving the model access to finer spatial details without exceeding hardware limits.

### Memory Management Strategy

To maintain the same effective batch size (16) at higher resolution:
- **BATCH_SIZE = 8** (halved from P.3's 16 to fit VRAM)
- **GRAD_ACCUM_STEPS = 2** (accumulate gradients over 2 mini-batches)
- **Effective batch = 8 x 2 = 16** (same gradient statistics as P.3)
- **cudnn.benchmark = True** (auto-tune convolution algorithms for 512x512)

This ensures the optimization dynamics remain comparable to P.3 — the model sees the same number of images per weight update.

---

## 2. What Changed from vR.P.3

| Aspect | vR.P.3 | vR.P.11 (This Version) |
|--------|--------|------------------------|
| **Resolution** | 384x384 | **512x512** |
| **BATCH_SIZE** | 16 | **8** (VRAM constraint) |
| **Gradient accumulation** | None | **2 steps** (effective batch=16) |
| **Loss function** | SoftBCEWithLogitsLoss + DiceLoss | **FocalLoss(alpha=0.25, gamma=2.0) + DiceLoss** |
| **EPOCHS** | 25 | **50** (more pixels = slower convergence) |
| **PATIENCE** | 7 | **10** (longer convergence allowance) |
| **NUM_WORKERS** | 2 | **4** |
| **DataLoader** | No prefetch_factor | **prefetch_factor=2** |
| **cudnn.benchmark** | default | **True** |

---

## 3. What DID NOT Change (Frozen)

- Architecture: UNet + ResNet-34 (SMP)
- Input type: ELA (Q=90, brightness-scaled)
- Normalization: ELA-specific mean/std (computed from training set)
- Encoder state: Frozen body + BN unfrozen
- Optimizer: Adam, single LR=1e-3, weight_decay=1e-5
- LR Scheduler: ReduceLROnPlateau (factor=0.5, patience=3)
- Early stopping: monitor=val_loss
- Seed: 42
- Data split: 70/15/15 (stratified)
- Dataset: sagnikkayalcse52/casia-spicing-detection-localization
- GT mask handling (same 3-tier fallback)
- AMP + TF32 enabled
- Evaluation: pixel-level + image-level metrics

---

## 4. Experiment Lineage

```
vR.P.0 (baseline)
  +-- P.1 (dataset fix)
       +-- P.1.5 (speed optimizations)
            +-- P.2 (gradual unfreeze, RGB)
                 +-- P.3 (ELA input, frozen + BN)  <- BEST Pixel F1 = 0.6920
                      +-- P.4 (RGB + ELA 4-channel)
                      +-- P.7 (ELA + extended training)
                      +-- P.8 (ELA + progressive unfreeze)
                      +-- P.9 (Focal + Dice loss)
                      +-- P.11 (Higher resolution 512x512)  <- THIS
            +-- P.5 (ResNet-50 encoder)
            +-- P.6 (EfficientNet-B0 encoder)
```

vR.P.11 tests whether **higher spatial resolution can improve segmentation quality** by preserving fine boundary details that are lost at 384x384. The Focal+Dice loss (from P.9 design) provides better hard-pixel focus, and gradient accumulation maintains the effective batch size despite VRAM constraints.

---

## 5. Why Focal+Dice Loss (Not BCE+Dice)

This experiment adopts the Focal+Dice loss from vR.P.9's design. At higher resolution, the class imbalance problem is amplified — more total pixels means more easy background pixels dominating the gradient. Focal loss with gamma=2.0 automatically down-weights these easy pixels, focusing training on the informative boundary region.

- **alpha=0.25**: Standard weighting for binary segmentation
- **gamma=2.0**: Easy pixels (p > 0.9) contribute only ~1% of the loss

---

## 6. Why Extended Training (50 Epochs)

Higher resolution means:
- **More spatial detail to learn** — the model must learn finer feature patterns
- **Smaller effective batch updates** — even with accumulation, each mini-batch sees fewer images
- **Focal loss converges slower** — by design, it down-weights easy examples, reducing overall gradient magnitude

50 epochs with patience=10 gives the model sufficient time to converge while still stopping early if it plateaus.
