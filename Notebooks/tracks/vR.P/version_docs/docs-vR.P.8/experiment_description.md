# Experiment Description — vR.P.8: ELA + Gradual Encoder Unfreeze

| Field | Value |
|-------|-------|
| **Version** | vR.P.8 |
| **Track** | Pretrained Localization (Track 2) |
| **Parent** | vR.P.3 (ELA as input, BN unfrozen) |
| **Change** | Progressive encoder unfreeze (frozen → layer4 → layer3+layer4) with ELA input |
| **Encoder** | ResNet-34 (ImageNet, progressive unfreeze schedule) |
| **Input** | ELA 384×384×3 (RGB ELA map, Q=90) |

---

## 1. Motivation

vR.P.3 demonstrated that ELA is a dramatically better forensic signal than RGB for pretrained UNet localization (Pixel F1 = 0.6920 vs vR.P.2's RGB baseline). However, vR.P.3's encoder was **completely frozen** — only BatchNorm layers adapted to ELA statistics. This leaves substantial untapped potential: the convolutional weights were trained on ImageNet's natural image distribution, which is fundamentally different from ELA's sparse, high-contrast compression artifact maps.

vR.P.2 showed that unfreezing encoder layers improves domain adaptation on RGB input by allowing the encoder to tune its features for the forensic domain. However, vR.P.2 unfroze layer3+layer4 **statically from epoch 1**, risking catastrophic forgetting before the decoder has learned useful representations.

vR.P.8 combines both insights: **ELA input (from P.3) with true progressive unfreezing**. The key innovation is a 3-stage schedule:

1. **Stage 0** (epochs 1–10): Encoder frozen. The decoder learns to decode ELA features using frozen ImageNet representations. This gives the decoder a stable foundation.
2. **Stage 1** (epochs 11–25): Unfreeze layer4 only. The deepest encoder block adapts its high-level, domain-specific features to ELA patterns. The 100× lower encoder LR (1e-5 vs 1e-3) prevents catastrophic weight updates.
3. **Stage 2** (epochs 26–40): Also unfreeze layer3. Mid-level features now adapt, giving the encoder a richer ELA-specific representation while preserving low-level edge/texture detectors (layer1/layer2).

This progressive approach is safer than P.2's static unfreeze because the decoder is already warm when encoder adaptation begins.

---

## 2. What Changed from vR.P.3

| Aspect | vR.P.3 | vR.P.8 (This Version) |
|--------|--------|----------------------|
| **Encoder state** | Frozen body + BN unfrozen | **Progressive unfreeze (3 stages)** |
| **Optimizer** | Single LR (1e-3) | **Dual param groups after Stage 0 (enc=1e-5, dec=1e-3)** |
| **Max epochs** | 25 | **40** |
| **Early stopping** | patience=7 | **patience=7, reset at stage transitions** |
| **NUM_WORKERS** | 2 | **4** |
| **DataLoader** | No prefetch_factor | **prefetch_factor=2** |
| **New code** | None | **5 helper functions, optimizer rebuild, stage logic** |
| **Checkpoint** | Standard | **+ current_stage for resume** |
| **History tracking** | Single `lr` | **`lr_encoder` + `lr_decoder`** |

---

## 3. What DID NOT Change (Frozen)

- Architecture: UNet + ResNet-34 (SMP)
- `IN_CHANNELS = 3` (ELA is 3-channel RGB)
- Input: ELA (Q=90, brightness-scaled)
- Normalization: ELA-specific mean/std (computed from training set)
- Loss: BCEDiceLoss (SoftBCEWithLogitsLoss + DiceLoss)
- LR Scheduler: ReduceLROnPlateau (factor=0.5, patience=3) — recreated at transitions
- Batch size: 16
- Seed: 42
- Data split: 70/15/15 (stratified)
- Dataset: sagnikkayalcse52/casia-spicing-detection-localization
- GT mask handling (same 3-tier fallback)
- AMP + TF32 enabled
- Evaluation: pixel-level + image-level metrics
- Classification threshold: MASK_AREA_THRESHOLD = 100 pixels

---

## 4. Progressive Unfreeze Schedule

| Stage | Epochs | What Unfreezes | Trainable Params | Optimizer Config |
|-------|--------|---------------|-----------------|-----------------|
| 0 | 1–10 | Encoder frozen, BN unfrozen | ~500K (decoder + BN) | Single group @ 1e-3 |
| 1 | 11–25 | + layer4 (3 BasicBlocks, 512-ch) | ~2M (+ layer4 conv/BN) | 2 groups: enc@1e-5, dec@1e-3 |
| 2 | 26–40 | + layer3 (6 BasicBlocks, 256-ch) | ~5M (+ layer3 conv/BN) | 2 groups: enc@1e-5, dec@1e-3 |

At each stage transition:
- Optimizer is rebuilt with fresh Adam (clean momentum for new layers)
- Scheduler is recreated (fresh patience counter)
- GradScaler is recreated (recalibrate to new gradient magnitudes)
- Early stopping patience is reset to 0 (don't penalize transition dips)
- `best_val_loss` is NOT reset (global best across all stages)

---

## 5. Experiment Lineage

```
vR.P.0 (baseline)
  └→ P.1 (dataset fix)
       └→ P.1.5 (speed optimizations)
            └→ P.2 (gradual unfreeze, RGB)
                 └→ P.3 (ELA input, frozen + BN)  ← BEST Pixel F1 = 0.6920
                      ├→ P.4 (RGB + ELA 4-channel)
                      ├→ P.7 (ELA + extended training, planned)
                      └→ P.8 (ELA + progressive unfreeze)  ← THIS
            ├→ P.5 (ResNet-50 encoder)
            └→ P.6 (EfficientNet-B0 encoder)
```

vR.P.8 branches from the P.3 lineage. It tests whether the ELA signal benefits from adapted encoder features, combining the best input representation (P.3) with controlled encoder adaptation (inspired by P.2).

---

## 6. Why Encoder LR Must Remain Small (1e-5)

The encoder's 21M parameters were trained on 1.2M ImageNet images. These weights encode robust visual features (edges, textures, shapes) that transferred remarkably well to ELA even without fine-tuning (P.3's success). A large encoder LR would:

1. **Destroy pretrained features** — catastrophic forgetting of useful low-level detectors
2. **Dominate gradient updates** — encoder has 40× more parameters than decoder
3. **Cause training instability** — large weight changes in early layers cascade through the network

The 100× ratio (1e-5 / 1e-3) ensures the encoder makes small, conservative adjustments while the decoder retains full learning capacity.
