# 03 — Model Architecture Evolution

## Purpose

Assess the U-Net/ResNet34 architecture choice against audit critiques and Run01 performance, and define architecture-level changes for v8.

---

## Phase 1: Docs7 Design

| Property | Value |
|---|---|
| Framework | Segmentation Models PyTorch (SMP) |
| Architecture | U-Net |
| Encoder | ResNet34 |
| Pretrained | ImageNet |
| Input channels | 3 (RGB) |
| Output | 1 channel, raw logits |
| Parameters | ~24.4M |
| Image size | 384×384 |

**Docs7 rationale:**
- U-Net is proven for dense prediction with strong skip connections
- ResNet34 balances parameter count with feature quality
- ImageNet pretraining compensates for CASIA's small size (~12K images)
- Fits within T4 GPU memory constraints
- SMP provides reliable, tested implementations

## Phase 2: Audit Critique

### Findings Against Architecture

| Finding | Severity | Summary |
|---|---|---|
| U-Net/ResNet34 is a convenience baseline, not forensic architecture | MEDIUM | Audit6 Pro §02 Finding 1 |
| No DeepLabV3+ comparison | MEDIUM | Audit6 Pro §02 Finding 2 |
| No transformer consideration | LOW | Audit6 Pro §02 Finding 3 |
| RGB-only input is a major forensic limitation | HIGH | Audit6 Pro §02 Finding 4 |
| Image-level heuristic instead of learned classification head | HIGH | Audit6 Pro §02 Finding 8 |

### The Core Objection

The audit's strongest critique was not that U-Net is a bad choice, but that it is an **unjustified** choice. The rationale in Docs7 was: "it's standard, pretrained, and fits on T4." The audit argued this is implementation convenience, not task alignment.

### RGB-Only Limitation

The most substantive architecture critique: image forensics often relies on signals invisible in standard RGB — compression artifacts, noise residuals, demosaicing patterns, frequency-domain anomalies. A model limited to RGB may learn semantic shortcuts (object boundaries, texture discontinuities) rather than forensic evidence.

Run01's robustness plateau (4 degradation conditions producing identical F1≈0.593) is consistent with this hypothesis.

## Phase 3: Run01 Evidence

### What the Architecture Achieved

| Metric | Value | Assessment |
|---|---|---|
| Splicing F1 | 0.5901 | Moderate — architecture captures large-region splicing |
| Copy-move F1 | 0.3105 | Near-failure — architecture struggles with subtle manipulation |
| Overall tampered-only F1 | 0.4101 | Below useful threshold |
| Image-level AUC | 0.8703 | Decent — heuristic works at image level |
| Robustness drop | ~13% | Significant — model is input-distribution sensitive |

### Architecture-Specific Observations

1. **U-Net skip connections work for large splicing regions** — splicing F1=0.59 is reasonable for a first-attempt baseline.
2. **Encoder fails on copy-move** — copy-move forgeries duplicate existing content, so RGB features may not distinguish pasted from original. This is an expected limitation of RGB-only input.
3. **Threshold=0.1327 is not an architecture problem** — it stems from the loss function (no pos_weight), not from the model's representational capacity.
4. **High variance (std≈0.41)** — some images are predicted well, others fail completely. This suggests the model has capacity but lacks consistent feature extraction.

---

## v8 Architecture Changes

### Retain: Core Architecture

**Decision: Keep U-Net/ResNet34 as the primary architecture for v8.**

Rationale:
- The architecture itself is not the primary performance bottleneck — loss design and augmentation are
- Changing architecture and loss simultaneously makes ablation impossible
- U-Net/ResNet34 has proven it can achieve F1=0.59 on splicing — the ceiling has not been reached
- SMP makes architecture swaps trivial once the training pipeline is stable

### Add: Explicit Architecture Justification

v8 documentation must include:

> "We retain U-Net/ResNet34 as a proven dense-prediction baseline. It is not chosen because it is optimal for forensic localization — it is chosen because it provides a stable reference point while we address higher-priority issues (loss calibration, augmentation, class imbalance). Architecture exploration (DeepLabV3+, EfficientNet encoders, hybrid transformers) is deferred to v9 once the training pipeline produces calibrated, well-regularized predictions."

### Consider: Minimal Architecture Improvements

These are **P2 (moderate priority)** — implement only after P0 loss/augmentation fixes:

**1. Frozen encoder warmup**
```python
# Freeze encoder for first N epochs to stabilize BatchNorm
for param in model.encoder.parameters():
    param.requires_grad = False
# Unfreeze after warmup_epochs
```
This addresses the BatchNorm instability concern (batch-size-4 with pretrained statistics).

**2. Encoder swap experiment**
```python
# SMP makes this trivial:
model = smp.Unet(encoder_name='efficientnet-b0', ...)  # vs resnet34
```
If v8 training pipeline is stable, run a single encoder comparison (ResNet34 vs EfficientNet-B0 vs ResNet50) as a documented experiment.

### Defer to v9+

- **DeepLabV3+:** Stronger global context via atrous convolutions. Worthwhile comparison but not needed for v8.
- **Forensic input streams:** SRM residuals, ELA, noise residual maps as additional input channels. Addresses RGB-only limitation but requires significant preprocessing pipeline changes.
- **Transformer encoders:** SegFormer or hybrid models. Addresses long-range context gap. Requires architecture pipeline restructuring.
- **Learned image-level head:** Replace `max(prob_map)` heuristic with a binary classification branch. Important for production use but not v8 priority.

---

## Interview Defense

**"Why U-Net and not DeepLabV3+?"**
> "U-Net is our implementation baseline. DeepLabV3+ is a valid alternative we plan to compare once the training pipeline produces calibrated results. SMP makes the swap trivial — the higher-priority work is fixing loss design and augmentation."

**"Why not transformers?"**
> "We made a deliberate scope decision: fix the training fundamentals first (loss, augmentation, scheduling), then explore architecture. A SegFormer comparison is planned for v9. The current U-Net/ResNet34 achieves F1=0.59 on splicing, suggesting the architecture has headroom we haven't exhausted."

**"Why RGB-only?"**
> "This is a known limitation. Run01's robustness results show ~13% F1 dependency on input distribution, which is consistent with partial shortcut learning from RGB artifacts. Forensic streams (SRM, ELA) are our top v9 architecture priority."
