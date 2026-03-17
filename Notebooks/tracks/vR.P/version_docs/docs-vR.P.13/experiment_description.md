# vR.P.13 — Experiment Description

## Combined Best-of Run (CBAM + Augmentation + Extended Training)

### Hypothesis
The individually positive changes from P.7 (extended training), P.10 (CBAM attention), and P.12 (data augmentation) will **stack additively** when combined, yielding the best localization performance in the pretrained track.

### Motivation
Each of these changes was tested in isolation against the P.3 baseline:
- **P.7 (50 epochs):** Pixel F1 0.7154 (+2.34pp from P.3) — POSITIVE
- **P.10 (CBAM attention):** Pixel F1 0.7277 (+3.57pp from P.3) — POSITIVE
- **P.12 (augmentation):** Untested but expected POSITIVE

These address **orthogonal** aspects:
- CBAM improves feature selection (architecture)
- Augmentation improves data diversity (regularization)
- Extended training allows convergence (optimization)

### Single Variable Changed from vR.P.3
This is a **multi-variable combination run**, not a single-variable ablation. It combines:
1. CBAM attention modules in all 5 decoder blocks (from P.10)
2. Albumentations augmentation pipeline (from P.12)
3. 50 epochs / patience 10 (from P.7)
4. Focal+Dice loss (from P.9/P.10)

### Architecture
UNet + ResNet-34 (ImageNet, frozen body + BN unfrozen) + CBAM in decoder

### Key Configuration

| Parameter | P.3 (parent) | P.13 (this) |
|-----------|-------------|-------------|
| CBAM attention | None | All 5 decoder blocks |
| Augmentation | None | Albumentations (6 transforms) |
| Epochs | 25 | 50 |
| Patience | 7 | 10 |
| Loss | BCE + Dice | Focal(0.25, 2.0) + Dice |
| Image size | 384 | 384 |
| Batch size | 16 | 16 |
