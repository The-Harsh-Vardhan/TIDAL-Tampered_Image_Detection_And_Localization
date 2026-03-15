# vR.P.14 — Experiment Description

## Test-Time Augmentation (TTA)

### Hypothesis
Averaging predictions from multiple augmented views of the same test image reduces prediction noise and improves boundary precision, yielding **+1–3pp Pixel F1** over the base model with **zero retraining cost**.

### Motivation
Test-Time Augmentation is a standard technique in segmentation competitions. The idea: instead of predicting once on the original image, predict on multiple augmented versions (flipped, rotated) and average the probability maps. This:
- Reduces directional bias in predictions
- Smooths noisy boundary predictions
- Improves precision at tampered region edges

TTA is "free" — no retraining, no architecture change, just modified inference.

### Single Variable Changed from vR.P.3
**Evaluation pipeline only** — adds TTA to the test-time prediction. All training is identical to P.3 (or whichever base model is loaded).

### TTA Strategy
For each test image, predict on 4 views:
1. Original
2. Horizontal flip → predict → flip back
3. Vertical flip → predict → flip back
4. Horizontal + Vertical flip → predict → flip back

Average the 4 probability maps, then threshold at 0.5.

### Key Configuration

| Parameter | P.3 (parent) | P.14 (this) |
|-----------|-------------|-------------|
| TTA views | 1 (original only) | 4 (orig + hflip + vflip + hvflip) |
| Training | Unchanged | Unchanged |
| Architecture | Unchanged | Unchanged |
| Inference time | 1× | ~4× per image |
| Everything else | Same | Same |
