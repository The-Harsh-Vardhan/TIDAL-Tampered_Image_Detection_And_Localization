# vR.P.24 — Expected Outcomes

## Scenarios

### Positive (30% confidence)
- **Pixel F1: 0.72–0.75** (+2-6pp over P.3 baseline 0.6920)
- Noiseprint residuals capture manipulation traces that ELA misses (e.g., copy-move from same-camera images)
- DnCNN pretrained weights generalize well to CASIA2 images
- The segmentation encoder learns to interpret noiseprint discontinuities as forgery boundaries

### Neutral (40% confidence)
- **Pixel F1: 0.65–0.72**
- Noiseprint provides some signal but not clearly better than ELA for this dataset
- Without properly pretrained DnCNN weights, the residual is mostly noise
- Comparable to ELA but with higher computational cost

### Negative (30% confidence)
- **Pixel F1: < 0.65**
- DnCNN without good pretrained weights produces uninformative residuals
- CASIA2 images are too heterogeneous for camera-model-based fingerprinting to work
- The noiseprint signal is too subtle for the segmentation encoder to learn from

## Primary Metric Targets

| Metric | Target | Stretch Goal |
|--------|--------|-------------|
| Pixel F1 | > 0.7120 (+2pp) | > 0.7520 (+6pp) |
| Pixel IoU | > 0.58 | > 0.62 |
| Pixel AUC | > 0.90 | > 0.93 |

## Secondary Metrics

- Image-level accuracy: >= P.3 baseline (~92%)
- Training convergence: should plateau within 30 epochs
- Per-image inference time: expected ~1.5-2x slower than ELA due to DnCNN forward pass

## Success Criteria

- POSITIVE verdict if Pixel F1 > 0.7277 (exceeds current best P.10)
- NEUTRAL if 0.65 < F1 < 0.7277
- Any Pixel F1 > 0.55 indicates noiseprint has localization signal worth exploring further

## Failure Modes

1. **DnCNN weight unavailability**: If pretrained weights cannot be loaded in Kaggle, random-init DnCNN will produce near-random residuals. Mitigation: include a lightweight training loop for DnCNN on clean images before main training.
2. **Computational overhead**: Noiseprint extraction requires a full DnCNN forward pass per image. May cause OOM or excessive training time. Mitigation: precompute noiseprints and cache to disk.
3. **Domain mismatch**: DnCNN trained on natural image denoising may not produce forensically meaningful residuals on CASIA2's heterogeneous sources.

## Comparison Baselines

- vR.P.1 (RGB baseline): Pixel F1 = 0.4546
- vR.P.3 (ELA input): Pixel F1 = 0.6920
- vR.P.10 (current best): Pixel F1 = 0.7277
- If P.24 exceeds P.3: Noiseprint is a viable alternative/complement to ELA
- If P.24 exceeds P.10: Noiseprint is a superior representation for this task
