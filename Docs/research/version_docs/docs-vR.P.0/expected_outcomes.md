# Expected Outcomes: vR.P.0 — ResNet-34 UNet Baseline

---

## Hypothesis

A pretrained ResNet-34 encoder (frozen) with a UNet decoder trained on CASIA v2.0 RGB images will produce meaningful pixel-level tampered region masks, establishing a localization baseline that the ETASR CNN cannot achieve.

The decoder should learn to map the encoder's hierarchical visual features to tampered region masks, leveraging the encoder's understanding of natural image statistics to detect visual inconsistencies.

---

## Predicted Outcomes

### Optimistic Scenario

| Metric | Predicted | Rationale |
|--------|-----------|-----------|
| Pixel F1 (Dice) | 0.35 - 0.45 | v6.5 achieved 0.41; this version has similar config |
| Pixel IoU | 0.25 - 0.35 | IoU ≈ F1 / (2 - F1) for reasonable F1 values |
| Pixel AUC | 0.80 - 0.90 | Good pixel-level discrimination |
| Classification Accuracy | 85 - 92% | Competitive with ETASR's 88.38% |
| Macro F1 (classification) | 0.85 - 0.92 | Balanced classes |
| Best epoch | 10 - 20 | Pretrained decoder converges more slowly |

### Expected Scenario

| Metric | Predicted | Rationale |
|--------|-----------|-----------|
| Pixel F1 (Dice) | 0.20 - 0.35 | Slightly lower than v6.5 due to GT mask quality |
| Pixel IoU | 0.15 - 0.25 | Reasonable segmentation |
| Pixel AUC | 0.70 - 0.80 | Above random |
| Classification Accuracy | 75 - 85% | May underperform ETASR if masks are noisy |
| Macro F1 (classification) | 0.75 - 0.85 | Lower than ETASR due to different task |
| Best epoch | 5 - 15 | Early training should show clear progress |

GT mask quality is the biggest uncertainty. If using ELA pseudo-masks (no official GT), the model is effectively learning to reproduce ELA thresholding, which limits the ceiling.

### Pessimistic Scenario

| Metric | Predicted | Rationale |
|--------|-----------|-----------|
| Pixel F1 (Dice) | 0.05 - 0.15 | Poor masks or convergence issues |
| Pixel IoU | 0.03 - 0.10 | Very limited overlap |
| Classification Accuracy | 60 - 75% | Mask-based classification underperforms |
| Best epoch | 1 - 5 | Possible convergence failure |

This would happen if:
1. No GT masks available AND ELA pseudo-masks are too noisy
2. Encoder features don't transfer well for this specific task
3. Training configuration (LR, loss) is suboptimal

**Even the pessimistic scenario is still POSITIVE** — it demonstrates localization capability that ETASR cannot provide, and establishes a baseline for improvement in vR.P.1+.

---

## Success Criteria

### POSITIVE verdict (proceed to vR.P.1):
- Pixel F1 ≥ 0.15 (model produces non-trivial localization)
- OR Classification accuracy ≥ 75% (competitive baseline)
- AND Training converges (best epoch > 1, val_loss improves over training)

### NEUTRAL verdict (still proceed, investigate issues):
- Pixel F1 between 0.05 and 0.15
- AND Classification accuracy between 60% and 75%
- May indicate GT mask quality issues or need for unfreezing

### NEGATIVE verdict (investigate before proceeding):
- Pixel F1 < 0.05 (model predicts trivial masks — all zeros or all ones)
- OR Training doesn't converge (best epoch = 1, no improvement)
- OR OOM / crash on T4 GPU

---

## Key Uncertainties

### 1. Ground Truth Mask Quality (HIGH impact)

The biggest unknown. CASIA v2.0 may or may not have pixel-level GT masks on Kaggle:
- **With GT masks:** Model has real targets to learn from → expect good Pixel F1
- **With ELA pseudo-masks:** Model learns to reproduce ELA thresholding → ceiling limited
- **No masks at all:** Fall back to all-zero (authentic) + pseudo (tampered) → noisiest

**Mitigation:** The notebook auto-detects GT masks and falls back gracefully.

### 2. ELA Pseudo-Mask Noise (MEDIUM impact)

If using ELA pseudo-masks:
- ELA thresholding is imprecise — it highlights compression artifacts, not exact tampered boundaries
- The model may learn the ELA pattern rather than true tampering artifacts
- Pixel-level metrics will be artificially limited by mask quality

**Mitigation:** Even noisy masks provide spatial signal. Gradual unfreeze (vR.P.1) may help the encoder adapt.

### 3. Classification vs ETASR (LOW impact)

Classification accuracy derived from masks may differ from ETASR's direct classification:
- ETASR directly optimizes for classification → likely higher accuracy
- vR.P.0 derives classification from mask area → indirect, may be lower
- This is expected and acceptable — the goal is localization, not beating ETASR on classification

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| No GT masks on Kaggle | Medium | High | ELA pseudo-mask fallback built in |
| OOM on T4 at 384×384 batch=16 | Low | Medium | Reduce to batch=8 |
| SMP install fails | Very Low | High | Notebook has isInternetEnabled=True |
| Frozen encoder too restrictive | Low | Medium | vR.P.1 tests unfreezing |
| Model predicts trivial masks | Low | High | BCEDiceLoss handles imbalance |
| Training doesn't converge | Very Low | High | Check LR, loss, normalization |

**Overall risk: LOW-MEDIUM.** The approach is proven (v6.5), the framework is mature (SMP), and the notebook has fallbacks for known issues. The main uncertainty is GT mask availability.

---

## Comparison: ETASR vR.1.x vs Pretrained vR.P.0

| Aspect | ETASR | Pretrained | Winner |
|--------|-------|-----------|--------|
| Classification accuracy | ~88% | ~75-85% (expected) | ETASR |
| Localization capability | None | Native | **Pretrained** |
| Assignment alignment | Partial | **Full** | **Pretrained** |
| Data efficiency | 1:3,343 | 1:57 | **Pretrained** |
| Augmentation compatibility | Failed (vR.1.2) | Likely works | **Pretrained** |
| Architecture ceiling | Low (fixed) | High (scalable) | **Pretrained** |

Even if vR.P.0 has lower classification accuracy than ETASR, it is the more valuable experiment because it addresses the assignment's core requirement.
