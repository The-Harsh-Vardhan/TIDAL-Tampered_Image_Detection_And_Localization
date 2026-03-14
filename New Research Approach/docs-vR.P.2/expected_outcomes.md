# Expected Outcomes: vR.P.2 — Gradual Encoder Unfreeze

---

## Hypothesis

Selectively unfreezing the last 2 encoder blocks (layer3 + layer4) of the pretrained ResNet-34 with a conservative learning rate (1e-5, 100x lower than the decoder) will allow the encoder to develop forensic-specific representations while preserving general visual features, improving pixel-level localization metrics over the fully-frozen baseline (vR.P.1).

---

## Predicted Outcomes

### Optimistic Scenario

| Metric | vR.P.1 (predicted baseline) | vR.P.2 Predicted | Rationale |
|--------|---------------------------|------------------|-----------|
| Pixel F1 | 0.20 - 0.35 | 0.30 - 0.50 | Domain-adapted features capture forensic artifacts |
| Pixel IoU | 0.15 - 0.25 | 0.20 - 0.35 | Better pixel-level localization |
| Pixel AUC | 0.70 - 0.80 | 0.80 - 0.90 | Improved discrimination at pixel level |
| Tampered F1 (cls) | 0.70 - 0.85 | 0.80 - 0.90 | Better masks → better classification |
| Test Accuracy | 75 - 85% | 80 - 90% | Improved overall performance |
| Best epoch | 5 - 15 | 10 - 20 | Slower convergence with more trainable params |

This happens if: encoder layer3+layer4 learn meaningful forensic features without catastrophic forgetting, and the 100x LR ratio is well-calibrated.

### Expected Scenario

| Metric | vR.P.1 (predicted baseline) | vR.P.2 Predicted | Rationale |
|--------|---------------------------|------------------|-----------|
| Pixel F1 | 0.20 - 0.35 | 0.25 - 0.40 | Modest improvement from domain adaptation |
| Pixel IoU | 0.15 - 0.25 | 0.18 - 0.30 | Slight improvement |
| Pixel AUC | 0.70 - 0.80 | 0.75 - 0.85 | Moderate improvement |
| Tampered F1 (cls) | 0.70 - 0.85 | 0.75 - 0.88 | Slightly better masks |
| Test Accuracy | 75 - 85% | 78 - 87% | Modest gain |
| Best epoch | 5 - 15 | 8 - 18 | Slightly slower convergence |

The 100x LR ratio is conservative — the encoder adapts slowly but meaningfully. The improvement is measurable but not dramatic because the CASIA v2.0 dataset is small (~8,800 training images) relative to the 5M trainable parameters.

### Pessimistic Scenario

| Metric | vR.P.1 (predicted baseline) | vR.P.2 Predicted | Rationale |
|--------|---------------------------|------------------|-----------|
| Pixel F1 | 0.20 - 0.35 | 0.15 - 0.30 | Overfitting degrades generalization |
| Pixel IoU | 0.15 - 0.25 | 0.10 - 0.20 | Worse than frozen encoder |
| Pixel AUC | 0.70 - 0.80 | 0.65 - 0.75 | Encoder features degraded |
| Test Accuracy | 75 - 85% | 70 - 80% | Overfitting hurts |
| Best epoch | 5 - 15 | 3 - 8 | Early stopping fires early |

This happens if:
1. **Overfitting:** 5M trainable params on ~8,800 images (1:570 data:param ratio) causes memorization
2. **Catastrophic forgetting:** Even 1e-5 is too aggressive, encoder loses valuable ImageNet features
3. **Training instability:** Two param groups with very different LRs create optimization difficulties

---

## Success Criteria

### POSITIVE verdict (Pixel F1 improvement ≥ +1 percentage point):
- Pixel F1 improves by ≥ 0.01 over vR.P.1
- OR Image-level Tampered F1 improves by ≥ 0.01
- AND training converges normally (no early collapse)

### NEUTRAL verdict (within ±1pp of vR.P.1):
- All metrics within ±0.01 of vR.P.1
- Suggests 100x LR ratio is too conservative — may need 10x in future version

### NEGATIVE verdict (Pixel F1 drops by > 1pp):
- Pixel F1 or Tampered F1 drops by > 0.01
- Indicates unfreezing hurts — keep encoder frozen for future versions
- If negative: vR.P.3+ branch from vR.P.1 (frozen encoder) instead

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Overfitting (5M params, small dataset) | **Medium** | Medium | Early stopping (patience=7), ReduceLROnPlateau, weight decay |
| Catastrophic forgetting | Low | High | 100x LR ratio is very conservative |
| Training instability (two very different LRs) | Low | Medium | Standard Adam handles multi-group well |
| Longer training time | Low | Low | ~30% slower; still fits in Kaggle's 12-hour limit |
| Scheduler affects both groups | Low | Low | ReduceLROnPlateau scales both groups proportionally — this is actually desirable |

**Primary risk: Overfitting.** The data:param ratio shifts from 1:57 (vR.P.1) to 1:570 (vR.P.2). This 10x increase in trainable parameters on the same dataset is the main concern. The mitigation (early stopping + weight decay + conservative LR) should be sufficient, but if Pixel F1 drops, overfitting is the likely cause.

**Overall risk: LOW-MEDIUM.** The conservative 100x LR ratio limits downside. Even if neutral, the experiment provides valuable information about whether domain adaptation helps for forensic detection on small datasets.

---

## What This Experiment Tells Us

| Result | Conclusion | Impact on Future Versions |
|--------|-----------|--------------------------|
| POSITIVE | Domain adaptation helps — encoder needs forensic-specific features | Consider 10x LR ratio in future, or unfreezing more layers |
| NEUTRAL | 100x is too conservative — encoder barely adapts | Try 10x LR ratio as separate ablation |
| NEGATIVE | Unfreezing hurts — ImageNet features are sufficient | Keep encoder frozen; focus on input changes (ELA, 4ch) instead |

This is a critical experiment because it determines whether downstream versions should modify the encoder or leave it frozen.
