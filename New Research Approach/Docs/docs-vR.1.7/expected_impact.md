# Expected Impact — vR.1.7: Global Average Pooling

## Hypothesis

Replacing Flatten with GlobalAveragePooling2D will dramatically reduce overfitting (from vR.1.6's ~5pp train-val gap) by cutting trainable parameters from 13.8M to ~64K (99.5% reduction). The model should maintain competitive accuracy despite the massive parameter reduction, because the convolutional layers already extract meaningful features and the Dense(256) classifier needs only 64 filter-level signals, not 53,824 spatial values.

---

## Predicted Outcomes

### Optimistic Scenario (POSITIVE)

| Metric | vR.1.6 | vR.1.7 Predicted | Rationale |
|--------|--------|------------------|-----------|
| Test Accuracy | 90.23% | 89-91% | GAP regularization compensates for info loss |
| Macro F1 | 0.9004 | 0.89-0.91 | Better generalization from reduced overfitting |
| ROC-AUC | 0.9657 | 0.96-0.97 | Similar discrimination |
| Train-Val Gap | ~5pp | **<2pp** | 64K params cannot memorize 8,829 images |
| Best Epoch | 13 | 15-25 | Slower convergence, better plateau |

This happens if: The 64 GAP features capture enough discriminative information. Tampered/authentic distinction is a global image property that doesn't require spatial precision. The massive regularization eliminates the observed overfitting.

### Expected Scenario (NEUTRAL)

| Metric | vR.1.6 | vR.1.7 Predicted | Rationale |
|--------|--------|------------------|-----------|
| Test Accuracy | 90.23% | 87-89% | Moderate accuracy loss from spatial info compression |
| Macro F1 | 0.9004 | 0.87-0.89 | Slight degradation |
| ROC-AUC | 0.9657 | 0.94-0.96 | Less confident predictions |
| Train-Val Gap | ~5pp | **<3pp** | Much less overfitting |
| Best Epoch | 13 | 10-20 | |

This is the most likely scenario. The 99.5% parameter reduction is aggressive — some performance loss is expected. But the reduced overfitting partially compensates, keeping accuracy within a tolerable range.

### Pessimistic Scenario (NEGATIVE)

| Metric | vR.1.6 | vR.1.7 Predicted | Rationale |
|--------|--------|------------------|-----------|
| Test Accuracy | 90.23% | <87% | Spatial info loss is critical for ELA detection |
| Macro F1 | 0.9004 | <0.87 | Both classes degrade |
| Best Epoch | 13 | 5-10 | Model learns quickly but plateaus at lower accuracy |

This happens if: ELA-based tampering detection fundamentally requires knowing **where** high-ELA pixels are located, not just **that** they exist. GAP discards this spatial information, leaving only 64 global feature averages — insufficient for fine-grained detection.

---

## Success Criteria

### POSITIVE verdict (≥ 89.73% test accuracy):
- Test accuracy within 0.5pp of vR.1.6 (≥ 89.73%)
- OR Macro F1 within 0.005 of vR.1.6 (≥ 0.8954)
- AND reduced overfitting (train-val gap < 3pp)
- This would prove that spatial pooling + massive param reduction is the right direction

### NEUTRAL verdict (87.23% - 89.73%):
- Moderate accuracy drop but still above vR.1.1 baseline (88.38%)
- Reduced overfitting validates the regularization benefit
- May suggest GAP is too aggressive — consider GAP + larger Dense, or spatial attention

### NEGATIVE verdict (< 87.23%):
- Accuracy drops > 3pp from vR.1.6
- Indicates ELA detection needs spatial structure that GAP destroys
- Direction requires rethinking — Flatten may be necessary, focus on other W10 mitigations

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Spatial info loss hurts ELA detection | Medium | High | Conv layers provide hierarchical spatial processing before GAP |
| Model underfits (too few params) | Medium | Medium | Dense(256) still provides nonlinear classification capacity |
| Training instability with tiny model | Low | Low | Adam is robust; BN stabilizes convolutions |
| All filters learn similar features | Low | Medium | Dropout(0.25) before GAP encourages diversity |

**Overall risk: MEDIUM.** The change is aggressive (99.5% param reduction), but GAP is proven in image classification. The main uncertainty is whether ELA features need spatial precision that GAP removes.

---

## Key Metric to Watch

**Train-val accuracy gap.** If vR.1.7 reduces the gap from ~5pp (vR.1.6) to <2pp while maintaining accuracy above 88%, the GAP approach is validated. Even if accuracy drops slightly, a tighter train-val gap proves the model generalizes better and opens the path for further improvements (more filters, deeper networks, augmentation retry).
