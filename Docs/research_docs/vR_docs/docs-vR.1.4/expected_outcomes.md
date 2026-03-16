# Expected Outcomes: vR.1.4 — BatchNormalization

---

## Hypothesis

Adding BatchNormalization after each Conv2D will:
1. **Stabilize training** — eliminate the epoch 11-type val_accuracy spikes seen in vR.1.0/1.1
2. **Reduce overfitting** — BN's implicit regularization effect narrows the train-val gap
3. **Maintain or slightly improve accuracy** — normalized activations learn more efficiently

---

## Quantitative Predictions

| Metric | vR.1.3 (Parent) | vR.1.4 (Predicted) | Reasoning |
|--------|-----------------|-------------------|-----------|
| Test Accuracy | TBD | +0 to +2pp | BN generally matches or improves accuracy |
| Tampered F1 | TBD | +0 to +1pp | Stabilized gradients → better minority class learning |
| Macro F1 | TBD | +0 to +1pp | Follows accuracy trend |
| ROC-AUC | TBD | Stable or +0.01 | Probability calibration may improve |
| Train-Val Gap | ~5-10% | **Narrower** | BN's regularization effect |
| Training Epochs | ~10-15 | ~10-20 | More stable → may train longer before stopping |
| Best Epoch Spikes | Likely present | **Reduced or eliminated** | Primary expected effect |

---

## Qualitative Predictions

### Training Curve Shape

**vR.1.3 (expected, based on vR.1.1):**
```
Val Accuracy
 ^
 |   ╱──╲──╱╲
 |  ╱        ╲
 | ╱
 +──────────────> Epoch
   Spiky, unstable
```

**vR.1.4 (predicted):**
```
Val Accuracy
 ^
 |      ────────
 |   ╱
 | ╱
 +──────────────> Epoch
   Smooth, stable plateau
```

### Loss Landscape

BatchNorm smooths the loss landscape by reducing internal covariate shift. This should manifest as:
- Smoother loss curves (less zig-zagging)
- More monotonic val_loss decrease before plateau
- Less sensitivity to the exact epoch at which early stopping triggers

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| BN + Dropout conflict | Low | Slight accuracy decrease | Monitor: if accuracy drops, consider removing Dropout(0.25) in next version |
| BN slows convergence | Very Low | More epochs needed | patience=5 handles this; max_epochs=50 is sufficient |
| BN changes batch sensitivity | Low | Results depend on batch composition | batch_size=32 is large enough for stable BN statistics |
| Accuracy regression | Low | NEGATIVE verdict | Roll back to vR.1.3, skip to vR.1.5 |

---

## Success Criteria

| Criterion | Threshold | Verdict |
|-----------|-----------|---------|
| Accuracy ≥ parent | ±0.5% | POSITIVE if improved, NEUTRAL if within range |
| Training more stable | Smoother val_loss curve | Qualitative assessment |
| No accuracy regression | > -0.5% from parent | NEGATIVE if violated |

---

## If POSITIVE → Next Step

Proceed to vR.1.5 (ReduceLROnPlateau scheduler), which builds on the stabilized training landscape from BatchNorm.

## If NEGATIVE → Fallback

- Reject vR.1.4
- Branch vR.1.5 from vR.1.3 (skip BatchNorm)
- Investigate whether BN+Dropout(0.25) conflict was the cause
