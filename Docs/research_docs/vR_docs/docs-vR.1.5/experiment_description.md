# Experiment Description — vR.1.5

| Field | Value |
|-------|-------|
| **Version** | vR.1.5 |
| **Title** | ETASR Ablation Study: Learning Rate Scheduler |
| **Parent** | vR.1.4 (BatchNormalization — NEUTRAL) |
| **Category** | Training Configuration |
| **Weakness Fixed** | W11 — No learning rate scheduler; contributes to training instability |
| **Single Change** | Add `ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)` |
| **Date** | 2026-03-15 |

---

## 1. Context

### Ablation Chain

```
vR.1.0 (baseline) → vR.1.1 (eval fix) → vR.1.2 (augmentation — REJECTED)
                                          ↓
                                   vR.1.3 (class weights — POSITIVE)
                                          ↓
                                   vR.1.4 (BatchNorm — NEUTRAL)
                                          ↓
                                   vR.1.5 (LR Scheduler) ← THIS VERSION
```

### Parent Results (vR.1.4)

| Metric | Value |
|--------|-------|
| Test Accuracy | 88.75% |
| Macro F1 | 0.8852 |
| ROC-AUC | 0.9536 |
| Tp Recall | 0.9194 (best in series) |
| Epochs | 8 (best at 3) |
| Verdict | NEUTRAL |

### Why This Change?

vR.1.4 revealed a critical training instability: **epoch 1 val_loss = 16.13** (BN warmup catastrophe). Despite recovering by epoch 3, training only lasted 8 epochs total — the shortest in the ablation study. This suggests the model converged too quickly to a suboptimal local minimum instead of exploring the loss landscape more thoroughly.

A learning rate scheduler addresses this by:
1. **Automatically reducing LR when learning plateaus** — prevents oscillation around minima
2. **Allowing longer training** — the model can train for more epochs before early stopping triggers
3. **Handling BN warmup** — if val_loss spikes, the scheduler reduces LR, dampening the instability

---

## 2. The Single Change

### Code Diff (vR.1.4 → vR.1.5)

```python
# vR.1.4 — Training callbacks
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True
)
model.fit(..., callbacks=[early_stopping], ...)

# vR.1.5 — Training callbacks (ONE CHANGE: add LR scheduler)
from tensorflow.keras.callbacks import ReduceLROnPlateau  # NEW import

early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True
)
lr_scheduler = ReduceLROnPlateau(        # ← NEW
    monitor='val_loss',                   # ← monitors loss, not accuracy
    factor=0.5,                           # ← halve LR on plateau
    patience=3,                           # ← wait 3 epochs before reducing
    min_lr=1e-6                           # ← floor: don't reduce below this
)
model.fit(..., callbacks=[early_stopping, lr_scheduler], ...)
```

### What Changes

| Aspect | vR.1.4 | vR.1.5 |
|--------|--------|--------|
| Callbacks | `[early_stopping]` | `[early_stopping, lr_scheduler]` |
| LR schedule | Fixed 1e-4 | Starts 1e-4, halves on val_loss plateau |
| Min LR | N/A | 1e-6 |

### What Stays Frozen

| Parameter | Value |
|-----------|-------|
| ELA quality | 90 |
| Image size | 128×128 |
| Seed | 42 |
| Optimizer | Adam (lr=1e-4 initial) |
| Batch size | 32 |
| Max epochs | 50 |
| Early stopping | patience=5, val_accuracy |
| Class weights | inverse-frequency (Au=0.8420, Tp=1.2310) |
| BatchNorm | After each Conv2D (from vR.1.4) |
| Architecture | Conv→BN→Conv→BN→Pool→Drop→Dense→Drop→Dense |
| Split | 70/15/15 stratified |

---

## 3. ETASR vs vR.1.5

| Aspect | Paper (ETASR) | vR.1.5 |
|--------|---------------|--------|
| LR schedule | Not mentioned | ReduceLROnPlateau |
| Initial LR | Not specified | 1e-4 (Adam default) |
| Optimizer | Adam | Adam |

The ETASR paper does not specify a learning rate scheduler. This ablation tests whether adaptive LR improves the CNN's convergence characteristics, which is a standard modern training technique.

---

## 4. Hypothesis

Adding ReduceLROnPlateau will:
1. Allow the model to train for more epochs (15-30 vs vR.1.4's 8)
2. Achieve better final metrics by fine-tuning at lower LR
3. Reduce the BN warmup instability (LR drops after epoch 1 spike)
4. Push test accuracy past 89%, potentially toward 90%
