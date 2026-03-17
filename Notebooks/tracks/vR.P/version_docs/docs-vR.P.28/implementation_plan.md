# vR.P.28 — Implementation Plan

## Core Implementation: Cosine Annealing LR Scheduler

### Scheduler Replacement

Replace `ReduceLROnPlateau` with `CosineAnnealingWarmRestarts`:

```python
# Old (P.3):
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
# scheduler.step(val_loss) — called per epoch after validation

# New (P.28):
scheduler = CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,       # first restart period: 10 epochs
    T_mult=2,     # double period after each restart: 10, 20, 40 epochs
    eta_min=1e-6  # minimum LR floor
)
# scheduler.step() — called per epoch (not per plateau)
```

### Training Loop Modification

```python
for epoch in range(EPOCHS):
    train_loss = train_one_epoch(...)
    val_loss = validate(...)
    scheduler.step()  # cosine step — no val_loss argument
    # early stopping still uses val_loss for patience tracking
```

### Cell Modification Map

| Cell | Section | Action |
|------|---------|--------|
| 0 | Title | "vR.P.28 — Cosine Annealing LR Scheduler" |
| 1 | Changelog | Add P.28 entry: cosine annealing with warm restarts |
| 2 | Setup | VERSION='vR.P.28', EPOCHS=50, PATIENCE=10, LR_SCHEDULER='cosine' |
| 13 | Training config | Describe cosine annealing strategy: T_0=10, T_mult=2, eta_min=1e-6; warm restarts allow escape from local minima |
| 14 | Scheduler/loss setup | Replace ReduceLROnPlateau with CosineAnnealingWarmRestarts(T_0=10, T_mult=2, eta_min=1e-6) |
| 15 | Training loop | Change `scheduler.step(val_loss)` to `scheduler.step()` — cosine scheduler steps per epoch, not per plateau event |
| 25 | Results table | Note "Cosine LR T_0=10 T_mult=2" in config column |
| 26 | Discussion | Cosine vs plateau hypothesis, LR curve analysis, warm restart effects |
| 27 | Save model | Config includes lr_scheduler='cosine', T_0=10, T_mult=2, eta_min=1e-6, epochs=50 |

### Unchanged Cells

Cells 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 17, 18, 19, 20, 21, 22, 23, 24 remain unchanged from P.3. This is a scheduler-only modification — no changes to data pipeline, model architecture, loss function, or evaluation.

### Key New Code

- Scheduler instantiation (~3 lines): CosineAnnealingWarmRestarts with configured params
- Training loop modification (~2 lines): change scheduler.step() call signature
- Hyperparameter changes: EPOCHS=50 (up from 30), PATIENCE=10 (up from 7) to accommodate cosine schedule
- Total changes are minimal (~5 lines), making this a clean scheduler ablation

### Verification Checklist

- [ ] CosineAnnealingWarmRestarts instantiates without errors
- [ ] `scheduler.step()` called once per epoch (no val_loss argument)
- [ ] LR follows expected cosine curve: starts at initial LR, decays to eta_min over T_0 epochs, then restarts
- [ ] Warm restart occurs at epoch 10 (first restart), epoch 30 (second restart at T_0*T_mult=20)
- [ ] Early stopping still functions correctly with PATIENCE=10
- [ ] Training runs for up to 50 epochs without timeout on Kaggle (budget ~12 hours)
- [ ] LR curve logged and can be plotted for verification
- [ ] All metric cells execute and produce valid Pixel F1 / IoU / AUC values

### Risks

- 50 epochs may hit Kaggle's 12-hour GPU time limit — monitor runtime per epoch
- Warm restarts may cause training instability if the restart LR is too high relative to model state
- PATIENCE=10 may be too generous — model could waste epochs after convergence
- Cosine schedule does not adapt to validation loss — may continue training past optimal point
