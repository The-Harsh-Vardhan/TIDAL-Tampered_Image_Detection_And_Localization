# vR.P.28 — Expected Outcomes

## Scenarios

### Positive (35% confidence)
- **Pixel F1: 0.70–0.71** (+1-2pp over P.3 baseline 0.6920)
- Cosine annealing with warm restarts helps the model escape local minima found by plateau scheduler
- The cyclic LR exploration finds a better loss basin in the extended 50-epoch budget
- Warm restarts at epochs 10 and 30 provide productive re-exploration of the loss landscape

### Neutral (45% confidence)
- **Pixel F1: 0.68–0.70**
- Cosine schedule reaches similar performance to plateau scheduler — the loss landscape is smooth enough that ReduceLROnPlateau finds a good minimum already
- Extended training (50 epochs) does not help because the model converges by epoch 20-25 regardless of scheduler
- Warm restarts temporarily increase loss but recover to approximately the same level

### Negative (20% confidence)
- **Pixel F1: < 0.68**
- Warm restarts are too disruptive — the LR jumps destabilize learned features
- The model oscillates between minima without settling into any one
- Early stopping triggers during a restart phase when val_loss spikes

## Primary Metric Targets

| Metric | Target | Stretch Goal |
|--------|--------|-------------|
| Pixel F1 | > 0.7020 (+1pp) | > 0.7120 (+2pp) |
| Pixel IoU | > 0.57 | > 0.58 |
| Pixel AUC | > 0.89 | > 0.91 |

## Secondary Metrics

- Convergence epoch: which epoch achieves best val loss (expected later than P.3 due to restarts)
- LR curve shape: verify cosine pattern with restarts at expected epochs
- Training loss after restart: should recover within 2-3 epochs of each restart
- Total training time: expected ~1.7x longer than P.3 (50 vs 30 epochs)
- Image-level accuracy: should remain >= 92%

## Success Criteria

- POSITIVE verdict if Pixel F1 > 0.7020 with clean convergence
- NEUTRAL if within +/- 1pp of P.3 (0.6820 < F1 < 0.7020)
- The cosine scheduler is a hyperparameter change — even neutral results inform future experiment design
- If positive, cosine annealing should be adopted as default scheduler for all future experiments

## Failure Modes

1. **Kaggle timeout**: 50 epochs at ~15 min/epoch = 12.5 hours, right at the limit. Mitigation: reduce EPOCHS to 40 or monitor runtime early.
2. **Restart destabilization**: LR jumping back to initial value at epoch 10 may undo learned features. Mitigation: if loss doesn't recover within 3 epochs after restart, the restarts are too aggressive. Consider T_mult=3 or higher eta_min.
3. **Early stopping during restart**: Val loss spike after restart may trigger early stopping. Mitigation: PATIENCE=10 should accommodate a single restart cycle (T_0=10), but verify.
4. **No improvement despite more epochs**: Model may have already converged by epoch 20-25 (same as P.3), and extra epochs are wasted compute. Mitigation: log and compare best-epoch timing vs P.3.

## Comparison Baselines

- vR.P.3 (ELA baseline, ReduceLROnPlateau, 30 epochs): Pixel F1 = 0.6920
- vR.P.10 (current best): Pixel F1 = 0.7277
- Cosine annealing literature reports 0.5-2pp improvements on image classification tasks
- Segmentation tasks may benefit less from LR scheduling since the loss landscape is typically smoother
- If P.28 > P.3: cosine annealing becomes the default scheduler going forward
- If P.28 = P.3: scheduler choice is not a bottleneck; focus optimization efforts elsewhere
