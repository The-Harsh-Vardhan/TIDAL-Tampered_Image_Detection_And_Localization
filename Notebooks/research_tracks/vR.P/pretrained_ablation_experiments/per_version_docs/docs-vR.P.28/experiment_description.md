# vR.P.28 -- Experiment Description

## Cosine Annealing LR Scheduler

### Hypothesis

Cosine annealing with warm restarts provides better learning rate dynamics than ReduceLROnPlateau, enabling the model to escape local minima through periodic LR increases. This is particularly beneficial for longer training runs (50+ epochs) where the model may plateau prematurely.

### Motivation

P.3 uses ReduceLROnPlateau (reactive LR reduction). Cosine annealing is proactive -- it follows a predetermined cosine curve that smoothly reduces LR from max to min over each cycle, then restarts. Benefits:
- Periodic restarts escape local minima
- Smooth LR reduction is less disruptive than step-wise plateau reduction
- Well-suited for the 50-epoch training budget (from P.7 findings)

This is standard in modern segmentation competitions (Kaggle) and has been shown to improve convergence quality.

### Single Variable Changed from vR.P.3

**LR scheduler** -- Replace ReduceLROnPlateau with CosineAnnealingWarmRestarts.

### Key Configuration

| Parameter | P.3 (parent) | P.28 (this) |
|-----------|-------------|-------------|
| LR scheduler | ReduceLROnPlateau(patience=3, factor=0.5) | CosineAnnealingWarmRestarts(T_0=10, T_mult=2, eta_min=1e-6) |
| Epochs | 25 | 50 (extended to benefit from restarts) |
| Patience | 7 | 10 (longer patience for cosine cycles) |
| Initial LR | 1e-3 | 1e-3 |
| Everything else | Same | Same |

### Pipeline

```
LR schedule over 50 epochs:
    Cycle 1: epochs 0-9   (10 epochs, LR: 1e-3 -> 1e-6 -> restart)
    Cycle 2: epochs 10-29 (20 epochs, LR: 1e-3 -> 1e-6 -> restart)
    Cycle 3: epochs 30-49 (20 epochs, LR: 1e-3 -> 1e-6)
```

### Expected Impact

+1-2pp Pixel F1 (training optimization experiment -- lower impact category but combines well with other improvements).

### Risk

Warm restarts can temporarily degrade metrics (LR spikes to 1e-3 at restart). Early stopping with patience=10 must accommodate this without triggering premature stop.
