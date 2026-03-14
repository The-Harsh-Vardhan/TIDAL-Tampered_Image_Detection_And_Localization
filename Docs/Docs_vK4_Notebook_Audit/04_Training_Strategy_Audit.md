# 04 - Training Strategy Audit

## Components Used

- Multi-task loss: classification + segmentation
- Classification loss: focal loss with class weights
- Segmentation loss: BCEWithLogits + Dice
- Class imbalance handling: class weights + pos_weight
- Optimizer: Adam with weight decay
- Scheduler: ReduceLROnPlateau
- Stabilizers: AMP, gradient accumulation, gradient clipping, early stopping

## What Is Good

1. The strategy is modern and practical for constrained GPU memory.
2. Segmentation imbalance handling is not ignored.
3. Early stopping and scheduler coupling are sane defaults.

## What Is Weak

## A) Too many methods changed simultaneously

This setup introduces many interacting controls at once.

### Why that hurts
When metrics move, causal attribution is impossible.

### Senior expectation
Incremental experiments with explicit deltas per change.

## B) No executed evidence of convergence

There are no output traces for loss curves, LR drops, or metric stabilization in the submitted artifact.

### Why that hurts
A training strategy is only valid if it demonstrably converges on the actual run.

### Senior expectation
Executed logs visible in notebook with checkpoint epoch and best metric.

## C) pos_weight logic may diverge from effective training distribution

Foreground ratio is estimated on raw masks, while heavy augmentation and resizing alter effective pixel distribution.

### Why that hurts
Weighting can become miscalibrated, producing unstable gradients.

### Senior expectation
Report final foreground ratio after preprocessing or validate sensitivity to pos_weight.

## D) Early stopping on a single metric without variance reporting

Early stopping monitors val F1 only; no seed variation shown.

### Why that hurts
Single-seed volatility can produce false confidence.

### Senior expectation
At least 3 seeds or confidence bounds for final claims.

## Verdict

- Technical strategy design: **Good on paper**
- Empirical validity in submission artifact: **Not proven**
- Final score: **6/10**
