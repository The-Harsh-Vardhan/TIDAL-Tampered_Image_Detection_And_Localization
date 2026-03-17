# 05_Training_Pipeline.md Review

## Purpose

Defines the loss, optimizer, scheduler, training loop, validation flow, checkpointing, and reproducibility setup.

## Technical Accuracy Score

`5/10`

## What Is Correct

- Technical correctness: BCE + Dice is a sound default loss for binary tampering segmentation with heavy class imbalance.
- Implementability on a single Colab notebook with T4: AMP, gradient accumulation, checkpointing, and early stopping are all appropriate for Colab if implemented carefully.
- Assignment alignment: The overall training structure is appropriate for the assignment and avoids unnecessary external systems.

## Issues Found

- Contradictions: The optimizer parameter groups reference `model.unet.encoder`, `model.unet.decoder`, and `model.unet.segmentation_head`, which do not match the plain `smp.Unet` baseline defined in `04_Model_Architecture.md`.
- Unsupported or hallucinated claims: The checkpoint-size and total-storage estimates are not reliable once optimizer, scheduler, and scaler states are included.
- Unnecessary complexity: `CosineAnnealingWarmRestarts` is presented as a baseline component even though other docs treat scheduler use as Stage 2 work.
- Missing technical details: The resume path restores `best_f1` but not enough state to resume early-stopping logic cleanly, such as `best_epoch` or patience history.
- Additional implementation risk: The loop has a leftover gradient-accumulation bug. If the number of train batches is not divisible by `ACCUMULATION_STEPS`, the final partial accumulation window is never stepped.
- Additional implementation risk: The optional `edge_loss` formulation is questionable because it applies BCE to logits and targets masked by an edge map rather than comparing explicit edge predictions. It should not be presented as a near-drop-in upgrade.

## Recommendations

- Fix the model API mismatch first. Either train plain `smp.Unet` or define a wrapper class and update every snippet to match.
- Flush the optimizer step after the loop if a partial accumulation window remains.
- Move the scheduler into an explicit optional section unless it is made baseline everywhere else.
- Drop the checkpoint-size numbers or replace them with a note that storage depends on model and optimizer state size.
- Keep edge loss out of the main path unless a clear failure mode justifies it.
