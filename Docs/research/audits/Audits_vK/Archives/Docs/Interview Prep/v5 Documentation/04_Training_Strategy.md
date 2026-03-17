# Training Strategy

## How to explain this in an interview

Start with this:

"The training pipeline was designed for a small imbalanced segmentation dataset running on a single GPU, so I optimized for stable learning, reproducibility, and efficient memory use."

## Training pipeline at a glance

The main training choices are:

- BCE + Dice loss
- AdamW optimizer
- mixed precision training
- gradient accumulation
- gradient clipping
- early stopping
- checkpointing

Each of those solves a specific engineering problem.

## BCE + Dice loss

### What it is

This combines two losses:

- Binary Cross-Entropy for pixel-wise classification
- Dice loss for region overlap

### What problem it solves

Tamper masks are highly imbalanced. Most pixels are authentic, and only a small fraction are tampered.

If I use BCE alone, the model can get biased toward the easy majority class. Dice loss helps because it directly rewards overlap between predicted and true tampered regions.

### Why it was chosen here

I wanted a loss that handled both:

- pixel-level classification
- small-region localization quality

That makes BCE + Dice a practical choice for forgery localization.

### Alternatives

- BCE only
- Focal loss
- Tversky loss
- Lovasz-style IoU surrogates

### Why those were not selected

- BCE only is too weak against class imbalance.
- Focal loss is useful, but the project did not yet need that extra tuning complexity.
- Tversky and related losses can be strong, but BCE + Dice is simpler and easier to explain for a baseline.

### Future improvement

If false negatives on small tampered regions became a major issue, I would test Focal loss or Tversky-style variants.

## AdamW optimizer

### What it is

AdamW is an adaptive optimizer with decoupled weight decay.

### What problem it solves

The project fine-tunes a pretrained encoder while also training randomly initialized decoder layers. That usually benefits from stable adaptive updates.

### Why it was chosen here

AdamW is a strong default for transfer learning because it:

- converges quickly
- is easy to tune
- works well with pretrained backbones
- handles notebook-scale experimentation well

The project also uses a smaller learning rate for the encoder and a larger one for the decoder and head. That protects pretrained features while still allowing the new layers to adapt quickly.

### Alternatives

- SGD with momentum
- Adam
- RMSProp

### Why those were not selected

- SGD can work well, but it usually needs more tuning and longer training.
- Adam is close, but AdamW handles regularization more cleanly.
- RMSProp is less standard for this kind of transfer-learning setup.

### Future improvement

For larger-scale experiments, I would compare AdamW against SGD with a scheduler.

## Mixed precision training

### What it is

Mixed precision uses lower-precision math where safe, usually with automatic loss scaling.

### What problem it solves

Segmentation models can be memory intensive, especially at `512 x 512` resolution. Mixed precision helps reduce GPU memory use and often speeds up training.

### Why it was chosen here

The project needed to fit comfortably on a Colab T4 without shrinking the model or input too aggressively.

### Alternatives

- full FP32 training

### Why it was not selected

FP32 is simpler, but it uses more memory and wastes performance on this hardware budget.

### Future improvement

If the project moved to larger models, mixed precision would become even more important.

## Gradient accumulation

### What it is

Gradient accumulation lets the model simulate a larger effective batch size by summing gradients across multiple smaller batches before stepping the optimizer.

### What problem it solves

With `512 x 512` images and a segmentation network, large batch sizes may not fit comfortably in memory.

### Why it was chosen here

It lets the project keep:

- higher image resolution
- stable optimization behavior
- a Colab-friendly memory footprint

### Alternatives

- smaller images
- smaller model
- true larger batch sizes on a bigger GPU

### Why those were not selected

Reducing resolution can hurt localization quality, and a bigger GPU was outside the project constraints.

## Gradient clipping

### What it is

Gradient clipping limits gradient magnitude before the optimizer step.

### What problem it solves

It protects training from unstable updates, especially when using mixed precision and accumulation.

### Why it was chosen here

It is a low-cost stability safeguard. On a notebook project, that is usually worth adding.

### Alternatives

- no clipping
- lower learning rate

### Why those were not selected

No clipping leaves more room for unstable spikes, and only lowering learning rate is a weaker safeguard.

### What would happen without it

Training would still often work, but it would be less robust to unstable batches.

## Early stopping

### What it is

Early stopping ends training when validation performance stops improving.

### What problem it solves

The dataset is not large, so overfitting is a real risk. Early stopping limits wasted epochs and protects the best checkpoint.

### Why it was chosen here

It is a practical regularization tool and saves compute.

### Important detail

The project uses threshold-aware validation, which means model selection is based on the best validation F1 from a threshold sweep rather than a fixed threshold.

That is important because the saved best model should match the actual operating point used later in evaluation.

## Checkpointing

### What it is

The notebook saves:

- `best_model.pt`
- `last_checkpoint.pt`
- periodic epoch checkpoints

### What problem it solves

Notebook and Colab workflows are fragile:

- sessions can disconnect
- runtimes can reset
- long runs can fail

Checkpointing makes the training process recoverable.

### Why it was chosen here

It is essential for any serious training workflow, especially in cloud notebooks.

### Alternatives

- save only the final model
- no periodic resume state

### Why those were not selected

They are too risky for long training runs and make iteration slower.

## What if different options were used

### If I used BCE only

The model would likely bias toward background pixels and underperform on small tampered regions.

### If I used SGD instead of AdamW

It might eventually perform well, but it would probably require more careful learning-rate scheduling and longer tuning.

### If I disabled mixed precision

Training would use more memory and likely run slower.

### If I removed gradient clipping

The pipeline would still be simpler, but less protected against unstable updates.

### If I skipped checkpointing

The workflow would be much less reliable on Colab-style infrastructure.

## How I would summarize the training strategy

"The training strategy was built around the real constraints of the project: small masks, class imbalance, limited GPU memory, and notebook instability. BCE + Dice handled the segmentation objective, AdamW made fine-tuning stable, mixed precision and accumulation kept the run feasible on a T4, and checkpointing plus early stopping made the workflow reliable."
