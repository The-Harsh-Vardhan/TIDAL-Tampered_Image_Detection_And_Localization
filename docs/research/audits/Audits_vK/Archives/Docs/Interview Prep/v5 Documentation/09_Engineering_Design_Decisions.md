# Engineering Design Decisions

## How to explain this in an interview

Start with this:

"A big part of this project was not just model selection, but making the whole pipeline reproducible and practical on notebook infrastructure. I treated it like a small ML system, not just a training script."

## Why engineering decisions matter

In interviews, strong projects usually stand out because of engineering quality:

- Can the data pipeline be trusted?
- Can the run be reproduced?
- Can it survive notebook resets?
- Can it run on constrained hardware?

This project was deliberately designed around those questions.

## Dataset pipeline design

### What it is

The dataset pipeline does:

- dynamic sample discovery
- validation of image and mask readability
- dimension checking
- binary mask generation
- stratified split creation
- split persistence

### What problem it solves

Tamper datasets are often messy. If the data pipeline is weak, the training results become hard to trust.

### Why it was chosen here

I wanted the notebook to be robust against:

- corrupted files
- mask mismatches
- hardcoded path assumptions
- accidental split drift across reruns

### Alternatives

- manual dataset curation outside the notebook
- hardcoded filename lists
- one-time split creation without reuse

### Why those were not selected

They are faster initially, but they make the project more fragile and less reproducible.

### Future improvement

I would add richer dataset auditing, like automatic counts of excluded samples and optional dataset version hashing.

## Reproducibility decisions

### What it is

The project uses:

- fixed random seed
- persisted `split_manifest.json`
- deterministic DataLoader setup
- saved checkpoints and exported results

### What problem it solves

Without these controls, two runs can silently differ in ways that make comparisons misleading.

### Why it was chosen here

Reproducibility is critical when:

- comparing hyperparameters
- tuning thresholds
- debugging regressions
- discussing results in interviews

### Alternatives

- reseeding only the model
- letting splits regenerate every run
- saving only final metrics

### Why those were not selected

They leave too much hidden variability in the workflow.

## Colab compatibility

### What it is

The main notebook is designed to run on Google Colab with a single T4 GPU and optional Google Drive persistence.

### What problem it solves

The assignment explicitly required a single-notebook workflow that is practical on cloud notebook hardware.

### Why it was chosen here

Building for Colab forces pragmatic decisions:

- moderate model size
- mixed precision
- checkpointing
- limited dependencies

That makes the project more realistic and easier to demo.

## Kaggle portability

The project also has a Kaggle-runtime variant in `tamper_detection_v5.1_kaggle.ipynb`.

Why that matters:

- It shows the system is not locked to one notebook platform.
- It demonstrates portability of the core pipeline.
- It separates the core ML design from runtime-specific storage and input-path details.

How to talk about it:

"The main system definition is the v5 notebook. The v5.1 Kaggle notebook is the same core pipeline adapted to Kaggle's mounted input and working directories."

That is the right way to frame it in an interview. It is a runtime adaptation, not a different modeling approach.

## Single-GPU optimization

### What it is

The project uses:

- mixed precision
- gradient accumulation
- moderate batch size
- a lightweight baseline architecture

### What problem it solves

Segmentation can be memory heavy, especially at `512 x 512` resolution.

### Why it was chosen here

I wanted the model to remain:

- trainable on a T4
- stable
- reasonably fast
- close to the original assignment constraints

### Alternatives

- smaller input resolution
- much smaller encoder
- larger cloud GPU

### Why those were not selected

Reducing resolution risks hurting localization quality, and larger hardware was outside the target environment.

## Artifact and checkpoint design

### What it is

The project saves:

- best checkpoint
- last checkpoint
- periodic checkpoints
- split manifest
- plots
- result summary JSON

### What problem it solves

Notebook runtimes are fragile. Artifacts need to survive interruptions and support later review.

### Why it was chosen here

This makes the project easier to:

- resume
- audit
- compare
- explain

### Alternatives

- save only the final model
- keep metrics only in notebook cells

### Why those were not selected

They are too fragile for a serious training workflow.

## Why these choices matter in production ML systems

Even though this is a notebook project, the same principles matter in production:

- data validation prevents silent failures
- reproducibility supports trustworthy comparison
- checkpointing improves reliability
- platform portability reduces environment lock-in
- explicit artifacts make debugging easier

That is why these decisions are worth discussing in interviews. They show that the project was designed like a small ML product, not just a one-off experiment.

## Future improvements

If I wanted to harden the system further, I would add:

- dataset version tracking
- formal config files
- automated regression checks on metrics
- cleaner artifact naming across all runtime variants
- packaging the pipeline outside a single notebook

## How I would summarize the engineering side

"The engineering goal was to make the project reliable under notebook constraints. I focused on data validation, reproducible splits, recoverable checkpoints, and single-GPU efficiency. I also kept the pipeline portable enough to run in both Colab and Kaggle with only runtime-specific adjustments."
