# Experiment Tracking

## How to explain this in an interview

Start with this:

"I used experiment tracking so I could compare runs systematically instead of relying on notebook memory. In this project, Weights & Biases is optional, but when enabled it logs the training and evaluation behavior end to end."

## What experiment tracking is

Experiment tracking means storing:

- hyperparameters
- metrics
- plots
- model artifacts
- run summaries

in a structured way so different experiments can be compared later.

## What problem it solves

Without tracking, it is easy to lose important context:

- Which learning rate worked best?
- Which checkpoint produced the final numbers?
- What threshold was selected?
- Did a change improve robustness or only the clean test set?

Tracking solves that by turning a notebook experiment into something reproducible and comparable.

## Why Weights & Biases was chosen

### What it is

Weights & Biases, or W&B, is an experiment-tracking platform for ML workflows.

### What problem it solves

It stores metrics, artifacts, and plots across runs in a way that is easier to compare than local notebook output.

### Why it was chosen here

I chose it because:

- it is widely used in industry
- it works well for notebook workflows
- it makes run comparison easy
- it can log both numeric metrics and visual artifacts

## What gets logged

The project is designed to log:

- training loss
- validation loss
- validation F1
- validation IoU
- selected validation threshold
- final test metrics
- prediction visualizations
- saved model artifact

That makes it easy to compare both model quality and training behavior.

## Why tracking is useful in this project

This project has several moving parts:

- threshold selection
- segmentation metrics
- robustness results
- visual outputs

If I only print numbers in a notebook, it becomes hard to compare experiments later. W&B helps keep the whole pipeline organized.

## Guarded `USE_WANDB` behavior

This is an important engineering detail.

The notebook uses a `USE_WANDB` flag. When it is:

- `True`: W&B logging is active
- `False`: the notebook still runs correctly and saves local artifacts

That is a good design because experiment tracking should improve the workflow, not become a hard dependency that breaks the project.

## Alternatives that could have been used

- TensorBoard
- MLflow
- plain CSV or JSON logging
- manual notebook outputs only

## Why they were not selected

- TensorBoard is useful, but W&B is often easier for quick notebook-based comparison and artifact sharing.
- MLflow is strong, but it is heavier than needed for a Colab-scale personal project.
- CSV or JSON logs are fine for simple metrics, but weak for visualization-heavy workflows.
- Manual notebook inspection does not scale once multiple runs exist.

## How experiments can be compared

The main comparison questions are:

- Which hyperparameter setting gave the best validation F1?
- Did a change improve IoU but hurt recall?
- Did robustness improve or only clean accuracy?
- Did the new threshold shift significantly?

That is exactly the kind of reasoning experiment tracking supports.

## Why this matters in interviews

Experiment tracking shows that I approached the project like an engineer, not just like someone who trained one model once.

It demonstrates:

- reproducibility
- debugging discipline
- metric awareness
- good ML workflow habits

## Future improvements

If the project grew, I would extend tracking to include:

- richer hyperparameter sweeps
- dataset-version metadata
- cross-run robustness dashboards
- clearer artifact lineage between checkpoints and reports

## How I would summarize experiment tracking

"W&B was used as an optional tracking layer, not as a hard dependency. That let me log training, evaluation, thresholds, plots, and artifacts in a structured way when I wanted run comparison, while still keeping the notebook runnable without external tracking."
