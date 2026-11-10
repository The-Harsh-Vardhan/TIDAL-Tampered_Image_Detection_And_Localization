# Review: Dataset.md

Source document path: `Docs/Dataset.md`

Purpose: Argue for Kaggle-first dataset handling over a Hugging Face-first workflow.

Validity score: 6/10

## Assignment alignment
- Strong on the need to demonstrate cleaning and alignment work.
- Useful as a short operational note.

## Technical correctness
- The Kaggle-first recommendation is reasonable.
- The "up to 1000x slower" HF claim is unverified (lines 23-25).
- The suggestion to use a balanced training subset of `1000` authentic and `2000` tampered images is unnecessary for a 5K-image project and may reduce coverage (lines 50-55).

## Colab T4 feasibility
- Kaggle download plus local file loading is feasible and simple on T4.

## Issues found
- Moderate: Unsupported platform-speed comparison (lines 23-25).
- Moderate: Unnecessary subsetting advice risks throwing away data (lines 50-55).
- Minor: The doc presents the Kaggle path as the only technically sound option, which is too absolute.

## Contradictions with other docs
- `Docs/Overall Flow Docs/13_Kaggle_vs_HuggingFace.md` makes the same conclusion more carefully.
- `Docs/Overall Flow Docs/14_HuggingFace_Platform.md` reintroduces HF Hub as an optional sharing layer.

## Recommendations
- Keep the raw-data-cleaning argument.
- Remove the 1000x claim and balanced-subset suggestion.
- Frame HF as optional post-project sharing rather than an enemy workflow.

## Severity summary
- Moderate
