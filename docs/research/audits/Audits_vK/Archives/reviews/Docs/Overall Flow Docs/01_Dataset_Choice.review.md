# Review: 01_Dataset_Choice.md

Source document path: `Docs/Overall Flow Docs/01_Dataset_Choice.md`

Purpose: Turn the dataset decision into notebook-ready implementation steps.

Validity score: 7/10

## Assignment alignment
- Strongly aligned with the need to choose a dataset and handle mask issues.

## Technical correctness
- The pairing rule, mask binarization, and misalignment handling are useful.
- Exact benchmark references for COVERAGE models are unverified in the repo (lines 121-125).
- The "HF can be up to 1000x slower" claim is not substantiated locally (lines 161-165).

## Colab T4 feasibility
- CASIA plus optional COVERAGE is realistic on T4.

## Issues found
- Moderate: Unverified benchmark values appear in a practical implementation guide (lines 121-125).
- Moderate: Platform-speed claims are stronger than the evidence available in the repo (lines 161-165).

## Contradictions with other docs
- Consistent with `Docs/04_Best_Dataset.md`.
- Tension remains with later HF Hub docs that try to make hosting part of the main workflow.

## Recommendations
- Keep the CASIA-first implementation guidance.
- Remove unsupported benchmark and speed claims from the core execution path.

## Severity summary
- Moderate
