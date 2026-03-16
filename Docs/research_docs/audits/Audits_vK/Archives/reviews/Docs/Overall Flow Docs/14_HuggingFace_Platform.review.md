# Review: 14_HuggingFace_Platform.md

Source document path: `Docs/Overall Flow Docs/14_HuggingFace_Platform.md`

Purpose: Explain how to use HF Hub for storage, sharing, and portfolio presentation.

Validity score: 4/10

## Assignment alignment
- Mostly out of scope for the assignment.
- Acceptable only as post-project portfolio material.

## Technical correctness
- The storage/sharing concepts are generally real.
- The "one-line loading" claim using `AutoModel.from_pretrained()` is incorrect for the repo's custom SMP-based model (lines 24-29).
- Recommending cleaned CASIA dataset upload ignores redistribution/licensing concerns (lines 34-37, 58-147).
- Free-storage numbers and similar service details are volatile external claims (lines 22-31, 331-339).

## Colab T4 feasibility
- Uploading artifacts is feasible.
- None of it is needed to satisfy the assignment.

## Issues found
- Major: Wrong model-loading implication for a custom architecture (lines 24-29).
- Major: Cleaned-dataset upload is presented too casually given licensing/redistribution uncertainty (lines 34-37, 58-147).
- Moderate: The whole doc adds scope without improving the notebook deliverable.

## Contradictions with other docs
- Conflicts with the notebook-first scope in `Docs/Assignment.md` and `Docs/Overall Flow Docs/08_The_Code.md`.
- `Docs/Overall Flow Docs/13_Kaggle_vs_HuggingFace.md` already settles the dataset-access decision without needing this extra layer.

## Recommendations
- Demote this to optional portfolio guidance.
- Do not make HF Hub part of the core assignment path.

## Severity summary
- Major
