# Review: 05_Resource_Constraints.md

Source document path: `Docs/Overall Flow Docs/05_Resource_Constraints.md`

Purpose: Explain how to keep training within Colab T4 memory and time limits.

Validity score: 6/10

## Assignment alignment
- Directly relevant to the Colab T4 requirement.

## Technical correctness
- AMP, gradient accumulation, and basic DataLoader tuning are good guidance.
- The memory budget and runtime numbers are estimates but are written with too much certainty (lines 24-48).
- "Ideal effective batch size 16" is a heuristic, not a requirement (lines 99-103).
- The gradient-checkpointing suggestion is version-sensitive and may not exist for all SMP/timm encoders (lines 253-259).

## Colab T4 feasibility
- The baseline advice is feasible.
- The doc is most useful when interpreted as approximate tuning guidance, not a guarantee.

## Issues found
- Moderate: Resource numbers are not measured from an actual notebook run (lines 24-48).
- Moderate: Some recommendations are more absolute than they should be (lines 99-103).
- Minor: The checkpointing API assumption may break depending on encoder/version (lines 253-259).

## Contradictions with other docs
- `Docs/Overall Flow Docs/17_Training_Optimisation.md` adds many extra knobs beyond what this doc already covers.

## Recommendations
- Keep AMP, sensible batch sizing, and checkpointing.
- Label all memory/time numbers as estimates.
- Do not escalate to advanced optimization unless the baseline actually fails.

## Severity summary
- Moderate
