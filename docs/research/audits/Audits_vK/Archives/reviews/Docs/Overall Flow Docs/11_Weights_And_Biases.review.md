# Review: 11_Weights_And_Biases.md

Source document path: `Docs/Overall Flow Docs/11_Weights_And_Biases.md`

Purpose: Explain how to use W&B for experiment tracking.

Validity score: 6/10

## Assignment alignment
- Optional enhancement only.
- Not required for a successful internship submission.

## Technical correctness
- The basic logging ideas are fine.
- The doc overstates necessity by ending with a strong "Use it" verdict (lines 239-249).
- Free-tier/storage statements are volatile and unverified (lines 29-35).

## Colab T4 feasibility
- Technically feasible, but it introduces an external dependency the assignment does not require.

## Issues found
- Moderate: W&B is framed like a near-default requirement rather than an optional convenience (lines 239-249).
- Minor: Storage and plan details are time-sensitive external claims (lines 29-35).

## Contradictions with other docs
- `Docs/Overall Flow Docs/09_Assets.md` already gives a complete submission path without W&B.
- `Docs/Overall Flow Docs/12_WandB_Sweeps.md` compounds the same optional tooling into overengineering.

## Recommendations
- Keep W&B as optional.
- Ensure the notebook remains complete and readable without external dashboards.

## Severity summary
- Moderate
