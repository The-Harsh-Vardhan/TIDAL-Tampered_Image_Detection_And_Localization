# Review: 12_WandB_Sweeps.md

Source document path: `Docs/Overall Flow Docs/12_WandB_Sweeps.md`

Purpose: Describe hyperparameter search using W&B Sweeps.

Validity score: 5/10

## Assignment alignment
- Low priority for this assignment.
- Clear overengineering risk.

## Technical correctness
- The general idea of a sweep is correct.
- The sample code defines `scheduler = ... T_max=SWEEP_EPOCHS` before `SWEEP_EPOCHS` is assigned (lines 149 and 155), so the snippet is not runnable as written.
- The time budget assumes 20 short runs plus final training, which is excessive for the project scope (lines 241-255).

## Colab T4 feasibility
- A tiny mini-sweep is possible.
- The full recommended workflow is not a good default for a one-week Colab assignment.

## Issues found
- Critical: Sample code bug due to `SWEEP_EPOCHS` being used before definition (lines 149, 155).
- Major: The proposed sweep budget is overkill for the assignment (lines 241-255).
- Moderate: Parallel-agent / multi-Colab framing assumes more infrastructure than many candidates will use (lines 47-50, 214-219).

## Contradictions with other docs
- Conflicts with the repo's repeated "single notebook, practical solution" theme.
- `Docs/Overall Flow Docs/11_Weights_And_Biases.md` already makes W&B optional; this doc pushes deeper optionality into the critical path.

## Recommendations
- Remove full sweep guidance from the core doc set.
- If tuning is needed, suggest 2-3 manual ablations or a 3-5 run mini-sweep only after the baseline is stable.

## Severity summary
- Critical
