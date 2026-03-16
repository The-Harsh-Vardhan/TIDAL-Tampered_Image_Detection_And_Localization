# Review: 17_Training_Optimisation.md

Source document path: `Docs/Overall Flow Docs/17_Training_Optimisation.md`

Purpose: Describe advanced optimization techniques beyond the basic T4 setup.

Validity score: 5/10

## Assignment alignment
- Mostly optional.
- Too advanced for the critical path.

## Technical correctness
- Some tips are fine, especially `non_blocking=True` and inference-mode guidance.
- The DALI sample is unsafe for segmentation because image and mask readers are shuffled independently (lines 58-59), which can desynchronize pairs.
- `torch.compile()` is described as near-free speedup, but the doc underplays compile overhead and instability for notebook workflows (lines 117-159).

## Colab T4 feasibility
- Small parts are feasible.
- DALI and many advanced knobs are not good defaults for this assignment.

## Issues found
- Major: The DALI example can break image-mask alignment (lines 58-59).
- Moderate: The document recommends many low-ROI optimizations for a project that first needs a clean baseline (lines 163-179, 309-340).
- Minor: Some snippets are Linux/Colab-version sensitive, such as the multiprocessing context advice (lines 167-179).

## Contradictions with other docs
- `Docs/Overall Flow Docs/05_Resource_Constraints.md` already covers the optimizations that actually matter.
- This doc reintroduces complexity that other docs try to avoid.

## Recommendations
- Keep only AMP, `set_to_none=True`, `non_blocking=True`, and inference-mode notes.
- Move DALI, channels-last, and `torch.compile()` to appendix-only status.

## Severity summary
- Major
