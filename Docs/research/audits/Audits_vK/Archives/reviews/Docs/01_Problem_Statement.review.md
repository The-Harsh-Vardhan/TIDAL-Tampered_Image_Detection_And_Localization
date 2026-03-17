# Review: 01_Problem_Statement.md

Source document path: `Docs/01_Problem_Statement.md`

Purpose: Frame the assignment and translate it into concrete engineering tasks.

Validity score: 7/10

## Assignment alignment
- Strong alignment on the dual objective: image-level detection plus pixel-level localization.
- Partial misalignment because optional design choices are written like required tasks.

## Technical correctness
- The forensic background is reasonable and useful.
- The engineering-task table becomes too prescriptive by treating a dual-stream architecture and a specific loss stack as defaults rather than choices (lines 118-120).
- The metric list grows beyond what the assignment needs and is not prioritized clearly (lines 125-126).

## Colab T4 feasibility
- Feasible if reduced to one dataset and one model.
- Scope becomes heavier if COVERAGE and more advanced model work are treated as mandatory (lines 139-140).

## Issues found
- Moderate: Architecture direction is hard-coded too early (lines 118-120).
- Moderate: The metric scope expands to AUC and MCC without a clear primary metric hierarchy (lines 125-126).
- Minor: Bonus work is phrased like a core engineering task (lines 139-140).

## Contradictions with other docs
- `Docs/Code-Generation-Instructions.md` and `Docs/Copilot-Engineering-Instructions.md` allow a simpler BCE + Dice path.
- `Docs/05_Best_Solution.md` later narrows the architecture even further to U-Net + EfficientNet-B1 + SRM.

## Recommendations
- Keep the problem framing and task breakdown.
- Rephrase model, loss, and COVERAGE items as options.
- Prioritize IoU, Dice/F1, and one image-level detection metric family.

## Severity summary
- Moderate
