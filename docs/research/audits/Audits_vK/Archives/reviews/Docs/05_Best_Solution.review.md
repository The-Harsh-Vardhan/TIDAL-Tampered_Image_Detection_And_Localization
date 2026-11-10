# Review: 05_Best_Solution.md

Source document path: `Docs/05_Best_Solution.md`

Purpose: Present the final architecture, loss, and training strategy.

Validity score: 6/10

## Assignment alignment
- Covers architecture, loss, training, and inference in one place.
- Too confident for a repo that contains no implementation or ablation.

## Technical correctness
- The high-level model structure is plausible.
- SRM is described as the "single biggest performance boost" without local evidence (lines 13-16).
- Expected performance targets and SOTA comparisons should be treated as `Unverified / likely hallucinated` (lines 246-253).
- The "any pixel above threshold means tampered" image-level rule is too sensitive to isolated false positives (lines 220-225).

## Colab T4 feasibility
- A U-Net-based model is feasible on T4.
- Adding SRM, edge loss, and ambitious upgrade paths before the baseline exists raises risk.

## Issues found
- Major: The document turns a specialized stack into the default submission path without proving that the added complexity is necessary (lines 5-16, 119-150).
- Moderate: Performance targets look more like benchmark lore than engineering acceptance criteria (lines 246-253).
- Moderate: The inference rule for image-level detection is brittle (lines 220-225).

## Contradictions with other docs
- `Docs/Code-Generation-Instructions.md` and `Docs/Copilot-Engineering-Instructions.md` both support a simpler BCE + Dice baseline.
- `Docs/Deep Research.md` later suggests SegFormer-B1 as the "optimal" T4 path instead.

## Recommendations
- Replace the default stack with a simpler RGB U-Net baseline plus BCE + Dice.
- Keep SRM and edge loss as ablations or bonus enhancements only.
- Remove benchmark targets unless they are externally verified.

## Severity summary
- Major
