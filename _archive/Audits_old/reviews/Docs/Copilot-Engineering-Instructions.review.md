# Review: Copilot-Engineering-Instructions.md

Source document path: `Docs/Copilot-Engineering-Instructions.md`

Purpose: Provide an implementation brief focused on engineering quality and notebook structure.

Validity score: 8/10

## Assignment alignment
- Strong overall alignment with the internship brief.
- One of the cleaner execution-oriented docs.

## Technical correctness
- The zero-mask rule, nearest-neighbor mask resizing, modular dataset class, and visualization requirements are all correct (lines 44-79, 182-211).
- The optional classification head adds scope beyond the simplest assignment path (lines 122-124).
- Blur is included as a robustness augmentation even though other docs correctly warn that strong blur can erase forensic evidence (lines 97-101).

## Colab T4 feasibility
- The core guidance is feasible on T4.
- Nothing here forces overengineering by itself.

## Issues found
- Moderate: The optional classification head can distract from a clean segmentation-first notebook (lines 122-124).
- Minor: Blur augmentation should be weakened or justified more carefully for forensic use (lines 97-101).
- Minor: The metric set omits image-level AUC or accuracy, which other docs rely on (lines 140-156).

## Contradictions with other docs
- Simpler than `Docs/05_Best_Solution.md`, which is a good thing.
- Slightly conflicts with `Docs/Overall Flow Docs/03_Augmentation.md`, which deliberately avoids heavy blur.

## Recommendations
- Keep this as a compact implementation brief.
- Add one image-level metric requirement.
- Treat classification head and blur as optional extras.

## Severity summary
- Moderate
