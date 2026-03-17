# Review: 02_Possible_Solutions.md

Source document path: `Docs/02_Possible_Solutions.md`

Purpose: Compare traditional, CNN, and transformer-based approaches for tampering localization.

Validity score: 6/10

## Assignment alignment
- Helpful as a survey doc.
- Too broad for a single-notebook internship implementation.

## Technical correctness
- The high-level taxonomy is useful.
- Exact performance ranges and feasibility labels are not validated anywhere in the repo (lines 144-145, 172-184).
- The claim that pure RGB models are "insufficient" is too absolute without local ablation evidence (lines 192-196).

## Colab T4 feasibility
- The "practical" path is feasible.
- The SegFormer, TruFor, REFORGE, and multimodal material is mostly research-context only.

## Issues found
- Major: Unverified benchmark table and exact performance numbers drive the reader toward conclusions that the repo does not prove (lines 172-184).
- Moderate: SegFormer is called the best balance for Colab T4, but the rest of the docs standardize on SMP U-Net (lines 144-145).
- Moderate: The backbone-vs-preprocessing ranking is asserted as fact instead of evidence (lines 192-202).

## Contradictions with other docs
- `Docs/05_Best_Solution.md` and `Docs/Overall Flow Docs/04_Architecture.md` settle on U-Net + EfficientNet-B1.
- `Docs/Deep Research.md` later pushes SegFormer-B1 again, re-opening the decision.

## Recommendations
- Keep only three options in the core path: RGB U-Net baseline, U-Net + SRM enhancement, and one appendix-only ambitious option.
- Remove exact benchmark ranges unless they are externally verified.

## Severity summary
- Major
