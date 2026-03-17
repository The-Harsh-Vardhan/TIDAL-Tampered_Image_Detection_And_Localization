# Review: 04_Architecture.md

Source document path: `Docs/Overall Flow Docs/04_Architecture.md`

Purpose: Provide the implementation guide for the selected model and loss.

Validity score: 6/10

## Assignment alignment
- Covers the right assignment area.
- Too opinionated for a repo without working code.

## Technical correctness
- The overall U-Net-based design is plausible.
- The SRM code is explicitly placeholder-heavy and pads/duplicates kernels instead of showing a real 30-filter bank (lines 81-133).
- The decoder sketch should not be treated as a literal SMP implementation contract (lines 192-200).
- The loss section again hard-codes edge supervision as part of the default path (lines 266-318).

## Colab T4 feasibility
- A U-Net on T4 is feasible.
- The extra SRM branch and edge loss increase complexity before the baseline exists.

## Issues found
- Major: Placeholder SRM implementation is presented too close to production guidance (lines 81-133).
- Moderate: Strong claims about SRM as "essential" are not backed by repo evidence (lines 17-20, 30-34).
- Moderate: The default loss stack is more complex than the assignment requires (lines 266-318).

## Contradictions with other docs
- `Docs/Code-Generation-Instructions.md` and `Docs/Copilot-Engineering-Instructions.md` both support a simpler BCE + Dice path.
- `Docs/Deep Research.md` pushes an even more complex SegFormer-based path.

## Recommendations
- Freeze the v1 architecture at RGB U-Net with a proven encoder.
- Treat SRM as a later ablation and edge loss as optional.

## Severity summary
- Major
