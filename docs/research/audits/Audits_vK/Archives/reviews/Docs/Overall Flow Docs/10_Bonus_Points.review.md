# Review: 10_Bonus_Points.md

Source document path: `Docs/Overall Flow Docs/10_Bonus_Points.md`

Purpose: Define optional robustness and generalization work for bonus credit.

Validity score: 6/10

## Assignment alignment
- Correctly targets the bonus part of the brief.
- Sometimes overstates bonus work as if it were core scope.

## Technical correctness
- JPEG/noise/resize robustness testing is useful.
- The doc calls robustness testing "essential" even though it is bonus work (lines 15-20).
- The tampering-type breakdown code uses `_S_` for splicing and `_C_` for copy-move (lines 265-268), which does not match the CASIA naming used elsewhere in the repo.
- COVERAGE setup is too vague to be fully reproducible (lines 175-186).

## Colab T4 feasibility
- The JPEG/noise/resize tests are feasible.
- COVERAGE is feasible only if the dataset path and mask naming are made precise.

## Issues found
- Major: CASIA tampering-type parsing is wrong in the sample code (lines 265-268).
- Moderate: Bonus work is framed too strongly (lines 15-20).
- Moderate: COVERAGE setup is under-specified (lines 175-186).

## Contradictions with other docs
- `Docs/Overall Flow Docs/01_Dataset_Choice.md` and `Docs/03_Dataset_Exploration.md` correctly describe `Tp_D` and `Tp_S`.

## Recommendations
- Keep bonus scope clearly separate from the baseline.
- Fix the tampering-type parsing logic and keep COVERAGE optional.

## Severity summary
- Major
