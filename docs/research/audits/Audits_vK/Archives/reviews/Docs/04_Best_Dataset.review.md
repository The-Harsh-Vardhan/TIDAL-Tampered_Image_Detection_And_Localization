# Review: 04_Best_Dataset.md

Source document path: `Docs/04_Best_Dataset.md`

Purpose: Lock the dataset choice and describe the cleaning plan.

Validity score: 8/10

## Assignment alignment
- Strong alignment with the brief.
- One of the better docs in the repo.

## Technical correctness
- The cleaning guidance is practical and usable.
- Two recurring unsupported claims remain: the "1000x slower" HF comparison and the mask-provenance language around the Kaggle version (lines 43-46).
- The later training-subset suggestion is unnecessary for a 5K-image project (lines 137-140).

## Colab T4 feasibility
- CASIA primary training on T4 is realistic.
- Optional COVERAGE evaluation is also realistic if kept small.

## Issues found
- Moderate: Speed and provenance claims are stronger than the repo can justify (lines 43-46).
- Moderate: Suggesting a training subset throws away usable data and can create needless inconsistency (lines 137-140).
- Minor: Exact target ranges from published SOTA are not needed for this document (lines 35-38).

## Contradictions with other docs
- `Docs/Dataset Selection.md` recommends a different split policy.
- `Docs/Dataset.md` and `Docs/Overall Flow.md` push balanced subsets more directly than this doc should.

## Recommendations
- Keep CASIA v2.0 as the final dataset choice.
- Use the full cleaned dataset unless a measured runtime constraint forces otherwise.
- Remove unsupported platform-performance language.

## Severity summary
- Moderate
