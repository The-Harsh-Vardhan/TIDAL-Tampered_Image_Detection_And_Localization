# Review: 03_Dataset_Exploration.md

Source document path: `Docs/03_Dataset_Exploration.md`

Purpose: Compare candidate tampering datasets and justify a primary dataset choice.

Validity score: 7/10

## Assignment alignment
- Good coverage of dataset availability, masks, known issues, and bonus-dataset options.
- Strong direct relevance to the assignment.

## Technical correctness
- The dataset profiles are generally sensible.
- The exact benchmark values for CASIA, COVERAGE, NIST16, and IMD2020 are not validated inside the repo and should be treated as `Unverified / likely hallucinated` unless externally checked (lines 63-69, 95-99, 154-157, 174-177, 187-193).
- The claim that HF `datasets` can be "1000x slower" is too strong without direct measurement (lines 205-210).

## Colab T4 feasibility
- The Kaggle-first recommendation is practical.
- CASIA primary plus optional COVERAGE evaluation is realistic on Colab.

## Issues found
- Moderate: Benchmark tables are too exact for unverified secondary-source content (lines 63-69, 187-193).
- Moderate: The platform-speed comparison overstates certainty (lines 205-210).
- Minor: Redistribution or license constraints are not discussed when talking about processed-dataset sharing.

## Contradictions with other docs
- Supports `Docs/04_Best_Dataset.md` well.
- Conflicts with later HF-platform docs that try to turn dataset hosting into part of the main workflow.

## Recommendations
- Keep the issue lists and dataset comparison.
- Drop exact SOTA tables from the implementation path.
- Keep CASIA as the only required dataset and move the rest to optional evaluation.

## Severity summary
- Moderate
