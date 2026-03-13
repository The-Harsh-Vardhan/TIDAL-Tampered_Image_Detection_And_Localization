# 03_Model_Architecture.md Review

## Purpose

Defines the MVP U-Net baseline, the SMP API contract, and optional ELA/SRM extensions.

## Accuracy Score

`8/10`

## What Improved Since Audit 2

- Locks the MVP image-level score to `max(prob_map)` and the single-threshold policy.
- Keeps the SMP API explicit and correct.
- Pushes SRM into clear Phase 3 experimental territory.
- Adds a concrete ELA option instead of only mentioning future forensic features abstractly.

## Issues Found

- The `tamper_score = prob_map.view(B, -1).max(dim=1).values` snippet uses `B` without defining it.
- The ELA rationale invokes research literature, but the doc does not cite any specific source from the repo.
- The ELA path is technically plausible, but it increases complexity and reduces the benefit of pretrained weights in the documented implementation path.

## Suggested Improvements

- Replace `B` with `prob_map.size(0)` or another explicit batch-dimension expression.
- Either cite the supporting research directly or rephrase the ELA rationale as a tentative optional hypothesis rather than a literature-backed claim.
