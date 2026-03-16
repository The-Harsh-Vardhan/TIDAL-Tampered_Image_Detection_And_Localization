# Docs8 vs Audit7 Pro Regression and Improvement Map

This map answers the only comparison question that matters:

Did `Docs8` actually respond to `Audit7 Pro`, or did it just become better at sounding self-aware?

## Comparison Table

| Audit7 Pro criticism | Audit7 Pro evidence | Docs8 response | Status | Audit note |
|---|---|---|---|---|
| Single Colab deliverable was not clean | `Audit7 Pro/00_Master_Report.md:19-25` | `Docs8` now marks single Colab notebook as partial and admits Colab is unverified | Acknowledged, not fixed | Better honesty, same compliance gap |
| CASIA was falsely treated as assignment-mandated | `Audit7 Pro/00_Master_Report.md:33-45` | `Docs8` stops the explicit old claim, but still says CASIA is the assignment's expected dataset | Partially acknowledged, still unresolved | Softer wording, same instinct |
| Project lacked clean problem framing and used a weak detector bolt-on | `Audit7 Pro/00_Master_Report.md:47-61` | `Docs8` admits heuristic image-level detection and proposes a learned head later | Acknowledged, not fixed | More honest, still weak on detection |
| CASIA is old, narrow, and leakage-prone | `Audit7 Pro/00_Master_Report.md:63-79` | `Docs8` openly documents CASIA age, leak-risk gaps, and need for pHash checks | Improved acknowledgement, not fixed | Real progress in honesty |
| U-Net/ResNet34 was a convenience baseline, not a strong forensic design | `Audit7 Pro/00_Master_Report.md:81-97` | `Docs8` explicitly rebrands it as a stable baseline and defers stronger alternatives | Acknowledged, not fixed | Cleaner architecture story, same baseline |
| Training logic lacked `pos_weight`, per-sample Dice, and stronger imbalance handling | `Audit7 Pro/00_Master_Report.md:99-115` | `Docs8` centers these issues and writes concrete fixes | Acknowledged, not fixed | Good diagnosis, no executed correction yet |
| Evaluation was inflated by mixed-set metrics and unstable image scoring | `Audit7 Pro/00_Master_Report.md:117-133` | `Docs8` now leads with tampered-only critique and documents `max(prob_map)` honestly | Improved acknowledgement, partially fixed in docs only | Reporting intent improved; executed stack still old |
| Validation experiments were documentation theater | `Audit7 Pro/00_Master_Report.md:135-149` | `Docs8` converts some of this into future experiments and monitoring plans | Acknowledged, not fixed | Still mostly aspirational |
| Engineering maturity was oversold by notebook scaffolding | `Audit7 Pro/00_Master_Report.md:169-185` | `Docs8` keeps practical engineering credit but still relies on plan/checklist language | Partially acknowledged | More sober tone, same notebook-first reality |
| Research awareness existed without implementation follow-through | `Audit7 Pro/00_Master_Report.md:187-203` | `Docs8` references more alternatives and sequences them into v9+ roadmap | Acknowledged, not fixed | Better roadmap, same implementation gap |

## Issues Docs8 Genuinely Improved

These are real improvements, not cosmetic ones:

1. **Mixed-set inflation is no longer buried.**
   `Docs8/05_Evaluation_Methodology_Evolution.md:54-77` makes the metric distortion explicit.

2. **Copy-move weakness is treated as a central failure mode.**
   `Docs8/11_Training_Failure_Cases.md:52-95` gives the problem a proper failure taxonomy instead of treating it as a minor footnote.

3. **Architecture humility improved.**
   `Docs8/03_Model_Architecture_Evolution.md:76-89` explicitly stops pretending U-Net/ResNet34 is a forensic optimum.

4. **Training flaws are identified precisely.**
   `Docs8/04_Training_Strategy_Evolution.md:28-39` and `:89-186` are far more technically coherent than the old defensive story.

5. **The project now uses actual `Run01` evidence to guide design discussion.**
   That is a major upgrade in seriousness over purely speculative documentation.

## Issues Docs8 Only Acknowledged

These are improvements in honesty, not in project state:

1. Single Colab notebook still not verified.
2. `pos_weight`, scheduler, per-sample Dice, and stronger augmentation still not integrated.
3. Heuristic image-level detection still not replaced.
4. Boundary F1 and mask-size stratification still not executed.
5. Duplicate/content leakage checks still not run.
6. Architecture comparisons still deferred.
7. Cross-dataset and multi-seed validation still future work.

## Issues That Remain Unresolved

1. Assignment compliance remains only partial because the final artifact is still not a verified single Colab submission.
2. Copy-move remains weak on the current evidence.
3. RGB-only input remains a structural limitation.
4. The current training and evaluation evidence still comes from the pre-v8 state.
5. The project still cannot cleanly claim that it truly solved both detection and localization at a strong level.

## New Docs8 Credibility Problems

These are not leftovers from `Docs7`. They are fresh problems introduced by how `Docs8` presents itself.

### 1. Claimed lineage versus cited lineage

`Docs8/00_Project_Evolution_Summary.md:5-11` says `Docs8` bridges `Docs7`, `Audit7 Pro`, and `Run01`.

`Docs8/10_References.md:13-18` instead formalizes `Docs7`, `Audit6 Pro`, and `Audit 6.5 Notebook` as the main upstream internal references.

That inconsistency makes the documentation history look selective.

### 2. "No bugs or data leakage detected" overshoots the evidence

`Docs8/00_Project_Evolution_Summary.md:78` is too strong given:

- pending duplicate checks (`Docs8/02_Dataset_Evolution.md:94-96`)
- path-only leak evidence (`Docs8/01_Assignment_Requirement_Alignment.md:17-18`)
- acknowledged `cudnn.benchmark` contradiction (`Docs8/00_Project_Evolution_Summary.md:96`, `Docs8/04_Training_Strategy_Evolution.md:164-171`)

### 3. Compliance scoring is more generous than the actual gaps justify

`Docs8/01_Assignment_Requirement_Alignment.md:29-32` marks runtime-on-Colab-class hardware as met while `:57-67` still marks the actual single Colab notebook deliverable as partial.

That split is not insane, but it is generous enough to warrant audit pressure.

## Bottom Line

`Docs8` is not a regression from `Audit7 Pro`. It is a better response than `Docs7` was capable of producing.

But the improvement is mostly this:

- fewer false claims
- better diagnosis
- stronger prioritization

It is **not** this:

- corrected final artifact
- validated v8 pipeline
- repaired assignment compliance

So the honest status is:

`Docs8` substantially improves the narrative and the self-critique, while leaving the underlying technical submission only partially repaired.
