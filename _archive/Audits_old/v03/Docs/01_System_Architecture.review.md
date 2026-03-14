# 01_System_Architecture.md Review

## Purpose

Defines the end-to-end system pipeline, major design decisions, and high-level Colab deployment shape.

## Accuracy Score

`7/10`

## What Improved Since Audit 2

- Locks the MVP image-level detection rule to `max(prob_map)`.
- Locks the system to a single validation-selected threshold for both pixel and image decisions.
- Adds blur robustness and optional W&B tracking into the overall system view.
- Gives a much clearer full-pipeline map than the prior doc set.

## Issues Found

- The main remaining cross-document contradiction is here: the diagram says Phase 2 ELA changes `in_channels` to 6, while `03_Model_Architecture.md` and `10_Project_Timeline.md` describe ELA as a 4th input channel.
- The VRAM estimate (`~6 GB with AMP`) is still an unmeasured heuristic and should not be treated as a verified budget.
- The design-decision table is otherwise solid, but it inherits the known fragility of `max(prob_map)` for image-level detection.

## Suggested Improvements

- Fix the ELA channel count and keep it consistent everywhere.
- Rephrase the VRAM line as an approximate expectation that must be confirmed in the notebook.
