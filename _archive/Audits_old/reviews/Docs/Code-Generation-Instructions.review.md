# Review: Code-Generation-Instructions.md

Source document path: `Docs/Code-Generation-Instructions.md`

Purpose: Give an LLM or assistant a concise brief for writing PyTorch code.

Validity score: 7/10

## Assignment alignment
- Good modular coding guidance for a notebook implementation.
- Partial because it underspecifies the assignment's image-level detection requirement.

## Technical correctness
- The dataset, training-loop, and metric sections are broadly sound.
- The doc never explicitly states that authentic images should receive zero masks.
- Best-model selection is framed around IoU or Dice only (lines 177-189), while the rest of the repo often uses validation F1.

## Colab T4 feasibility
- Feasible and lightweight.
- Does not drag the project into unnecessary platforms or tooling.

## Issues found
- Moderate: Image-level detection is only optional here, even though the assignment requires it (lines 87-92, 193-205).
- Moderate: Zero-mask handling for authentic images is missing from the dataset brief (lines 29-47).
- Minor: Checkpoint-selection language conflicts with the repo's F1-first evaluation path (lines 177-189).

## Contradictions with other docs
- `Docs/Copilot-Engineering-Instructions.md` explicitly includes authentic zero masks.
- `Docs/06_Best_Practices.md` recommends saving by validation F1.

## Recommendations
- Add the zero-mask rule and a simple image-level score derivation.
- Standardize on one checkpoint-selection metric.

## Severity summary
- Moderate
