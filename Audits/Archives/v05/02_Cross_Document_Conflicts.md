# Cross-Document Conflicts

This file lists only real contradictions or implementation drift across `Docs5/`, `notebooks/tamper_detection_v5.ipynb`, and the audited supporting material.

## Summary

No material documentation-versus-notebook conflicts were found in the current v5 state. The earlier v4-era drift on artifact names, notebook structure, threshold policy, and W&B behavior has been resolved.

## Remaining Drift

| Conflicting artifacts | Conflict | Why it matters | Recommended resolution |
|---|---|---|---|
| `Docs5/08_Engineering_Practices.md`, `Docs5/12_Complete_Notebook_Structure.md`, and `notebooks/tamper_detection_v5.ipynb` | The docs correctly describe a Drive-or-local artifact path, but the notebook's final disabled-W&B print message still says artifacts were saved "to Google Drive" even when the runtime falls back to the local artifact directory. | This is a minor handoff issue. It can mislead users reviewing a local run and looking in the wrong place for outputs. | Make the final print message report `CHECKPOINT_DIR` directly or branch the message on the actual storage target. |

## Explicitly Checked and Found Aligned

- `results_summary_v5.json` is the documented and implemented results artifact name.
- The notebook is documented as `tamper_detection_v5.ipynb`, not v4.
- `Docs5/12_Complete_Notebook_Structure.md` matches the v5 notebook's 17 sections and 61 cells.
- Threshold selection is validation-only and reused for test and robustness evaluation.
- W&B is integrated throughout the notebook and remains guarded behind `USE_WANDB`.
- The image-level score is documented as top-k mean rather than `max(prob_map)`.
