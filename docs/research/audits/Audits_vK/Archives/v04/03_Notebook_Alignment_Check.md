# Notebook Alignment Check

This file checks structural alignment between `Docs4/` and `notebooks/tamper_detection_v4.ipynb`. It is not a runtime validation report. Alignment does not prove the notebook executes successfully end to end.

| Notebook area | Notebook evidence | Docs4 anchor | Status | Notes |
|---|---|---|---|---|
| Setup and environment | `## 1. Setup & Environment`, quoted install line `!pip install -q kaggle segmentation-models-pytorch "albumentations>=1.3.1,<2.0"` | `08_Engineering_Practices.md`, `01_System_Architecture.md` | Aligned | Quoted `albumentations` range and Kaggle setup now match the docs. |
| W&B setup and guarded flow | `USE_WANDB`, guarded `wandb.login()`, guarded `wandb.log`, guarded `wandb.summary.update`, guarded `wandb.finish()` | `09_Experiment_Tracking.md`, `04_Training_Strategy.md` | Aligned | Behavior is aligned. The only drift is that notebook v4 integrates W&B throughout instead of keeping a standalone section. |
| Dataset download and discovery | Kaggle download marker, dynamic discovery, split manifest path, `dimension_mismatch` exclusion path | `02_Dataset_and_Preprocessing.md` | Partially aligned | The notebook implements the promised dimension check, but the Docs4 code snippet does not show it. |
| Preprocessing and split | `## 3. Preprocessing & Data Split`, `split_manifest.json` | `02_Dataset_and_Preprocessing.md`, `01_System_Architecture.md` | Aligned | The documented split-and-persist workflow is present. |
| Dataset class and loaders | `## 4. Dataset Class & DataLoaders` | `04_Training_Strategy.md`, `01_System_Architecture.md` | Aligned | MVP loader stage and transform path are present. |
| Model definition | `## 5. Model Definition`, `smp.Unet(` | `03_Model_Architecture.md` | Aligned | The baseline SMP U-Net path is implemented as documented. |
| Loss and optimizer | `## 6. Loss Function & Optimizer`, `BCEDiceLoss`, AdamW | `04_Training_Strategy.md` | Aligned | Matches the documented optimizer and loss choices. |
| Training loop and checkpointing | `## 8. Training Loop`, `best_model.pt`, `last_checkpoint.pt`, periodic checkpoints | `04_Training_Strategy.md` | Aligned | Checkpoint names and guarded logging behavior are present. |
| Threshold selection | `find_best_threshold(...)`, `## 9. Threshold Selection` | `05_Evaluation_Methodology.md`, `03_Model_Architecture.md` | Aligned | Validation-only threshold sweep is present. |
| Evaluation and reporting | `## 10. Evaluation on Test Set`, mixed/tampered-only reporting, `max(prob_map)` image metrics | `05_Evaluation_Methodology.md` | Aligned | The main reporting views match the docs. |
| Visualization | `## 11. Visualization`, saved `training_curves.png`, `prediction_grid.png`, W&B image logging | `07_Visualization_and_Results.md` | Aligned | Required visuals are implemented. Optional feature-map inspection and ELA visualization are not required for MVP alignment. |
| Robustness testing | `## 12. Robustness Testing (Bonus - Phase 3)`, `ResizeDegradationDataset`, `ImageCompression`, `GaussNoise`, `GaussianBlur` | `06_Robustness_Testing.md` | Aligned | The documented degradation suite is structurally implemented. |
| Save and export | `## 13. Save & Export`, artifact logging, `results_summary_v4.json` | `08_Engineering_Practices.md`, `09_Experiment_Tracking.md`, `10_Project_Timeline.md` | Partially aligned | Export exists, but Docs4 says `results_summary.json` while notebook v4 writes `results_summary_v4.json`. |

## Notebook Markers Confirmed

- Quoted `albumentations` install
- Kaggle download command
- `USE_WANDB`
- `wandb.login()`
- `smp.Unet`
- `find_best_threshold`
- `best_model.pt`
- `last_checkpoint.pt`
- `ResizeDegradationDataset`

## Optional Feature Note

Optional Phase 2 or Phase 3 features do not count as mismatches when omitted, as long as Docs4 keeps them outside the MVP path. Notebook v4 is already stronger than notebook v3 because W&B support is integrated throughout while still remaining optional.
