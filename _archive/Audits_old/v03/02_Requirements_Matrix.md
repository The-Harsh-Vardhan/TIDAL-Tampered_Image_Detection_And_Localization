# Requirements Matrix

This matrix maps the audit objectives and assignment requirements to `Docs3/` and `notebooks/tamper_detection_v3.ipynb`.

| Requirement | Source of truth | Docs3 / notebook coverage | Status | Gap or risk | Required action |
|---|---|---|---|---|---|
| Problem definition and expected outputs | User prompt, `Assignment.md` | `00_Master_Report.md`, `01_System_Architecture.md`, notebook overview sections | Pass | Problem and outputs are clearly defined | None |
| Dataset source and tampering types | User prompt, `Assignment.md` | `02_Dataset_and_Preprocessing.md`, notebook discovery section | Pass | CASIA source and tampering types are explicit | None |
| Dataset preprocessing pipeline | User prompt | `01_System_Architecture.md`, `02_Dataset_and_Preprocessing.md`, notebook preprocessing section | Pass | Dynamic discovery, binarization, zero-mask handling, and split manifest are clear | None |
| Train / validation / test split | User prompt | `02_Dataset_and_Preprocessing.md`, `01_System_Architecture.md`, notebook split section | Pass | 85 / 7.5 / 7.5 policy is explicit and reproducible | None |
| Dataset bias / leakage risk | User prompt | `02_Dataset_and_Preprocessing.md`, `00_Master_Report.md` | Partial | Source-group leakage remains a known CASIA limitation | Keep the limitation explicit; do not overclaim generalization |
| Augmentation strategy | User prompt | `04_Training_Strategy.md`, `10_Project_Timeline.md`, notebook loader / robustness sections | Pass | MVP vs optional augmentation is clearly separated | None |
| Segmentation architecture | User prompt, `Assignment.md` | `01_System_Architecture.md`, `03_Model_Architecture.md`, notebook model section | Pass | U-Net + ResNet34 is appropriate for Colab and the task | None |
| Architecture consistency | User prompt | `01_System_Architecture.md`, `03_Model_Architecture.md`, `10_Project_Timeline.md` | Partial | ELA channel-count contradiction remains | Fix the ELA channel count everywhere |
| Training strategy | User prompt | `04_Training_Strategy.md`, notebook optimizer and training sections | Pass | Loss, optimizer, AMP, accumulation, clipping, checkpointing, and early stopping are covered | None |
| Evaluation methodology | User prompt | `05_Evaluation_Methodology.md`, notebook threshold and evaluation sections | Pass | IoU, precision, recall, F1, mixed/tampered-only views, and frozen thresholding are clear | None |
| Robustness testing | User prompt | `06_Robustness_Testing.md`, notebook robustness section | Pass | JPEG, noise, resize, and blur are all covered with implementable paths | Tone down uncited research-context claims |
| Explainability | User prompt | `07_Visualization_and_Explainability.md`, notebook visualization section | Partial | Visualization is strong, but true explainability methods such as attribution or attention maps are missing | Add a real explainability method or rename the doc to reflect its actual scope |
| Experiment tracking | User prompt | `09_Experiment_Tracking.md`, notebook W&B section | Partial | Optional W&B design is good, but the doc setup flow contradicts that optional status | Align the doc with the guarded notebook flow |
| Engineering practices | User prompt | `08_Engineering_Practices.md`, notebook setup and save sections | Partial | One setup command is incorrect if copied literally because the version range is unquoted | Quote the version range in the doc |
| Colab / T4 feasibility | User prompt, `Assignment.md` | `01_System_Architecture.md`, `08_Engineering_Practices.md`, notebook structure | Partial | Feasible overall, but the VRAM number is still a heuristic and setup docs need one fix | Rephrase the VRAM figure as an estimate and fix the install command |
| Prediction visualization | User prompt, `Assignment.md` | `07_Visualization_and_Explainability.md`, notebook visualization section | Pass | Binary predicted mask, overlay, and training curves are clearly required | None |
| Notebook alignment | User prompt | `03_Notebook_Alignment_Check.md`, `notebooks/tamper_detection_v3.ipynb` | Partial | Notebook is strongly aligned overall, but docs-to-notebook drift remains in setup and W&B flow | Fix the two mismatches and keep optional features clearly marked |
| Overengineering / tooling scope | User prompt | `01_System_Architecture.md`, `08_Engineering_Practices.md`, `09_Experiment_Tracking.md`, `10_Project_Timeline.md` | Pass | Extras are phased and kept off the MVP path | None |
| Single-notebook implementation path | `Assignment.md` | `01_System_Architecture.md`, `08_Engineering_Practices.md`, `10_Project_Timeline.md`, notebook itself | Pass | One notebook remains the delivery model throughout | None |
