# Requirements Matrix

This matrix maps the assignment requirements and the second-audit checklist to `Docs 2/`.

| Requirement | Source of truth | Docs 2 coverage | Audit status | Gap or risk | Required action |
|---|---|---|---|---|---|
| Dataset preparation | `Assignment.md` Section 1 | `01_Assignment_Overview.md`, `02_Dataset_and_Preprocessing.md` | Pass | Core dataset workflow is clearly documented | None |
| Mask pairing | `Assignment.md` Section 1 | `02_Dataset_and_Preprocessing.md`, `03_Data_Pipeline.md` | Pass | Pairing rule is explicit and consistent | None |
| Mask binarization | `Assignment.md` Section 1 | `02_Dataset_and_Preprocessing.md`, `03_Data_Pipeline.md` | Pass | Fixed thresholding is clearly defined | None |
| Dataset validation | User audit checklist | `02_Dataset_and_Preprocessing.md`, `03_Data_Pipeline.md` | Pass | Alignment checks and file-read errors are covered | None |
| Train/val/test split policy | `Assignment.md` Section 1 | `02_Dataset_and_Preprocessing.md`, `10_Project_Timeline.md`, `12_Final_Submission_Checklist.md` | Pass | Split procedure is explicit and reproducible | None |
| Leakage prevention | User audit checklist | `02_Dataset_and_Preprocessing.md`, `11_Limitations_and_Future_Work.md` | Partial | Leakage risk is acknowledged but cannot be fully prevented without source grouping | Keep limitation explicit and persist the split manifest |
| Augmentation policy | `Assignment.md` Section 1 | `03_Data_Pipeline.md`, `10_Project_Timeline.md`, `12_Final_Submission_Checklist.md` | Pass | MVP versus optional augmentation is now aligned | None |
| Segmentation architecture | `Assignment.md` Section 2 | `01_Assignment_Overview.md`, `04_Model_Architecture.md`, `10_Project_Timeline.md` | Pass | U-Net with pretrained encoder is clear and appropriate | None |
| Pretrained backbone choice | User audit checklist | `04_Model_Architecture.md`, `10_Project_Timeline.md` | Pass | ResNet34 is now the locked MVP baseline | None |
| Loss function | Expected architecture | `01_Assignment_Overview.md`, `05_Training_Strategy.md` | Pass | BCE + Dice is consistently documented | None |
| Training loop description | User audit checklist | `05_Training_Strategy.md`, `10_Project_Timeline.md` | Pass | Core training flow is now complete enough to implement | None |
| Optimizer and checkpointing | User audit checklist | `05_Training_Strategy.md`, `12_Final_Submission_Checklist.md` | Pass | AdamW, best/last checkpoints, and resume fields are defined | None |
| Mixed precision usage | User audit checklist | `05_Training_Strategy.md`, `09_Engineering_Practices.md` | Pass | AMP is clearly part of the baseline | None |
| Evaluation metrics | `Assignment.md` Section 3 | `06_Evaluation_Methodology.md`, `12_Final_Submission_Checklist.md` | Pass | IoU, Dice/F1, precision, recall, and image metrics are covered | None |
| Threshold handling | User audit checklist | `04_Model_Architecture.md`, `06_Evaluation_Methodology.md`, `10_Project_Timeline.md`, `12_Final_Submission_Checklist.md` | Partial | Validation-only thresholding is clear, but pixel and image thresholds are not fully separated in the example code | Define `pixel_threshold` and `image_threshold` separately if they differ |
| Validation vs test separation | User audit checklist | `04_Model_Architecture.md`, `06_Evaluation_Methodology.md`, `10_Project_Timeline.md` | Pass | Validation selects thresholds; test reports final metrics | None |
| Prediction visualization | `Assignment.md` Section 3 | `07_Visualization_and_Results.md`, `12_Final_Submission_Checklist.md` | Pass | Binary predicted mask is restored as the required main output | None |
| Overlay visualization | `Assignment.md` Section 3 | `07_Visualization_and_Results.md`, `12_Final_Submission_Checklist.md` | Pass | Overlay output is clear and assignment-aligned | None |
| Robustness testing design | Bonus requirement | `08_Robustness_Testing.md`, `10_Project_Timeline.md` | Partial | Core design is sound, but resize degradation is not fully shown in the evaluation loop | Add one concrete resize-evaluation implementation path |
| Concise and focused documentation | User audit checklist | All `Docs 2/` docs | Pass | The rewritten set is much tighter and avoids prior sprawl | None |
| Avoid unnecessary tools / overengineering | User audit checklist | `01_Assignment_Overview.md`, `09_Engineering_Practices.md`, `11_Limitations_and_Future_Work.md` | Pass | Nonessential tooling is now clearly excluded from the core path | None |
| Avoid speculative performance claims | User audit checklist | `01_Assignment_Overview.md`, `04_Model_Architecture.md`, `08_Robustness_Testing.md` | Pass | Prior unsupported targets and expected robustness drops were removed | None |
| Google Colab feasibility | `Assignment.md` Section 2 | `01_Assignment_Overview.md`, `05_Training_Strategy.md`, `09_Engineering_Practices.md`, notebook note | Partial | Pipeline is realistic, but setup docs omit `kaggle` in the default install and still rely on version-sensitive augmentation APIs | Add `kaggle` to default setup or document it as a prerequisite; pin tested `albumentations` version |
| Single-notebook implementation path | `Assignment.md` Section 4 | `01_Assignment_Overview.md`, `09_Engineering_Practices.md`, `10_Project_Timeline.md`, notebook note | Pass | The docs now support one clear notebook-first path | None |
