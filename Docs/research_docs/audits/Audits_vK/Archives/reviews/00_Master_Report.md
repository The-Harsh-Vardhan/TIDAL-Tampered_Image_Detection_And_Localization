# Audit Review Master Report

This audit reviews 35 Markdown documents under `Docs/`. It is documentation-only: the repo does not currently contain a Colab notebook or implementation to validate against the written guidance.

## 1. Overall Project Assessment

The documentation is technically mixed. The core path is mostly sound: use CASIA v2.0, pair images with masks, binarize masks, run a segmentation model on Colab, evaluate with overlap metrics, and show predicted masks. That is a valid assignment strategy.

The problem is that the doc set is much larger and more confident than the assignment needs. It contains unsupported benchmark numbers, over-specific architecture claims, multiple optional platform/tooling digressions, and several sample-code bugs. As written, the docs are not yet a clean execution plan for a single Colab notebook. After pruning, the project is suitable for the internship assignment.

## 2. Document-by-Document Review

Detailed findings live in the linked per-file reviews.

### Core, research, and planning docs

| Document | Purpose | Score | Top issue | Review |
|---|---|---:|---|---|
| `01_Problem_Statement.md` | Problem framing and task map | 7 | Treats architecture/loss choices as mandatory | [Review](Docs/01_Problem_Statement.review.md) |
| `02_Possible_Solutions.md` | Solution-space survey | 6 | Unverified performance tables and strong ranking claims | [Review](Docs/02_Possible_Solutions.review.md) |
| `03_Dataset_Exploration.md` | Dataset comparison | 7 | Unverified benchmark tables and platform-speed claims | [Review](Docs/03_Dataset_Exploration.review.md) |
| `04_Best_Dataset.md` | Final dataset selection | 8 | Good direction, but still relies on unsupported external claims | [Review](Docs/04_Best_Dataset.review.md) |
| `05_Best_Solution.md` | Final architecture and loss choice | 6 | Over-prescriptive U-Net + SRM + Edge-loss default | [Review](Docs/05_Best_Solution.review.md) |
| `06_Best_Practices.md` | Engineering standards | 6 | Metric definitions drift and older AMP API examples | [Review](Docs/06_Best_Practices.review.md) |
| `07_Industry_Relevance.md` | SOTA and positioning | 4 | Heavy use of unverified named models and exact scores | [Review](Docs/07_Industry_Relevance.review.md) |
| `Assignment.md` | Assignment source of truth | 9 | Minor encoding corruption only | [Review](Docs/Assignment.review.md) |
| `Code-Generation-Instructions.md` | Code-writing brief | 7 | Omits image-level detection handling and zero-mask rule | [Review](Docs/Code-Generation-Instructions.review.md) |
| `Copilot-Engineering-Instructions.md` | Engineering execution brief | 8 | Small scope creep from optional classification head and blur | [Review](Docs/Copilot-Engineering-Instructions.review.md) |
| `Dataset Selection.md` | Short dataset recommendation | 6 | Conflicting split recommendation and broken formatting | [Review](Docs/Dataset Selection.review.md) |
| `Dataset.md` | Kaggle vs HF recommendation | 6 | Unnecessary subsetting and unsupported speed claim | [Review](Docs/Dataset.review.md) |
| `Deep Research.md` | Research-heavy background note | 4 | Citation-heavy and not trustworthy enough for implementation decisions | [Review](Docs/Deep Research.review.md) |
| `Overall Flow.md` | One-page project sequence | 6 | Adds extra optimizers/features and has formatting breakage | [Review](Docs/Overall Flow.review.md) |
| `Tips.md` | Extra guidance | 1 | Empty file | [Review](Docs/Tips.review.md) |

### Overall Flow docs

| Document | Purpose | Score | Top issue | Review |
|---|---|---:|---|---|
| `01_Dataset_Choice.md` | Dataset implementation guide | 7 | Good core guidance but unsupported benchmark/platform claims | [Review](Docs/Overall Flow Docs/01_Dataset_Choice.review.md) |
| `02_Data_Pipeline.md` | Pairing, cleaning, and splits | 8 | Missing source-leakage / dedup control | [Review](Docs/Overall Flow Docs/02_Data_Pipeline.review.md) |
| `03_Augmentation.md` | Augmentation policy | 7 | Some transforms and claims are stronger than needed | [Review](Docs/Overall Flow Docs/03_Augmentation.review.md) |
| `04_Architecture.md` | Architecture implementation guide | 6 | Placeholder SRM implementation presented as full solution | [Review](Docs/Overall Flow Docs/04_Architecture.review.md) |
| `05_Resource_Constraints.md` | T4 optimization guide | 6 | Overconfident memory/runtime estimates | [Review](Docs/Overall Flow Docs/05_Resource_Constraints.review.md) |
| `06_Performance_Metrics.md` | Evaluation guide | 6 | Pixel-metric inflation risk and Oracle-F1 misuse risk | [Review](Docs/Overall Flow Docs/06_Performance_Metrics.review.md) |
| `07_Visual_Results.md` | Visualization guide | 8 | Mostly good; should match assignment outputs more literally | [Review](Docs/Overall Flow Docs/07_Visual_Results.review.md) |
| `08_The_Code.md` | Notebook blueprint | 7 | Single notebook is at risk of becoming too large | [Review](Docs/Overall Flow Docs/08_The_Code.review.md) |
| `09_Assets.md` | Deliverable and sharing guide | 7 | Practical, but some estimates are unverified | [Review](Docs/Overall Flow Docs/09_Assets.review.md) |
| `10_Bonus_Points.md` | Bonus-evaluation guide | 6 | Contains a CASIA tampering-type parsing bug | [Review](Docs/Overall Flow Docs/10_Bonus_Points.review.md) |
| `11_Weights_And_Biases.md` | Experiment tracking guide | 6 | Treated as more necessary than it really is | [Review](Docs/Overall Flow Docs/11_Weights_And_Biases.review.md) |
| `12_WandB_Sweeps.md` | Hyperparameter search guide | 5 | Sample code bug and clear overengineering for the assignment | [Review](Docs/Overall Flow Docs/12_WandB_Sweeps.review.md) |
| `13_Kaggle_vs_HuggingFace.md` | Dataset-source decision guide | 7 | Some environment and HF dataset examples are speculative | [Review](Docs/Overall Flow Docs/13_Kaggle_vs_HuggingFace.review.md) |
| `14_HuggingFace_Platform.md` | HF Hub storage/sharing guide | 4 | Wrong loading guidance for custom SMP model and scope drift | [Review](Docs/Overall Flow Docs/14_HuggingFace_Platform.review.md) |
| `15_Model_Checkpoints.md` | Checkpointing guide | 8 | Strong doc, only mildly over-detailed | [Review](Docs/Overall Flow Docs/15_Model_Checkpoints.review.md) |
| `16_DuckDB_Cache_DynamoDB.md` | Data-management tech assessment | 6 | Verdict is good, but sample caching code is broken | [Review](Docs/Overall Flow Docs/16_DuckDB_Cache_DynamoDB.review.md) |
| `17_Training_Optimisation.md` | Advanced optimization guide | 5 | DALI example can desynchronize image-mask pairs | [Review](Docs/Overall Flow Docs/17_Training_Optimisation.review.md) |
| `18_Generalised_Reusable_Scripts.md` | Refactoring/abstraction guide | 7 | Helpful ideas, but too framework-like for a notebook project | [Review](Docs/Overall Flow Docs/18_Generalised_Reusable_Scripts.review.md) |
| `19_HF_Deployment.md` | Demo deployment guide | 4 | Placeholder app code and major scope drift | [Review](Docs/Overall Flow Docs/19_HF_Deployment.review.md) |
| `20_DataBricks.md` | Databricks assessment | 6 | Mostly accurate but irrelevant to the assignment | [Review](Docs/Overall Flow Docs/20_DataBricks.review.md) |

## 3. Major Architectural Risks

- Unsupported benchmark numbers are used as design evidence across multiple docs. This makes the architecture selection look more validated than it really is.
- The documentation treats a forensic-specialized stack (`SRM + U-Net + EfficientNet-B1 + Edge loss`) as the default even though the repo contains no ablation or implementation proving that this complexity is needed.
- Evaluation guidance mixes pixel-level and image-level metrics inconsistently, and some docs risk threshold leakage by discussing Oracle-F1 on the test set.
- Image-level detection is often derived from `max(pred_mask)` or "any pixel above threshold", which is highly sensitive to isolated false positives.
- The data-split guidance is stratified but not group-aware. If source-image relationships leak across splits, reported performance can be optimistic.
- The repo has extensive documentation but no notebook. The biggest current project risk is documentation outpacing implementation.

## 4. Unnecessary Complexity

- W&B Sweeps for large hyperparameter search
- Hugging Face dataset hosting as part of the core workflow
- Hugging Face Spaces deployment for the assignment submission
- DuckDB, Redis, SQLite, or DynamoDB discussions inside the main project path
- Databricks as a planning artifact for a 5K-image Colab project
- DALI, advanced compile modes, channels-last tuning, and similar secondary optimizations before a baseline exists
- Noiseprint++, TruFor-style reliability heads, BayarConv, or SegFormer dual-stream work as v1 requirements
- Building a generalized framework instead of one clean notebook

These items are not all wrong in isolation. They are wrong for the critical path of this assignment.

## 5. Missing Components

- A real single Google Colab notebook implementing the chosen pipeline
- A minimal baseline architecture decision that is simpler than the current "best solution" stack
- Clear policy for preventing split leakage from related images or near-duplicates
- A metric policy that separates tampered-only localization quality from authentic-image false-positive behavior
- A threshold-selection policy that uses validation only and does not tune on the test set
- Actual deliverables: notebook link, weights, and generated figures

## 6. Recommended Final Architecture

Use one notebook-first pipeline and treat everything else as optional.

1. Dataset
   - Use CASIA v2.0 as the only required dataset.
   - Download from Kaggle inside the notebook.
   - Pair tampered images to masks by filename, generate all-zero masks for authentic images, binarize masks with a fixed threshold, and log or exclude misaligned pairs.

2. Preprocessing and augmentation
   - Resize images and masks to `512 x 512`.
   - Use synchronized `albumentations` transforms.
   - Keep training augmentation conservative: horizontal flip, optional `RandomRotate90`, light brightness/contrast, light JPEG compression, light Gaussian noise.

3. Model
   - Start with `U-Net` plus an ImageNet-pretrained encoder that is known to fit on T4, such as `ResNet34`, `EfficientNet-B0`, or `EfficientNet-B1`.
   - Make the first shipped version RGB-only.
   - Treat SRM preprocessing as an optional ablation or bonus enhancement, not as a hard dependency for v1.

4. Loss and training
   - Use `BCEWithLogitsLoss + Dice loss` as the default.
   - Keep focal loss or edge loss optional and only add them if a baseline failure mode clearly justifies it.
   - Train with PyTorch, AMP, checkpoint the best model by validation F1/Dice, and use early stopping.

5. Evaluation
   - Primary localization metrics: IoU, Dice/F1, pixel precision, pixel recall.
   - Primary image-level metric: one derived tamper score reported as accuracy and AUC/F1, with threshold chosen on validation only.
   - Report robust visualizations: original image, ground-truth mask, predicted probability map, and overlay/binary mask.
   - Bonus robustness testing should be limited to JPEG compression, Gaussian noise, and resizing.

This pipeline is enough to satisfy the assignment and demonstrate engineering discipline without pretending to ship a research platform.

## 7. Cleaned Project Plan

Replace the current sprawl with a smaller documentation set:

1. `01_Assignment_and_Success_Criteria.md`
2. `02_Dataset_and_Cleaning.md`
3. `03_Data_Pipeline_and_Augmentation.md`
4. `04_Model_and_Loss.md`
5. `05_Training_and_Checkpointing.md`
6. `06_Evaluation_and_Visual_Results.md`
7. `07_Robustness_and_Bonus_Work.md`
8. `08_Submission_Assets.md`
9. `Appendix_Optional_Tooling.md`

Move W&B, HF Hub, HF deployment, database systems, Databricks, and advanced optimization notes into the appendix or remove them entirely. The main review packet should describe one implementation path, not every platform adjacent to it.
