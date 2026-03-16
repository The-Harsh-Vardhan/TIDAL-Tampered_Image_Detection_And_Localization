# Requirements Matrix

This matrix maps the assignment requirements to the current documentation set and shows whether coverage is actually reliable enough to implement directly.

| Requirement | Assignment anchor | Main docs covering it | Audit status | Gap or risk | Required resolution |
|---|---|---|---|---|---|
| Public dataset with authentic images, tampered images, and masks | `Docs/Assignment.md` lines 8-19 | `Docs/03_Dataset_Exploration.md`, `Docs/04_Best_Dataset.md`, `Docs/Overall Flow Docs/01_Dataset_Choice.md` | Pass | Core dataset choice is covered well enough. | Keep CASIA v2.0 as the default dataset. |
| Mask pairing and alignment | `Docs/Assignment.md` lines 8-19 | `Docs/04_Best_Dataset.md`, `Docs/Overall Flow Docs/02_Data_Pipeline.md` | Partial | Good handling of the known 17 misaligned CASIA pairs, but no group-leakage or dedup policy. | Add a split-integrity check and keep the pairing logic simple and explicit. |
| Mask binarization | `Docs/Assignment.md` lines 8-19 | `Docs/03_Dataset_Exploration.md`, `Docs/04_Best_Dataset.md`, `Docs/Overall Flow Docs/02_Data_Pipeline.md` | Pass | Thresholding policy is consistently described. | Use one threshold and document it once. |
| Train/validation/test split | `Docs/Assignment.md` lines 8-19 | `Docs/04_Best_Dataset.md`, `Docs/Overall Flow Docs/02_Data_Pipeline.md`, `Docs/Dataset Selection.md` | Partial | Split ratios conflict, and there is no discussion of leakage between related images. | Pick one split ratio and add a leakage/dedup check. |
| Augmentation | `Docs/Assignment.md` lines 8-19 | `Docs/06_Best_Practices.md`, `Docs/Overall Flow Docs/03_Augmentation.md`, `Docs/Copilot-Engineering-Instructions.md` | Partial | Core augmentation guidance is good, but some docs add more photometric distortion than necessary. | Keep augmentation conservative and aligned with forensic signals. |
| Segmentation architecture | `Docs/Assignment.md` lines 20-27 | `Docs/05_Best_Solution.md`, `Docs/Overall Flow Docs/04_Architecture.md`, `Docs/02_Possible_Solutions.md` | Partial | Too many candidate architectures and too much unsupported certainty around the chosen one. | Freeze one simple baseline architecture for v1. |
| Loss for class imbalance | `Docs/Assignment.md` lines 20-27 | `Docs/05_Best_Solution.md`, `Docs/Code-Generation-Instructions.md`, `Docs/Copilot-Engineering-Instructions.md` | Partial | BCE/Dice, BCE/Dice/Edge, and focal-style ideas all appear. | Default to BCE + Dice; make extras optional. |
| Runnable on Colab T4 | `Docs/Assignment.md` lines 20-27 | `Docs/Overall Flow Docs/05_Resource_Constraints.md`, `Docs/Overall Flow Docs/08_The_Code.md`, `Docs/Overall Flow Docs/17_Training_Optimisation.md` | Partial | Baseline path is plausible, but many optional docs add needless runtime and setup burden. | Keep only notebook-native optimizations on the core path. |
| Evaluation metrics | `Docs/Assignment.md` lines 28-34 | `Docs/06_Best_Practices.md`, `Docs/Overall Flow Docs/06_Performance_Metrics.md`, `Docs/Copilot-Engineering-Instructions.md` | Partial | Metric scope and definitions drift across docs. | Standardize around IoU, Dice/F1, precision, recall, and one image-level metric family. |
| Visualization of predicted masks | `Docs/Assignment.md` lines 28-34 | `Docs/Overall Flow Docs/07_Visual_Results.md`, `Docs/Copilot-Engineering-Instructions.md` | Pass | Visualization guidance is strong overall. | Make sure the notebook shows the four assignment views clearly. |
| Single Google Colab notebook | `Docs/Assignment.md` lines 35-49 | `Docs/Overall Flow Docs/08_The_Code.md`, `Docs/Overall Flow Docs/09_Assets.md` | Partial | Many other docs assume optional scripts, dashboards, hosting platforms, or deployment layers. | Re-center every design choice on one self-contained notebook. |
| Checkpoint saving | `Docs/Assignment.md` lines 35-49 | `Docs/06_Best_Practices.md`, `Docs/Overall Flow Docs/15_Model_Checkpoints.md`, `Docs/Overall Flow Docs/09_Assets.md` | Pass | Slightly over-detailed, but operational guidance is solid. | Keep `last` and `best` checkpoint handling only. |
| Bonus robustness testing | `Docs/Assignment.md` lines 51-57 | `Docs/Overall Flow Docs/10_Bonus_Points.md`, `Docs/Overall Flow Docs/03_Augmentation.md`, `Docs/Overall Flow.md` | Partial | Bonus scope is sometimes treated like core scope, and crop handling is inconsistent. | Limit bonus evaluation to JPEG, noise, and resize unless time remains. |
| Assets: notebook link and weights | `Docs/Assignment.md` lines 35-49 | `Docs/Overall Flow Docs/09_Assets.md`, `Docs/Overall Flow Docs/15_Model_Checkpoints.md` | Partial | The docs describe how to share assets, but the assets do not exist in the repo yet. | Produce the actual notebook and checkpoint before claiming completion. |

## Summary

- Reliable coverage exists for dataset choice, pairing logic, augmentation basics, checkpointing, and visualization.
- Partial coverage dominates the rest because the docs add too many options, tools, and unverified claims.
- The missing artifact is not another document. It is the notebook itself.
