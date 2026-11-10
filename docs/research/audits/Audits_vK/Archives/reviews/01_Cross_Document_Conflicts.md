# Cross-Document Conflicts

This matrix captures the highest-impact conflicts across the documentation set and the recommended resolution for the final submission path.

| Conflict | Sources | Why it matters | Recommended resolution |
|---|---|---|---|
| Split policy is inconsistent | `Docs/04_Best_Dataset.md`, `Docs/Overall Flow Docs/02_Data_Pipeline.md`, `Docs/Dataset Selection.md` | A moving split policy makes experiments hard to compare and weakens reproducibility. | Use one policy everywhere. Keep `85 / 7.5 / 7.5` if that is the chosen repo standard, and document it once. |
| Architecture direction changes between "practical" and "ambitious" | `Docs/02_Possible_Solutions.md`, `Docs/05_Best_Solution.md`, `Docs/Overall Flow Docs/04_Architecture.md`, `Docs/Deep Research.md` | The docs oscillate between U-Net, SegFormer, SRM fusion, and more advanced forensic models. | Ship one baseline: `U-Net + pretrained encoder + BCE/Dice`. Treat SRM as optional and reject SegFormer for v1. |
| Loss policy drifts | `Docs/05_Best_Solution.md`, `Docs/Overall Flow Docs/04_Architecture.md`, `Docs/Copilot-Engineering-Instructions.md`, `Docs/Code-Generation-Instructions.md` | Loss choice directly changes implementation complexity and tuning burden. | Default to `BCEWithLogits + Dice`. Only add focal or edge loss if a measured failure mode justifies it. |
| Metric definitions are inconsistent | `Docs/06_Best_Practices.md`, `Docs/Overall Flow Docs/06_Performance_Metrics.md`, `Docs/01_Problem_Statement.md` | Pixel AUC, image AUC, Oracle-F1, and MCC are mixed without priority. | Use Pixel F1/Dice, IoU, precision, recall, and one image-level metric family. Keep Oracle-F1 as appendix only. Drop MCC from the default path. |
| Image-level detection strategy is unclear | `Docs/Copilot-Engineering-Instructions.md`, `Docs/06_Best_Practices.md`, `Docs/Overall Flow Docs/06_Performance_Metrics.md` | A separate classification head vs mask-derived score changes the model interface and notebook scope. | Do not add a classification head in v1. Derive image-level detection from the predicted mask and calibrate on validation only. |
| Full dataset vs balanced subset | `Docs/04_Best_Dataset.md`, `Docs/Dataset.md`, `Docs/Overall Flow.md`, `Docs/Dataset Selection.md` | Downsampling without a clear reason throws away already limited data. | Use the full cleaned dataset. Handle imbalance with loss design and careful reporting, not aggressive subsetting. |
| W&B is optional in some docs and effectively mandatory in others | `Docs/Overall Flow Docs/11_Weights_And_Biases.md`, `Docs/Overall Flow Docs/12_WandB_Sweeps.md`, `Docs/Overall Flow Docs/09_Assets.md` | Extra tooling should not become a hidden dependency for the assignment. | Make W&B optional. The notebook must be complete and interpretable without external dashboards. |
| Kaggle-first vs HF-heavy workflow | `Docs/Dataset.md`, `Docs/Overall Flow Docs/13_Kaggle_vs_HuggingFace.md`, `Docs/Overall Flow Docs/14_HuggingFace_Platform.md` | The assignment needs one data-access path, not two parallel ecosystems. | Download from Kaggle in the notebook. Treat HF Hub as post-project portfolio work only. |
| Single notebook requirement vs many external platform docs | `Docs/Assignment.md`, `Docs/Overall Flow Docs/14_HuggingFace_Platform.md`, `Docs/Overall Flow Docs/19_HF_Deployment.md`, `Docs/Overall Flow Docs/20_DataBricks.md` | The deliverable is a single Colab notebook, not a small MLOps platform. | Keep the critical path notebook-only. Remove or demote all platform extras from the main documentation. |
| Basic optimization vs advanced optimization | `Docs/Overall Flow Docs/05_Resource_Constraints.md`, `Docs/Overall Flow Docs/17_Training_Optimisation.md` | DALI, `torch.compile`, channels-last, and similar knobs can distract from baseline correctness. | Use AMP, reasonable batch sizing, and checkpointing. Treat the rest as optional appendix material only. |
| Bonus work is framed like core scope in several docs | `Docs/01_Problem_Statement.md`, `Docs/Overall Flow Docs/10_Bonus_Points.md`, `Docs/07_Industry_Relevance.md` | Bonus work should not block the core assignment. | Core deliverable: CASIA-only notebook with strong evaluation and visuals. Bonus work comes after the baseline is stable. |

## Final resolution

The final project path should be:

`CASIA v2.0 -> cleaning and mask binarization -> synchronized augmentation -> U-Net baseline on T4 -> BCE/Dice -> checkpointed training -> IoU/F1 plus image-level detection -> visual results -> optional JPEG/noise/resize robustness`

Everything else should be described as optional follow-up work, not part of the main implementation contract.
