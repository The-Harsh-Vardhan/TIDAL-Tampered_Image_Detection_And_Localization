# Notebook Alignment Check

This file compares `Docs6/` against the actual implementation now present in the repo:

- `notebooks/tamper_detection_v6_colab.ipynb`
- `notebooks/tamper_detection_v6_kaggle.ipynb`

It is a static alignment report. It does not prove that the notebooks execute successfully end to end.

## Shared Pipeline Alignment

| Area | Docs6 claim | v6 notebook evidence | Status | Notes |
|---|---|---|---|---|
| Dataset family | CASIA localization dataset | Both notebooks use CASIA-style localization discovery | Aligned | Core dataset choice is consistent. |
| Image size | 384 × 384 | `CONFIG['image_size'] = 384` in both notebooks | Aligned | |
| Split policy | 70 / 15 / 15 | `CONFIG['train_ratio'] = 0.70` with 70/15/15 split logic in both notebooks | Aligned | |
| Mask binarization | `> 0` | `mask = (mask > 0).astype(np.uint8)` in both notebooks | Aligned | |
| Model | `smp.Unet(resnet34)` | `smp.Unet(... encoder_name='resnet34' ...)` in both notebooks | Aligned | |
| Loss | `BCEDiceLoss` | `class BCEDiceLoss` in both notebooks | Aligned | |
| Optimizer | AdamW with differential LR | `torch.optim.AdamW([...])` in both notebooks | Aligned | |
| Threshold sweep | Validation-only threshold sweep | `find_best_threshold(... np.linspace(0.1, 0.9, 50))` in both notebooks | Aligned | |
| Checkpoint artifacts | `best_model.pt`, `last_checkpoint.pt`, `split_manifest.json`, `results_summary.json` | Present in both notebooks | Aligned | |
| Grad-CAM safety framing | Grad-CAM with safety checks | Safety checks present in both notebooks | Aligned | |
| Image-level score | Docs6 says top-k mean | Both v6 notebooks use `probs[i].view(-1).max().item()` | Conflicting | This is the main evaluation-method mismatch. |
| Notebook structure | Docs6 says 61 cells / 17 sections | Both v6 notebooks have 66 cells / 22 sections | Conflicting | Structure docs are stale. |

## Colab-Specific Alignment

| Area | Docs6 claim | v6 Colab evidence | Status | Notes |
|---|---|---|---|---|
| Runtime coverage | Docs6 is Kaggle-native | Colab notebook exists and is fully implemented | Missing | Docs6 does not explain the Colab variant as a first-class runtime. |
| Output storage | Docs6 focuses on `/kaggle/working/` | Colab notebook uses Drive-backed `OUTPUT_DIR` and derived checkpoint/results dirs | Missing | Reproducibility for Colab is under-documented. |
| Dataset access | Docs6 assumes Kaggle pre-mounted input | Colab notebook uses Kaggle credentials via `google.colab.userdata` | Missing | Dataset access for Colab is not documented in Docs6. |
| W&B auth | Docs6 documents `kaggle_secrets` | Colab notebook uses `google.colab.userdata` with interactive fallback | Conflicting | Runtime-specific W&B auth is not split correctly in Docs6. |
| Notebook section map | Docs6 says 17 sections | Colab notebook has 22 sections including GPU Verification, Evaluation Metrics, Validation Loop, Final Summary | Conflicting | |

## Kaggle-Specific Alignment

| Area | Docs6 claim | v6 Kaggle evidence | Status | Notes |
|---|---|---|---|---|
| Kaggle runtime | Kaggle T4 with `/kaggle/input/` and `/kaggle/working/` | Present in the Kaggle v6 notebook | Aligned | |
| Dataset loading | Case-insensitive discovery under `/kaggle/input/` | Present in Kaggle notebook | Aligned | |
| W&B auth | `kaggle_secrets` | Present in Kaggle notebook | Aligned | |
| Artifact paths | `/kaggle/working/...` | Present in Kaggle notebook | Aligned | |
| Notebook section map | 61 cells / 17 sections | Kaggle v6 notebook has 66 cells / 22 sections | Conflicting | |
| Image-level score | Top-k mean | Kaggle v6 notebook uses max pixel probability | Conflicting | |

## Stale Docs6 References

| Docs6 artifact | Current claim | Real repo state | Status |
|---|---|---|---|
| `00_Master_Report.md` | Primary notebook is `tamper_detection_v5.1_kaggle.ipynb` | Real implementation includes two v6 notebooks | Conflicting |
| `08_Engineering_Practices.md` | Current notebook is `tamper_detection_v5.1_kaggle.ipynb` with 61 cells / 17 sections | Real v6 notebooks are 66 cells / 22 sections | Conflicting |
| `10_Project_Timeline.md` | Reference notebook is `tamper_detection_v5.1_kaggle.ipynb` | v6 notebooks now exist | Conflicting |
| `12_Complete_Notebook_Structure.md` | Authoritative notebook is `tamper_detection_v5.1_kaggle.ipynb` with 61 cells / 17 sections | v6 notebooks supersede this | Conflicting |

## Conclusion

`Docs6` still aligns reasonably well with the **older Kaggle v5.1-style baseline design**, but it is not aligned with the actual v6 implementation source of truth in the repo. The largest problems are:

- stale notebook identity
- stale notebook structure
- wrong image-level scoring rule
- no proper Colab-runtime coverage
