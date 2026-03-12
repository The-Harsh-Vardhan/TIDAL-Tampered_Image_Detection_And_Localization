# Audit6 Master Report

This audit reviews `Docs6/` against the actual implementation now present in the repository:

- `notebooks/tamper_detection_v6_colab.ipynb`
- `notebooks/tamper_detection_v6_kaggle.ipynb`

It is a static documentation audit. It checks technical correctness, consistency, reproducibility, and interview readiness from repository inspection only. It does not prove that either notebook was executed end to end.

## 1. Overall Documentation Score (1–10)

`5.8/10`

`Docs6` contains a large amount of technically useful material, and many sections still describe the intended v6 baseline correctly: CASIA localization, `smp.Unet(resnet34)`, BCE + Dice, AdamW, AMP, Grad-CAM, and robustness testing are all present. The problem is that the documentation is no longer aligned with the real implementation source of truth. The repo now contains two v6 notebooks, but `Docs6` is still anchored to `tamper_detection_v5.1_kaggle.ipynb`, a 61-cell / 17-section structure, and a Kaggle-only runtime story.

## 2. Major Strengths

- The documentation explains the core segmentation framing clearly and consistently.
- Dataset discovery, mask binarization, split persistence, and corruption checks are described in practical detail.
- The training stack is mostly well documented: BCE + Dice, AdamW, AMP, gradient accumulation, clipping, checkpointing, and early stopping all match the v6 notebooks.
- Explainability language is appropriately cautious. Grad-CAM is described as a lightweight diagnostic tool rather than rigorous causal explainability.
- `13_References.md` is a useful repository index for datasets, papers, and reference notebooks.
- The v6 notebooks themselves are structurally consistent with each other: both are 66-cell, 22-section pipelines with the same core model, split, artifact names, and training logic.

## 3. Critical Issues

- `Docs6` still identifies `tamper_detection_v5.1_kaggle.ipynb` as the primary notebook, while the real implementation source of truth is now `tamper_detection_v6_colab.ipynb` and `tamper_detection_v6_kaggle.ipynb`.
- `Docs6` still documents **61 cells / 17 sections**, but both v6 notebooks contain **66 cells / 22 sections**. The notebook structure documentation is no longer trustworthy for navigation or maintenance.
- `Docs6` documents image-level detection using a **top-k mean** score, but both v6 notebooks actually use **max pixel probability** as the tamper score. This is a real evaluation-method mismatch, not a wording issue.
- `Docs6` presents itself as a Kaggle-native final revision, but the implementation now has both Colab and Kaggle v6 variants. Colab-specific setup, dataset access, W&B auth flow, and artifact storage are not documented.

## 4. Medium Issues

- `10_Project_Timeline.md` conflicts with `02_Dataset_and_Preprocessing.md`: the timeline says to download the Kaggle dataset slug, while the dataset doc says the dataset is pre-mounted and no download step is needed.
- `10_Project_Timeline.md` and `11_Research_Alignment.md` describe a larger robustness suite including brightness, contrast, saturation, and combined degradation, but the actual v6 notebooks and `06_Robustness_Testing.md` only implement JPEG, Gaussian noise, Gaussian blur, and resize degradation.
- `11_Research_Alignment.md` uses paper IDs such as `P1`, `P2`, `P4`, `P20`, and `P21`, but `13_References.md` does not define a consistent paper-ID scheme for those references. The citation chain is ambiguous.
- `03_Model_Architecture.md` explains the chosen baseline well, but it does not meaningfully discuss ViT or DeepLabV3 tradeoffs even though those are central interview questions for this project.
- Several docs still use Kaggle-only wording where the repo now contains a Colab variant with materially different environment setup.

## 5. Minor Improvements

- Add a short runtime matrix explaining what is shared between Colab and Kaggle and what differs.
- Add one authoritative v6 artifact table that covers both runtime variants.
- Separate “implemented now” from “future work” more aggressively in `10_Project_Timeline.md` and `11_Research_Alignment.md`.
- Add a direct interview-facing comparison paragraph for `ResNet34 vs EfficientNet vs ViT vs DeepLabV3`.
- Make the references document map named papers to the `P#` notation used elsewhere, or remove the `P#` notation entirely.

## 6. Notebook–Documentation Mismatches

- `Docs6` still points to `tamper_detection_v5.1_kaggle.ipynb`.
- `Docs6` still documents 61 cells / 17 sections instead of the v6 notebooks’ 66 cells / 22 sections.
- `Docs6` documents image-level scoring via top-k mean, but both v6 notebooks use `probs[i].view(-1).max().item()`.
- `Docs6` is Kaggle-only in environment, storage, and W&B-auth descriptions, but the repo now contains a Colab v6 notebook with Drive mounting, Kaggle API credential setup, and `google.colab.userdata`.
- `10_Project_Timeline.md` lists robustness extras that are not implemented in the v6 notebooks.

See `01_Cross_Document_Conflicts.md`, `02_Notebook_Alignment_Check.md`, and `06_Runtime_Variant_Consistency.md`.

## 7. Missing Information for Reproducibility

- No authoritative `Docs6` path explains how to run the Colab v6 notebook.
- `Docs6` does not explain how to obtain dataset credentials and download data for the Colab variant.
- `Docs6` does not document the Colab artifact directory and Drive behavior.
- The notebook-structure doc is stale, so another engineer cannot rely on it to navigate the actual v6 notebooks.
- The reference mapping is ambiguous because `11_Research_Alignment.md` and `13_References.md` do not share a stable citation scheme.

Kaggle reproduction is partially documented. Colab reproduction is not.

## 8. Interview Readiness Evaluation

Current interview readiness is **moderate**.

Strong areas:
- Why segmentation instead of plain classification
- Why BCE + Dice
- Why Grad-CAM should be treated cautiously
- What the dataset limitations are

Weak areas:
- Why ResNet34 instead of ViT or DeepLabV3
- Why the project now has both Colab and Kaggle variants
- Why the documented image-level detection rule differs from the real implementation
- How to explain the runtime story without contradicting the current docs

See `04_Interview_Readiness.md`.

## 9. Recommended Fixes

1. Update all `Docs6` notebook references from v5.1 to the real v6 notebooks.
2. Rewrite `12_Complete_Notebook_Structure.md` around the 66-cell / 22-section v6 structure.
3. Correct the image-level detection description everywhere: either update the docs to `max(prob_map)` or change both v6 notebooks back to top-k mean.
4. Split runtime documentation into:
   - shared v6 pipeline
   - Colab-specific setup
   - Kaggle-specific setup
5. Fix `10_Project_Timeline.md` so it only claims implemented robustness conditions in the current pipeline.
6. Harmonize `11_Research_Alignment.md` with `13_References.md` by defining or removing the `P#` citation scheme.
