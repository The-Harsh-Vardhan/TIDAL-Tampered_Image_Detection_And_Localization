# Implementation Risks

This report isolates the main implementation risks that remain after the v5 alignment fixes. It distinguishes between items that are now mitigated in code and items that remain open runtime or research limitations.

| Risk area | Current state | Residual risk | Recommended guardrail |
|---|---|---|---|
| Empty-mask precision/recall behavior | Mitigated | The notebook now treats empty-empty per-image cases as undefined for per-image precision/recall and reports mixed-set precision/recall from global pixel counts. | Keep using the global mixed-set metric path and avoid naively averaging per-image precision/recall on authentic images. |
| Checkpoint and threshold coupling | Mitigated | Validation sweeps now drive checkpoint selection and early stopping, so the saved best model matches the real operating point. | Do not reintroduce fixed-threshold model selection in later edits. |
| Dataset-root ambiguity | Mitigated | Dataset extraction and root resolution are scoped to the Kaggle slug path rather than the first `Image/Mask` tree under `/content`. | Keep dataset-root selection explicit if alternate datasets are added later. |
| Split-manifest reuse | Mitigated | `split_manifest.json` is reused on compatible reruns and acts as the split source of truth. | Keep the compatibility check strict if dataset contents change. |
| Corrupt sample handling | Mostly mitigated | Discovery excludes corrupt images, corrupt masks, unknown forgery types, and dimension mismatches. | Log excluded-sample counts in run summaries so data quality issues remain visible. |
| `unknown` stratification edge cases | Mitigated | Unknown forgery types are excluded before stratified splitting. | Fail loudly if unknown categories start appearing in meaningful volume. |
| Colab / T4 operational reliability | Open | Kaggle auth, Drive mounting, dependency installs, limited VRAM, and session timeouts can still interrupt runs. | Treat the notebook as a reproducible baseline, not a guaranteed uninterrupted runtime. Save checkpoints frequently. |
| Source-image leakage in CASIA | Open | CASIA still lacks source-group metadata, so related content may span splits even when the split is reproducible. | Keep generalization claims conservative and document the limitation wherever results are reported. |
| Heuristic image-level scoring | Open | The top-k mean score is stabler than `max(prob_map)` but is still a handcrafted rule. | Consider a dual-head classification extension if image-level performance becomes a priority. |
| Explainability depth | Open | Grad-CAM, overlays, and failure cases help inspection but do not provide formal causal explanations. | Continue labeling these outputs as lightweight diagnostics rather than definitive explainability. |

## Current Blocker Assessment

No implementation risk above should block an initial baseline training run. The open risks affect runtime reliability, empirical generalization, or future model quality claims rather than the basic technical validity of the v5 pipeline.
