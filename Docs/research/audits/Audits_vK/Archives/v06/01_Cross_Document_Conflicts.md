# Cross-Document Conflicts

This file lists only real contradictions or meaningful drift across `Docs6/` and the actual v6 notebooks.

| Conflicting artifacts | Conflict | Why it matters | Recommended resolution |
|---|---|---|---|
| `Docs6/00_Master_Report.md`, `Docs6/08_Engineering_Practices.md`, `Docs6/10_Project_Timeline.md`, `Docs6/12_Complete_Notebook_Structure.md` vs `notebooks/tamper_detection_v6_colab.ipynb` and `notebooks/tamper_detection_v6_kaggle.ipynb` | Docs6 still references `tamper_detection_v5.1_kaggle.ipynb` and a 61-cell / 17-section structure, but the real implementation now consists of two v6 notebooks with 66 cells / 22 sections. | This makes the docs unreliable for navigation, maintenance, and interview explanation. | Update all notebook references to the real v6 notebooks and rebuild the structure map from the v6 notebooks. |
| `Docs6/01_System_Architecture.md`, `Docs6/03_Model_Architecture.md`, `Docs6/05_Evaluation_Methodology.md`, `Docs6/11_Research_Alignment.md`, `Docs6/12_Complete_Notebook_Structure.md` vs both v6 notebooks | Docs6 says image-level detection uses a top-k mean tamper score, but both v6 notebooks use max pixel probability as the tamper score. | This is a real model-behavior mismatch that affects evaluation, interpretation, and interview answers. | Standardize on one image-level scoring rule and update either the notebooks or the docs everywhere. |
| `Docs6/02_Dataset_and_Preprocessing.md` vs `Docs6/10_Project_Timeline.md` | The dataset doc says Kaggle data is pre-mounted and no download step is needed, but the timeline says to download the Kaggle dataset slug as a Phase 1 task. | This creates confusion about the actual runtime flow. | Decide whether Docs6 is Kaggle-only or dual-runtime. If dual-runtime, split the steps by platform. |
| `Docs6/06_Robustness_Testing.md` and `Docs6/01_System_Architecture.md` vs `Docs6/10_Project_Timeline.md` and `Docs6/11_Research_Alignment.md` | The robustness doc and notebooks cover JPEG, Gaussian noise, blur, and resize. The timeline and research-alignment doc additionally claim brightness, contrast, saturation, and combined degradation conditions. | Readers cannot tell what is implemented versus planned. | Keep implemented robustness conditions in the main docs and move the extra conditions to explicit future work. |
| `Docs6/11_Research_Alignment.md` vs `Docs6/13_References.md` | `11_Research_Alignment.md` uses `P#` citations such as `P1`, `P2`, `P4`, `P20`, and `P21`, but `13_References.md` does not define a matching ID system for those citations. | This weakens citation traceability and makes research claims harder to verify. | Add a stable paper-ID mapping in `13_References.md` or remove the `P#` shorthand from `11_Research_Alignment.md`. |
| `Docs6/00_Master_Report.md`, `Docs6/01_System_Architecture.md`, `Docs6/08_Engineering_Practices.md`, `Docs6/09_Experiment_Tracking.md`, `Docs6/10_Project_Timeline.md` vs the presence of both v6 notebooks | Docs6 presents the project as Kaggle-native and final, but the repo now contains both Kaggle and Colab v6 variants. | The docs under-document the actual implementation surface. | Split runtime behavior into shared-core, Colab-specific, and Kaggle-specific sections. |

## Summary

The most serious conflicts are:

- stale v5.1 notebook references
- stale 61/17 notebook structure claims
- top-k mean versus max-probability image scoring

These should be treated as first-priority cleanup items before interview use or submission.
