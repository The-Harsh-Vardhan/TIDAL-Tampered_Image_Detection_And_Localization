# 10. Reproducibility Audit

Review target: `vK.7.1 Image Detection and Localisation.ipynb`

Reproducibility is weak.

The split process uses fixed `random_state=42` (cell 27), but there is no global seeding for Python, NumPy, or PyTorch anywhere in the notebook. That means training remains non-deterministic across reruns.

The saved artifact is also not reproducible as a clean experiment trace. The effective training block is unexecuted, yet later cells assume the existence of `best_model_path`, `TRAINING_HISTORY`, and other runtime state when producing visualizations (cells 71, 73, 75, 77, 81). That suggests the notebook was not saved from a coherent single run.

Dependency control is only partial. Albumentations and OpenCV are pinned in the notebook (cells 31, 53), but core framework versions are not fully frozen. W&B online logging further depends on Kaggle secrets (cell 50), which adds another external prerequisite.

Another user could probably rerun the implementation with some effort. Another user could not confidently treat the saved notebook itself as a reproducible research artifact.
