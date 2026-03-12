# 06 - Shortcut Learning and Robustness Audit

## Bottom line

The notebook knows shortcut learning is a risk. It does not actually test that risk with enough rigor to earn the confidence it tries to project. Most of the "checks" are weak proxies, and several are biased in ways that make the model look cleaner than it is.

## 1. The risk is real

This project is especially vulnerable to shortcut learning because the model is:

- RGB-only,
- trained on one dataset family,
- evaluated on the same dataset family,
- asked to localize manipulations that may carry compression or boundary artifacts.

That means the model can easily learn:

- mask-edge style cues,
- compression inconsistencies,
- color mismatches,
- resampling artifacts,
- dataset-specific annotation habits.

In other words, it can look forensic while actually behaving like a dataset artifact detector.

## 2. The robustness suite is not enough

Cells 46 through 48 apply JPEG compression, Gaussian noise, blur, and resize degradations. That is a useful nuisance test. It is not a shortcut-learning verdict. It only shows whether the model is sensitive to a few perturbations on the same test set.

That leaves major gaps:

- no cross-dataset evaluation,
- no content-matched control set,
- no source-camera or acquisition split,
- no test for manipulation provenance leakage,
- no check that predictions survive when obvious artifact cues are suppressed.

The notebook is testing convenience corruptions, not forensic generalization.

## 3. Robustness metrics are still vulnerable to authentic-image inflation

`run_robustness_eval()` in Cell 47 computes Pixel-F1 on the full loader, which includes authentic images. So the exact same mixed-set inflation problem from the main evaluation leaks into the robustness section. If many authentic images remain empty under degradation, the robustness averages can still look decent while tampered localization falls apart.

That is a major design flaw. A robustness section that does not lead with tampered-only behavior is asking to be misread.

## 4. The mask-randomization test is weak bordering on silly

Cell 50 compares predictions against random binary masks and expects F1 near `0.0-0.1`. That expectation is arbitrary. The random masks are not matched to the true tamper prevalence, not matched to mask sparsity, not matched to object boundaries, and not tied to a meaningful null hypothesis.

This test can only tell you something extremely crude: the model is not perfectly aligned with random noise. That is such a low bar it is barely worth writing down.

Calling it a shortcut-learning check is generous.

## 5. Boundary sensitivity analysis cherry-picks survivors

Also in Cell 50, `boundary_sensitivity_analysis()` keeps only tampered predictions with `pixel_f1 > 0.1`, then truncates to the first 50 samples. That filters out the hardest failures before the analysis even starts.

So the analysis is biased toward examples where the model already worked at least a little. That makes the reported sensitivity look calmer than the full model behavior probably is.

A real stress test includes the failures. This one quietly discards them.

## 6. Grad-CAM does not rescue the story

Cells 42 and 43 generate Grad-CAM using `output.mean()` as the backprop target on encoder `layer4`. That can produce nice heatmaps. It does not prove the model is focusing on genuine tamper evidence. It mostly proves the notebook can draw explainability pictures.

When a project is already vulnerable to artifact learning, pretty overlays are not evidence. They are presentation.

## 7. What a stronger shortcut-learning audit would look like

A senior reviewer would expect some combination of:

- tampered-only robustness reporting,
- duplicate-aware or source-aware split validation,
- content-matched controls,
- artifact suppression tests,
- cross-dataset transfer,
- boundary metrics rather than only region overlap,
- copy-move-specific stress tests.

The current notebook does not do that. It gestures toward the topic and then stops where the implementation gets inconvenient.

## Verdict

The notebook is right to worry about shortcut learning. It is not right to imply that the current checks seriously constrain that risk. The robustness and shortcut sections are useful as rough exploratory tools, but they are nowhere near strong enough to support trust claims about what the model has actually learned.
