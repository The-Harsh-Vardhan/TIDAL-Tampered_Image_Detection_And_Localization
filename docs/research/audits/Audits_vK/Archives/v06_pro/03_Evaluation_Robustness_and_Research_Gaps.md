# Evaluation, Robustness, and Research Gaps

This note focuses on whether the metrics prove what the project claims, whether the explainability and robustness sections are technically convincing, and where the project sits relative to modern tamper-detection research.

## Finding 1: IoU and F1 are necessary, but far from sufficient

**Claim in docs**

- `Docs6/05_Evaluation_Methodology.md` treats Pixel-F1 as the primary localization metric and IoU as the secondary metric.

**Technical objection**

Those are standard segmentation metrics, but they do not show that the model is detecting tampering for the right reasons. They measure overlap against a legacy benchmark's masks, not forensic validity.

**Why it matters**

A model can achieve decent F1 by exploiting dataset shortcuts, coarse regions, or mask priors without being robust to real manipulations.

**Stronger answer or remediation**

Keep F1 and IoU, but add analyses by mask size, edge quality, and failure category. If the claim is forensic detection, show evidence beyond region overlap.

## Finding 2: Empty-mask handling can inflate mixed-set performance

**Claim in docs**

- `Docs6/05_Evaluation_Methodology.md` returns 1.0 for F1, IoU, precision, and recall when both prediction and ground truth are empty.

**Technical objection**

This is mathematically consistent, but it makes authentic images score as perfect localization examples. If the mixed test set has many authentic images, average localization metrics can look better than the model's real tampered-region quality.

**Why it matters**

A reviewer may interpret mixed-set Pixel-F1 as tamper-localization skill when part of that score is simply "correctly predicted no tampering."

**Stronger answer or remediation**

Always lead with tampered-only localization metrics, and treat mixed-set numbers as secondary context.

## Finding 3: One threshold for both masks and image-level decisions is unjustified

**Claim in docs**

- `Docs6/05_Evaluation_Methodology.md` uses one validation-selected threshold for pixel binarization, image-level decisions, and robustness evaluation.

**Technical objection**

This is operationally simple, but segmentation and classification are different decision problems. The threshold that maximizes mean Pixel-F1 is not automatically the threshold that optimizes image-level precision, recall, or analyst utility.

**Why it matters**

The image-level performance may be weaker than it appears simply because it inherits a threshold optimized for another objective.

**Stronger answer or remediation**

Use separate calibration for image-level detection, or explicitly state that image-level results are approximate because they reuse the segmentation threshold.

## Finding 4: The explainability story is honest in tone, but weak in evidence

**Claim in docs**

- `Docs6/07_Visualization_and_Explainability.md` says Grad-CAM is only a diagnostic tool.

**Technical objection**

That caveat is correct, but the notebook implementation still uses `output.mean()` as the target scalar for Grad-CAM. For segmentation, that can produce diffuse heatmaps dominated by generic foreground activation rather than evidence for the specific tampered region.

**Why it matters**

The heatmap may look plausible while saying very little about why the model localized a particular area.

**Stronger answer or remediation**

Use class- and region-specific targets if explainability is important, or state plainly that Grad-CAM is used only as a rough sanity check.

## Finding 5: There is no quantitative evaluation of explanations

**Claim in docs**

- `Docs6/07_Visualization_and_Explainability.md` relies on overlays, Grad-CAM, and failure-case review.

**Technical objection**

No deletion/insertion test, pointing-game style overlap, sanity check, or counterfactual analysis is used. The report asks the reader to visually trust the explanations.

**Why it matters**

Visual plausibility is not the same as explanation validity. This is especially risky in interviews because many candidates overclaim on XAI.

**Stronger answer or remediation**

Keep the language modest and avoid any statement that suggests the explanations validate causal reasoning.

## Finding 6: The robustness suite is narrow

**Claim in docs**

- `Docs6/06_Robustness_Testing.md` evaluates JPEG, Gaussian noise, Gaussian blur, and resize degradation.

**Technical objection**

These are nuisance transforms, not modern manipulation families. They test whether the model tolerates image corruption, not whether it generalizes to new tampering mechanisms.

**Why it matters**

A reviewer can reasonably say: "You tested compression robustness, not forgery robustness."

**Stronger answer or remediation**

Rename the section mentally as post-processing robustness, then add at least a discussion of missing semantic edits such as diffusion inpainting, localized relighting, face swaps, and content-aware fill.

## Finding 7: The project does not test cross-dataset or out-of-distribution generalization

**Claim in docs**

- `Docs6/05_Evaluation_Methodology.md` uses a single held-out test split from the same dataset family.

**Technical objection**

That answers only one question: can the model separate held-out examples from the same benchmark under the same annotation regime?

**Why it matters**

A model can look strong in-domain and collapse out-of-domain. Without cross-dataset validation, generalization claims should stay very narrow.

**Stronger answer or remediation**

Add a second benchmark, a synthetic challenge set, or at least a deliberately shifted holdout regime.

## Finding 8: The project is behind current research directions

**Claim in docs**

- `Docs6/11_Research_Alignment.md` cites stronger transformer and multi-trace papers as future work.

**Technical objection**

The gap is not cosmetic. Modern tamper localization increasingly benefits from:

- transformer or hybrid global-context modeling
- noise residual or frequency-domain streams
- edge-aware supervision
- multi-task learning for image-level and pixel-level outputs
- cross-dataset robustness evaluation

**Why it matters**

If the project is presented as research-aligned rather than assignment-aligned, the reviewer will see a gap between cited literature and implemented system.

**Stronger answer or remediation**

Say the project is literature-informed, not literature-competitive.

## Interview-defense questions

1. If your mixed-set F1 includes authentic images scoring 1.0, what is the tampered-only F1 and why should I care about the mixed number?
2. Why did you not report boundary metrics when tamper localization quality often depends on edge accuracy?
3. If the same threshold is used for segmentation and image-level detection, what evidence shows that both operating points are appropriate?
4. Why should Grad-CAM on `output.mean()` tell me anything useful about a segmentation decision?
5. What failure patterns appear when the tampered region is very small?
6. How would your evaluation change if the test set contained only tampered images?
7. How would you show that the model is not just learning CASIA-specific compression or annotation artifacts?
8. Why do JPEG, blur, and noise count as robustness tests for tamper detection rather than just corruption tests?
9. How would you evaluate this model against diffusion-based local edits?
10. What metric would you use if the product cared about not missing any tampered image, even at the cost of more false positives?
11. What is your plan for calibration or uncertainty if this system is used for analyst triage?
12. If a new benchmark contradicts your CASIA results, which conclusion would you trust more and why?

## Bottom line

The evaluation stack is competent for a class project baseline, but not strong enough to support broad claims about real-world tamper detection. The safest honest conclusion is:

"The reported metrics show in-domain segmentation performance on a legacy benchmark, not robust forensic performance in the wild."
