# Explainable AI

## How to explain this in an interview

Start with this:

"Because this is a forensic-style vision task, I did not want the model to be a pure black box. I added lightweight explainability tools like overlays and Grad-CAM so I could check whether the model was focusing on plausible tampered regions."

## Why explainability matters here

Tamper detection is not a typical image-classification problem where a label alone is enough.

In this kind of system, people usually want evidence:

- Where is the manipulation?
- What part of the image triggered the model?
- Is the model focusing on the suspicious region or on irrelevant textures?

That is why explainability is useful. It improves:

- debugging
- trust
- error analysis
- interview clarity

## What explainability components were used

The project uses three main explainability or diagnostic views:

- predicted masks
- overlay visualizations
- Grad-CAM heatmaps

## Predicted masks

### What they are

These are the model's binary tamper localizations after thresholding.

### What problem they solve

They show the exact area the model considers manipulated.

### Why they were chosen here

The whole project is built around segmentation, so the predicted mask is the most direct explanation of model behavior.

### Alternatives

- pure image-level saliency
- classification heatmaps without segmentation

### Why those were not selected

They are less direct than a true predicted mask for a localization task.

## Overlay visualizations

### What they are

An overlay blends the predicted tamper region with the original image.

### What problem they solve

A raw mask can be hard to interpret by itself. The overlay makes it easier to see whether the highlighted region actually corresponds to a suspicious visual area.

### Why they were chosen here

They are simple, intuitive, and very effective in interviews. A non-specialist can immediately understand them.

### Alternatives

- separate mask-only panels
- contour drawings

### Why they were not enough alone

They are still useful, but overlays communicate the result faster and more naturally.

## Grad-CAM

### What it is

Grad-CAM is a gradient-based visualization method that highlights which spatial regions in the feature maps contributed most to the output.

### What problem it solves

It helps answer:

- Is the encoder paying attention to the tampered area?
- Is the model reacting to meaningful forensic cues?
- Is it focusing on unrelated objects or textures instead?

### Why it was chosen here

Grad-CAM is lightweight and easy to add without redesigning the whole model. That made it a good fit for a notebook-based baseline.

### Alternatives

- Integrated Gradients
- SHAP
- occlusion sensitivity
- attention map inspection in transformer models

### Why those were not selected

- They are often more computationally expensive or harder to present cleanly in this pipeline.
- Some are more natural for classification than for dense segmentation outputs.
- Grad-CAM gave enough diagnostic value for the first version.

## Why explainability was chosen for this project

I added explainability because tamper detection systems can fail in deceptive ways.

For example, a model may appear correct because the final mask looks reasonable, but internally it may be reacting to:

- high-contrast object edges
- repetitive textures
- compression artifacts unrelated to tampering

Explainability helps catch that.

## What Grad-CAM can tell us

Grad-CAM is useful for checking whether the model is broadly attending to the correct area.

It is good for:

- qualitative sanity checks
- comparing successful and failed examples
- understanding failure cases

## What Grad-CAM cannot tell us

This is the key limitation to mention in interviews.

Grad-CAM does not prove causality. It does not mean:

- the highlighted area alone caused the decision
- the model truly understands tampering
- the highlighted region is a precise explanation of every pixel in the predicted mask

It is a diagnostic tool, not a formal explanation method.

## Why that limitation is acceptable here

For this project, the goal was not to build a full XAI research system. The goal was to add practical interpretability that helps:

- debug model behavior
- present the model clearly
- inspect whether outputs are plausible

That makes Grad-CAM a good engineering choice even with its limitations.

## Future improvements

If I extended the explainability side, I would consider:

- more systematic failure-case clustering
- occlusion sensitivity studies
- boundary-focused explanation tools
- a separate analysis of false positives and false negatives
- stronger explanation methods if the architecture became transformer-based

## How I would summarize explainability

"I used the predicted mask as the primary explanation, overlays as the easiest human-readable view, and Grad-CAM as a lightweight diagnostic on top of that. I would be careful not to oversell Grad-CAM - it is useful for sanity checking and debugging, but it is not a perfect explanation of the model's reasoning."
