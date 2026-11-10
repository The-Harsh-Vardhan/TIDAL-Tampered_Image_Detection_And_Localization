# 4. Model Architecture Review

Review target: `vK.7.1 Image Detection and Localisation.ipynb`

The architecture is a valid multitask baseline. The encoder-decoder backbone is structurally correct for localization, the skip connections are appropriate, and the classifier head attached to the bottleneck is a reasonable way to add image-level detection without building a separate network (cell 61).

## Strengths

- It is conceptually correct for joint detection and localization.
- The U-Net structure is a sensible starting point for binary mask prediction.
- The shared feature backbone keeps the model simple and assignment-friendly.
- The classification head is computationally cheap relative to the segmentation trunk.

## Weaknesses

- It is a generic vision architecture, not a forensic-specialized one.
- There is no pretrained encoder, so the model must learn everything from scratch.
- There is no explicit mechanism to emphasize subtle manipulation artifacts, boundaries, or compression inconsistencies.
- Shared features can favor coarse image-level discrimination over fine mask delineation.

The likely failure mode is that the model learns broad tamper cues and obvious splice regions but struggles on small, subtle, low-contrast, or boundary-sensitive manipulations. The notebook also saves the best checkpoint using validation classification accuracy rather than localization quality (cell 71), which further biases the final selected model toward the detection task instead of the localization task the assignment also requires.
