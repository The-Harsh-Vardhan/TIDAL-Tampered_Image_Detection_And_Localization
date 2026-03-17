# Common Interview Questions

## How to use this document

Do not memorize these answers word for word. Use them as speaking points. A strong interview answer should sound natural, short, and confident.

## 1. What problem does this project solve?

Sample answer:

"It solves two related problems: image-level tamper detection and pixel-level tamper localization. Instead of only saying an image might be manipulated, the model also highlights the suspicious region, which makes the result more useful for forensic or verification workflows."

## 2. Why did you frame this as a segmentation problem?

Sample answer:

"Because the real deliverable was localization. A classifier can only tell me whether an image is suspicious, but a segmentation model tells me where the tampering is. That also makes visualization and debugging much easier."

## 3. Why did you choose U-Net?

Sample answer:

"U-Net is a strong baseline for dense prediction because it combines semantic context with fine spatial detail through skip connections. That is useful for tamper localization, where the manipulated region can be small or boundary-sensitive."

## 4. Why did you choose ResNet34 as the encoder?

Sample answer:

"ResNet34 gave me a good balance between capacity and compute cost. It is pretrained, stable to fine-tune, and light enough for a Colab T4 workflow. I wanted a practical baseline, not the heaviest possible backbone."

## 5. Why not use DeepLabV3 instead of U-Net?

Sample answer:

"DeepLabV3 is a strong alternative, especially for multi-scale context, but it is heavier and more complex. For this project, U-Net was easier to train, easier to explain, and already well aligned with the localization requirement."

## 6. Why not use Vision Transformers?

Sample answer:

"Transformers are attractive because of their global context modeling, but they usually need more compute and often benefit from larger datasets or stronger pretraining. Since this project used a relatively small forensic dataset and a single T4 GPU, a CNN-based baseline was the safer and more practical first choice."

## 7. Why did you use BCE + Dice loss?

Sample answer:

"Tampered pixels are a small minority of the image, so class imbalance is a real issue. BCE handles pixel-wise classification, while Dice helps optimize overlap on small regions. Together they are a better fit than BCE alone."

## 8. Why did you use AdamW instead of SGD?

Sample answer:

"AdamW is a strong default for transfer learning because it converges faster and needs less tuning in a notebook-scale project. SGD might work well too, but it usually requires more scheduler tuning and a longer optimization cycle."

## 9. Why was mixed precision training important?

Sample answer:

"Segmentation at `512 x 512` is memory intensive. Mixed precision reduced memory usage and improved training efficiency, which helped keep the pipeline feasible on a Colab T4 without shrinking the architecture too aggressively."

## 10. How did you choose the prediction threshold?

Sample answer:

"I swept thresholds on the validation set only and selected the one that maximized validation Pixel-F1. Then I froze that threshold for test evaluation and robustness testing. That avoids leaking information from the test set into model selection."

## 11. Why is IoU important in this project?

Sample answer:

"Because this is a localization problem, I care about how much the predicted region overlaps the true manipulated region. IoU is a direct measure of spatial overlap, so it is more meaningful than something like raw pixel accuracy."

## 12. How did you handle authentic images with empty masks?

Sample answer:

"Authentic images are treated as all-zero masks. For evaluation, I was careful with metric design because per-image precision and recall are awkward for empty masks. So the pipeline reports mixed-set precision and recall as global pixel metrics and also reports tampered-only metrics separately."

## 13. What are the main limitations of the dataset?

Sample answer:

"The main limitations are dataset size, possible annotation noise, focus on classical tampering only, and the fact that CASIA does not provide source-image grouping metadata. So even though the split is reproducible, there can still be hidden leakage risk across splits."

## 14. Why did you derive image-level detection from the mask instead of training a separate classification head?

Sample answer:

"For the MVP I wanted one simple model that prioritized localization. I used a top-k mean score from the probability map as a lightweight image-level decision rule. It is not as strong as a dedicated classification head, but it keeps the baseline simpler and easier to debug."

## 15. How did you make the project reproducible?

Sample answer:

"I used a fixed seed, persisted the split manifest, reused the same split on reruns, used deterministic DataLoader setup, and saved checkpoints plus exported result artifacts. That way I could compare runs without accidental split drift."

## 16. Why did you include Grad-CAM if the model already outputs masks?

Sample answer:

"The predicted mask shows the final output, but Grad-CAM helps me inspect what the encoder was focusing on internally. It is useful for sanity checking whether the model is looking at plausible tampered regions rather than unrelated textures or objects."

## 17. What are the limitations of Grad-CAM?

Sample answer:

"It is a qualitative diagnostic, not a proof of causality. It can show broadly important regions, but it does not fully explain the model's reasoning or provide exact pixel-level causal attribution."

## 18. Why did you test robustness against JPEG, noise, blur, and resizing?

Sample answer:

"Because tampered images are rarely seen in perfect benchmark form. They are often compressed, resized, or degraded by upload pipelines. Robustness testing checks whether the model still works under those realistic transformations."

## 19. How would you improve the model?

Sample answer:

"The most immediate upgrade would be adding a dual-head architecture with a learned image-level classifier. After that, I would test stronger encoders, multi-scale segmentation, edge-aware supervision, and possibly transformer-based hybrids if compute and dataset scale allowed."

## 20. How would you scale this system for production?

Sample answer:

"I would move the pipeline out of a notebook into modular training and inference code, add dataset versioning and experiment registry support, use a service for batched inference, monitor drift and false-positive rates, and likely add a dedicated image-level classification head for more reliable triage."

## 21. What did this project teach you as an ML engineer?

Sample answer:

"It reinforced that model choice is only part of the job. Data validation, reproducibility, threshold selection, robustness, and explainability all matter if you want results that are trustworthy and easy to discuss."

## 22. What would you say if an interviewer pushes on the fact that this is not state of the art?

Sample answer:

"I would agree with that framing. This is a strong baseline, not a frontier forensic model. That was intentional. I optimized for correctness, clarity, reproducibility, and feasibility on limited hardware first, then documented clear next steps toward more advanced models."
