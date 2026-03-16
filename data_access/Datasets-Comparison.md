# Dataset Comparison for Tampered Image Detection & Localization

This document compares the 12 candidate datasets against the current project setup in this repository: a dual-head CNN model for image-level tamper detection and pixel-level localization, trained in Kaggle with limited GPU resources. The current working assumption is the same one used in the v8 notebook pipeline: `384x384` inputs, a `ResNet34` encoder, a U-Net style decoder, and a segmentation-first training loop with an auxiliary classification head.

For project context, see the [assignment brief](../Assignment.md) and the current [Kaggle notebook](../notebooks/v8-tampered-image-detection-localization-kaggle-run-01.ipynb).

## Evaluation Criteria

The comparison below uses six project-specific criteria:

- **Architecture compatibility**: whether the dataset can support a CNN segmentation model with both classification and localization outputs.
- **Dataset quality**: scale, class balance, annotation availability, and manipulation coverage.
- **Training usefulness**: whether it is a good fit for training the current model under Kaggle limits.
- **Difficulty**: how realistic and generalization-relevant the manipulations are.
- **Compute feasibility**: whether it is practical to train on directly, needs sub-sampling, or is better kept for test-only use.
- **Research usage**: how commonly the dataset appears in tampering detection or localization papers and benchmark discussions.

### Rating language used in this report

- **Ground truth masks**:
  - `Yes`: official pixel-level or region-level annotations are available and can be converted into binary masks.
  - `Yes (derived)`: localization masks are commonly used in practice, but they come from corrected or third-party annotations rather than the original official release.
  - `No`: no reliable pixel-level supervision for a segmentation model.
- **Research usage**: `High / Medium / Low`
- **Training suitability**: `Strong / Limited / Not recommended`
- **Testing suitability**: `Strong / Useful / Limited`

Important interpretation detail: the mask column answers the practical question, "Can I realistically put this into `images/`, `masks/`, and `metadata.csv` for a segmentation model?" It does not mean that every dataset ships in the same binary-mask format out of the box.

## Master Comparison Table

| Dataset | Year | Number of images | Tampering types | Ground truth masks | Research usage | Training suitability | Testing suitability | Final recommendation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CASIA v1.0 | 2013 | 1,721 (800 authentic + 921 tampered) | Mostly splicing / copy-paste style tampering | Yes (derived) | Medium | Limited | Useful | Optional legacy benchmark only |
| CASIA v2.0 | 2013 | 12,614 (7,491 authentic + 5,123 tampered) | Splicing, copy-move, post-processed composite edits | Yes (derived) | High | Strong | Useful | **Core training dataset** |
| CoMoFoD | 2013 | 260 base sets; 10,400 small-set post-processed images | Copy-move with rotation, scaling, distortion, compression, blur, noise, brightness, contrast | Yes | High | Strong | Useful | **Controlled supplemental training subset** |
| Coverage Dataset | 2016 | 200 (100 original + 100 forged) | Copy-move with similar-but-genuine objects | Yes | Medium | Limited | Strong | **External validation/test** |
| Columbia Image Splicing Dataset | 2006 | 363 (183 authentic + 180 spliced) | Splicing | Yes | High | Limited | Strong | **External validation/test** |
| CG-1050 | 2019 | 1,150 images plus 1,380 masks | Copy-move, cut-paste, retouching, colorizing | Yes | Low | Limited | Useful | Optional future ablation dataset |
| MICC-F220 | 2011 | 220 (110 authentic + 110 tampered) | Copy-move | No | Medium | Not recommended | Limited | Do not use for segmentation training |
| MICC-F2000 | 2011 | 2,000 (1,300 authentic + 700 tampered) | Copy-move | No | Medium | Not recommended | Limited | Do not use for segmentation training |
| IMD2020 | 2020 | 70,000 synthetic images + 2,010 real-life manipulated pairs | Copy-paste, splicing, retouching, inpainting, post-processed local edits | Yes | High | Strong | Strong | **Supplemental training + external benchmark** |
| NIST Nimble 2016 (NC16) | 2016 | 1,124 commonly used local-manipulation images from the 1,200-image kickoff dev set | Splicing, cloning/copy-move, removal, other local manipulations | Yes | High | Limited | Strong | **External benchmark, not core train** |
| SMIFD-1000 | 2022 | 1,000 (500 authentic + 500 manipulated) | Real social-media manipulations, mostly compositing and context edits | Yes | Medium | Limited | Strong | **External validation/test** |
| Fantastic Reality Dataset | 2019 | ~32,000 (16k real + 16k fake) | Splicing / compositing | Yes | Medium | Limited | Useful | Optional future large-scale splice pretraining |

## Detailed Dataset Notes

### 1. CASIA v1.0

- **Architecture compatibility**: Only partially compatible. The official release is a legacy tampering benchmark without native segmentation masks, so your model can only use it after adopting a derived mask set. That adds annotation risk for a localization assignment.
- **Dataset quality**: Small by modern standards at 1,721 images. It is useful historically, but the manipulation diversity is narrow and the images are fixed-resolution legacy JPEGs.
- **Manipulation realism and difficulty**: Mostly classic splice or copy-paste style edits. Good for a baseline sanity check, but weaker for modern generalization.
- **Kaggle feasibility**: Directly feasible. The issue is not size, it is annotation confidence and limited diversity.
- **Best use**: Legacy supplementary experiment or small benchmark. Not a core training set for your current dual-head localizer.
- **Recommendation for this assignment**: Do not anchor the project on CASIA v1.0. If used at all, treat it as a secondary legacy comparison dataset with clearly disclosed derived masks.
- **Sources**: [CASIA paper](https://doi.org/10.1109/ChinaSIP.2013.6625374), [official CASIA site](http://forensics.idealtest.org/), [derived CASIA v1 masks](https://github.com/namtpham/casia1groundtruth)

### 2. CASIA v2.0

- **Architecture compatibility**: Strong practical fit. Although the official release does not ship clean binary localization masks, the corrected and commonly reused derived mask sets make it workable for a segmentation model. This is already the family of dataset your repo notebook is built around.
- **Dataset quality**: Large enough for a Kaggle-first baseline at 12,614 total images. It covers authentic images plus both splice-like and copy-move style tampering. The scale is still modest compared with newer synthetic datasets, but it is strong enough to train a ResNet34 U-Net.
- **Manipulation realism and difficulty**: More realistic than CASIA v1.0, but still a classical forensics benchmark with legacy biases and known annotation noise. It is useful, not sufficient by itself for strong generalization claims.
- **Kaggle feasibility**: Directly feasible. This is the safest full training set for your current `384x384` notebook.
- **Best use**: Core training dataset, with internal train/val/test splits and careful leakage controls.
- **Recommendation for this assignment**: Make CASIA v2.0 localization your anchor dataset. It is the most assignment-safe choice because it is widely cited, practical to unify, and already matched to your current pipeline.
- **Sources**: [CASIA paper](https://doi.org/10.1109/ChinaSIP.2013.6625374), [official CASIA site](http://forensics.idealtest.org/), [derived CASIA v2 masks](https://github.com/namtpham/casia2groundtruth), [corrected ground-truth repo](https://github.com/SunnyHaze/CASIA2.0-Corrected-Groundtruth)

### 3. CoMoFoD

- **Architecture compatibility**: Strong for segmentation, especially if you want explicit copy-move localization. It ships with both colored masks and binary masks.
- **Dataset quality**: Very good for copy-move-specific benchmarking. The key nuance is that the dataset has 260 base scenes but many transformed and post-processed variants, so naive full use can overweight one synthetic manipulation family.
- **Manipulation realism and difficulty**: The transformations are intentionally diverse and useful for robustness, but still synthetic. It improves coverage of rotation, scaling, distortion, blur, noise, contrast, brightness, and JPEG effects better than CASIA does.
- **Kaggle feasibility**: Feasible only as a controlled subset. The full post-processed expansion is not what you want to dump into a first-pass Kaggle run.
- **Best use**: Supplemental training for copy-move robustness, plus targeted evaluation.
- **Recommendation for this assignment**: Use the small `512x512` category only, and sample a restrained subset per base image so the training distribution does not collapse into CoMoFoD-style synthetic copy-move artifacts.
- **Sources**: [official project page](https://www.vcl.fer.hr/comofod/), [dataset description](https://www.vcl.fer.hr/comofod/comofod.html), [download page](https://www.vcl.fer.hr/comofod/download.html), [paper](https://ieeexplore.ieee.org/document/6658316)

### 4. Coverage Dataset

- **Architecture compatibility**: Good for localization, but you need a small preprocessing step because the duplicated region and forged region masks are annotated separately and should be merged into one binary tamper mask for your unified pipeline.
- **Dataset quality**: Small but high-value. Its main strength is not size; it is difficulty. The dataset was built to expose failure modes when similar-but-genuine objects create copy-move ambiguity.
- **Manipulation realism and difficulty**: Challenging and useful for generalization, especially against false positives in semantically repetitive scenes.
- **Kaggle feasibility**: Easy to evaluate on; too small to matter as a main training set.
- **Best use**: External validation or testing, especially for copy-move-specific failure analysis.
- **Recommendation for this assignment**: Keep Coverage out of the core training pool and use it as an external benchmark to show that your model is not only fitting CASIA-style shortcuts.
- **Sources**: [official dataset repo](https://github.com/wenbihan/coverage), [paper](https://doi.org/10.1109/ICIP.2016.7532339)

### 5. Columbia Image Splicing Dataset

- **Architecture compatibility**: Good, but not plug-and-play. Columbia provides region-style edgemasks rather than a plain ready-made binary foreground map, so you need to convert its annotation format to your binary mask convention.
- **Dataset quality**: High quality but small: 183 authentic and 180 spliced images. The images are uncompressed and high-resolution for the time, which makes the dataset clean but also distribution-shifted relative to modern web and social-media imagery.
- **Manipulation realism and difficulty**: Strong for splice benchmarking, weak for manipulation diversity. No copy-move, no removal, no platform recompression.
- **Kaggle feasibility**: Very easy to use as an external evaluation set. Not large enough for meaningful training by itself.
- **Best use**: External validation/test set for splicing localization.
- **Recommendation for this assignment**: Use Columbia as a clean external splice benchmark, not as a training dataset.
- **Sources**: [official dataset page](https://www.ee.columbia.edu/ln/dvmm/downloads/authsplcuncmp/), [download form](https://www.ee.columbia.edu/ln/dvmm/downloads/authsplcuncmp/dlform.html), [paper](https://ieeexplore.ieee.org/document/4036658)

### 6. CG-1050

- **Architecture compatibility**: Technically compatible because official masks are available, but it is not a neat match to your task. The dataset mixes copy-move, cut-paste, retouching, and colorizing, which broadens manipulation coverage but also blends in categories that are less central to classical tamper localization benchmarks.
- **Dataset quality**: Moderate size with 100 originals, 1,050 tampered images, and 1,380 masks. It includes both color and grayscale originals, and many manipulated versions per base image.
- **Manipulation realism and difficulty**: Useful for stress-testing broad manipulation sensitivity, but more synthetic and less standard than CASIA, Columbia, NIST, or IMD2020.
- **Kaggle feasibility**: Feasible after resizing, but the raw high-resolution images make preprocessing heavier than the final training value justifies for a first assignment run.
- **Best use**: Optional ablation or future robustness study.
- **Recommendation for this assignment**: Do not include CG-1050 in the first-pass core training recipe. It is more useful later if you specifically want to test cross-manipulation robustness beyond splice and copy-move.
- **Sources**: [Mendeley dataset](https://data.mendeley.com/datasets/dk84bmnyw9/2), [paper](https://doi.org/10.1016/j.dib.2019.104864)

### 7. MICC-F220

- **Architecture compatibility**: Poor for your current model because there are no official pixel-level masks for F220. That makes it unsuitable for segmentation supervision.
- **Dataset quality**: Tiny at 220 images. Historically important in copy-move papers, but too small and too weakly annotated for a modern multi-task localizer.
- **Manipulation realism and difficulty**: Classical copy-move benchmark with some geometric variation. Good for historical comparison, not for a serious segmentation training setup.
- **Kaggle feasibility**: Easy to handle, but there is no point for mask-supervised training.
- **Best use**: At most, image-level copy-move detection experiments or literature comparison outside the main assignment path.
- **Recommendation for this assignment**: Exclude it from the segmentation training pool.
- **Sources**: [MICC lab page](https://lci.micc.unifi.it/labd/2015/01/copy-move-forgery-detection-and-localization/), [paper](https://ieeexplore.ieee.org/document/5734842)

### 8. MICC-F2000

- **Architecture compatibility**: Same core problem as MICC-F220: no official pixel-level masks for the F2000 subset.
- **Dataset quality**: Larger than F220 at 2,000 images, but still only useful for copy-move image-level evaluation. The images are also high-resolution, which raises preprocessing cost without solving the missing-mask problem.
- **Manipulation realism and difficulty**: Better than tiny toy datasets, but still narrow and not a modern localization benchmark.
- **Kaggle feasibility**: Directly feasible as files, not useful for your segmentation objective.
- **Best use**: Historical copy-move image-level benchmark only.
- **Recommendation for this assignment**: Do not use it for the core model because it cannot supervise the segmentation head.
- **Sources**: [MICC lab page](https://lci.micc.unifi.it/labd/2015/01/copy-move-forgery-detection-and-localization/), [paper](https://ieeexplore.ieee.org/document/5734842)

### 9. IMD2020

- **Architecture compatibility**: Strong. IMD2020 is one of the best matches for a modern localization model because it combines large-scale authentic imagery with manipulated subsets that have binary masks.
- **Dataset quality**: Very strong. The official site exposes 35,000 clean images, 35,000 inpainting images, and 2,010 real-life manipulated images paired with originals and masks. It is much broader than CASIA in camera diversity and manipulation realism.
- **Manipulation realism and difficulty**: Strong for generalization. The real-life subset is particularly valuable because it reflects uncontrolled internet manipulations rather than only lab-synthesized edits.
- **Kaggle feasibility**: Not realistic to train in full for an internship assignment on Kaggle unless heavily staged. It should be sub-sampled.
- **Best use**: Supplemental training and strong external evaluation, especially if you want to make credible generalization claims beyond CASIA.
- **Recommendation for this assignment**: Use the full real-life manipulated subset if possible, and only a capped synthetic subset for the large-scale manipulated pool. IMD2020 is the best modernization dataset to add after CASIA.
- **Sources**: [official project page](https://staff.utia.cas.cz/novozada/db/), [WACV 2020 paper](https://doi.org/10.1109/WACVW50321.2020.9096940), [open-access paper page](https://openaccess.thecvf.com/content_WACVW_2020/html/w4/Novozamsky_IMD2020_A_Large-Scale_Annotated_Dataset_Tailored_for_Detecting_Manipulated_Images_WACVW_2020_paper.html)

### 10. NIST Nimble 2016 (NC16)

- **Architecture compatibility**: Strong when the task subset includes local manipulations and released reference masks. The dataset is built for detection and localization evaluation, but it is more complex operationally than academic plug-and-play datasets.
- **Dataset quality**: High benchmark value. NIST-style data is widely used in media forensics evaluation, but it is heterogeneous and challenge-oriented rather than curated for easy notebook training.
- **Manipulation realism and difficulty**: High. It includes realistic manipulation workflows and is significantly harder than small legacy benchmarks.
- **Kaggle feasibility**: Better treated as test-only. Access, preprocessing, metadata handling, and dataset size make it a poor candidate for your first training pool.
- **Best use**: External benchmark for reporting credible evaluation on a literature-standard challenge dataset.
- **Recommendation for this assignment**: Keep NC16 out of core training and use it, if access is available, as a held-out benchmark to demonstrate robustness.
- **Sources**: [OpenMFC / NIST resource page](https://mfc.nist.gov/), [MFC datasets paper](https://ieeexplore.ieee.org/document/8638296)

### 11. SMIFD-1000

- **Architecture compatibility**: Good. The paper explicitly describes binary pixel-level annotations and rich metadata, which makes it compatible with a segmentation setup.
- **Dataset quality**: Moderate size but high-value realism. The main benefit is not volume; it is the fact that the images are collected from social-media settings where recompression and low quality make localization harder.
- **Manipulation realism and difficulty**: High relative difficulty for its size. This is one of the better datasets for checking whether your model survives beyond curated academic image quality.
- **Kaggle feasibility**: Fully feasible. The question is strategic priority, not resource fit.
- **Best use**: External validation/test set, or a small fine-tuning stage after a base model is already trained.
- **Recommendation for this assignment**: Use SMIFD-1000 as a realism-oriented external benchmark. If you have extra time, it is a better fine-tuning candidate than a brand-new full training anchor.
- **Sources**: [paper](https://doi.org/10.1016/j.fsidi.2022.301392), [dataset repo mentioned by the paper](https://github.com/rana23/SMIFD-1000)

### 12. Fantastic Reality Dataset

- **Architecture compatibility**: Strong in principle. It provides pixel-level manipulated-region annotations and was designed for segmentation-style splice localization.
- **Dataset quality**: Large and useful, with about 16k real and 16k fake images plus additional object annotations. It is much closer to a modern large-scale splice dataset than CASIA or Columbia.
- **Manipulation realism and difficulty**: Strong for splice localization, but still narrower than a broad tampering benchmark because it is centered on splicing/compositing rather than the full space of manipulations.
- **Kaggle feasibility**: Possible, but not the right first move for this assignment. It is large enough to increase storage, preprocessing, and training time noticeably, and it does not cover copy-move the way CoMoFoD does.
- **Best use**: Future large-scale pretraining or a second-phase splice-focused study.
- **Recommendation for this assignment**: Keep Fantastic Reality out of the first-pass training stack. It is attractive, but it is a better v2 dataset than a first submission dataset.
- **Sources**: [NeurIPS paper page](https://papers.neurips.cc/paper/8315-the-point-where-reality-meets-fantasy-mixed-adversarial-generators-for-image-splice-detection), [paper PDF](https://papers.nips.cc/paper/8315-the-point-where-reality-meets-fantasy-mixed-adversarial-generators-for-image-splice-detection.pdf), [historical dataset site referenced by the paper](http://zefirus.org/MAG)

## Final Recommendation

### Best dataset combination for this project

- **Core training**: CASIA v2.0 localization derivative
- **Supplemental training**: CoMoFoD small-category subset
- **Supplemental training for realism**: IMD2020 real-life manipulated subset plus a capped synthetic subset
- **External validation/testing**: Columbia, Coverage, SMIFD-1000, and NIST NC16 when access is available
- **Optional future only**: Fantastic Reality

### Why this is the best balance

- **Training diversity**: CASIA v2.0 gives you a broad and practical base. CoMoFoD adds difficult copy-move structure that CASIA alone does not model well. IMD2020 adds camera diversity and more modern manipulation realism.
- **Localization capability**: Every recommended core dataset has usable localization supervision. Datasets without reliable masks are excluded from the main training set because they do not help the segmentation head.
- **Kaggle compute limits**: CASIA is already proven feasible in your current notebook. CoMoFoD and IMD2020 are recommended only as capped subsets, not as full raw merges.
- **Assignment fit**: This combination directly supports both image-level detection and pixel-level localization without changing your model family or moving away from the current Kaggle notebook design.

### Datasets that should not be core training anchors

- **CASIA v1.0**: too small and mask provenance is derived
- **MICC-F220 / MICC-F2000**: no official localization masks
- **CG-1050**: interesting but not benchmark-critical for a first submission
- **Fantastic Reality**: attractive, but too large and splice-focused for the first Kaggle-limited training recipe

## Practical Kaggle Strategy

Use the dataset mix in phases rather than trying to train on everything at once.

### Phase 1: Stable baseline

- Train on the full CASIA v2.0 localization derivative at `384x384`.
- Keep the current dual-head `ResNet34 + U-Net` architecture and your current augmentation pipeline.
- Validate internally with a fixed train/val split and retain a held-out test split.

### Phase 2: Improve copy-move robustness

- Add CoMoFoD small-set data as a **controlled subset**, not the entire expanded post-processed pool.
- Prefer a small number of variants per base scene so the model does not overfit to CoMoFoD-specific synthetic distortions.
- Use this stage to test whether copy-move localization improves without destabilizing splice performance.

### Phase 3: Improve realism and cross-dataset generalization

- Add the full IMD2020 real-life manipulated subset if storage allows.
- Optionally add a capped synthetic IMD2020 subset rather than the full 35k manipulated pool.
- Keep Columbia, Coverage, SMIFD-1000, and NIST NC16 for evaluation, not training.

### Practical recommendation for a first internship submission

- **Minimum credible setup**: CASIA v2.0 only
- **Better setup if time allows**: CASIA v2.0 + CoMoFoD subset
- **Best setup under the current plan**: CASIA v2.0 + CoMoFoD subset + IMD2020 real-life subset, with Columbia/Coverage/SMIFD/NIST used only for external testing

## References

- CASIA: [paper](https://doi.org/10.1109/ChinaSIP.2013.6625374), [official site](http://forensics.idealtest.org/), [CASIA v1 derived masks](https://github.com/namtpham/casia1groundtruth), [CASIA v2 derived masks](https://github.com/namtpham/casia2groundtruth), [corrected CASIA v2 ground truth](https://github.com/SunnyHaze/CASIA2.0-Corrected-Groundtruth)
- CoMoFoD: [project](https://www.vcl.fer.hr/comofod/), [dataset page](https://www.vcl.fer.hr/comofod/comofod.html), [download page](https://www.vcl.fer.hr/comofod/download.html), [paper](https://ieeexplore.ieee.org/document/6658316)
- Coverage: [dataset repo](https://github.com/wenbihan/coverage), [paper](https://doi.org/10.1109/ICIP.2016.7532339)
- Columbia: [official dataset page](https://www.ee.columbia.edu/ln/dvmm/downloads/authsplcuncmp/), [download form](https://www.ee.columbia.edu/ln/dvmm/downloads/authsplcuncmp/dlform.html), [paper](https://ieeexplore.ieee.org/document/4036658)
- CG-1050: [dataset](https://data.mendeley.com/datasets/dk84bmnyw9/2), [paper](https://doi.org/10.1016/j.dib.2019.104864)
- MICC: [dataset page](https://lci.micc.unifi.it/labd/2015/01/copy-move-forgery-detection-and-localization/), [paper](https://ieeexplore.ieee.org/document/5734842)
- IMD2020: [official site](https://staff.utia.cas.cz/novozada/db/), [WACV paper](https://doi.org/10.1109/WACVW50321.2020.9096940)
- NIST NC16 / OpenMFC: [official resource page](https://mfc.nist.gov/), [MFC datasets paper](https://ieeexplore.ieee.org/document/8638296)
- SMIFD-1000: [paper](https://doi.org/10.1016/j.fsidi.2022.301392), [dataset repo referenced by authors](https://github.com/rana23/SMIFD-1000)
- Fantastic Reality: [NeurIPS paper page](https://papers.neurips.cc/paper/8315-the-point-where-reality-meets-fantasy-mixed-adversarial-generators-for-image-splice-detection), [paper PDF](https://papers.nips.cc/paper/8315-the-point-where-reality-meets-fantasy-mixed-adversarial-generators-for-image-splice-detection.pdf), [historical dataset site](http://zefirus.org/MAG)

## Notes and caveats

- The `research usage`, `training suitability`, `testing suitability`, and `final recommendation` columns are project judgments for this assignment, not official dataset labels.
- `Yes (derived)` is used deliberately for CASIA because the practical localization masks commonly used in segmentation work are not part of the native official release.
- NIST counts vary by release and subset. This document uses the commonly cited local-manipulation subset relevant to image localization, while also noting that the official kickoff development release is larger.
- Columbia and Coverage both require annotation normalization before they fit a unified binary-mask pipeline.
- If you need the simplest possible first submission, use CASIA v2.0 only and treat every other dataset as external evaluation or later-stage improvement.
