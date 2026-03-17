# Audit6 Pro - Skeptical ML Teardown

This report is a deliberate stress test of the tampered-image localization project documented in `Docs6/` and implemented in:

- `notebooks/tamper_detection_v6_kaggle.ipynb`
- `notebooks/tamper_detection_v6_colab.ipynb`

The goal is not to decide whether the project is "good enough" for an assignment. The goal is to identify where a senior interviewer or ML architecture reviewer would push back, what claims are weakly supported, and where the system is likely to fail outside the narrow CASIA setup.

## 1. Overall Project Maturity Score (1-10)

**Overall score: 5.0 / 10**

This is a coherent baseline, not a production-ready tamper localization system and not a research-competitive forensic model. The pipeline is internally understandable, but the problem framing is thin, the dataset is narrow and leakage-prone, the model choice is justified mostly by convenience, and the evaluation stack is vulnerable to inflated conclusions.

### Scoring rubric

| Dimension | Weight | Score | Weighted contribution |
|---|---:|---:|---:|
| Problem framing | 10% | 6.0 | 0.60 |
| Dataset choice + leakage risk | 25% | 4.0 | 1.00 |
| Architecture + loss + training stability | 20% | 5.5 | 1.10 |
| Evaluation + explainability + robustness | 25% | 4.8 | 1.20 |
| Engineering maturity + research alignment | 20% | 5.5 | 1.10 |
| **Total** | **100%** |  | **5.00** |

## 2. What the project does well

- It correctly frames tamper localization as a segmentation problem rather than stopping at image-level classification.
- It documents practical engineering details that many student projects omit: corruption checks, split persistence, threshold sweep, checkpointing, and deterministic loader setup.
- It acknowledges several limitations openly, especially CASIA scope limits and the fact that Grad-CAM is only a diagnostic tool.
- It uses a reasonable starter baseline: pretrained U-Net, BCE plus Dice, AdamW, AMP, and early stopping are all defensible baseline choices.

## 3. Major weaknesses

- The project never really defines the operational problem. It says "tamper detection and localization," but does not specify the end user, decision workflow, acceptable false positive rate, or whether localization is actually required in the target setting.
- CASIA is too old and too narrow to support strong claims about modern tampering. The dataset primarily covers classical splicing and copy-move artifacts, not diffusion edits, semantic inpainting, GAN retouching, or composite edits from current tools.
- Leakage control is weak relative to the claim surface. File-path disjointness is verified, but source-image lineage, near-duplicate content, and derivative manipulations are not controlled.
- The model choice is justified mostly by "baseline" and "assignment constraints." That is an execution reason, not a technical reason that ResNet34 U-Net is the right bias for forensic localization.
- The image-level detector is a heuristic bolted onto the segmentation output instead of a properly learned head. That makes the detection story much weaker than the localization story.
- The loss discussion is incomplete. BCE plus Dice is reasonable, but the implementation still ignores explicit positive reweighting, hard-example emphasis, boundary supervision, and small-region sensitivity.
- Evaluation is exposed to optimistic reporting. Mixed-set metrics can look strong simply because authentic images with empty masks are treated as perfect true negatives.
- Robustness testing is mostly post-processing corruption, not robustness to new manipulation families. That is resilience to nuisance transforms, not robustness to modern forgery generation.
- Explainability claims are modest in wording, but the project still risks overselling diagnostic heatmaps that are not quantitatively validated and are not naturally matched to segmentation.
- The engineering story is notebook-first and single-run oriented. It is not yet a scalable training or inference system.

## 4. Architecture critiques

- `Docs6/03_Model_Architecture.md` defends U-Net plus ResNet34 as "proven" and transfer-learning friendly, but that does not answer the harder question: why is a natural-image pretrained RGB encoder the right feature extractor for forensic traces that often live in compression noise, boundary inconsistencies, or frequency artifacts rather than semantic content?
- The project does not justify why DeepLabV3+, HRNet, SegFormer, Mask2Former, or hybrid transformer decoders were rejected. That omission matters because tampered regions are often small, irregular, and context-dependent.
- The architecture is single-stream RGB only. `Docs6/11_Research_Alignment.md` itself admits that stronger papers rely on edge, noise, or multi-domain fusion, which undercuts the claim that the chosen model is anything more than a convenience baseline.
- The image-level decision path is particularly weak. `Docs6/03_Model_Architecture.md` describes a handcrafted score derived from pixel probabilities rather than a dedicated classification head. Even if acceptable for an MVP, it is hard to defend as a serious system design.
- Training uses a micro-batch of 4. With a ResNet34 encoder full of BatchNorm layers, gradient accumulation does not fix noisy batch-statistics behavior because BatchNorm still sees the micro-batch, not the effective batch.
- The Dice component shown in `Docs6/04_Training_Strategy.md` and the v6 notebooks computes intersection and union across the full batch rather than per image. That lets large masks dominate small masks and weakens the claim that the loss is tuned for small tampered regions.

## 5. Dataset limitations

- `Docs6/02_Dataset_and_Preprocessing.md` explicitly relies on CASIA-derived image-mask pairs because they are convenient on Kaggle. That makes the project easier to run, but convenience is not the same as representativeness.
- `Docs6/13_References.md` anchors the dataset to CASIA 2.0 from 2013. That is a major warning sign for any claim about current tamper detection relevance.
- The dataset covers classical splicing and copy-move only. The documentation itself excludes GAN edits, deepfakes, and AI-generated manipulations, which means the project is not testing the failure modes that matter most in 2026.
- Annotation quality is uncertain. The docs mention coarse or noisy mask boundaries, but the project does not estimate annotation noise or boundary uncertainty, so reported IoU and F1 may partly measure label quality limits rather than model quality.
- The split is stratified only by `forgery_type`. It does not stratify by source image, scene, camera, object category, mask size, or post-processing condition.
- File-overlap assertions do not address content leakage. If CASIA contains related derivatives, crops, or multiple tampered variants from the same source image, the current split strategy will not detect it.
- Resizing everything to 384 x 384 may destroy some of the very forensic evidence the model is supposed to learn, especially for tiny spliced regions and local boundary artifacts.

## 6. Evaluation weaknesses

- `Docs6/05_Evaluation_Methodology.md` uses Pixel-F1 and IoU as the main success story. Those are standard segmentation metrics, but they do not automatically prove forensic quality. A model can achieve decent overlap while learning dataset-specific shortcuts.
- The true-negative policy returns 1.0 for F1, IoU, precision, and recall when both prediction and ground truth are empty. That is mathematically defensible, but on a mixed authentic-plus-tampered test set it can materially inflate average localization metrics.
- The project does not separate boundary quality from region quality. Tamper localization often fails at edges, but there is no boundary F1, contour accuracy, Hausdorff-style metric, or calibration analysis.
- The same threshold is reused for pixel-level binarization and image-level detection. That is a convenience decision, not a principled one. The optimal operating point for mask quality is not necessarily the right operating point for binary image detection.
- The evaluation story depends on a single in-domain test split. There is no cross-dataset validation, no leave-source-out protocol, and no external benchmark to show that performance survives beyond CASIA.
- The project reports AUC-ROC for image-level detection, but the image-level signal is heuristic rather than learned. That makes the AUC less persuasive than it would be in a properly supervised detection head.
- Failure analysis is qualitative. There is no error taxonomy tied to object size, boundary complexity, image content, or manipulation pipeline.

## 7. Research gaps

- `Docs6/11_Research_Alignment.md` openly lists stronger model families, but the project remains far from them: no transformer encoder-decoder, no frequency-domain stream, no SRM/noise residual branch, no edge supervision, and no multi-trace fusion.
- The project cites modern tamper-localization papers as future work without testing even a minimal ablation against one stronger baseline. That leaves the architecture choice weakly defended.
- There is no cross-manipulation generalization study. Modern tamper detection research increasingly asks whether a model trained on one manipulation family transfers to another.
- There is no attempt to distinguish semantic edits from low-level compositing artifacts. That is a major gap for any discussion of diffusion-era tampering.
- The project has no calibration, uncertainty, or abstention mechanism, which is a problem if the intended deployment setting involves human review or high-stakes triage.

## 8. Interview questions the author must be ready to answer

1. Why is localization the right objective here instead of first solving image-level tamper detection well?
2. Why did you choose CASIA when it is old, limited to classical manipulations, and known to have possible source-image leakage issues?
3. What evidence do you have that file-path disjointness is enough to prevent leakage in a derived-forgery dataset?
4. Why is U-Net plus ResNet34 better for this task than DeepLabV3+, HRNet, SegFormer, or a transformer-based segmentation model?
5. ImageNet pretraining helps semantics. Why should it help forensic traces such as noise inconsistency or recompression artifacts?
6. Why is BCE plus Dice the right loss, and why did you not try Focal, Tversky, Lovasz, or boundary-aware losses?
7. Your Dice implementation aggregates across the whole batch. How does that affect small masks versus large masks?
8. With a batch size of 4 and a BatchNorm encoder, how do you know the training statistics are stable?
9. Why should IoU and F1 on CASIA convince anyone that the system can detect modern GAN or diffusion edits?
10. Why do you return perfect scores for empty-mask authentic images, and how much does that inflate your mixed-set metrics?
11. Why should Grad-CAM on a segmentation model be trusted when the target is not quantitatively validated?
12. How would this system fail in production if you ran it on millions of social-media images from unseen cameras and editing tools?

## 9. Recommendations for strengthening the project

- Rewrite the problem statement around a concrete use case: moderation triage, newsroom verification, insurance fraud review, or forensic analyst assistance. Then define the actual failure costs.
- Add a second evaluation regime that isolates tampered images only and reports boundary-sensitive metrics, size-stratified results, and calibration metrics.
- Replace or augment CASIA with at least one newer benchmark, or state much more aggressively that the system only targets classical manipulations.
- Implement a source-aware or perceptual-duplicate-aware split if lineage metadata is unavailable.
- Add at least one stronger comparison baseline: DeepLabV3+, SegFormer, or a lightweight dual-stream forensic model.
- Revisit the loss stack: per-sample Dice, `pos_weight` or Focal/Tversky variants, and possibly boundary supervision for sharper masks.
- Replace the heuristic image-level detector with a learned classification head or a separately calibrated detector.
- Expand robustness beyond JPEG/noise/blur/resize to semantic inpainting, style transfer, diffusion edits, color harmonization, and compositing pipelines.
- Treat explainability as diagnostic only unless you add quantitative sanity checks.
- Move the project out of notebook-only form into a config-driven training pipeline if you want to claim engineering maturity.

## Supporting files

- `01_Problem_Framing_and_Dataset_Risks.md`
- `02_Model_Training_and_Engineering_Critique.md`
- `03_Evaluation_Robustness_and_Research_Gaps.md`
- `04_Doc_vs_Notebook_Consistency_Notes.md`
