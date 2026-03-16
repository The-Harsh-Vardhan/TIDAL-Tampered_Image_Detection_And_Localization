# Resource 07: document-forensics-using-ela-and-rpa.ipynb

## 1. Resource Overview
- Title: document-forensics-using-ela-and-rpa
- Source: Kaggle notebook
- Author: Author not identified from local resource
- Year: Year not identified from local resource

This notebook is a cheap forensic baseline. Its goal is to classify images as genuine or tampered using Error Level Analysis and a threshold rule, not to localize tampered pixels.

## 2. Technical Summary
The notebook implements an `ela(...)` function that resizes an image, re-saves it as JPEG, and computes a grayscale difference image. It then applies an `rpa(...)` function that measures the standard deviation of that ELA output and compares it against a threshold. A `detectFraud(...)` wrapper returns either `Genuine` or `Tampered`.

The dataset path points to CASIA 2.0 on Kaggle. The notebook performs a grid search over a small set of ELA scale, JPEG quality, and threshold values, then reports precision, recall, and F1. That is practical and fast, but it is still a heuristic classifier with zero localization ability.

## 3. Key Techniques Used
- Error Level Analysis as a handcrafted forensic preprocessing step
- Threshold-based decision rule using standard deviation of the ELA map
- Small grid search over preprocessing hyperparameters

ELA is useful because recompression can exaggerate tampered-region inconsistencies. The problem is that ELA alone is often brittle and heavily tied to JPEG behavior.

## 4. Strengths of the Approach
This notebook is lightweight and honest about what it is. It can run easily on Kaggle or Colab, and it offers a quick sanity check before any deep model is trained.

It is also useful as a preprocessing reference because it makes the ELA transform concrete. That matters if the main project wants to test RGB plus ELA channels later.

## 5. Weaknesses or Limitations
It does not solve the assignment. There is no mask prediction, no learned localization, and no real generalization argument. It is basically a tuned forensic threshold wrapped in a notebook.

It also risks threshold overfitting to CASIA-specific compression artifacts. The grid search may look rigorous, but it is still just parameter tuning around a fragile handcrafted signal.

## 6. Alignment With Assignment
Alignment: Medium

It is partially aligned because it addresses tamper detection and is extremely compatible with Kaggle hardware. It is not fully aligned because it completely misses the localization requirement.

## 7. Relevance to My Project
Useful parts:
- ELA preprocessing idea
- Cheap baseline for whether the dataset carries obvious compression cues

Unnecessary parts:
- The threshold-rule detector as a final system
- Any claim that this notebook validates localization performance

## 8. Should This Be Used?
Use partially for inspiration.

Use it as a preprocessing reference or sanity-check baseline. Do not let it become the core design of a localization assignment.

## 9. Integration Ideas
- Add an RGB plus ELA ablation in the main project.
- Use the notebook as a non-deep baseline to show why mask supervision is needed.
- If ELA helps, fuse it as an auxiliary channel instead of replacing the segmentation model with heuristics.

## 10. Citation
document-forensics-using-ela-and-rpa. Local notebook copy: `Research Papers/document-forensics-using-ela-and-rpa.ipynb`. Kaggle notebook. Author and year not identified from local resource.
