You are acting as a Principal AI Engineer performing a technical audit of a machine learning project documentation set.

The project is titled:

"Tampered Image Detection & Localization"

The documentation represents the final consolidated design for an internship assignment.

Your job is to audit the documentation and determine whether it is:

1. technically correct
2. aligned with the assignment requirements
3. internally consistent
4. realistically implementable in Google Colab with a T4 GPU
5. free from hallucinated claims or unnecessary complexity

The goal is to validate whether the documentation represents a **credible implementation plan for a deep learning computer vision project**.

---

# Project Goal

Build a deep learning system that:

1. Detects whether an image is tampered.
2. Localizes manipulated regions with a pixel-level segmentation mask.

The solution must run in a **single Google Colab notebook**.

Expected pipeline:

Dataset
→ preprocessing
→ augmentation
→ segmentation model
→ training
→ evaluation
→ visualization
→ optional robustness testing

---

# Final Documentation Set

The final documentation includes the following files:

Docs/

01_Assignment_Overview.md  
02_Dataset_and_Cleaning.md  
03_Data_Pipeline.md  
04_Model_Architecture.md  
05_Training_Strategy.md  
06_Evaluation_Metrics.md  
07_Visual_Results.md  
08_Robustness_Testing.md  
09_Engineering_Practices.md  
10_Project_Timeline.md  
11_Final_Submission_Checklist.md  

---

# Ground Truth Architecture

The expected baseline architecture is:

Dataset:
CASIA v2.0

Pipeline:

CASIA dataset
→ dataset cleaning
→ mask binarization
→ augmentation
→ segmentation model

Model:

U-Net
+ pretrained encoder (ResNet34 or EfficientNet-B0/B1)

Loss:

Binary Cross Entropy + Dice Loss

Evaluation:

IoU
Dice / F1
Precision
Recall
Image-level detection derived from mask

Visualization:

Original Image
Ground Truth Mask
Predicted Mask
Overlay Visualization

Bonus:

robustness tests

JPEG compression  
Gaussian noise  
resizing

---

# Audit Tasks

Perform a **full documentation audit**.

For each document:

1. Verify that the technical explanations are correct.
2. Confirm that the instructions are implementable.
3. Check whether the document contradicts any other document.
4. Detect hallucinated claims or unsupported benchmarks.
5. Identify unnecessary complexity.
6. Identify missing technical details.

---

# Required Checks

Validate the following key components:

Dataset
- CASIA dataset usage
- mask pairing
- mask binarization
- dataset split policy

Data Pipeline
- preprocessing steps
- augmentation policy
- leakage prevention

Model
- segmentation architecture
- pretrained backbone
- compatibility with Colab GPU memory

Training
- optimizer
- checkpointing
- mixed precision usage

Evaluation
- correct metric definitions
- proper threshold handling
- separation of validation vs test

Visualization
- predicted mask visualization
- overlay outputs

Bonus Experiments
- robustness testing design

---

# Detect These Common Problems

Flag if the documentation contains:

1. Over-engineered architectures
2. unnecessary ML tools
3. unsupported research claims
4. evaluation mistakes
5. data leakage risks
6. unrealistic training setups

---

# Output Format

Return a structured audit report with the following sections.

## 1. Overall Assessment

- Is the documentation technically valid?
- Is the design appropriate for the assignment?

Score the project from 1–10.

---

## 2. Document-by-Document Review

For each document:

Document Name  
Purpose  
Technical Accuracy Score (1–10)  
Issues Found  
Recommendations

---

## 3. Cross-Document Consistency

List contradictions between documents.

---

## 4. Implementation Risks

Identify areas where the documentation may fail during implementation.

---

## 5. Missing Components

List technical items that are not sufficiently documented.

---

## 6. Suggested Improvements

Provide concrete improvements to make the documentation stronger.

---

# Important Instruction

Be skeptical.

Assume the documents may contain LLM hallucinations or overly confident claims.

Your goal is to ensure the documentation forms a **credible and executable engineering plan**, not just theoretical descriptions.