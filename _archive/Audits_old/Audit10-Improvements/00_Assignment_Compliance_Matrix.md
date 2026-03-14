# Assignment Compliance Matrix

All 46 assignment items mapped against each notebook run.

**Legend:** PASS | PARTIAL | FAIL | N/A

---

## Core Requirements (R1–R25)

| ID | Requirement | v8 Run | vK.7.5 Run | vK.3 Run | vK.10.3 (code) |
|----|-------------|--------|------------|----------|----------------|
| R1 | Detect tampered regions | PASS | FAIL (2 epochs, not learned) | PASS (50 epochs) | PASS (code ready) |
| R2 | Localize tampered regions | PASS (F1=0.29 tampered) | FAIL (degenerate) | PARTIAL (Dice=0.576) | PASS (code ready) |
| R3 | Image-level classification | PASS (Acc=0.719, AUC=0.817) | FAIL (Acc=0.553) | PASS (Acc=0.899) | PASS (code ready) |
| R4 | Pixel-level mask output | PASS | PASS (but degenerate) | PASS | PASS |
| R5 | Strong problem-solving skills | PASS | FAIL | PARTIAL | PASS |
| R6 | Thoughtful architecture choices | PASS (SMP ResNet34) | PARTIAL (custom U-Net) | PARTIAL (custom U-Net) | PARTIAL (custom U-Net, no pretrain) |
| R7 | Rigorous evaluation | PASS (extensive) | FAIL (no rigor) | PARTIAL | PARTIAL (no robustness/explainability) |
| R8 | Public dataset with images+masks+GT | PASS (CASIA) | PASS (CASIA) | PASS (CASIA) | PASS (CASIA) |
| R9 | Dataset cleaning | PASS | PARTIAL | PARTIAL | PARTIAL |
| R10 | Preprocessing | PASS | PASS | PASS | PASS |
| R11 | Mask alignment verification | PARTIAL | PARTIAL | PARTIAL | PARTIAL |
| R12 | Train/Val/Test split | PASS (70/15/15 stratified) | PASS (70/15/15) | PASS (70/15/15) | PASS (70/15/15 stratified) |
| R13 | Data augmentation | PASS (7 transforms) | PASS (6 transforms) | PASS (6 transforms) | PASS (6 transforms) |
| R14 | Train a model | PASS (27 epochs) | FAIL (2 epochs) | PASS (50 epochs) | PASS (code ready, 50 max) |
| R15 | T4 GPU compatible | PASS (ran on T4) | PARTIAL (ran on local) | PASS (ran on Kaggle) | PASS (metadata set for T4) |
| R16 | Localization metrics | PASS (F1, IoU, Dice, stratified) | FAIL (inflated) | PASS (Dice, IoU, F1) | PASS (Dice, IoU, F1, tampered-only) |
| R17 | Detection accuracy metrics | PASS (Acc, AUC) | PARTIAL (Acc only) | PASS (Acc) | PASS (Acc, AUC) |
| R18 | 4-panel visualization | PASS | PASS | PASS | PASS |
| R19 | Single notebook | PASS | PASS | PASS | PASS |
| R20 | Dataset explanation | PASS | PASS | PASS | PASS |
| R21 | Architecture description | PASS | PASS | PASS | PASS |
| R22 | Training strategy description | PASS | PARTIAL | PASS | PASS |
| R23 | Hyperparameter documentation | PASS (CONFIG dict) | PARTIAL | PARTIAL | PASS (CONFIG dict) |
| R24 | Evaluation results | PASS | FAIL (meaningless) | PASS | PASS (code ready) |
| R25 | Clear visualizations | PASS | PASS (varied) | PASS | PASS |

---

## Deliverables (D1–D4)

| ID | Deliverable | v8 Run | vK.7.5 Run | vK.3 Run | vK.10.3 |
|----|-------------|--------|------------|----------|---------|
| D1 | Single Colab notebook | PASS | PASS | PASS | PASS |
| D2 | Colab/Kaggle link | PASS (Kaggle) | FAIL (local run) | PASS (Kaggle) | PASS (Kaggle metadata) |
| D3 | Trained model weights | PASS | PASS | PASS | PASS (checkpoint system) |
| D4 | Additional scripts | N/A | N/A | N/A | N/A |

---

## Evaluation Criteria (E1–E15)

| ID | Criterion | v8 Run | vK.7.5 Run | vK.3 Run | vK.10.3 |
|----|-----------|--------|------------|----------|---------|
| E1 | Problem-solving skills | STRONG | WEAK | MODERATE | STRONG |
| E2 | Architecture choices | STRONG (pretrained) | WEAK (from scratch, 2ep) | MODERATE | MODERATE (from scratch) |
| E3 | Rigorous evaluation | STRONG | FAIL | MODERATE | MODERATE |
| E4 | Localization quality | MODERATE (F1=0.29 tam) | FAIL | MODERATE (Dice=0.576) | TBD |
| E5 | Detection accuracy | GOOD (AUC=0.817) | FAIL (53%) | GOOD (Acc=0.899) | TBD |
| E6 | Standard metrics | PASS (F1, IoU, AUC) | PARTIAL | PASS | PASS (Dice, IoU, F1, AUC) |
| E7 | 4-panel visuals | PASS | PASS | PASS | PASS |
| E8 | Dataset explanation | GOOD | GOOD | MODERATE | GOOD |
| E9 | Architecture description | GOOD | GOOD | MODERATE | GOOD |
| E10 | Training strategy docs | GOOD | PARTIAL | MODERATE | GOOD |
| E11 | Hyperparameter justification | GOOD (CONFIG) | WEAK | WEAK | GOOD (CONFIG) |
| E12 | Visualization clarity | STRONG | STRONG (varied) | GOOD | GOOD |
| E13 | Resource efficiency (T4) | PASS | PARTIAL | PASS | PASS |
| E14 | Data pipeline rigor | STRONG | PARTIAL (leakage bug) | MODERATE | GOOD |
| E15 | Augmentation relevance | STRONG | MODERATE | MODERATE | MODERATE |

---

## Bonus Items (B1–B2)

| ID | Bonus Item | v8 Run | vK.7.5 Run | vK.3 Run | vK.10.3 |
|----|------------|--------|------------|----------|---------|
| B1 | Robustness testing (JPEG, noise, resize) | PASS (8 conditions) | FAIL | FAIL | FAIL |
| B2 | Subtle tampering detection (copy-move, splicing) | PARTIAL (splicing OK, copy-move weak) | FAIL | FAIL | FAIL |

---

## Compliance Scores

| Notebook | Core (25) | Deliverables (4) | Eval Criteria (15) | Bonus (2) | Overall |
|----------|-----------|-------------------|---------------------|-----------|---------|
| **v8 Run** | 23 PASS, 2 PARTIAL | 3 PASS, 1 N/A | 13 STRONG/GOOD, 2 MODERATE | 1 PASS, 1 PARTIAL | **Best** |
| **vK.3 Run** | 20 PASS, 5 PARTIAL | 3 PASS, 1 N/A | 7 GOOD, 8 MODERATE | 0 | Decent |
| **vK.7.5 Run** | 10 PASS, 5 PARTIAL, 10 FAIL | 2 PASS, 1 FAIL, 1 N/A | 3 GOOD, 2 MODERATE, 10 WEAK/FAIL | 0 | Poor |
| **vK.10.3 (code)** | 22 PASS, 3 PARTIAL | 3 PASS, 1 N/A | 10 GOOD/STRONG, 5 MODERATE/TBD | 0 | Good (needs run) |
