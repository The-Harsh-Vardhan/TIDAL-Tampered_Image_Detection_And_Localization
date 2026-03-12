# Doc vs Notebook Consistency Notes

This appendix is intentionally secondary. The main audit is about ML quality, not paperwork. Even so, a strong interviewer will treat documentation drift as a credibility problem because it makes every technical claim harder to trust.

Only mismatches explicitly confirmed from `Docs6/` and the v6 notebooks are included here.

## 1. Primary notebook mismatch

**Docs say**

- `Docs6/00_Master_Report.md` lists `tamper_detection_v5.1_kaggle.ipynb` as the primary notebook.
- `Docs6/12_Complete_Notebook_Structure.md` says the authoritative notebook is `tamper_detection_v5.1_kaggle.ipynb` and describes a 61-cell, 17-section structure.

**Notebook reality**

- The repo now contains `notebooks/tamper_detection_v6_kaggle.ipynb` and `notebooks/tamper_detection_v6_colab.ipynb`.
- Both v6 notebooks have 66 cells and 22 sections.

**Why it matters**

If the docs point to the wrong implementation artifact, every downstream claim about runtime, structure, evaluation, and saved artifacts becomes suspect.

**Fix**

Make the v6 notebooks the explicit source of truth and rewrite the notebook-structure doc around the real 66-cell layout.

## 2. Image-level detection logic mismatch

**Docs say**

- `Docs6/01_System_Architecture.md`
- `Docs6/03_Model_Architecture.md`
- `Docs6/05_Evaluation_Methodology.md`
- `Docs6/11_Research_Alignment.md`
- `Docs6/12_Complete_Notebook_Structure.md`

All describe image-level detection as a top-k mean over pixel probabilities.

**Notebook reality**

- In both v6 notebooks, the evaluation logic uses `tamper_score = probs[i].view(-1).max().item()`.

**Why it matters**

This is not a wording issue. It changes how image-level detection behaves, how sensitive the detector is to single hot pixels, and how any claimed threshold rationale should be interpreted.

**Fix**

Either update the docs to match `max(prob_map)` or change the notebooks back to top-k mean. Leaving them inconsistent weakens the whole evaluation story.

## 3. Runtime story mismatch: Kaggle-only docs vs actual Colab variant

**Docs say**

- `Docs6/00_Master_Report.md`, `Docs6/01_System_Architecture.md`, `Docs6/08_Engineering_Practices.md`, `Docs6/09_Experiment_Tracking.md`, `Docs6/10_Project_Timeline.md`, and `Docs6/12_Complete_Notebook_Structure.md` present a Kaggle-native story with:
  - dataset pre-mounted at `/kaggle/input/`
  - artifacts saved to `/kaggle/working/`
  - W&B auth via Kaggle Secrets
  - no Colab-specific operations

**Notebook reality**

- `notebooks/tamper_detection_v6_colab.ipynb` uses:
  - Kaggle API download via `KAGGLE_USERNAME` and `KAGGLE_KEY`
  - Colab Secrets access
  - Google Drive mount and Drive-based checkpoint storage

**Why it matters**

The runtime story is materially different, not just a path rename. Reproduction steps, secrets handling, storage behavior, and failure modes differ across Kaggle and Colab.

**Fix**

Split documentation into:

- shared pipeline behavior
- Kaggle-specific setup
- Colab-specific setup

## Bottom line

The consistency issues do not prove the model is wrong, but they do weaken technical credibility. In an interview, the safest response is:

"The docs lagged behind a v6 implementation update. The core pipeline stayed similar, but the documentation needs to be realigned before I would treat it as authoritative."
