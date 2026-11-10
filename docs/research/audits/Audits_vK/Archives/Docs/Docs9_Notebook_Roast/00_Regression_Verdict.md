# 00 — Regression Verdict

**Reviewer:** Arshad Siddiqui, Principal AI Engineer, BigVision LLC  
**Date:** 2026-03-13  
**Subject:** v9 vs v8 — Tampered Image Detection & Localization  

---

## Verdict: REGRESSION

This is not a close call. v9 is categorically worse than v8 as an assignment deliverable, and worse than v8 as a piece of evidence that any training happened at all.

---

## Primary Evidence

### v8 notebook (run-01)
- **Execution count cells:** 20+ cells with non-null `execution_count` values (1, 2, 3, 4…)
- **Output cells:** 20+ cells with non-empty `"outputs"` arrays containing actual terminal output, training logs, and Kaggle console text
- **Line count:** 16,976 lines — includes model training logs, metric tables, visualisation output
- **Status:** A real, executed artifact with recoverable evidence

### v9 notebook (colab)
- **Execution count cells:** Every single code cell shows `"execution_count": null`
- **Output cells:** All 14 tracked output arrays are **empty** — `"outputs": []`
- **Line count:** 2,569 lines — structurally complete code but zero runtime evidence
- **Status:** An unexecuted shell. It has never been trained.

This is not a "partial regression." The v9 notebook has never produced a prediction, never computed a metric, never rendered a visualisation. It is a well-organised draft, not a submission artifact.

---

## Regression Scorecard

| Dimension | v8 run-01 | v9 Colab | Verdict |
|-----------|-----------|----------|---------|
| Executed end-to-end | ✅ Yes | ❌ No | **REGRESSION** |
| Training outputs present | ✅ Yes | ❌ No | **REGRESSION** |
| Metric evidence in notebook | ✅ Yes | ❌ No | **REGRESSION** |
| Visual output cells | ✅ Yes | ❌ No | **REGRESSION** |
| Colab-compatible paths | ⚠️ Kaggle only | ✅ Yes (code) | Improvement (in code only) |
| Loss function complexity | BCE + Dice (2 terms) | BCE + Dice + edge + cls (4 terms) | Worsened — untested |
| Batch size for T4 Colab | 64 (Kaggle 2×T4) | 4 — dramatically reduced | Regression (throughput) |
| pHash leakage guard | ❌ None | ✅ Full Hamming search | Improvement (in code only) |
| Learned cls head | ❌ Heuristic max | ✅ FC head | Improvement (in code only) |
| ELA input channel | ❌ None | ✅ 4-ch RGB+ELA | Unknown — never tested |
| Architecture comparison | ❌ None | Defined but disabled | No evidence |

---

## Why "Improvements in Code Only" Counts as Nothing

Every improvement listed in v9's favour is implemented in an unexecuted notebook. There are no outputs, no loss curves, no metric tables, no threshold analysis, no predicted masks — nothing. The code could work, could silently fail, could OOM on the first training batch at 384×384 with a 4-channel ResNet34 + Dice + edge loss + cls head stack.

**Writing an improvement is not the same as demonstrating one.**

v8 run-01 has real loss curves. It has a real threshold scan. It has a real 4-panel visualisation grid. Those are material evidence. v9 has none of that.

---

## Qualitative Regression: Architecture Complexity Without Validation

v8 carried a disciplined, single-purpose segmentation loss (BCE + per-sample Dice). v9 piles on:

- classification BCE loss (weight 0.5)
- edge-weighted BCE loss (weight 0.3)
- ELA 4th channel (needs its own normalisation path)
- pHash grouping before splitting (correct, but expensive at 12,614 images)
- full Boundary-F1 evaluation
- PR curves
- per-forgery-type loss tracking

None of this has been validated against each other. Adding 3 loss terms simultaneously without ablation is how training instability gets introduced and never diagnosed. The project promoted from "let's fix one known gap" to "let's fix everything at once" — and then never ran it.

---

## Summary

v9 introduced real improvements on paper. Learned classification head, cleaner leakage control, ELA as a forensic cue, richer evaluation. These are directionally correct.

None of them produced evidence. The assignment requires a working, demonstrable system. v9 is not that. v8 run-01 is. The regression verdict stands.

**v9 must be executed, validated, and retained with outputs before it can replace v8 run-01 as the submission artifact.**
