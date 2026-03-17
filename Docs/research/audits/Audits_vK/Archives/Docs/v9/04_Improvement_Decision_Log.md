# 04 — Improvement Decision Log

## Purpose

Consolidated decision table for every proposed improvement evaluated in Docs9. This is the authoritative record of what is in, what is out, and why.

---

## Decision Legend

| Symbol | Meaning |
|---|---|
| ✅ Approved | Implement in v9 |
| ⏸️ Deferred | Technically sound but deferred beyond v9 scope |
| ❌ Rejected | Not implemented — insufficient benefit or too risky |

---

## Full Decision Table

| # | Technique | Assignment Alignment | Feasibility (Colab/Kaggle) | Expected Benefit | Implementation Difficulty | Decision | Rationale |
|---|---|---|---|---|---|---|---|
| 1 | Learned classification head | **High** — assignment requires detection + localization | Easy — adds ~512 parameters | Moderate-High — replaces heuristic detector | Easy | ✅ Approved | Directly addresses assignment's dual-task requirement and Audit8 Pro's strongest criticism |
| 2 | ELA auxiliary input channel | **High** — improves forensic detection capability | Easy — adds 1 channel, ~2ms/image | Moderate-High — provides forensic signal for copy-move | Easy-Medium | ✅ Approved | Most feasible forensic signal addition; external resources confirm value |
| 3 | Auxiliary edge loss | **High** — improves boundary localization | Easy — single conv + loss term | Moderate — targets boundary quality | Easy-Medium | ✅ Approved | Backed by EMT-Net and ME-Net research; improves Boundary F1 |
| 4 | DeepLabV3+ comparison | **High** — provides architecture justification | Easy — one-line SMP swap | Low-Moderate — comparison evidence | Easy | ✅ Approved | Required to justify architecture choice; Audit8 Pro demands comparison |
| 5 | DataLoader optimization | **Medium** — runtime improvement | Easy — config changes | Low-Moderate — 5-15% faster training | Easy | ✅ Approved | Free performance improvement |
| 6 | Boundary F1 metric | **High** — localization quality metric | Easy — evaluation-time only | High (informational) | Easy-Medium | ✅ Approved | Assignment asks for rigorous evaluation; measures boundary precision |
| 7 | Precision-recall curves | **High** — evaluation rigor | Easy — sklearn one-liner | High (informational) | Easy | ✅ Approved | Shows full operating characteristic |
| 8 | Multi-seed validation (3 seeds) | **High** — evaluation credibility | Feasible — 3× training time | High — confidence intervals | Low (code), High (compute) | ✅ Approved | Single-seed results are anecdotal; 3 seeds provide robustness |
| 9 | Mask randomization test | **High** — validates real learning | Easy — shuffle + re-evaluate | High — shortcut falsification | Easy | ✅ Approved | Concrete test to replace pseudo-quantitative shortcut claims |
| 10 | pHash near-duplicate check | **High** — data integrity | Easy — CPU-only, minutes | High — leakage credibility | Easy | ✅ Approved | Audit8 Pro flagged missing content-level leak check |
| 11 | Augmentation ablation | **Medium** — validates pipeline | Medium — multiple runs | High (informational) | Medium | ✅ Approved | Confirms augmentation choices with evidence |
| 12 | Classification loss (for dual head) | **High** — required for #1 | Easy — standard BCE | Required | Easy | ✅ Approved | Dependent on #1 |
| 13 | Corrected dataset framing | **High** — honesty | Easy — text changes | High (credibility) | Easy | ✅ Approved | Stop calling CASIA "expected"; it is "chosen" |
| 14 | Colab end-to-end verification | **Critical** — submission gate | Easy — run notebook | Essential | Easy | ✅ Approved | Assignment deliverable is a Colab notebook |
| 15 | Per-forgery-type loss tracking | **High** — diagnostic | Easy — log split | Moderate (diagnostic) | Easy | ✅ Approved | Monitor copy-move convergence separately |
| 16 | Focal Loss (replace BCE) | **Medium** — alternative loss | Easy — swap in existing code | Moderate — uncertain marginal benefit over pos_weight | Easy | ⏸️ Deferred | v8 pos_weight + Dice already addresses imbalance; test only if small regions still fail |
| 17 | Tversky Loss | **Medium** — alternative loss | Easy — swap in existing code | Low-Moderate | Easy | ⏸️ Deferred | Similar intent to Focal; not enough expected marginal gain |
| 18 | SRM noise residuals | **High** — forensic input | Medium-Hard — filter bank implementation | High for copy-move | Medium-Hard | ⏸️ Deferred | ELA first; SRM if ELA insufficient |
| 19 | CbCr chrominance channels | **Medium** — alternative input | Easy — color conversion | Low-Moderate | Easy | ⏸️ Deferred | Lower priority than ELA; marginal expected benefit |
| 20 | Multi-scale training | **Medium** — robustness | Medium — dynamic resize pipeline | Low-Moderate | Medium | ⏸️ Deferred | CASIA images are consistent size; benefit uncertain |
| 21 | Multi-scale inference (TTA) | **Medium** — evaluation bonus | Easy-Medium — TTA wrapper | Moderate — 1-3% improvement | Easy-Medium | ⏸️ Deferred | Nice-to-have; increases inference time 3× |
| 22 | EfficientNet encoder | **Medium** — marginal architecture | Easy — one-line swap | Low | Easy | ⏸️ Deferred | Marginal improvement; DeepLabV3+ comparison is more informative |
| 23 | Cosine scheduler | **Medium** — alternative scheduler | Easy — swap | Uncertain | Easy | ⏸️ Deferred | ReduceLROnPlateau is working well |
| 24 | Gradient accumulation tuning | **Low** — minor tuning | Easy — config change | Uncertain | Easy | ⏸️ Deferred | Current setting is reasonable |
| 25 | Cross-dataset evaluation | **Medium** — generalization | Medium — new dataset prep | High (credibility) | Medium | ⏸️ Deferred | Valuable but not assignment-required; stabilize CASIA first |
| 26 | Transformer encoder (SegFormer) | **Low** — uncertain benefit | Medium-Hard — custom integration | Uncertain | Hard | ❌ Rejected | Implementation complexity + uncertain benefit for Colab |
| 27 | Stronger geometric augmentation | **Low** — risk of harm | Easy — Albumentations | Low — may destroy forensic signal | Easy | ❌ Rejected | Elastic/grid distortion creates artificial boundary artifacts |
| 28 | Full multi-branch forensic architecture | **Medium** — high complexity | Hard — exceeds Colab constraints | High | Hard | ❌ Rejected | EMT-Net/ME-Net scale is beyond internship scope and Colab memory |
| 29 | Large-scale dataset replacement | **Low** — unnecessary risk | Medium-Hard — new dataset pipeline | Uncertain | Hard | ❌ Rejected | CASIA is a valid choice; changing datasets adds risk without clear benefit |

---

## Implementation Priority Order

The approved items should be implemented in this order based on dependencies and impact:

### Phase 1: Core Architecture Changes (Highest Impact)

1. **Learned classification head** (#1, #12) — Changes model architecture and training loop
2. **ELA auxiliary channel** (#2) — Changes data pipeline and model input

### Phase 2: Loss & Training Refinement

3. **Auxiliary edge loss** (#3) — New loss component
4. **Per-forgery-type loss tracking** (#15) — Diagnostic logging
5. **DataLoader optimization** (#5) — Training speed

### Phase 3: Evaluation Stack

6. **pHash near-duplicate check** (#10) — Data validation (run once, before training)
7. **Boundary F1 metric** (#6) — New evaluation metric
8. **Precision-recall curves** (#7) — New evaluation visualization
9. **Mask randomization test** (#9) — Shortcut falsification

### Phase 4: Experiments & Validation

10. **Multi-seed validation** (#8) — Run primary config with 3 seeds
11. **DeepLabV3+ comparison** (#4) — Architecture comparison experiment
12. **Augmentation ablation** (#11) — Validate augmentation choices

### Phase 5: Documentation & Delivery

13. **Corrected dataset framing** (#13) — Notebook text updates
14. **Colab end-to-end verification** (#14) — Final verification

---

## Expected Cumulative Impact

| Phase | Key Changes | Expected Tampered-Only F1 Impact |
|---|---|---|
| v8 baseline | pos_weight, scheduler, augmentation, per-sample Dice | 0.50–0.60 |
| + Phase 1 (dual head, ELA) | Forensic input, learned detection | +0.03–0.08 |
| + Phase 2 (edge loss) | Better boundaries | +0.01–0.03 |
| + Phase 3 (evaluation) | No F1 change, better reporting | Informational |
| + Phase 4 (multi-seed) | Establishes confidence interval | Informational |
| **v9 total** | All approved improvements | **0.55–0.65** |

---

## What v9 Does NOT Include (And Why)

1. **No transformer architecture** — Uncertain benefit, high complexity, not justified for Colab.
2. **No SRM noise streams** — ELA tested first as simpler alternative.
3. **No multi-scale training** — CASIA images are consistent size; benefit uncertain.
4. **No cross-dataset evaluation** — Valuable but scope-creep for assignment submission.
5. **No dataset replacement** — CASIA is appropriate; switching adds risk.
6. **No loss function experiments beyond edge loss** — Current BCE+Dice+pos_weight is strong; avoid over-tuning.
