# Final 48-Hour Execution Plan

| Field | Value |
|-------|-------|
| **Deadline** | ~2 days from 2026-03-14 |
| **Deliverable** | Executed notebook + trained model weights |
| **Platform** | Kaggle (preferred) or Google Colab |
| **Strategy** | **Two-Track: ETASR Classification + Pretrained Localization** |

---

## Strategic Overview

The project now runs two parallel experiment tracks:

| Track | Purpose | Framework | Output | Assignment Value |
|-------|---------|-----------|--------|-----------------|
| **Track 1 (ETASR, vR.1.x)** | Ablation study, paper reproduction | TensorFlow/Keras | Binary classification | Methodology documentation |
| **Track 2 (Pretrained, vR.P.x)** | Pixel-level localization | PyTorch + SMP | Tampered region masks | **Final submission** |

**Priority:** Track 2 (Pretrained) is the priority for the assignment deliverable. Track 1 continues in parallel for ablation documentation value.

---

## Phase 1: Complete ETASR Ablation (Hours 0-4)

### Tasks
1. Run `vR.1.3 — ETASR Run-02 Image Detection and Localisation.ipynb` on Kaggle
2. Record results and compare against vR.1.1 baseline
3. Determine verdict (POSITIVE/NEUTRAL/NEGATIVE)
4. If time permits, continue vR.1.4 (BatchNorm) → vR.1.5 (LR scheduler)

### Exit Criteria
- vR.1.3 has been run and results recorded
- Updated tracking table in ablation_master_plan.md
- Decision made on whether to continue Track 1 or focus on Track 2

---

## Phase 2: Build Pretrained Notebook vR.P.0 (Hours 4-8)

### Tasks
1. Generate `vR.P.0` notebook (PyTorch + SMP + ResNet-34 + UNet)
2. Verify notebook structure: imports, SMP install, data loading, model build, training loop, evaluation
3. Ensure ImageNet normalization applied to RGB inputs
4. Confirm encoder frozen, decoder trainable (~500K params)
5. Verify ground truth mask availability in CASIA v2.0
6. If no masks: use classification-first approach (ForensicClassifier with GAP head)

### Exit Criteria
- vR.P.0 notebook ready for Kaggle execution
- Model compiles and encoder is properly frozen
- Data pipeline loads RGB images at 384×384 with ImageNet normalization

---

## Phase 3: Run Pretrained Baseline on Kaggle (Hours 8-14)

### Tasks
1. Upload vR.P.0 notebook to Kaggle
2. Run all cells: SMP install → data loading → model build → training → evaluation
3. Training should complete in ~15-20 minutes (25 epochs, batch=16, 384×384)
4. Monitor: val_loss should decrease, pixel-F1 should improve
5. Capture all metrics: Pixel-F1, IoU, AUC, classification accuracy

### Exit Criteria
- Training completes without errors or OOM
- Best epoch > 1 (model actually learned)
- Pixel-F1 ≥ 0.30 or classification accuracy ≥ 85% (whichever mode used)
- All visualizations render (training curves, predictions, overlays)

### Red Flags (Abort Criteria)
- OOM at batch=16 → reduce to batch=8
- Pixel-F1 stuck at 0 → check mask quality / loss function
- Training not converging → check ImageNet normalization applied

---

## Phase 4: Analyze and Iterate (Hours 14-20)

### Tasks
1. Audit vR.P.0 results — compare against Track 1 (ETASR vR.1.1: 88.38% acc, 0.9601 AUC)
2. If vR.P.0 is promising: proceed to vR.P.1 (gradual unfreeze)
3. If vR.P.0 underperforms: diagnose (wrong normalization? mask quality? frozen too aggressively?)
4. Record results in pretrained tracking table
5. Generate and run vR.P.1 if time allows

### Exit Criteria
- vR.P.0 audited and results recorded
- Decision made on next pretrained version
- At least one successful pretrained run completed

---

## Phase 5: Polish Best Model for Submission (Hours 20-30)

### Tasks
1. Select the best model across both tracks
2. Clean up the best notebook for submission:
   - Clear, well-structured markdown explanations
   - All visualizations present: training curves, ROC, confusion matrix
   - For pretrained: Original / GT / Predicted / Overlay visualization grid
3. Ensure model weights saved (.keras or .pth)
4. Write final results discussion comparing ETASR baseline vs pretrained

### Exit Criteria
- Submission notebook runs cleanly end-to-end
- All assignment requirements addressed
- Model weights saved and downloadable

---

## Phase 6: Final Review and Documentation (Hours 30-36)

### Tasks
1. Re-run best notebook top-to-bottom ("Run All") to ensure clean execution
2. Verify no cell has errors and all outputs are visible
3. Update ablation master plan with all final results
4. Prepare documentation folder for submission
5. Final review of all visualizations and metrics

### Exit Criteria
- Clean end-to-end execution of submission notebook
- All outputs visible in notebook
- Model weights saved and downloadable
- Documentation complete

---

## Time Budget Summary

| Phase | Hours | Focus | Risk Level |
|-------|-------|-------|-----------|
| Complete ETASR ablation | 0-4 | Track 1 (vR.1.3+) | Low |
| Build pretrained notebook | 4-8 | Track 2 (vR.P.0) | Low |
| Run pretrained on Kaggle | 8-14 | Track 2 execution | Medium |
| Analyze and iterate | 14-20 | Both tracks | Low |
| Polish for submission | 20-30 | Best model | Low |
| Final review | 30-36 | Documentation | Low |
| **Buffer** | **36-48** | **Issues / extra iterations** | **Available** |

**Total active work: ~36 hours. Buffer: ~12 hours.**

---

## Contingency Plans

### If pretrained accuracy < ETASR baseline
1. Check ImageNet normalization is applied (mean/std)
2. Verify encoder is frozen (print param.requires_grad)
3. Try unfreezing last 2 encoder blocks with low LR (1e-5)
4. Fall back to classification-first approach (ForensicClassifier)

### If no CASIA ground truth masks available
1. Use classification approach with pretrained encoder (ForensicClassifier)
2. Generate pseudo-masks from ELA thresholding
3. Use the classification comparison as the primary result

### If OOM on T4 at 384×384
1. Reduce batch size to 8
2. Reduce image size to 256×256
3. Use EfficientNet-B0 (smallest memory footprint)

### If SMP install fails on Kaggle
1. Use `!pip install segmentation-models-pytorch` in first cell
2. Verify PyTorch version compatibility
3. Fall back to manual UNet decoder in Keras with ResNet50

### If training doesn't converge (pretrained)
1. Check decoder LR (should be 1e-3)
2. Verify loss function computes gradient (check for NaN)
3. Try unfreezing encoder earlier
4. Verify data augmentation is NOT applied (clean baseline first)

---

## Deliverables Checklist

### Track 1 (ETASR Classification)
- [ ] vR.1.3 (class weights) executed and results recorded
- [ ] vR.1.4+ if time permits
- [ ] Ablation tracking table updated

### Track 2 (Pretrained Localization) — PRIMARY
- [ ] vR.P.0 notebook generated and executed
- [ ] vR.P.1 (gradual unfreeze) if time permits
- [ ] Trained model weights (.pth file)
- [ ] Pixel-level prediction visualizations

### Documentation
- [ ] Updated DocsR1/ folder (all files updated with pretrained findings)
- [ ] Pretrained Models/ analysis folder (4 documents)
- [ ] Audit reports for all executed runs
- [ ] Final comparison table (ETASR vs Pretrained)

### Submission Notebook
- [ ] Clean end-to-end execution
- [ ] All assignment requirements met:
  - [ ] Image tampering detection
  - [ ] ELA preprocessing documented
  - [ ] CNN/encoder model trained
  - [ ] Proper evaluation metrics
  - [ ] Tampered region masks (if localization achieved)
  - [ ] Original / GT / Predicted / Overlay visualization
