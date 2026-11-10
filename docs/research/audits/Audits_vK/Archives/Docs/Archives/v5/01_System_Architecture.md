# System Architecture

End-to-end data flow for the tampered image detection and localization system. This is a **baseline system aligned with assignment constraints**, not a frontier research architecture. Stronger models (edge-enhanced, multi-trace, transformer-based) are documented as future work in `11_Research_Alignment.md`.

---

## High-Level Pipeline

```
┌──────────────────────────────────────────────────────────────┐
│  1. DATA ACQUISITION                                          │
│     Kaggle API → CASIA Splicing Detection + Localization     │
│     Dynamic discovery → (image, mask, label, forgery_type)   │
│     Dimension validation for image–mask pairs                │
└──────────────────┬───────────────────────────────────────────┘
                   ▼
┌──────────────────────────────────────────────────────────────┐
│  2. PREPROCESSING                                             │
│     Mask binarization (threshold > 128)                      │
│     Authentic → zero mask                                    │
│     Stratified split 85 / 7.5 / 7.5 (seed=42)               │
│     Split manifest persisted to JSON                         │
└──────────────────┬───────────────────────────────────────────┘
                   ▼
┌──────────────────────────────────────────────────────────────┐
│  3. DATA PIPELINE                                             │
│     TamperingDataset(pairs, transform)                       │
│     MVP: spatial augmentation only                           │
│     Phase 2: + photometric, + optional ELA (4th channel)     │
│     DataLoader(batch=4, workers=2, pin_memory=True)          │
└──────────────────┬───────────────────────────────────────────┘
                   ▼
┌──────────────────────────────────────────────────────────────┐
│  4. MODEL                                                     │
│     smp.Unet(resnet34, imagenet, in_channels=3, classes=1)   │
│     Output: raw logits (B, 1, 512, 512)                      │
│     Phase 2: in_channels=4 if ELA concatenated as 4th ch     │
└──────────────────┬───────────────────────────────────────────┘
                   ▼
┌──────────────────────────────────────────────────────────────┐
│  5. TRAINING                                                  │
│     BCEDiceLoss (BCE + Dice, equal weight)                   │
│     AdamW: encoder 1e-4, decoder 1e-3                        │
│     AMP + gradient accumulation (4 steps → eff. batch 16)   │
│     Early stopping on threshold-aware val Pixel-F1           │
│     Checkpoints → Drive or local artifact dir                │
│     W&B logging (optional, guarded via USE_WANDB)            │
└──────────────────┬───────────────────────────────────────────┘
                   ▼
┌──────────────────────────────────────────────────────────────┐
│  6. THRESHOLD SELECTION (validation set only)                 │
│     Sweep 50 thresholds [0.1, 0.9] on val Pixel-F1          │
│     Selected threshold used for all downstream evaluation    │
└──────────────────┬───────────────────────────────────────────┘
                   ▼
┌──────────────────────────────────────────────────────────────┐
│  7. EVALUATION (test set)                                     │
│     Mixed-set metrics + tampered-only metrics                │
│     Pixel-F1, IoU, Precision, Recall                         │
│     Image Accuracy, Image AUC-ROC                            │
│     Forgery-type breakdown (splicing vs copy-move)           │
└──────────────────┬───────────────────────────────────────────┘
                   ▼
┌──────────────────────────────────────────────────────────────┐
│  8. VISUALIZATION & EXPLAINABILITY                            │
│     4-column grid: Original | GT | Predicted Mask | Overlay  │
│     Training curves (loss, F1, IoU)                          │
│     F1-vs-threshold sweep plot                               │
│     Grad-CAM heatmaps on encoder features                    │
│     Failure case analysis                                    │
└──────────────────┬───────────────────────────────────────────┘
                   ▼
┌──────────────────────────────────────────────────────────────┐
│  9. ROBUSTNESS TESTING (Phase 3 bonus)                       │
│     JPEG QF 70/50, Gaussian noise, blur, resize 0.75/0.5×   │
│     Image-only degradation; masks stay clean                 │
│     Reuse validation-selected threshold                      │
└──────────────────┬───────────────────────────────────────────┘
                   ▼
┌──────────────────────────────────────────────────────────────┐
│  10. EXPERIMENT TRACKING (optional, guarded via USE_WANDB)    │
│     W&B integrated throughout notebook (not standalone)      │
│     Log train/val loss, F1, IoU per epoch                    │
│     Log final test metrics, prediction visualizations        │
│     Log best model checkpoint as artifact                    │
│     All calls behind USE_WANDB flag                          │
└──────────────────────────────────────────────────────────────┘
```

---

## Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Single notebook | Yes | Assignment requirement; simplifies delivery |
| Dataset | CASIA Splicing Detection + Localization | Pre-organized Image/Mask directories; direct filename correspondence; covers splicing + copy-move |
| Architecture | U-Net + ResNet34 | Proven segmentation backbone; pretrained encoder compensates for small dataset. Consistent with transfer-learning patterns described in survey papers (P14, P15) |
| Loss | BCE + Dice | Handles class imbalance (<5% tampered pixels); complementary gradients. Validated by reference notebook `image-detection-with-mask.ipynb` |
| Image-level detection | Top-k mean of pixel probabilities | More stable than `max(prob_map)` while keeping the MVP single-head |
| Threshold policy | Single threshold from val sweep | Avoids data snooping on test set; one threshold for pixel and image-level |
| Experiment tracking | W&B (optional, guarded) | Industry standard; notebook works without it; all calls behind `USE_WANDB` flag |
| Explainability | Grad-CAM + mask overlays | Lightweight methods that verify model focus on tampered regions without heavy frameworks |

---

## Environment

- **Runtime:** Google Colab (free tier)
- **GPU:** T4 (target; compatible CUDA GPU acceptable)
- **VRAM budget:** ~6 GB estimated with AMP (must be confirmed at runtime; fits T4's 15 GB comfortably)
- **Persistent storage:** Google Drive when available, otherwise the local artifact directory

---

## ELA Channel Convention (Cross-Document Reference)

If ELA is enabled in Phase 2, it is concatenated as a **4th input channel** (RGB + ELA grayscale → 4 channels). This changes `in_channels` from 3 to 4. ImageNet pretrained weights cannot be used directly when `in_channels != 3`.

This approach is inspired by research showing ELA can amplify compression artifacts invisible in RGB (P1, P7 from Research Paper Analysis Report), and is validated by the `document-forensics-using-ela-and-rpa.ipynb` reference notebook which demonstrates ELA preprocessing for forensic classification.

SRM features (Phase 3) use 3 high-pass kernels concatenated with RGB → `in_channels=6`. SRM and ELA are separate experimental paths and are not combined.
