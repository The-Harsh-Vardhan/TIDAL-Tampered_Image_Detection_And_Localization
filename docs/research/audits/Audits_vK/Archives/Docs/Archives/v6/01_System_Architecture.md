# System Architecture

End-to-end data flow for the tampered image detection and localization system. This is a **baseline system aligned with assignment constraints**, not a frontier research architecture. Stronger models (edge-enhanced, multi-trace, transformer-based) are documented as future work in `11_Research_Alignment.md`.

---

## High-Level Pipeline

```
┌──────────────────────────────────────────────────────────────┐
│  1. DATA ACQUISITION                                          │
│     Kaggle pre-mounted dataset at /kaggle/input/             │
│     Case-insensitive discovery → IMAGE/ + MASK/ directories  │
│     Dynamic pair discovery → (image, mask, label, forgery)   │
│     Corruption guard + dimension validation                  │
└──────────────────┬───────────────────────────────────────────┘
                   ▼
┌──────────────────────────────────────────────────────────────┐
│  2. PREPROCESSING                                             │
│     Mask binarization (threshold > 0)                        │
│     Authentic → zero mask                                    │
│     Stratified split 70 / 15 / 15 (seed=42)                 │
│     Data leakage verification (set-intersection assertions)  │
│     Split manifest persisted to JSON                         │
└──────────────────┬───────────────────────────────────────────┘
                   ▼
┌──────────────────────────────────────────────────────────────┐
│  3. DATA PIPELINE                                             │
│     TamperingDataset(pairs, transform)                       │
│     Spatial augmentation (flip, rotate, resize to 384×384)   │
│     ImageNet normalization                                   │
│     DataLoader(batch=4, workers=2, pin_memory=True)          │
└──────────────────┬───────────────────────────────────────────┘
                   ▼
┌──────────────────────────────────────────────────────────────┐
│  4. MODEL                                                     │
│     smp.Unet(resnet34, imagenet, in_channels=3, classes=1)   │
│     Output: raw logits (B, 1, 384, 384)                      │
└──────────────────┬───────────────────────────────────────────┘
                   ▼
┌──────────────────────────────────────────────────────────────┐
│  5. TRAINING                                                  │
│     BCEDiceLoss (BCE + Dice, equal weight)                   │
│     AdamW: encoder 1e-4, decoder 1e-3                        │
│     AMP + gradient accumulation (4 steps → eff. batch 16)   │
│     Gradient clipping (max_norm=1.0)                         │
│     Early stopping on val Pixel-F1 (patience=10)             │
│     Checkpoints → /kaggle/working/checkpoints/               │
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
│     Grad-CAM heatmaps (with safety checks)                   │
│     Diagnostic overlays (TP=green, FP=red, FN=blue)         │
│     Failure case analysis                                    │
└──────────────────┬───────────────────────────────────────────┘
                   ▼
┌──────────────────────────────────────────────────────────────┐
│  9. ROBUSTNESS TESTING                                        │
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
| Runtime | Kaggle (T4 GPU) | Pre-mounted dataset, consistent GPU, no Drive mount needed |
| Dataset | CASIA Splicing Detection + Localization | Pre-organized IMAGE/MASK directories; covers splicing + copy-move |
| Architecture | U-Net + ResNet34 | Proven segmentation backbone; pretrained encoder compensates for small dataset |
| Image size | 384 × 384 | Reduced from 512 for T4 VRAM headroom; within safe operating range |
| Loss | BCE + Dice | Handles class imbalance (<5% tampered pixels); complementary gradients |
| Data split | 70/15/15 | More balanced evaluation than 85/7.5/7.5 |
| Mask binarization | `> 0` | Captures all annotated pixels; avoids losing low-intensity annotations |
| Image-level detection | Top-k mean of pixel probabilities | More stable than `max(prob_map)` while keeping the MVP single-head |
| Threshold policy | Single threshold from val sweep | Avoids data snooping on test set; one threshold for pixel and image-level |
| W&B authentication | Kaggle Secrets API | Secure; no interactive login prompt needed |
| Explainability | Grad-CAM + mask overlays | Lightweight methods that verify model focus on tampered regions |

---

## Environment

- **Runtime:** Kaggle Notebook
- **GPU:** T4 (15 GB VRAM)
- **Image resolution:** 384 × 384
- **VRAM budget:** ~4–6 GB estimated with AMP at 384² resolution
- **Output storage:** `/kaggle/working/` (checkpoints, results, plots)
- **Dataset mount:** `/kaggle/input/` (read-only, pre-mounted by Kaggle)
