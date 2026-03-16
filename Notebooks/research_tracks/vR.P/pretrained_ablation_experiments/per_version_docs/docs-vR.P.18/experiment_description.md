# vR.P.18 — Compression Robustness Testing

## Experiment Description

**Version:** vR.P.18
**Track:** Pretrained Localization (Track 2)
**Parent:** vR.P.3 (uses P.3 trained model — no new training)
**Single Variable Changed:** Test-time evaluation protocol — evaluate under JPEG recompression at Q=70, Q=80, Q=90, Q=95

### Hypothesis

ELA-based forgery detection degrades when test images undergo additional JPEG compression, because recompression adds noise that obscures original manipulation artifacts. Higher compression (lower Q) destroys more forensic signal. The degradation pattern quantifies the model's practical robustness.

### Protocol

```
For each quality Q in [Original, Q=95, Q=90, Q=80, Q=70]:
    |
    v
  Test image → JPEG recompress at Q → Compute ELA (Q=90) → Predict → Measure
    |
    v
  Compare: Pixel F1, IoU, AUC, Image Accuracy across all Q levels
```

This experiment does NOT train a new model. It loads the best checkpoint from vR.P.3 and evaluates under 5 different conditions.

### What Changes from P.3

| Aspect | P.3 | P.18 |
|--------|-----|------|
| Training | 25 epochs | NONE (load P.3 checkpoint) |
| Test protocol | Single evaluation | 5 evaluations (Original + Q95 + Q90 + Q80 + Q70) |
| Preprocessing | ELA on original | ELA on recompressed image |
| Output | Single metrics set | Degradation curves + comparison table |
| Everything else | Same | Same |
