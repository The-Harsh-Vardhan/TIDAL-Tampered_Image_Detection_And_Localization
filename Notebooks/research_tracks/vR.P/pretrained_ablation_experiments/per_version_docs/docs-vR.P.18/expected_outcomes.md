# vR.P.18 — Expected Outcomes

## Expected Degradation Pattern

| Condition | Expected Pixel F1 | Expected Drop | Reasoning |
|-----------|-------------------|---------------|-----------|
| Original | ~0.6920 (P.3 baseline) | — | No degradation |
| Q=95 | ~0.67–0.69 | < 2pp | Mild compression preserves most artifacts |
| Q=90 | ~0.63–0.67 | 3–6pp | Moderate — this is the same Q used for ELA computation |
| Q=80 | ~0.55–0.63 | 7–15pp | Significant artifact masking |
| Q=70 | ~0.45–0.55 | 15–25pp | Heavy compression destroys most forensic signal |

## Key Questions This Experiment Answers

1. **How robust is ELA-based detection to post-tampering compression?**
   - Real-world tampered images are often recompressed when shared online
   - Social media platforms typically recompress at Q=75–85

2. **Is there a "cliff edge" where performance collapses?**
   - Graceful degradation suggests robust features
   - Sudden collapse at a specific Q indicates fragile reliance on specific artifact patterns

3. **Does the model detect different things at different Q levels?**
   - Per-forgery-type analysis at each Q reveals which manipulation types are most compression-resilient

## Success Criteria

This is a measurement experiment, not an optimization. There is no POSITIVE/NEUTRAL/NEGATIVE verdict. Success means:
- Complete metrics for all 5 conditions
- Clear degradation curves plotted
- Actionable conclusions about practical robustness
- Recommendations for deployment scenarios (what minimum Q is required?)
