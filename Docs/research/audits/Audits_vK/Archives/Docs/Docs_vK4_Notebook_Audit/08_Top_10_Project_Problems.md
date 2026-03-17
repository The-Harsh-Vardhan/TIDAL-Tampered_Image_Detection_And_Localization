# 08 - Top 10 Project Problems

1. Notebook is unexecuted; outputs are absent.
2. Evaluation claims are not evidence-backed in the artifact.
3. Visualization deliverable is effectively missing in executed form.
4. Assignment asks Colab notebook; implementation is Kaggle-focused.
5. Architecture justification is descriptive, not decision-driven.
6. No explicit split leakage assertions.
7. Tampered-mask integrity is permissive instead of fail-fast.
8. Excessive training complexity without staged ablations.
9. No multi-seed variance or confidence reporting.
10. Strong engineering features are present but submission discipline is incomplete.

## Severity Ranking

- Critical: 1, 2, 3
- High: 4, 5, 6, 7
- Medium: 8, 9, 10

## Immediate Fix Order

1. Execute notebook end-to-end and save outputs.
2. Preserve quantitative outputs and qualitative panels.
3. Add data integrity assertions and leakage checks.
4. Add short architecture tradeoff rationale with one ablation.
