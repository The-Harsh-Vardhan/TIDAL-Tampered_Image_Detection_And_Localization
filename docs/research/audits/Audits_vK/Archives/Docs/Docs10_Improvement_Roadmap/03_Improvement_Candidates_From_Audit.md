# 3. Improvement Candidates From Audit

The audit-derived candidates fall into three ranked groups.

## Highest Expected Impact

- Localization-aware checkpoint selection
- Pretrained encoder within the same dual-task U-Net pattern
- Leakage-aware split strategy
- Boundary-preserving augmentation policy

These are the candidates most likely to alter localization quality, generalization quality, or both.

## Medium Expected Impact

- Invalid-mask filtering and dataset integrity gate
- Tampered-only localization reporting plus ROC-AUC
- Global seeding and deterministic run manifest

These candidates mostly improve result validity, supervision quality, and experiment trustworthiness.

## Lower Metric Impact but High Engineering Value

- AMP / mixed precision
- Output-complete execution discipline
- Hardcoded-path cleanup and notebook simplification

These are important for runtime efficiency, artifact quality, and maintainability even when they do not directly raise scores.

## Interpretation Rule

- Some candidates improve performance.
- Some improve evaluation validity.
- Some improve reproducibility or runtime efficiency.
- All should be documented.
- Only some should become merged model changes.
