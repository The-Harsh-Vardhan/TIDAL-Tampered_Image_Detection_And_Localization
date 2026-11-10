# 07 — Engineering Quality Audit

**Reviewer:** Arshad Siddiqui, Principal AI Engineer, BigVision LLC

---

## Code Readability — ✅ Good

- Clear section headers with Unicode box-drawing characters (`──`)
- Consistent naming conventions (`snake_case` throughout)
- Functions are well-scoped (single responsibility)
- Type hints absent but acceptable for a notebook

## Comment Clarity — ⚠️ Mixed

- Section banners are informative
- Some inline comments explain **what** but not **why** (e.g., `ALPHA = CONFIG['cls_loss_weight']` — no explanation of why 1.5)
- No docstrings on critical functions: `train_one_epoch()`, `validate()`, `evaluate_test()`

## Notebook Structure — ✅ Good

Clear numbered sections: Environment → Dataset → Augmentation → Model → Loss → Training → Curves → Evaluation → Visualization → Grad-CAM → Robustness → Shortcut Checks → Save → Conclusion. This is well-organized.

## Dead Code and Unused Imports — 🔴 Issues

| Item | File Location | Issue |
|---|---|---|
| `is_valid_image()` | Cell 10, line 263 | Defined, never called |
| `excluded = []` | Cell 10, line 284 | Initialized, never populated or used |
| `roc_auc_score` | Cell 3, line 69 | Imported, never used |
| `math` | Cell 3, line 50 | Imported, never used |
| `contextlib` | Cell 3, line 50 | Imported, never used |
| `pd` (pandas) | Cell 3, line 55 | Imported, never used |
| `Image` (PIL) | Cell 3, line 66 | Imported, used only in dead `is_valid_image()` |

**Count: 7 unused imports/functions.** This signals hastily assembled code.

## Error Handling — ⚠️ Weak

- `cv2.imread` failures silently return None; only mask reading checks for None (Cell 14 line 447), but image reading (line 444) does **not** check for None — would crash with `AttributeError: 'NoneType' object has no attribute 'shape'` on corrupt images
- W&B setup has try/except (good), but most other cells have no error handling

## Stderr Suppression — 🔴 Dangerous

Cell 3, lines 76-77:
```python
_devnull = os.open(os.devnull, os.O_WRONLY)
os.dup2(_devnull, 2)
```

This silences **ALL stderr output for the entire process**, including CUDA errors, Python tracebacks going to stderr, and library warnings. This is extremely dangerous in a training notebook — critical errors could be silently swallowed.

## W&B Integration — ✅ Good

Conditional W&B with graceful fallback. Uses `kaggle_secrets` for API key. Logs comprehensive metrics and uploads model artifact.

## Experiment Tracking — ⚠️ Limited

History dict tracks metrics, but:
- No gradient norm logging (v8 has this)
- No learning rate at each step (only epoch-level)
- No per-sample loss analysis

---

## Summary

| Aspect | Rating |
|---|---|
| Code readability | 7/10 |
| Comment quality | 5/10 |
| Structure | 8/10 |
| Dead code | 3/10 — 7 unused items |
| Error handling | 4/10 |
| Stderr suppression | 2/10 — dangerous |

**Overall Engineering Quality: 5/10**

**Severity: MEDIUM** — functional but sloppy for a principal-level review.
