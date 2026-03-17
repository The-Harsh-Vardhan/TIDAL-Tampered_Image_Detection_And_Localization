# Reproducibility Gaps

This report evaluates whether another engineer could reproduce the current project from `Docs6/` alone.

## Summary

Reproducibility is **partial**.

- Kaggle v6 reproduction: partially documented
- Colab v6 reproduction: weakly documented

The main problem is not missing hyperparameters. The main problem is that the docs no longer identify the correct notebooks or runtime variants.

## Reproducibility Strengths

- Core hyperparameters are documented: image size, batch size, accumulation steps, patience, optimizer LR, and loss.
- Dataset preprocessing rules are documented: case-insensitive discovery, mask binarization, 70/15/15 split, leakage assertions.
- Artifact names such as `split_manifest.json`, `best_model.pt`, `last_checkpoint.pt`, and `results_summary.json` are documented.
- Deterministic seeding and split persistence are described.

## Gaps

| Area | Docs6 status | Impact | Recommended fix |
|---|---|---|---|
| Notebook selection | Docs6 still points to v5.1 instead of v6 | Engineers may open the wrong notebook and follow stale structure notes | Update all notebook references to the v6 notebooks immediately |
| Runtime coverage | Docs6 describes Kaggle almost exclusively | Colab reproduction is not documented even though a real v6 Colab notebook exists | Add a runtime selection section and separate Colab/Kaggle instructions |
| Colab dataset access | Not documented in Docs6 | Another engineer cannot follow the Colab notebook setup from docs alone | Document Kaggle credential retrieval via `google.colab.userdata` and the dataset download flow |
| Colab artifact storage | Not documented in Docs6 | Output locations and resume paths are unclear for Colab | Add Colab output directory and Drive behavior to engineering docs |
| Notebook structure map | Stale 61/17 map | Navigation and verification steps are unreliable | Rebuild the structure doc from the v6 notebooks |
| Image-level scoring rule | Docs say top-k mean; notebooks use max | Reproduced metrics may disagree with the docs | Standardize the scoring rule in one place and propagate it |
| Reference traceability | `P#` citations are not defined consistently | Research-backed design claims are harder to verify | Add explicit paper ID mapping in `13_References.md` |

## Can Another Engineer Reproduce the Project?

### Kaggle variant

Probably yes, but with friction.

Why:
- the Kaggle runtime story is mostly documented
- the main pipeline choices are documented
- artifact paths are documented

What blocks trust:
- the docs still claim the authoritative notebook is v5.1
- the structure doc is stale
- the image-level scoring rule is wrong

### Colab variant

Not from Docs6 alone.

Why:
- no proper Colab setup path is documented
- no Drive output path is documented
- no Colab-specific dataset credential flow is documented
- W&B auth is described only through `kaggle_secrets`

## Reproducibility Verdict

Docs6 is not yet a reliable reproduction guide for the current repo state. It is closest to a Kaggle-only design document for a pre-v6 notebook state.
