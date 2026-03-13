# Runtime Variant Consistency

This report compares:

- `notebooks/tamper_detection_v6_colab.ipynb`
- `notebooks/tamper_detection_v6_kaggle.ipynb`

The goal is to separate expected platform differences from accidental functional drift.

## Shared Core Behavior

From static inspection, both v6 notebooks appear functionally aligned on the core ML pipeline:

- 66 cells / 22 sections
- `image_size = 384`
- `train_ratio = 0.70`
- `smp.Unet(resnet34)`
- `BCEDiceLoss`
- `AdamW`
- `split_manifest.json`
- `results_summary.json`
- threshold sweep on validation set
- image-level score via **max pixel probability**
- Grad-CAM safety checks
- robustness suite of JPEG, Gaussian noise, Gaussian blur, and resize degradation

## Expected Runtime Differences

| Area | Colab v6 | Kaggle v6 | Expected? |
|---|---|---|---|
| Dataset access | Uses Kaggle credentials via `google.colab.userdata` and a download flow | Uses pre-mounted `/kaggle/input` | Yes |
| Output storage | Drive-backed output directory | `/kaggle/working/` | Yes |
| W&B auth | `google.colab.userdata` with interactive fallback | `kaggle_secrets.UserSecretsClient()` | Yes |
| Environment helpers | `google.colab` imports and Drive mount | Kaggle paths and Kaggle Secrets | Yes |

## Accidental Functional Drift Checked

No material core-pipeline drift was found from static inspection in:

- model architecture
- loss and optimizer
- image size
- split policy
- artifact naming
- robustness core suite
- evaluation threshold flow

## Documentation Impact

The presence of two aligned runtime variants means the docs should now separate:

- shared pipeline behavior
- Colab-specific operations
- Kaggle-specific operations

Right now `Docs6` mostly documents the Kaggle path only, which under-describes the actual repo state.
