# vR.P.30.1 Inference Bundle

This bundle packages the production inference assets for the `vR.P.30.1` notebook pipeline.

Contents:
- `best_model.pt`: deployable checkpoint for the backend
- `vR.P.30.1_unet_resnet34_mqela_cbam.pth`: versioned copy of the same checkpoint
- `manifest.json`: bundle metadata, notebook source, and preprocessing stats
- `run_backend.ps1`: starts the FastAPI backend against this bundle

Pipeline summary:
- Grayscale multi-quality ELA with `Q=[75, 85, 95]`
- `384x384` input
- UNet with `resnet34` encoder
- CBAM injected into all decoder blocks
- Notebook-style postprocessing controls for pixel threshold, area threshold, and review flagging

Primary notebook source:
- `Notebooks/tracks/vR.P/final_runs/vr_p_30_1_multi_quality_ela_cbam_attention_run_01 (1).ipynb`

Quick start:

```powershell
pwsh -File .\models\inference\vR.P.30.1\run_backend.ps1 -Device cpu
```
