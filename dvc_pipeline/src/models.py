"""
dvc_pipeline/src/models.py
===========================
Model builder for TIDAL: UNet + ResNet-34 with 9-channel input.
"""

from __future__ import annotations

import torch
import segmentation_models_pytorch as smp


def build_model(
    encoder: str = "resnet34",
    encoder_weights: str | None = "imagenet",
    in_channels: int = 9,
    num_classes: int = 1,
    freeze_strategy: str = "body_frozen_bn_unfrozen",
) -> smp.Unet:
    """Build UNet + ResNet encoder with 9-channel conv1 initialization.

    For 9-channel input, pretrained 3ch ImageNet weights are tiled 3×
    and scaled by 1/3, following the vR.P.19 initialization strategy.

    Args:
        encoder: Encoder backbone name.
        encoder_weights: Pretrained weights to use for initialization.
        in_channels: Number of input channels (9 for Multi-Q RGB ELA).
        num_classes: Number of output segmentation classes.
        freeze_strategy: Freezing strategy for encoder.

    Returns:
        Configured UNet model.
    """
    model = smp.Unet(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
    )

    # Initialize conv1 for 9 channels from pretrained 3ch weights
    if in_channels == 9 and encoder_weights is not None:
        with torch.no_grad():
            pretrained_3ch = smp.Unet(
                encoder_name=encoder,
                encoder_weights=encoder_weights,
                in_channels=3,
                classes=num_classes,
            ).encoder.conv1.weight.data.clone()

            # Tile 3ch weights 3× for 9ch, scale by 1/3
            model.encoder.conv1.weight.data = pretrained_3ch.repeat(1, 3, 1, 1) / 3.0

    # Apply freeze strategy
    if freeze_strategy == "body_frozen_bn_unfrozen":
        for name, param in model.encoder.named_parameters():
            if "bn" in name or "conv1" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        # Decoder always trainable
        for param in model.decoder.parameters():
            param.requires_grad = True
        for param in model.segmentation_head.parameters():
            param.requires_grad = True

    return model


def get_model_info(model: smp.Unet) -> dict:
    """Get model parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "total_params": total,
        "trainable_params": trainable,
        "frozen_params": total - trainable,
        "trainable_pct": round(100 * trainable / total, 1),
    }
