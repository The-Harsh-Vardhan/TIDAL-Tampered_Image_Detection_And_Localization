"""Architecture builders for TIDAL inference models."""

from __future__ import annotations

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

ENCODER_NAME = "resnet34"
NUM_CLASSES = 1
IN_CHANNELS = 3
DECODER_CHANNELS = (256, 128, 64, 32, 16)
ATTENTION_REDUCTION = 16
CBAM_KERNEL_SIZE = 7


class ChannelAttention(nn.Module):
    """Channel attention module used by CBAM."""

    def __init__(self, channels: int, reduction: int = ATTENTION_REDUCTION):
        super().__init__()
        hidden_channels = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, channels, _, _ = x.shape
        avg_logits = self.fc(self.avg_pool(x).view(batch_size, channels))
        max_logits = self.fc(self.max_pool(x).view(batch_size, channels))
        scale = self.sigmoid(avg_logits + max_logits).view(batch_size, channels, 1, 1)
        return x * scale


class SpatialAttention(nn.Module):
    """Spatial attention module used by CBAM."""

    def __init__(self, kernel_size: int = CBAM_KERNEL_SIZE):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_map = x.mean(dim=1, keepdim=True)
        max_map = x.max(dim=1, keepdim=True)[0]
        descriptor = torch.cat((avg_map, max_map), dim=1)
        attention = self.sigmoid(self.conv(descriptor))
        return x * attention


class CBAMBlock(nn.Module):
    """Sequential channel and spatial attention."""

    def __init__(
        self,
        channels: int,
        reduction: int = ATTENTION_REDUCTION,
        kernel_size: int = CBAM_KERNEL_SIZE,
    ):
        super().__init__()
        # Keep attribute names aligned with the notebook-trained checkpoint.
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x


def build_vrp301_model() -> nn.Module:
    """Build the vR.P.30.1 model architecture used for inference."""

    model = smp.Unet(
        encoder_name=ENCODER_NAME,
        encoder_weights=None,
        in_channels=IN_CHANNELS,
        classes=NUM_CLASSES,
        activation=None,
    )
    for block_index, block in enumerate(model.decoder.blocks):
        channels = DECODER_CHANNELS[block_index]
        block.attention2 = CBAMBlock(channels)
    return model
