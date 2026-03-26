"""
tests/conftest.py
==================
Shared fixtures for TIDAL test suite.
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def sample_image() -> Image.Image:
    """Create a sample 100x100 RGB image for testing."""
    arr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    return Image.fromarray(arr)


@pytest.fixture
def sample_image_bytes(sample_image: Image.Image) -> bytes:
    """Create sample image as JPEG bytes."""
    buf = io.BytesIO()
    sample_image.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


@pytest.fixture
def mock_model():
    """Create a mock UNet model that returns random predictions."""
    import torch

    model = MagicMock()
    model.eval = MagicMock(return_value=model)
    model.to = MagicMock(return_value=model)

    def mock_forward(x):
        batch_size = x.shape[0]
        return torch.randn(batch_size, 1, 384, 384)

    model.__call__ = mock_forward
    model.parameters = MagicMock(return_value=iter([torch.randn(10)]))
    return model
