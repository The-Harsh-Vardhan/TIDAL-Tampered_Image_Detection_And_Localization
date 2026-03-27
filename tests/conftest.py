"""Shared fixtures."""

import io

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def sample_image():
    return Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))


@pytest.fixture
def sample_image_bytes(sample_image):
    buf = io.BytesIO()
    sample_image.save(buf, format="JPEG", quality=85)
    return buf.getvalue()
