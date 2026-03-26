"""
tests/test_infer.py
====================
FastAPI endpoint integration tests.
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def jpeg_bytes():
    """Create a small JPEG image as bytes."""
    img = Image.fromarray(np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


@pytest.fixture
def png_bytes():
    """Create a small PNG image as bytes."""
    img = Image.fromarray(np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_returns_200(self):
        from fastapi.testclient import TestClient
        from backend.app import app

        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "alive"
        assert "timestamp" in data


class TestReadyEndpoint:
    """Tests for GET /ready."""

    def test_ready_returns_status(self):
        from fastapi.testclient import TestClient
        from backend.app import app

        client = TestClient(app)
        response = client.get("/ready")
        # May return 200 or 503 depending on model availability
        assert response.status_code in (200, 503)


class TestVersionEndpoint:
    """Tests for GET /version."""

    def test_version_returns_info(self):
        from fastapi.testclient import TestClient
        from backend.app import app

        client = TestClient(app)
        response = client.get("/version")
        assert response.status_code == 200
        data = response.json()
        assert "api_version" in data
        assert "model_version" in data
        assert "cuda_available" in data


class TestMetricsEndpoint:
    """Tests for GET /metrics."""

    def test_metrics_returns_prometheus_format(self):
        from fastapi.testclient import TestClient
        from backend.app import app

        client = TestClient(app)
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "tidal_requests_total" in response.text
