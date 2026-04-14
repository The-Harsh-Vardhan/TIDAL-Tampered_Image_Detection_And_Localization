"""FastAPI endpoint tests."""

from pathlib import Path

from fastapi.testclient import TestClient

from backend.app import app


def _make_client():
    return TestClient(app)


def test_health():
    r = _make_client().get("/health")
    assert r.status_code == 200 and r.json()["status"] == "alive"


def test_ready():
    r = _make_client().get("/ready")
    assert r.status_code in (200, 503)


def test_version():
    r = _make_client().get("/version")
    assert r.status_code == 200
    payload = r.json()
    assert payload["api_version"] == "1.0.0"
    assert payload["model_version"] == "vR.P.30.1"
    assert Path(payload["model_bundle_path"]).name == "best_model.pt"
    assert Path(payload["model_dir"]).name == "vR.P.30.1"


def test_metrics():
    r = _make_client().get("/metrics")
    assert r.status_code == 200 and "tidal_requests_total" in r.text


def test_infer_returns_legacy_and_diagnostic_fields(sample_image_bytes):
    with _make_client() as client:
        response = client.post(
            "/infer",
            files={"file": ("sample.jpg", sample_image_bytes, "image/jpeg")},
        )
    assert response.status_code == 200
    payload = response.json()
    assert "is_tampered" in payload
    assert "confidence" in payload
    assert "confidence_mean_prob" in payload
    assert "raw_tampered_pixel_count" in payload
    assert "tampered_pixel_count" in payload
    assert "needs_review" in payload
    assert "threshold_sensitivity" in payload
    assert "mask_base64" in payload
    assert "overlay_base64" in payload
    assert payload["model_version"] == "vR.P.30.1"
    assert payload["applied_settings"]["threshold_sensitivity_preset"] == "balanced"


def test_infer_accepts_runtime_knobs(sample_image_bytes):
    with _make_client() as client:
        response = client.post(
            "/infer",
            data={
                "pixel_threshold": "0.004",
                "mask_area_threshold": "10000",
                "min_prediction_area_pixels": "15000",
                "review_confidence_threshold": "0.80",
                "threshold_sensitivity_preset": "strict",
            },
            files={"file": ("sample.jpg", sample_image_bytes, "image/jpeg")},
        )
    assert response.status_code == 200
    payload = response.json()
    settings = payload["applied_settings"]
    assert settings["pixel_threshold"] == 0.004
    assert settings["mask_area_threshold"] == 10000
    assert settings["min_prediction_area_pixels"] == 15000
    assert settings["review_confidence_threshold"] == 0.8
    assert settings["threshold_sensitivity_preset"] == "strict"
    assert settings["threshold_sensitivity_levels"] == [0.5, 0.7, 0.85]
    assert isinstance(payload["overlay_base64"], str)


def test_infer_rejects_out_of_range_knobs(sample_image_bytes):
    with _make_client() as client:
        response = client.post(
            "/infer",
            data={"pixel_threshold": "1.20"},
            files={"file": ("sample.jpg", sample_image_bytes, "image/jpeg")},
        )
    assert response.status_code == 422
