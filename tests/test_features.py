"""Tests for Features 1-6: EXIF, heatmap, ELA stats, cache, PDF report, batch inference."""

from __future__ import annotations

import base64
import hashlib
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.app import app, _cache
from backend.cache import InferenceCache
from backend.inference.exif_analyzer import extract_exif
from backend.inference.preprocessing import compute_ela_stats


# ── Shared helpers ─────────────────────────────────────────────────────────

def _make_client():
    return TestClient(app)


def _jpeg_bytes(w=100, h=100) -> bytes:
    img = Image.fromarray(np.random.randint(0, 255, (h, w, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _png_bytes(w=100, h=100) -> bytes:
    img = Image.fromarray(np.random.randint(0, 255, (h, w, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════════════════
# Feature 1 — EXIF Metadata Analysis
# ══════════════════════════════════════════════════════════════════════════

class TestExifAnalyzer:
    """Unit tests for the EXIF extraction module."""

    def test_jpeg_no_exif_returns_flag(self):
        """Plain JPEG with no EXIF returns a 'no metadata' flag."""
        result = extract_exif(_jpeg_bytes())
        assert result["has_exif"] is False
        assert len(result["forensic_flags"]) >= 1
        assert any("no exif" in f.lower() or "stripped" in f.lower() or "skipped" not in f.lower()
                   for f in result["forensic_flags"])

    def test_png_no_exif_handled_gracefully(self):
        """PNG images don't have EXIF — must not raise."""
        result = extract_exif(_png_bytes())
        assert isinstance(result["raw"], dict)
        assert isinstance(result["forensic_flags"], list)
        assert isinstance(result["has_exif"], bool)
        assert isinstance(result["has_gps"], bool)

    def test_exif_structure(self):
        """Return dict always has the required keys."""
        result = extract_exif(_jpeg_bytes())
        assert set(result.keys()) == {"raw", "has_exif", "has_gps", "forensic_flags"}


class TestExifInInferResponse:
    """/infer response includes the exif field with correct structure."""

    def test_exif_key_present(self, sample_image_bytes):
        with _make_client() as client:
            r = client.post(
                "/infer",
                files={"file": ("sample.jpg", sample_image_bytes, "image/jpeg")},
            )
        assert r.status_code == 200
        payload = r.json()
        assert "exif" in payload
        exif = payload["exif"]
        assert "raw" in exif
        assert "has_exif" in exif
        assert "has_gps" in exif
        assert "forensic_flags" in exif
        assert isinstance(exif["forensic_flags"], list)


# ══════════════════════════════════════════════════════════════════════════
# Feature 2 — Probability Heatmap
# ══════════════════════════════════════════════════════════════════════════

class TestHeatmap:
    def test_heatmap_base64_present(self, sample_image_bytes):
        with _make_client() as client:
            r = client.post(
                "/infer",
                files={"file": ("sample.jpg", sample_image_bytes, "image/jpeg")},
            )
        assert r.status_code == 200
        payload = r.json()
        assert "heatmap_base64" in payload
        assert isinstance(payload["heatmap_base64"], str)
        assert len(payload["heatmap_base64"]) > 100  # non-trivial base64

    def test_heatmap_is_valid_image(self, sample_image_bytes):
        """Decoded heatmap_base64 must be a valid PNG image."""
        with _make_client() as client:
            r = client.post(
                "/infer",
                files={"file": ("sample.jpg", sample_image_bytes, "image/jpeg")},
            )
        heatmap_bytes = base64.b64decode(r.json()["heatmap_base64"])
        img = Image.open(io.BytesIO(heatmap_bytes))
        assert img.mode == "RGB"
        assert img.size[0] > 0 and img.size[1] > 0


# ══════════════════════════════════════════════════════════════════════════
# Feature 3 — ELA Energy Stats
# ══════════════════════════════════════════════════════════════════════════

class TestElaStats:
    """Unit tests for compute_ela_stats."""

    def test_returns_three_entries(self):
        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        stats = compute_ela_stats(img)
        assert len(stats) == 3

    def test_entry_keys(self):
        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        stats = compute_ela_stats(img)
        for entry in stats:
            assert set(entry.keys()) == {"quality", "mean_energy", "max_energy", "scale_factor"}

    def test_qualities_are_correct(self):
        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        stats = compute_ela_stats(img)
        assert [s["quality"] for s in stats] == [75, 85, 95]

    def test_energy_values_non_negative(self):
        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        for stat in compute_ela_stats(img):
            assert stat["mean_energy"] >= 0
            assert stat["max_energy"] >= 0
            assert stat["scale_factor"] > 0


class TestElaStatsInInferResponse:
    def test_ela_stats_in_response(self, sample_image_bytes):
        with _make_client() as client:
            r = client.post(
                "/infer",
                files={"file": ("sample.jpg", sample_image_bytes, "image/jpeg")},
            )
        assert r.status_code == 200
        payload = r.json()
        assert "ela_stats" in payload
        assert len(payload["ela_stats"]) == 3
        for entry in payload["ela_stats"]:
            assert "quality" in entry
            assert "mean_energy" in entry
            assert "max_energy" in entry


# ══════════════════════════════════════════════════════════════════════════
# Feature 4 — In-Memory LRU Cache
# ══════════════════════════════════════════════════════════════════════════

class TestInferenceCacheUnit:
    """Unit tests for the InferenceCache class."""

    def test_miss_returns_none(self):
        c = InferenceCache(max_size=4)
        assert c.get("missing_key") is None

    def test_set_and_get(self):
        c = InferenceCache(max_size=4)
        c.set("k1", {"data": 42})
        assert c.get("k1") == {"data": 42}

    def test_lru_eviction(self):
        c = InferenceCache(max_size=2)
        c.set("a", 1)
        c.set("b", 2)
        c.get("a")       # touch 'a' so 'b' becomes LRU
        c.set("c", 3)    # should evict 'b'
        assert c.get("b") is None
        assert c.get("a") == 1
        assert c.get("c") == 3

    def test_stats_accuracy(self):
        c = InferenceCache(max_size=4)
        c.set("x", 1)
        c.get("x")  # hit
        c.get("y")  # miss
        s = c.stats()
        assert s["hits"] == 1
        assert s["misses"] == 1
        assert s["hit_rate"] == 0.5
        assert s["size"] == 1


class TestCacheIntegration:
    """Integration tests using the /infer and /cache/stats endpoints."""

    def setup_method(self):
        _cache.clear()

    def test_second_request_is_cache_hit(self, sample_image_bytes):
        with _make_client() as client:
            r1 = client.post("/infer", files={"file": ("s.jpg", sample_image_bytes, "image/jpeg")})
            r2 = client.post("/infer", files={"file": ("s.jpg", sample_image_bytes, "image/jpeg")})
        assert r1.status_code == 200
        assert r2.status_code == 200
        assert r1.json()["cache_hit"] is False
        assert r2.json()["cache_hit"] is True

    def test_cache_key_is_sha256(self, sample_image_bytes):
        with _make_client() as client:
            r = client.post("/infer", files={"file": ("s.jpg", sample_image_bytes, "image/jpeg")})
        expected_key = hashlib.sha256(sample_image_bytes).hexdigest()
        assert r.json()["cache_key"] == expected_key

    def test_cache_stats_endpoint(self, sample_image_bytes):
        with _make_client() as client:
            client.post("/infer", files={"file": ("s.jpg", sample_image_bytes, "image/jpeg")})
            r = client.get("/cache/stats")
        assert r.status_code == 200
        stats = r.json()
        assert "size" in stats
        assert "hits" in stats
        assert "misses" in stats
        assert "hit_rate" in stats
        assert stats["size"] >= 1


# ══════════════════════════════════════════════════════════════════════════
# Feature 5 — PDF Report Export
# ══════════════════════════════════════════════════════════════════════════

class TestPdfReport:
    def setup_method(self):
        _cache.clear()

    def test_report_returns_pdf(self, sample_image_bytes):
        with _make_client() as client:
            infer_r = client.post(
                "/infer", files={"file": ("s.jpg", sample_image_bytes, "image/jpeg")}
            )
            assert infer_r.status_code == 200
            cache_key = infer_r.json()["cache_key"]
            report_r = client.get(f"/report/{cache_key}")
        assert report_r.status_code == 200
        assert report_r.headers["content-type"] == "application/pdf"
        assert len(report_r.content) > 1024  # non-trivial PDF

    def test_report_starts_with_pdf_magic(self, sample_image_bytes):
        with _make_client() as client:
            infer_r = client.post(
                "/infer", files={"file": ("s.jpg", sample_image_bytes, "image/jpeg")}
            )
            cache_key = infer_r.json()["cache_key"]
            report_r = client.get(f"/report/{cache_key}")
        assert report_r.content[:4] == b"%PDF"

    def test_report_404_for_unknown_key(self):
        with _make_client() as client:
            r = client.get("/report/0000000000000000000000000000000000000000000000000000000000000000")
        assert r.status_code == 404

    def test_report_content_disposition_header(self, sample_image_bytes):
        with _make_client() as client:
            infer_r = client.post(
                "/infer", files={"file": ("s.jpg", sample_image_bytes, "image/jpeg")}
            )
            cache_key = infer_r.json()["cache_key"]
            report_r = client.get(f"/report/{cache_key}")
        cd = report_r.headers.get("content-disposition", "")
        assert "attachment" in cd
        assert ".pdf" in cd


# ══════════════════════════════════════════════════════════════════════════
# Feature 6 — Batch Inference
# ══════════════════════════════════════════════════════════════════════════

class TestBatchInference:
    def setup_method(self):
        _cache.clear()

    def test_batch_two_images(self, sample_image_bytes):
        img2 = _jpeg_bytes(120, 120)
        with _make_client() as client:
            r = client.post(
                "/infer/batch",
                files=[
                    ("files", ("a.jpg", sample_image_bytes, "image/jpeg")),
                    ("files", ("b.jpg", img2, "image/jpeg")),
                ],
            )
        assert r.status_code == 200
        payload = r.json()
        assert payload["total"] == 2
        assert payload["succeeded"] == 2
        assert payload["failed"] == 0
        assert len(payload["results"]) == 2

    def test_batch_result_has_required_fields(self, sample_image_bytes):
        with _make_client() as client:
            r = client.post(
                "/infer/batch",
                files=[("files", ("a.jpg", sample_image_bytes, "image/jpeg"))],
            )
        result = r.json()["results"][0]
        assert "is_tampered" in result
        assert "confidence" in result
        assert "mask_base64" in result
        assert "heatmap_base64" in result
        assert "cache_key" in result
        assert "exif" in result
        assert "ela_stats" in result

    def test_batch_second_call_uses_cache(self, sample_image_bytes):
        """Submitting the same image twice in batch should hit cache on 2nd call."""
        with _make_client() as client:
            r1 = client.post(
                "/infer/batch",
                files=[("files", ("a.jpg", sample_image_bytes, "image/jpeg"))],
            )
            r2 = client.post(
                "/infer/batch",
                files=[("files", ("a.jpg", sample_image_bytes, "image/jpeg"))],
            )
        assert r1.json()["results"][0]["cache_hit"] is False
        assert r2.json()["results"][0]["cache_hit"] is True

    def test_batch_rejects_too_many_files(self, sample_image_bytes):
        with _make_client() as client:
            files = [("files", ("f.jpg", sample_image_bytes, "image/jpeg"))] * 9
            r = client.post("/infer/batch", files=files)
        assert r.status_code == 422

    def test_batch_rejects_empty(self):
        """Zero files must return 422 (no files provided)."""
        with _make_client() as client:
            # Send the field with an empty value to trigger our guard
            r = client.post("/infer/batch", data={})
        assert r.status_code == 422
