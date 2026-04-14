"""Inference helper tests."""

import numpy as np
from PIL import Image

from backend.inference.engine import (
    DEFAULT_THRESHOLD_SENSITIVITY_PRESET,
    InferenceSettings,
    apply_prediction_area_filter,
    build_overlay_image,
    compute_threshold_sensitivity,
    is_tampered_prediction,
)


def test_default_settings_match_balanced_preset():
    settings = InferenceSettings()
    assert settings.threshold_sensitivity_preset == DEFAULT_THRESHOLD_SENSITIVITY_PRESET
    assert settings.threshold_sensitivity_levels == [0.30, 0.50, 0.70]


def test_low_pixel_threshold_is_preserved_at_four_decimals():
    settings = InferenceSettings(pixel_threshold=0.004)
    assert settings.to_dict()["pixel_threshold"] == 0.004


def test_apply_prediction_area_filter_suppresses_small_masks():
    pred_bin = np.ones((4, 4), dtype=np.uint8)
    filtered, triggered = apply_prediction_area_filter(pred_bin, min_area_pixels=20)
    assert triggered is True
    assert filtered.sum() == 0


def test_threshold_sensitivity_uses_filtered_counts():
    prob_map = np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32)
    rows = compute_threshold_sensitivity(prob_map, thresholds=[0.3, 0.7], min_area_pixels=2)
    assert rows[0]["final_pixels"] == 3
    assert rows[0]["area_filtered"] is False
    assert rows[1]["final_pixels"] == 0
    assert rows[1]["area_filtered"] is True


def test_zero_mask_area_threshold_means_any_surviving_pixel():
    assert is_tampered_prediction(0, 0) is False
    assert is_tampered_prediction(1, 0) is True


def test_build_overlay_image_only_tints_masked_pixels():
    image = Image.fromarray(np.full((2, 2, 3), 100, dtype=np.uint8))
    mask = np.array([[1, 0], [0, 1]], dtype=np.uint8)

    overlay = build_overlay_image(image, mask, overlay_alpha=0.42)

    assert overlay is not None
    assert overlay.shape == (2, 2, 3)
    assert tuple(overlay[0, 0]) != (100, 100, 100)
    assert tuple(overlay[0, 1]) == (100, 100, 100)
    assert tuple(overlay[1, 0]) == (100, 100, 100)


def test_build_overlay_image_returns_none_for_empty_mask():
    image = Image.fromarray(np.full((2, 2, 3), 100, dtype=np.uint8))
    mask = np.zeros((2, 2), dtype=np.uint8)

    assert build_overlay_image(image, mask) is None
