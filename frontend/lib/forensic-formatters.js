export const TAB_ORIGINAL = "original";
export const TAB_MASK = "mask";
export const TAB_MASK_ON_ORIGINAL = "mask_on_original";
export const ANALYTICS_MODE_SIMPLE = "simple";
export const ANALYTICS_MODE_ADVANCED = "advanced";
export const DEMO_IMAGE_PATH = "/demo/tamper-sample.jpg";

export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE ||
  (typeof window !== "undefined" &&
  (window.location.hostname === "localhost" ||
    window.location.hostname === "127.0.0.1")
    ? "http://localhost:8000"
    : "https://the-harsh-vardhan-tidal-api.hf.space");

export const DEFAULT_SETTINGS = {
  pixelThreshold: 0.7,
  maskAreaThreshold: 400,
  minPredictionAreaPixels: 0,
  reviewConfidenceThreshold: 0.65,
  thresholdSensitivityPreset: "balanced",
};

export const PRESET_LABELS = {
  lenient: "0.20 / 0.35 / 0.50",
  balanced: "0.30 / 0.50 / 0.70",
  strict: "0.50 / 0.70 / 0.85",
};

export const VALID_FILE_TYPES = ["image/jpeg", "image/png", "image/webp"];
export const MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024;

export function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

export function formatCount(value) {
  return Number(value || 0).toLocaleString();
}

export function formatPercent(value) {
  return `${(Number(value || 0) * 100).toFixed(1)}%`;
}

export function formatRatioPercent(value) {
  return `${(Number(value || 0) * 100).toFixed(2)}%`;
}

export function getFileSizeBucket(size) {
  if (size < 1_000_000) {
    return "lt_1mb";
  }
  if (size < 5_000_000) {
    return "1mb_to_5mb";
  }
  if (size < 10_000_000) {
    return "5mb_to_10mb";
  }
  return "10mb_to_20mb";
}

export function getAppliedSettings(settings) {
  return {
    pixel_threshold: Number(settings.pixelThreshold).toFixed(2),
    mask_area_threshold: String(Math.round(Number(settings.maskAreaThreshold))),
    min_prediction_area_pixels: String(
      Math.round(Number(settings.minPredictionAreaPixels))
    ),
    review_confidence_threshold: Number(
      settings.reviewConfidenceThreshold
    ).toFixed(2),
    threshold_sensitivity_preset: settings.thresholdSensitivityPreset,
  };
}

export function getInferencePayload(data, settings, activeVisualTab) {
  const appliedSettings = data.applied_settings || getAppliedSettings(settings);

  return {
    verdict: data.is_tampered ? "tampered" : "authentic",
    needs_review: Boolean(data.needs_review),
    confidence: Number(data.confidence || 0),
    confidence_mean_prob: Number(data.confidence_mean_prob || 0),
    tampered_ratio: Number(data.tampered_ratio || 0),
    raw_tampered_pixel_count: Number(data.raw_tampered_pixel_count || 0),
    tampered_pixel_count: Number(data.tampered_pixel_count || 0),
    inference_time_ms: Number(data.inference_time_ms || 0),
    pixel_threshold: Number(appliedSettings.pixel_threshold || 0),
    mask_area_threshold: Number(appliedSettings.mask_area_threshold || 0),
    min_prediction_area_pixels: Number(
      appliedSettings.min_prediction_area_pixels || 0
    ),
    review_confidence_threshold: Number(
      appliedSettings.review_confidence_threshold || 0
    ),
    threshold_sensitivity_preset:
      appliedSettings.threshold_sensitivity_preset || "balanced",
    active_visual_tab: activeVisualTab,
  };
}
