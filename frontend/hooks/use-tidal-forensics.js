"use client";

import { useEffect, useRef, useState } from "react";

import { trackTidalEvent } from "@/lib/analytics";
import {
  ANALYTICS_MODE_SIMPLE,
  API_BASE,
  DEFAULT_SETTINGS,
  DEMO_IMAGE_PATH,
  MAX_FILE_SIZE_BYTES,
  TAB_ORIGINAL,
  VALID_FILE_TYPES,
  getAppliedSettings,
  getFileSizeBucket,
  getInferencePayload,
} from "@/lib/forensic-formatters";
import { buildComparisonViews } from "@/lib/forensic-visuals";

const EMPTY_COMPARISON_VIEWS = {
  originalSrc: "",
  detectedRegionSrc: "",
  overlaySrc: "",
  hasMask: false,
};

function readFileAsDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (event) => resolve(event.target?.result || "");
    reader.onerror = () => reject(new Error("Failed to read the selected file."));
    reader.readAsDataURL(file);
  });
}

async function fetchHealthStatus() {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 5000);

  try {
    const response = await fetch(`${API_BASE}/health`, {
      signal: controller.signal,
    });

    return response.ok ? "online" : "offline";
  } catch {
    return "offline";
  } finally {
    clearTimeout(timer);
  }
}

function getFailureKind(error) {
  if (error?.name === "AbortError") {
    return "aborted";
  }

  if (String(error?.message || "").startsWith("HTTP ")) {
    return "http";
  }

  return "network";
}

export function useTidalForensics() {
  const [healthStatus, setHealthStatus] = useState("checking");
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewDataUrl, setPreviewDataUrl] = useState("");
  const [settings, setSettings] = useState(DEFAULT_SETTINGS);
  const [resultsVisible, setResultsVisible] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [resultData, setResultData] = useState(null);
  const [errorMessage, setErrorMessage] = useState("");
  const [visualTab, setVisualTab] = useState(TAB_ORIGINAL);
  const [analyticsMode, setAnalyticsMode] = useState(ANALYTICS_MODE_SIMPLE);
  const [isDemoLoading, setIsDemoLoading] = useState(false);
  const [soundEnabled, setSoundEnabled] = useState(false);
  const [comparisonViews, setComparisonViews] = useState(EMPTY_COMPARISON_VIEWS);

  const requestRef = useRef({
    controller: null,
    token: 0,
  });
  const uploadSourceRef = useRef("browse");
  const committedSettingsRef = useRef(DEFAULT_SETTINGS);
  const audioContextRef = useRef(null);
  const soundEnabledRef = useRef(false);

  useEffect(() => {
    let active = true;

    async function checkHealth() {
      const status = await fetchHealthStatus();
      if (active) {
        setHealthStatus(status);
      }
    }

    checkHealth();
    const intervalId = setInterval(checkHealth, 15000);

    return () => {
      active = false;
      clearInterval(intervalId);
    };
  }, []);

  useEffect(() => {
    let cancelled = false;

    async function renderViews() {
      const maskDataUrl = resultData?.mask_base64
        ? `data:image/png;base64,${resultData.mask_base64}`
        : "";

      try {
        const views = await buildComparisonViews(previewDataUrl, maskDataUrl);
        if (!cancelled) {
          setComparisonViews(views);
        }
      } catch {
        if (!cancelled) {
          setComparisonViews(
            previewDataUrl
              ? {
                  originalSrc: previewDataUrl,
                  detectedRegionSrc: "",
                  overlaySrc: "",
                  hasMask: false,
                }
              : EMPTY_COMPARISON_VIEWS
          );
        }
      }
    }

    renderViews();

    return () => {
      cancelled = true;
    };
  }, [previewDataUrl, resultData]);

  useEffect(() => {
    return () => {
      requestRef.current.controller?.abort();
    };
  }, []);

  useEffect(() => {
    soundEnabledRef.current = soundEnabled;
  }, [soundEnabled]);

  async function playTone(kind) {
    if (!soundEnabledRef.current) {
      return;
    }

    const AudioContextCtor =
      window.AudioContext || window.webkitAudioContext;

    if (!AudioContextCtor) {
      return;
    }

    const context =
      audioContextRef.current || new AudioContextCtor();
    audioContextRef.current = context;

    if (context.state === "suspended") {
      await context.resume();
    }

    const oscillator = context.createOscillator();
    const gain = context.createGain();

    const now = context.currentTime;
    const baseFrequency = kind === "error" ? 220 : 740;
    const endFrequency = kind === "error" ? 140 : 980;

    oscillator.type = "sine";
    oscillator.frequency.setValueAtTime(baseFrequency, now);
    oscillator.frequency.exponentialRampToValueAtTime(
      endFrequency,
      now + 0.2
    );

    gain.gain.setValueAtTime(0.0001, now);
    gain.gain.exponentialRampToValueAtTime(0.16, now + 0.02);
    gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.24);

    oscillator.connect(gain);
    gain.connect(context.destination);

    oscillator.start(now);
    oscillator.stop(now + 0.26);
  }

  async function submitImage(file, nextSettings = committedSettingsRef.current) {
    if (!file) {
      return;
    }

    requestRef.current.controller?.abort();

    const controller = new AbortController();
    const token = requestRef.current.token + 1;
    requestRef.current = {
      controller,
      token,
    };

    const appliedSettings = getAppliedSettings(nextSettings);
    const formData = new FormData();
    formData.append("file", file);
    Object.entries(appliedSettings).forEach(([key, value]) => {
      formData.append(key, value);
    });

    setIsAnalyzing(true);
    setResultsVisible(true);
    setResultData(null);
    setErrorMessage("");

    trackTidalEvent("inference_requested", {
      upload_source: uploadSourceRef.current,
      file_mime: file.type,
      file_size_bucket: getFileSizeBucket(file.size),
      active_visual_tab: visualTab,
      pixel_threshold: Number(appliedSettings.pixel_threshold),
      mask_area_threshold: Number(appliedSettings.mask_area_threshold),
      min_prediction_area_pixels: Number(
        appliedSettings.min_prediction_area_pixels
      ),
      review_confidence_threshold: Number(
        appliedSettings.review_confidence_threshold
      ),
      threshold_sensitivity_preset: appliedSettings.threshold_sensitivity_preset,
    });

    try {
      const response = await fetch(`${API_BASE}/infer`, {
        method: "POST",
        body: formData,
        signal: controller.signal,
      });

      if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        throw new Error(error.detail || `HTTP ${response.status}`);
      }

      const data = await response.json();

      if (token !== requestRef.current.token) {
        return;
      }

      setResultData(data);
      playTone("success");
      trackTidalEvent(
        "inference_completed",
        getInferencePayload(data, nextSettings, visualTab)
      );
    } catch (error) {
      if (controller.signal.aborted && token !== requestRef.current.token) {
        return;
      }

      if (token !== requestRef.current.token) {
        return;
      }

      const failureKind = getFailureKind(error);
      const message =
        failureKind === "http"
          ? error.message
          : "The API rejected the request or the forensic model is unavailable.";

      setErrorMessage(message);
      setVisualTab(TAB_ORIGINAL);
      playTone("error");
      trackTidalEvent("inference_failed", {
        failure_kind: failureKind,
        upload_source: uploadSourceRef.current,
        file_mime: file.type,
        file_size_bucket: getFileSizeBucket(file.size),
        active_visual_tab: visualTab,
      });
    } finally {
      if (token === requestRef.current.token) {
        setIsAnalyzing(false);
      }
    }
  }

  async function selectFile(file, source = "browse") {
    if (!VALID_FILE_TYPES.includes(file.type)) {
      window.alert("Please upload a JPEG, PNG, or WebP image.");
      return false;
    }

    if (file.size > MAX_FILE_SIZE_BYTES) {
      window.alert("File must be under 20 MB.");
      return false;
    }

    uploadSourceRef.current = source;
    committedSettingsRef.current = settings;

    const nextPreviewDataUrl = await readFileAsDataUrl(file);

    setSelectedFile(file);
    setPreviewDataUrl(nextPreviewDataUrl);
    setResultsVisible(true);
    setVisualTab(TAB_ORIGINAL);

    trackTidalEvent("image_upload_selected", {
      upload_source: source,
      file_mime: file.type,
      file_size_bucket: getFileSizeBucket(file.size),
    });

    await submitImage(file, settings);
    return true;
  }

  async function runDemo() {
    trackTidalEvent("demo_inference_requested", {
      active_visual_tab: visualTab,
      had_selected_file: Boolean(selectedFile),
    });

    setIsDemoLoading(true);
    setErrorMessage("");

    try {
      const response = await fetch(DEMO_IMAGE_PATH, {
        cache: "no-store",
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const blob = await response.blob();
      const demoFile = new File([blob], "tidal-demo-tamper-sample.jpg", {
        type: blob.type || "image/jpeg",
      });

      await selectFile(demoFile, "demo");
    } catch {
      setResultsVisible(true);
      setResultData(null);
      setVisualTab(TAB_ORIGINAL);
      setErrorMessage(
        "The bundled demo image could not be loaded. Please try a manual upload."
      );
    } finally {
      setIsDemoLoading(false);
    }
  }

  function clearUpload() {
    if (selectedFile) {
      trackTidalEvent("image_upload_cleared", {
        upload_source: uploadSourceRef.current,
        had_result: Boolean(resultData || errorMessage),
      });
    }

    requestRef.current.controller?.abort();
    requestRef.current = {
      controller: null,
      token: requestRef.current.token + 1,
    };

    setSelectedFile(null);
    setPreviewDataUrl("");
    setResultData(null);
    setErrorMessage("");
    setResultsVisible(false);
    setIsAnalyzing(false);
    setComparisonViews(EMPTY_COMPARISON_VIEWS);
    setVisualTab(TAB_ORIGINAL);
  }

  function updateSetting(name, value) {
    setSettings((current) => ({
      ...current,
      [name]: value,
    }));
  }

  function commitSetting(name, value) {
    const nextSettings = {
      ...committedSettingsRef.current,
      [name]: value,
    };

    if (committedSettingsRef.current[name] === value) {
      return;
    }

    committedSettingsRef.current = nextSettings;
    setSettings((current) => ({
      ...current,
      [name]: value,
    }));

    trackTidalEvent("forensic_control_changed", {
      control_name: name,
      control_value: typeof value === "number" ? Number(value) : value,
      has_selected_file: Boolean(selectedFile),
      active_visual_tab: visualTab,
    });

    if (selectedFile) {
      submitImage(selectedFile, nextSettings);
    }
  }

  function updateVisualTab(nextTab, shouldTrack = false) {
    setVisualTab(nextTab);

    if (shouldTrack) {
      trackTidalEvent("results_visual_tab_changed", {
        active_visual_tab: nextTab,
        has_result: Boolean(resultData),
      });
    }
  }

  function updateAnalyticsMode(nextMode) {
    if (nextMode === analyticsMode) {
      return;
    }

    setAnalyticsMode(nextMode);
    trackTidalEvent("analytics_mode_changed", {
      analytics_mode: nextMode,
      has_selected_file: Boolean(selectedFile),
      has_result: Boolean(resultData),
      active_visual_tab: visualTab,
    });
  }

  function toggleSound() {
    setSoundEnabled((current) => !current);
  }

  return {
    analyticsMode,
    comparisonViews,
    errorMessage,
    healthStatus,
    isAnalyzing,
    isDemoLoading,
    previewDataUrl,
    resultData,
    resultsVisible,
    selectedFile,
    settings,
    soundEnabled,
    visualTab,
    clearUpload,
    commitSetting,
    runDemo,
    selectFile,
    submitImage,
    toggleSound,
    updateAnalyticsMode,
    updateSetting,
    updateVisualTab,
  };
}
