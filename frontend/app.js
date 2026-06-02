(() => {
  "use strict";

  const API_BASE =
    location.hostname === "localhost" || location.hostname === "127.0.0.1"
      ? "http://localhost:8000"
      : "https://the-harsh-vardhan-tidal-api.hf.space";

  const DEMO_IMAGE_PATH = "./public/demo/gemini-demo.png";
  const DEFAULT_SETTINGS = {
    pixelThreshold: 0.7,
    maskAreaThreshold: 400,
    minPredictionAreaPixels: 0,
    reviewConfidenceThreshold: 0.65,
    thresholdSensitivityPreset: "balanced",
  };
  const DEMO_SETTINGS = {
    pixelThreshold: 0.004,
    maskAreaThreshold: 10000,
    minPredictionAreaPixels: 15000,
    reviewConfidenceThreshold: 0.65,
    thresholdSensitivityPreset: "balanced",
  };
  const PRESET_LABELS = {
    lenient: "0.20 / 0.35 / 0.50",
    balanced: "0.30 / 0.50 / 0.70",
    strict: "0.50 / 0.70 / 0.85",
  };
  const ICONS = {
    authentic: '<polyline points="20 6 9 17 4 12"></polyline>',
    tampered:
      '<line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line>',
    loading:
      '<line x1="12" y1="2" x2="12" y2="6"></line><line x1="12" y1="18" x2="12" y2="22"></line><line x1="4.93" y1="4.93" x2="7.76" y2="7.76"></line><line x1="16.24" y1="16.24" x2="19.07" y2="19.07"></line><line x1="2" y1="12" x2="6" y2="12"></line><line x1="18" y1="12" x2="22" y2="12"></line><line x1="4.93" y1="19.07" x2="7.76" y2="16.24"></line><line x1="16.24" y1="7.76" x2="19.07" y2="4.93"></line>',
    error:
      '<circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line>',
  };

  const $ = (id) => document.getElementById(id);
  const elements = {
    fileInput: $("fileInput"),
    browseBtn: $("browseBtn"),
    demoBtn: $("demoBtn"),
    clearBtn: $("clearBtn"),
    previewImage: $("previewImage"),
    previewEmpty: $("previewEmpty"),
    originalImage: $("originalImage"),
    originalEmpty: $("originalEmpty"),
    maskImage: $("maskImage"),
    maskEmpty: $("maskEmpty"),
    overlayImage: $("overlayImage"),
    overlayEmpty: $("overlayEmpty"),
    progressBar: $("progressBar"),
    progressFill: $("progressFill"),
    resultsArea: $("resultsArea"),
    resultVerdict: $("resultVerdict"),
    verdictIconSvg: $("verdictIconSvg"),
    verdictText: $("verdictText"),
    verdictDetail: $("verdictDetail"),
    metricConfidence: $("metricConfidence"),
    metricRatio: $("metricRatio"),
    metricTime: $("metricTime"),
    diagnosticBanner: $("diagnosticBanner"),
    diagnosticModel: $("diagnosticModel"),
    diagnosticNeedsReview: $("diagnosticNeedsReview"),
    diagnosticRawPixels: $("diagnosticRawPixels"),
    diagnosticFinalPixels: $("diagnosticFinalPixels"),
    diagnosticAreaFilter: $("diagnosticAreaFilter"),
    diagnosticMeanProb: $("diagnosticMeanProb"),
    settingsPills: $("settingsPills"),
    sensitivityTable: $("sensitivityTable"),
    statusDot: $("statusDot"),
    statusText: $("statusText"),
    pixelThreshold: $("pixelThreshold"),
    pixelThresholdValue: $("pixelThresholdValue"),
    maskAreaThreshold: $("maskAreaThreshold"),
    maskAreaThresholdValue: $("maskAreaThresholdValue"),
    minPredictionAreaPixels: $("minPredictionAreaPixels"),
    minPredictionAreaPixelsValue: $("minPredictionAreaPixelsValue"),
    reviewConfidenceThreshold: $("reviewConfidenceThreshold"),
    reviewConfidenceThresholdValue: $("reviewConfidenceThresholdValue"),
    thresholdSensitivityPreset: $("thresholdSensitivityPreset"),
    inputHelper: $("inputHelper"),
  };

  let currentFile = null;
  let currentSource = "browse";
  let currentPreviewDataUrl = "";
  let requestToken = 0;

  function formatCount(value) {
    return Number(value || 0).toLocaleString();
  }

  function formatThreshold(value, precision = 4) {
    return Number(value || 0)
      .toFixed(precision)
      .replace(/\.?0+$/, "")
      .replace(/^-0$/, "0");
  }

  function formatApiIssuePath(loc) {
    if (!Array.isArray(loc)) {
      return "";
    }

    return loc
      .filter((part) => typeof part === "string" && part !== "body")
      .join(".");
  }

  function formatApiIssue(issue) {
    if (typeof issue === "string") {
      return issue.trim();
    }

    if (!issue || typeof issue !== "object") {
      return "";
    }

    const path = formatApiIssuePath(issue.loc);
    const message =
      typeof issue.msg === "string" ? issue.msg.trim() : "";

    if (path && message) {
      return `${path}: ${message}`;
    }

    return message || path;
  }

  function createHttpError(status, payload = {}) {
    const detail = payload && payload.detail;
    const detailMessage = Array.isArray(detail)
      ? detail.map((issue) => formatApiIssue(issue)).filter(Boolean).join(" ")
      : typeof detail === "string"
      ? detail.trim()
      : "";
    const error = new Error(detailMessage || `HTTP ${status}`);

    error.kind = "http";
    error.status = status;

    return error;
  }

  function updateControlOutputs() {
    elements.pixelThresholdValue.textContent = formatThreshold(
      elements.pixelThreshold.value
    );
    elements.maskAreaThresholdValue.textContent = `${formatCount(
      elements.maskAreaThreshold.value
    )} px`;
    elements.minPredictionAreaPixelsValue.textContent = `${formatCount(
      elements.minPredictionAreaPixels.value
    )} px`;
    elements.reviewConfidenceThresholdValue.textContent = Number(
      elements.reviewConfidenceThreshold.value
    ).toFixed(2);
  }

  function applySettings(settings) {
    elements.pixelThreshold.value = settings.pixelThreshold;
    elements.maskAreaThreshold.value = settings.maskAreaThreshold;
    elements.minPredictionAreaPixels.value = settings.minPredictionAreaPixels;
    elements.reviewConfidenceThreshold.value =
      settings.reviewConfidenceThreshold;
    elements.thresholdSensitivityPreset.value =
      settings.thresholdSensitivityPreset;
    updateControlOutputs();
  }

  function getCurrentSettings() {
    return {
      pixel_threshold: Number(elements.pixelThreshold.value).toFixed(4),
      mask_area_threshold: String(
        Math.round(Number(elements.maskAreaThreshold.value))
      ),
      min_prediction_area_pixels: String(
        Math.round(Number(elements.minPredictionAreaPixels.value))
      ),
      review_confidence_threshold: Number(
        elements.reviewConfidenceThreshold.value
      ).toFixed(2),
      threshold_sensitivity_preset: elements.thresholdSensitivityPreset.value,
    };
  }

  function setVerdict(kind, label, detail) {
    elements.resultVerdict.className = `verdict is-${kind}`.trim();
    elements.verdictIconSvg.innerHTML = ICONS[kind];
    elements.verdictText.textContent = label;
    elements.verdictDetail.textContent = detail;
  }

  function setDiagnosticBanner(message, variant = "") {
    elements.diagnosticBanner.textContent = message;
    elements.diagnosticBanner.className = `diagnostic-banner ${variant}`.trim();
  }

  function setImageState(imageEl, emptyEl, src, alt, emptyText) {
    if (src) {
      imageEl.src = src;
      imageEl.alt = alt;
      imageEl.hidden = false;
      emptyEl.hidden = true;
      return;
    }

    imageEl.removeAttribute("src");
    imageEl.hidden = true;
    emptyEl.hidden = false;
    emptyEl.textContent = emptyText;
  }

  function clearDiagnostics() {
    elements.metricConfidence.textContent = "—";
    elements.metricRatio.textContent = "—";
    elements.metricTime.textContent = "—";
    elements.diagnosticModel.textContent = "—";
    elements.diagnosticNeedsReview.textContent = "—";
    elements.diagnosticRawPixels.textContent = "—";
    elements.diagnosticFinalPixels.textContent = "—";
    elements.diagnosticAreaFilter.textContent = "—";
    elements.diagnosticMeanProb.textContent = "—";
    elements.settingsPills.innerHTML = "";
    elements.sensitivityTable.innerHTML = "";
  }

  function syncOriginalPreview() {
    setImageState(
      elements.originalImage,
      elements.originalEmpty,
      currentPreviewDataUrl,
      "Original image used for inference",
      "Waiting for inference"
    );
  }

  function setLoadingState() {
    elements.resultsArea.hidden = false;
    elements.progressBar.hidden = false;
    elements.progressFill.className = "progress-fill indeterminate";
    setVerdict("loading", "Analyzing…", "Applying forensic controls");
    setDiagnosticBanner("Running the vR.P.30.1 forensic pipeline…");
    clearDiagnostics();
    syncOriginalPreview();
    setImageState(
      elements.maskImage,
      elements.maskEmpty,
      "",
      "",
      "Waiting for inference"
    );
    setImageState(
      elements.overlayImage,
      elements.overlayEmpty,
      "",
      "",
      "Waiting for inference"
    );
  }

  function finishLoading() {
    elements.progressFill.className = "progress-fill";
    elements.progressFill.style.width = "100%";
    window.setTimeout(() => {
      elements.progressBar.hidden = true;
      elements.progressFill.style.width = "0%";
    }, 350);
  }

  function renderSettingsPills(appliedSettings) {
    elements.settingsPills.innerHTML = "";
    [
      `Pixel ${formatThreshold(appliedSettings.pixel_threshold)}`,
      `Image area ${formatCount(appliedSettings.mask_area_threshold)} px`,
      `Min area ${formatCount(appliedSettings.min_prediction_area_pixels)} px`,
      `Review ${Number(appliedSettings.review_confidence_threshold || 0).toFixed(
        2
      )}`,
      `${appliedSettings.threshold_sensitivity_preset || "balanced"} · ${
        PRESET_LABELS[
          appliedSettings.threshold_sensitivity_preset || "balanced"
        ]
      }`,
    ].forEach((label) => {
      const pill = document.createElement("span");
      pill.className = "pill";
      pill.textContent = label;
      elements.settingsPills.appendChild(pill);
    });
  }

  function renderSensitivity(rows) {
    elements.sensitivityTable.innerHTML = "";
    if (!rows.length) {
      elements.sensitivityTable.textContent =
        "Sensitivity data will appear after inference.";
      return;
    }

    const maxPixels = Math.max(
      ...rows.map((row) => Number(row.final_pixels || 0)),
      1
    );

    rows.forEach((row) => {
      const item = document.createElement("div");
      item.className = "sensitivity-row";
      const width = Math.max(
        (Number(row.final_pixels || 0) / maxPixels) * 100,
        3
      );
      item.innerHTML = `
        <span>&gt;${Number(row.threshold).toFixed(2)}</span>
        <div class="sensitivity-bar-track">
          <div class="sensitivity-bar" style="width: ${width}%"></div>
        </div>
        <span>${formatCount(row.final_pixels)} px</span>
        <span>${row.area_filtered ? "filtered" : "kept"}</span>
      `;
      elements.sensitivityTable.appendChild(item);
    });
  }

  function renderResponse(data) {
    const appliedSettings = data.applied_settings || {};
    const overlayDataUrl = data.overlay_base64
      ? `data:image/png;base64,${data.overlay_base64}`
      : "";
    const maskDataUrl = data.mask_base64
      ? `data:image/png;base64,${data.mask_base64}`
      : "";
    const finalPixels = Number(data.tampered_pixel_count || 0);
    const hasRegion = finalPixels > 0;

    setVerdict(
      data.is_tampered ? "tampered" : "authentic",
      data.is_tampered ? "Tampered" : "Authentic",
      data.is_tampered
        ? `${(Number(data.tampered_ratio || 0) * 100).toFixed(
            2
          )}% of image shows tampering`
        : "No tampering detected"
    );
    elements.metricConfidence.textContent = `${(
      Number(data.confidence || 0) * 100
    ).toFixed(1)}%`;
    elements.metricRatio.textContent = `${(
      Number(data.tampered_ratio || 0) * 100
    ).toFixed(2)}%`;
    elements.metricTime.textContent = `${Number(
      data.inference_time_ms || 0
    ).toFixed(0)}ms`;

    setImageState(
      elements.maskImage,
      elements.maskEmpty,
      hasRegion ? maskDataUrl : "",
      "Binary mask generated by the inference backend",
      "No detected region"
    );
    setImageState(
      elements.overlayImage,
      elements.overlayEmpty,
      overlayDataUrl,
      "Server-generated red overlay rendered on top of the source image",
      hasRegion ? "Overlay unavailable" : "No detected region"
    );

    setDiagnosticBanner(
      data.needs_review
        ? "Borderline or unstable evidence detected. Manual review is recommended."
        : data.is_tampered
        ? "Stable suspicious region detected under the current forensic settings."
        : "No stable suspicious region detected under the current forensic settings.",
      data.needs_review ? "" : data.is_tampered ? "banner-alert" : "banner-ok"
    );
    elements.diagnosticModel.textContent = data.model_version || "vR.P.30.1";
    elements.diagnosticNeedsReview.textContent = data.needs_review ? "Yes" : "No";
    elements.diagnosticRawPixels.textContent = formatCount(
      data.raw_tampered_pixel_count
    );
    elements.diagnosticFinalPixels.textContent = formatCount(
      data.tampered_pixel_count
    );
    elements.diagnosticAreaFilter.textContent = data.area_filter_triggered
      ? "Yes"
      : "No";
    elements.diagnosticMeanProb.textContent = Number(
      data.confidence_mean_prob || 0
    ).toFixed(4);
    renderSettingsPills(appliedSettings);
    renderSensitivity(data.threshold_sensitivity || []);
  }

  async function checkHealth() {
    try {
      const response = await fetch(`${API_BASE}/health`, {
        signal: AbortSignal.timeout(5000),
      });
      if (!response.ok) {
        throw new Error("offline");
      }
      elements.statusDot.className = "status-dot online";
      elements.statusText.textContent = "API connected";
    } catch {
      elements.statusDot.className = "status-dot offline";
      elements.statusText.textContent = "API offline";
    }
  }

  async function submitImage(file) {
    if (!file) {
      return;
    }

    requestToken += 1;
    const token = requestToken;
    currentFile = file;

    setLoadingState();

    const formData = new FormData();
    formData.append("file", file);
    Object.entries(getCurrentSettings()).forEach(([key, value]) => {
      formData.append(key, value);
    });

    try {
      const response = await fetch(`${API_BASE}/infer`, {
        method: "POST",
        body: formData,
      });
      if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        throw createHttpError(response.status, error);
      }

      const data = await response.json();
      if (token !== requestToken) {
        return;
      }
      renderResponse(data);
    } catch (error) {
      if (token !== requestToken) {
        return;
      }
      const message =
        error && error.kind === "http"
          ? error.message
          : "The API rejected the request or the forensic model is unavailable.";
      setVerdict("error", "Error", message);
      setDiagnosticBanner(
        message,
        "banner-alert"
      );
      setImageState(
        elements.maskImage,
        elements.maskEmpty,
        "",
        "",
        "No detected region"
      );
      setImageState(
        elements.overlayImage,
        elements.overlayEmpty,
        "",
        "",
        "No detected region"
      );
      clearDiagnostics();
      syncOriginalPreview();
    } finally {
      if (token === requestToken) {
        finishLoading();
      }
    }
  }

  function readPreview(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = (event) => resolve(event.target?.result || "");
      reader.onerror = () =>
        reject(new Error("Failed to decode the selected image."));
      reader.readAsDataURL(file);
    });
  }

  async function selectFile(file, source = "browse", settings = null) {
    if (!file) {
      return;
    }

    if (!["image/jpeg", "image/png", "image/webp"].includes(file.type)) {
      window.alert("Please upload a JPEG, PNG, or WebP image.");
      return;
    }

    if (file.size > 20 * 1024 * 1024) {
      window.alert("File must be under 20 MB.");
      return;
    }

    currentSource = source;
    if (settings) {
      applySettings(settings);
    }

    currentPreviewDataUrl = await readPreview(file);
    setImageState(
      elements.previewImage,
      elements.previewEmpty,
      currentPreviewDataUrl,
      "Selected source image",
      "Select an image or run the Gemini demo sample."
    );
    syncOriginalPreview();
    elements.resultsArea.hidden = false;
    elements.inputHelper.textContent =
      source === "demo"
        ? "Running the Gemini sample through the backend with the validated demo profile."
        : "Manual upload selected. Any control change reruns inference with the current settings.";
    await submitImage(file);
  }

  async function runDemo() {
    elements.demoBtn.disabled = true;
    try {
      const response = await fetch(DEMO_IMAGE_PATH, { cache: "no-store" });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const blob = await response.blob();
      const demoFile = new File([blob], "tidal-gemini-demo.png", {
        type: blob.type || "image/png",
      });
      await selectFile(demoFile, "demo", DEMO_SETTINGS);
    } catch {
      elements.resultsArea.hidden = false;
      setVerdict(
        "error",
        "Error",
        "The bundled demo image could not be loaded."
      );
      setDiagnosticBanner(
        "The bundled demo image could not be loaded. Please try a manual upload.",
        "banner-alert"
      );
    } finally {
      elements.demoBtn.disabled = false;
    }
  }

  function clearUpload() {
    const resetToDefaults = currentSource === "demo";
    requestToken += 1;
    currentFile = null;
    currentSource = "browse";
    currentPreviewDataUrl = "";
    elements.fileInput.value = "";
    elements.resultsArea.hidden = true;
    elements.progressBar.hidden = true;
    elements.progressFill.className = "progress-fill";
    elements.progressFill.style.width = "0%";
    setImageState(
      elements.previewImage,
      elements.previewEmpty,
      "",
      "",
      "Select an image or run the Gemini demo sample."
    );
    setImageState(
      elements.originalImage,
      elements.originalEmpty,
      "",
      "",
      "Waiting for inference"
    );
    setImageState(
      elements.maskImage,
      elements.maskEmpty,
      "",
      "",
      "No detected region"
    );
    setImageState(
      elements.overlayImage,
      elements.overlayEmpty,
      "",
      "",
      "No detected region"
    );
    clearDiagnostics();
    setVerdict("authentic", "Authentic", "No tampering detected");
    setDiagnosticBanner("Waiting for inference…");
    elements.inputHelper.textContent =
      "Demo uses the validated profile: pixel threshold 0.004, image area threshold 10000, minimum predicted area 15000.";
    if (resetToDefaults) {
      applySettings(DEFAULT_SETTINGS);
    }
  }

  function rerunIfPossible() {
    if (currentFile) {
      submitImage(currentFile);
    }
  }

  elements.browseBtn.addEventListener("click", () => elements.fileInput.click());
  elements.fileInput.addEventListener("change", (event) => {
    const [file] = event.target.files || [];
    selectFile(file);
  });
  elements.demoBtn.addEventListener("click", runDemo);
  elements.clearBtn.addEventListener("click", clearUpload);
  [
    elements.pixelThreshold,
    elements.maskAreaThreshold,
    elements.minPredictionAreaPixels,
    elements.reviewConfidenceThreshold,
  ].forEach((input) => {
    input.addEventListener("input", updateControlOutputs);
    input.addEventListener("change", rerunIfPossible);
  });
  elements.thresholdSensitivityPreset.addEventListener("change", rerunIfPossible);

  applySettings(DEFAULT_SETTINGS);
  clearUpload();
  checkHealth();
  window.setInterval(checkHealth, 15000);
})();
