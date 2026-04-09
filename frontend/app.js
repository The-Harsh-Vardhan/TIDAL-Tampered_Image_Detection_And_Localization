/**
 * TIDAL Frontend — Application Logic
 * Tampered Image Detection & Localization
 *
 * Handles:
 * - API health polling
 * - Image upload (drag-drop + file picker)
 * - Notebook-style forensic control submission
 * - Diagnostic result rendering
 * - Scroll-triggered fade-in animations
 * - Mobile nav toggle
 */

(() => {
  "use strict";

  const API_BASE =
    location.hostname === "localhost" || location.hostname === "127.0.0.1"
      ? "http://localhost:8000"
      : "https://the-harsh-vardhan-tidal-api.hf.space";

  const $ = (id) => document.getElementById(id);

  const uploadArea = $("uploadArea");
  const uploadContent = $("uploadContent");
  const uploadPreview = $("uploadPreview");
  const fileInput = $("fileInput");
  const browseBtn = $("browseBtn");
  const clearBtn = $("clearBtn");
  const previewImage = $("previewImage");
  const progressBar = $("progressBar");
  const progressFill = $("progressFill");

  const resultsArea = $("resultsArea");
  const resultVerdict = $("resultVerdict");
  const verdictIconSvg = $("verdictIconSvg");
  const verdictText = $("verdictText");
  const verdictDetail = $("verdictDetail");
  const metricConfidence = $("metricConfidence");
  const metricRatio = $("metricRatio");
  const metricTime = $("metricTime");
  const maskImage = $("maskImage");

  const diagnosticBanner = $("diagnosticBanner");
  const diagnosticModel = $("diagnosticModel");
  const diagnosticNeedsReview = $("diagnosticNeedsReview");
  const diagnosticRawPixels = $("diagnosticRawPixels");
  const diagnosticFinalPixels = $("diagnosticFinalPixels");
  const diagnosticAreaFilter = $("diagnosticAreaFilter");
  const diagnosticMeanProb = $("diagnosticMeanProb");
  const settingsPills = $("settingsPills");
  const sensitivityTable = $("sensitivityTable");

  const pixelThreshold = $("pixelThreshold");
  const pixelThresholdValue = $("pixelThresholdValue");
  const maskAreaThreshold = $("maskAreaThreshold");
  const maskAreaThresholdValue = $("maskAreaThresholdValue");
  const minPredictionAreaPixels = $("minPredictionAreaPixels");
  const minPredictionAreaPixelsValue = $("minPredictionAreaPixelsValue");
  const reviewConfidenceThreshold = $("reviewConfidenceThreshold");
  const reviewConfidenceThresholdValue = $("reviewConfidenceThresholdValue");
  const thresholdSensitivityPreset = $("thresholdSensitivityPreset");

  const statusDot = $("statusDot");
  const statusText = $("statusText");
  const navToggle = $("navToggle");
  const navLinks = $("navLinks");

  const ICON_CHECK = '<polyline points="20 6 9 17 4 12"/>';
  const ICON_X =
    '<line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>';
  const ICON_LOADER =
    '<line x1="12" y1="2" x2="12" y2="6"/><line x1="12" y1="18" x2="12" y2="22"/><line x1="4.93" y1="4.93" x2="7.76" y2="7.76"/><line x1="16.24" y1="16.24" x2="19.07" y2="19.07"/><line x1="2" y1="12" x2="6" y2="12"/><line x1="18" y1="12" x2="22" y2="12"/><line x1="4.93" y1="19.07" x2="7.76" y2="16.24"/><line x1="16.24" y1="7.76" x2="19.07" y2="4.93"/>';
  const ICON_ALERT =
    '<circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>';

  const presetLabels = {
    lenient: "0.20 / 0.35 / 0.50",
    balanced: "0.30 / 0.50 / 0.70",
    strict: "0.50 / 0.70 / 0.85",
  };

  let currentFile = null;

  function setVerdictIcon(svgContent, className) {
    verdictIconSvg.innerHTML = svgContent;
    resultVerdict.className = `result-verdict ${className}`.trim();
  }

  function setDiagnosticBanner(message, variant = "") {
    diagnosticBanner.textContent = message;
    diagnosticBanner.className = `diagnostic-banner ${variant}`.trim();
  }

  function setDiagnosticPlaceholders() {
    diagnosticModel.textContent = "—";
    diagnosticNeedsReview.textContent = "—";
    diagnosticRawPixels.textContent = "—";
    diagnosticFinalPixels.textContent = "—";
    diagnosticAreaFilter.textContent = "—";
    diagnosticMeanProb.textContent = "—";
    settingsPills.innerHTML = "";
    sensitivityTable.innerHTML = "";
  }

  function formatCount(value) {
    return Number(value || 0).toLocaleString();
  }

  function updateControlOutputs() {
    pixelThresholdValue.textContent = Number(pixelThreshold.value).toFixed(2);
    maskAreaThresholdValue.textContent = `${formatCount(maskAreaThreshold.value)} px`;
    minPredictionAreaPixelsValue.textContent = `${formatCount(
      minPredictionAreaPixels.value
    )} px`;
    reviewConfidenceThresholdValue.textContent = Number(
      reviewConfidenceThreshold.value
    ).toFixed(2);
  }

  function getCurrentSettings() {
    return {
      pixel_threshold: Number(pixelThreshold.value).toFixed(2),
      mask_area_threshold: String(Math.round(Number(maskAreaThreshold.value))),
      min_prediction_area_pixels: String(
        Math.round(Number(minPredictionAreaPixels.value))
      ),
      review_confidence_threshold: Number(reviewConfidenceThreshold.value).toFixed(2),
      threshold_sensitivity_preset: thresholdSensitivityPreset.value,
    };
  }

  function maybeRerunInference() {
    if (currentFile) {
      submitImage(currentFile);
    }
  }

  async function checkHealth() {
    try {
      const res = await fetch(`${API_BASE}/health`, {
        signal: AbortSignal.timeout(5000),
      });
      if (res.ok) {
        statusDot.className = "status-dot online";
        statusText.textContent = "API connected";
        return true;
      }
    } catch {
      // Ignore.
    }
    statusDot.className = "status-dot offline";
    statusText.textContent = "API offline";
    return false;
  }

  checkHealth();
  setInterval(checkHealth, 15000);

  navToggle.addEventListener("click", () => {
    navLinks.classList.toggle("open");
  });

  navLinks.querySelectorAll("a").forEach((link) => {
    link.addEventListener("click", () => {
      navLinks.classList.remove("open");
    });
  });

  uploadArea.addEventListener("dragover", (event) => {
    event.preventDefault();
    uploadArea.classList.add("drag-active");
  });

  uploadArea.addEventListener("dragleave", () => {
    uploadArea.classList.remove("drag-active");
  });

  uploadArea.addEventListener("drop", (event) => {
    event.preventDefault();
    uploadArea.classList.remove("drag-active");
    if (event.dataTransfer.files.length) {
      handleFile(event.dataTransfer.files[0]);
    }
  });

  browseBtn.addEventListener("click", (event) => {
    event.stopPropagation();
    fileInput.click();
  });

  fileInput.addEventListener("change", () => {
    if (fileInput.files.length) {
      handleFile(fileInput.files[0]);
    }
  });

  clearBtn.addEventListener("click", (event) => {
    event.stopPropagation();
    resetUpload();
  });

  [pixelThreshold, maskAreaThreshold, minPredictionAreaPixels, reviewConfidenceThreshold].forEach(
    (input) => {
      input.addEventListener("input", updateControlOutputs);
      input.addEventListener("change", maybeRerunInference);
    }
  );

  thresholdSensitivityPreset.addEventListener("change", maybeRerunInference);
  updateControlOutputs();

  function handleFile(file) {
    const validTypes = ["image/jpeg", "image/png", "image/webp"];
    if (!validTypes.includes(file.type)) {
      alert("Please upload a JPEG, PNG, or WebP image.");
      return;
    }
    if (file.size > 20 * 1024 * 1024) {
      alert("File must be under 20 MB.");
      return;
    }

    currentFile = file;
    const reader = new FileReader();
    reader.onload = (event) => {
      previewImage.src = event.target.result;
      uploadContent.hidden = true;
      uploadPreview.hidden = false;
    };
    reader.readAsDataURL(file);
    submitImage(file);
  }

  function resetUpload() {
    currentFile = null;
    fileInput.value = "";
    uploadContent.hidden = false;
    uploadPreview.hidden = true;
    resultsArea.hidden = true;
    progressBar.hidden = true;
    progressFill.className = "progress-bar-fill";
    progressFill.style.width = "0%";
    setDiagnosticBanner("Waiting for inference…");
    setDiagnosticPlaceholders();
  }

  function renderSettingsPills(appliedSettings = {}) {
    settingsPills.innerHTML = "";
    const pills = [
      `Pixel ${Number(appliedSettings.pixel_threshold).toFixed(2)}`,
      `Image area ${formatCount(appliedSettings.mask_area_threshold)} px`,
      `Min area ${formatCount(appliedSettings.min_prediction_area_pixels)} px`,
      `Review ${Number(appliedSettings.review_confidence_threshold).toFixed(2)}`,
      `${
        appliedSettings.threshold_sensitivity_preset || "balanced"
      } · ${presetLabels[appliedSettings.threshold_sensitivity_preset || "balanced"]}`,
    ];
    pills.forEach((text) => {
      const pill = document.createElement("span");
      pill.className = "settings-pill";
      pill.textContent = text;
      settingsPills.appendChild(pill);
    });
  }

  function renderSensitivityTable(rows = []) {
    sensitivityTable.innerHTML = "";
    const maxPixels = Math.max(...rows.map((row) => row.final_pixels), 1);

    rows.forEach((row) => {
      const rowEl = document.createElement("div");
      rowEl.className = "sensitivity-row";

      const threshold = document.createElement("span");
      threshold.className = "sensitivity-threshold";
      threshold.textContent = `>${Number(row.threshold).toFixed(2)}`;

      const barWrap = document.createElement("div");
      barWrap.className = "sensitivity-bar-wrap";
      const bar = document.createElement("div");
      bar.className = "sensitivity-bar";
      bar.style.width = `${Math.max((row.final_pixels / maxPixels) * 100, 3)}%`;
      barWrap.appendChild(bar);

      const count = document.createElement("span");
      count.className = "sensitivity-count";
      count.textContent = `${formatCount(row.final_pixels)} px`;

      const filtered = document.createElement("span");
      filtered.className = "sensitivity-filter";
      filtered.textContent = row.area_filtered ? "filtered" : "kept";

      rowEl.append(threshold, barWrap, count, filtered);
      sensitivityTable.appendChild(rowEl);
    });
  }

  function renderDiagnostics(data) {
    diagnosticModel.textContent = data.model_version || "vR.P.30.1";
    diagnosticNeedsReview.textContent = data.needs_review ? "Yes" : "No";
    diagnosticRawPixels.textContent = formatCount(data.raw_tampered_pixel_count);
    diagnosticFinalPixels.textContent = formatCount(data.tampered_pixel_count);
    diagnosticAreaFilter.textContent = data.area_filter_triggered ? "Yes" : "No";
    diagnosticMeanProb.textContent = Number(data.confidence_mean_prob || 0).toFixed(4);

    const bannerMessage = data.needs_review
      ? "Borderline or unstable evidence detected. Manual review is recommended."
      : data.is_tampered
      ? "Stable suspicious region detected under the current forensic settings."
      : "No stable suspicious region detected under the current forensic settings.";
    const bannerVariant = data.needs_review
      ? ""
      : data.is_tampered
      ? "banner-alert"
      : "banner-ok";
    setDiagnosticBanner(bannerMessage, bannerVariant);

    renderSettingsPills(data.applied_settings);
    renderSensitivityTable(data.threshold_sensitivity || []);
  }

  function displayResults(data) {
    if (data.is_tampered) {
      setVerdictIcon(ICON_X, "verdict-tampered");
      verdictText.textContent = "Tampered";
      verdictDetail.textContent = `${(data.tampered_ratio * 100).toFixed(
        1
      )}% of image shows tampering`;
    } else {
      setVerdictIcon(ICON_CHECK, "verdict-authentic");
      verdictText.textContent = "Authentic";
      verdictDetail.textContent = "No tampering detected";
    }

    metricConfidence.textContent = `${(data.confidence * 100).toFixed(1)}%`;
    metricRatio.textContent = `${(data.tampered_ratio * 100).toFixed(2)}%`;
    metricTime.textContent = `${data.inference_time_ms.toFixed(0)}ms`;

    if (data.mask_base64) {
      maskImage.src = `data:image/png;base64,${data.mask_base64}`;
      maskImage.alt =
        "Tamper localization heatmap showing highlighted tampered regions";
    }

    renderDiagnostics(data);
  }

  async function submitImage(file) {
    progressBar.hidden = false;
    progressFill.className = "progress-bar-fill indeterminate";

    resultsArea.hidden = false;
    setVerdictIcon(ICON_LOADER, "");
    verdictText.textContent = "Analyzing…";
    verdictDetail.textContent = "Applying forensic controls";
    metricConfidence.textContent = "—";
    metricRatio.textContent = "—";
    metricTime.textContent = "—";
    maskImage.src = "";
    maskImage.alt = "Processing…";
    setDiagnosticBanner("Running the vR.P.30.1 forensic pipeline…");
    setDiagnosticPlaceholders();

    const formData = new FormData();
    formData.append("file", file);
    const settings = getCurrentSettings();
    Object.entries(settings).forEach(([key, value]) => {
      formData.append(key, value);
    });

    try {
      const res = await fetch(`${API_BASE}/infer`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${res.status}`);
      }

      const data = await res.json();
      displayResults(data);
    } catch (err) {
      setVerdictIcon(ICON_ALERT, "");
      verdictText.textContent = "Error";
      verdictDetail.textContent = err.message;
      setDiagnosticBanner(
        "The API rejected the request or the forensic model is unavailable.",
        "banner-alert"
      );
    } finally {
      progressFill.className = "progress-bar-fill";
      progressFill.style.width = "100%";
      setTimeout(() => {
        progressBar.hidden = true;
        progressFill.style.width = "0%";
      }, 600);
    }
  }

  const fadeElements = document.querySelectorAll(".fade-in");

  if ("IntersectionObserver" in window) {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add("visible");
            observer.unobserve(entry.target);
          }
        });
      },
      {
        threshold: 0.15,
        rootMargin: "0px 0px -40px 0px",
      }
    );

    fadeElements.forEach((el, index) => {
      el.style.transitionDelay = `${index * 80}ms`;
      observer.observe(el);
    });
  } else {
    fadeElements.forEach((el) => el.classList.add("visible"));
  }

  function animateCounter(element, target, duration = 1500) {
    const isFloat = String(target).includes(".");
    const start = 0;
    const startTime = performance.now();

    function update(currentTime) {
      const elapsed = currentTime - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const ease = 1 - Math.pow(1 - progress, 3);
      const current = start + (target - start) * ease;

      if (isFloat) {
        element.textContent = current.toFixed(4);
      } else {
        element.textContent = `${Math.round(current)}+`;
      }

      if (progress < 1) {
        requestAnimationFrame(update);
      } else if (isFloat) {
        element.textContent = target.toFixed(4);
      } else {
        element.textContent = `${target}+`;
      }
    }

    requestAnimationFrame(update);
  }

  const metricCards = document.querySelectorAll(".metric-card-value[data-count]");

  if ("IntersectionObserver" in window && metricCards.length) {
    const counterObserver = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            const target = parseFloat(entry.target.dataset.count);
            animateCounter(entry.target, target);
            counterObserver.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.5 }
    );

    metricCards.forEach((el) => counterObserver.observe(el));
  }
})();
