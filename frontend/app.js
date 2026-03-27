/**
 * TIDAL Frontend — Application Logic
 * Tampered Image Detection & Localization
 *
 * Handles:
 * - API health polling
 * - Image upload (drag-drop + file picker)
 * - Inference request + result display
 * - Scroll-triggered fade-in animations
 * - Mobile nav toggle
 */

(() => {
  "use strict";

  // ─── Config ────────────────────────────────────
  const API_BASE =
    location.hostname === "localhost" || location.hostname === "127.0.0.1"
      ? "http://localhost:8000"
      : "https://the-harsh-vardhan-tidal-api.hf.space";

  // ─── DOM Refs ──────────────────────────────────
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
  const verdictIconWrap = $("verdictIconWrap");
  const verdictIconSvg = $("verdictIconSvg");
  const verdictText = $("verdictText");
  const verdictDetail = $("verdictDetail");
  const metricConfidence = $("metricConfidence");
  const metricRatio = $("metricRatio");
  const metricTime = $("metricTime");
  const maskImage = $("maskImage");

  const statusDot = $("statusDot");
  const statusText = $("statusText");
  const navToggle = $("navToggle");
  const navLinks = $("navLinks");

  // ─── SVG Templates ─────────────────────────────
  const ICON_CHECK =
    '<polyline points="20 6 9 17 4 12"/>';
  const ICON_X =
    '<line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>';
  const ICON_LOADER =
    '<line x1="12" y1="2" x2="12" y2="6"/><line x1="12" y1="18" x2="12" y2="22"/><line x1="4.93" y1="4.93" x2="7.76" y2="7.76"/><line x1="16.24" y1="16.24" x2="19.07" y2="19.07"/><line x1="2" y1="12" x2="6" y2="12"/><line x1="18" y1="12" x2="22" y2="12"/><line x1="4.93" y1="19.07" x2="7.76" y2="16.24"/><line x1="16.24" y1="7.76" x2="19.07" y2="4.93"/>';
  const ICON_ALERT =
    '<circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>';

  function setVerdictIcon(svgContent, className) {
    verdictIconSvg.innerHTML = svgContent;
    resultVerdict.className = `result-verdict ${className}`;
  }

  // ─── Health Check ──────────────────────────────
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
      // ignore
    }
    statusDot.className = "status-dot offline";
    statusText.textContent = "API offline";
    return false;
  }

  checkHealth();
  setInterval(checkHealth, 15000);

  // ─── Mobile Nav ────────────────────────────────
  navToggle.addEventListener("click", () => {
    navLinks.classList.toggle("open");
  });

  // Close on link click
  navLinks.querySelectorAll("a").forEach((link) => {
    link.addEventListener("click", () => {
      navLinks.classList.remove("open");
    });
  });

  // ─── File Upload ───────────────────────────────
  uploadArea.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadArea.classList.add("drag-active");
  });

  uploadArea.addEventListener("dragleave", () => {
    uploadArea.classList.remove("drag-active");
  });

  uploadArea.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadArea.classList.remove("drag-active");
    if (e.dataTransfer.files.length) {
      handleFile(e.dataTransfer.files[0]);
    }
  });

  browseBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    fileInput.click();
  });

  fileInput.addEventListener("change", () => {
    if (fileInput.files.length) {
      handleFile(fileInput.files[0]);
    }
  });

  clearBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    resetUpload();
  });

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

    const reader = new FileReader();
    reader.onload = (e) => {
      previewImage.src = e.target.result;
      uploadContent.hidden = true;
      uploadPreview.hidden = false;
    };
    reader.readAsDataURL(file);
    submitImage(file);
  }

  function resetUpload() {
    fileInput.value = "";
    uploadContent.hidden = false;
    uploadPreview.hidden = true;
    resultsArea.hidden = true;
    progressBar.hidden = true;
    progressFill.className = "progress-bar-fill";
    progressFill.style.width = "0%";
  }

  // ─── Inference ─────────────────────────────────
  async function submitImage(file) {
    // Show progress
    progressBar.hidden = false;
    progressFill.className = "progress-bar-fill indeterminate";

    // Show results panel in loading state
    resultsArea.hidden = false;
    setVerdictIcon(ICON_LOADER, "");
    verdictText.textContent = "Analyzing…";
    verdictDetail.textContent = "Processing your image";
    metricConfidence.textContent = "—";
    metricRatio.textContent = "—";
    metricTime.textContent = "—";
    maskImage.src = "";
    maskImage.alt = "Processing…";

    const formData = new FormData();
    formData.append("file", file);

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
    } finally {
      // Hide progress
      progressFill.className = "progress-bar-fill";
      progressFill.style.width = "100%";
      setTimeout(() => {
        progressBar.hidden = true;
        progressFill.style.width = "0%";
      }, 600);
    }
  }

  function displayResults(data) {
    const tampered = data.is_tampered;

    if (tampered) {
      setVerdictIcon(ICON_X, "verdict-tampered");
      verdictText.textContent = "Tampered";
      verdictDetail.textContent = `${(data.tampered_ratio * 100).toFixed(1)}% of image shows tampering`;
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
      maskImage.alt = "Tamper localization heatmap showing highlighted tampered regions";
    }
  }

  // ─── Scroll Fade-in (IntersectionObserver) ─────
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

    fadeElements.forEach((el, i) => {
      el.style.transitionDelay = `${i * 80}ms`;
      observer.observe(el);
    });
  } else {
    // Fallback: show all immediately
    fadeElements.forEach((el) => el.classList.add("visible"));
  }

  // ─── Animated Number Counters ──────────────────
  function animateCounter(element, target, duration = 1500) {
    const isFloat = String(target).includes(".");
    const start = 0;
    const startTime = performance.now();

    function update(currentTime) {
      const elapsed = currentTime - startTime;
      const progress = Math.min(elapsed / duration, 1);
      // Ease out cubic
      const ease = 1 - Math.pow(1 - progress, 3);
      const current = start + (target - start) * ease;

      if (isFloat) {
        element.textContent = current.toFixed(4);
      } else {
        element.textContent = `${Math.round(current)}+`;
      }

      if (progress < 1) {
        requestAnimationFrame(update);
      } else {
        // Restore original text
        if (isFloat) {
          element.textContent = target.toFixed(4);
        } else {
          element.textContent = `${target}+`;
        }
      }
    }

    requestAnimationFrame(update);
  }

  // Observe metric cards for counter animation
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
