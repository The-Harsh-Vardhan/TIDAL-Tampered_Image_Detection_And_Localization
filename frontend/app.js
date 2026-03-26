/**
 * frontend/app.js
 * ================
 * TIDAL Frontend — API client, health polling, drag-and-drop upload.
 */

(() => {
    "use strict";

    // ── Configuration ─────────────────────────────────────────────────────
    const API_BASE = window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1"
        ? "http://localhost:8000"
        : "";
    const HEALTH_POLL_INTERVAL = 15000; // 15 seconds

    // ── DOM Elements ──────────────────────────────────────────────────────
    const uploadArea = document.getElementById("uploadArea");
    const uploadContent = document.getElementById("uploadContent");
    const uploadPreview = document.getElementById("uploadPreview");
    const fileInput = document.getElementById("fileInput");
    const browseBtn = document.getElementById("browseBtn");
    const clearBtn = document.getElementById("clearBtn");
    const previewImage = document.getElementById("previewImage");
    const resultsArea = document.getElementById("resultsArea");
    const statusIndicator = document.getElementById("statusIndicator");
    const statusText = document.getElementById("statusText");

    // Result elements
    const resultVerdict = document.getElementById("resultVerdict");
    const verdictIcon = document.getElementById("verdictIcon");
    const verdictText = document.getElementById("verdictText");
    const verdictDetail = document.getElementById("verdictDetail");
    const metricConfidence = document.getElementById("metricConfidence");
    const metricRatio = document.getElementById("metricRatio");
    const metricTime = document.getElementById("metricTime");
    const maskImage = document.getElementById("maskImage");

    let currentFile = null;

    // ── Health Check ──────────────────────────────────────────────────────
    async function checkHealth() {
        try {
            const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(5000) });
            if (res.ok) {
                statusIndicator.className = "status-indicator online";
                statusText.textContent = "API connected";
                return true;
            }
        } catch {
            // ignore
        }
        statusIndicator.className = "status-indicator offline";
        statusText.textContent = "API offline — start the backend server";
        return false;
    }

    // Poll health
    checkHealth();
    setInterval(checkHealth, HEALTH_POLL_INTERVAL);

    // ── Drag & Drop ──────────────────────────────────────────────────────
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
        const files = e.dataTransfer.files;
        if (files.length > 0) handleFile(files[0]);
    });

    // ── File Selection ────────────────────────────────────────────────────
    browseBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        fileInput.click();
    });

    fileInput.addEventListener("change", () => {
        if (fileInput.files.length > 0) handleFile(fileInput.files[0]);
    });

    clearBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        clearUpload();
    });

    // ── Handle File ──────────────────────────────────────────────────────
    function handleFile(file) {
        // Validate type
        const validTypes = ["image/jpeg", "image/png", "image/webp"];
        if (!validTypes.includes(file.type)) {
            showError("Please upload a JPEG, PNG, or WebP image.");
            return;
        }

        // Validate size (20 MB)
        if (file.size > 20 * 1024 * 1024) {
            showError("File too large. Maximum size is 20 MB.");
            return;
        }

        currentFile = file;

        // Show preview
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            uploadContent.hidden = true;
            uploadPreview.hidden = false;
        };
        reader.readAsDataURL(file);

        // Auto-submit
        submitInference(file);
    }

    function clearUpload() {
        currentFile = null;
        fileInput.value = "";
        uploadContent.hidden = false;
        uploadPreview.hidden = true;
        resultsArea.hidden = true;
    }

    // ── Submit Inference ──────────────────────────────────────────────────
    async function submitInference(file) {
        resultsArea.hidden = true;

        // Show loading state
        verdictText.textContent = "Analyzing...";
        verdictDetail.textContent = "Running ELA + UNet pipeline";
        verdictIcon.textContent = "⏳";
        resultVerdict.className = "result-verdict";
        metricConfidence.textContent = "—";
        metricRatio.textContent = "—";
        metricTime.textContent = "—";
        resultsArea.hidden = false;
        maskImage.src = "";

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
            displayResult(data);
        } catch (err) {
            showResultError(err.message);
        }
    }

    // ── Display Result ───────────────────────────────────────────────────
    function displayResult(data) {
        const tampered = data.is_tampered;

        resultVerdict.className = `result-verdict ${tampered ? "verdict-tampered" : "verdict-authentic"}`;
        verdictIcon.textContent = tampered ? "✗" : "✓";
        verdictText.textContent = tampered ? "Tampered" : "Authentic";
        verdictDetail.textContent = tampered
            ? `Tampering detected in ${(data.tampered_ratio * 100).toFixed(1)}% of the image`
            : "No tampering evidence found";

        metricConfidence.textContent = `${(data.confidence * 100).toFixed(1)}%`;
        metricRatio.textContent = `${(data.tampered_ratio * 100).toFixed(2)}%`;
        metricTime.textContent = `${data.inference_time_ms.toFixed(0)}ms`;

        // Show mask
        if (data.mask_base64) {
            maskImage.src = `data:image/png;base64,${data.mask_base64}`;
        }
    }

    function showResultError(message) {
        verdictIcon.textContent = "⚠";
        verdictText.textContent = "Error";
        verdictDetail.textContent = message;
        resultVerdict.className = "result-verdict";
    }

    function showError(message) {
        alert(message);
    }
})();
