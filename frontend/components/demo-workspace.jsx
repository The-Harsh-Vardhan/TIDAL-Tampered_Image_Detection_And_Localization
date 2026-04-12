"use client";
/* eslint-disable @next/next/no-img-element */

import { useRef, useState } from "react";

import { FadeIn } from "@/components/fade-in";
import { ResultsPanel } from "@/components/results-panel";
import { PRESET_LABELS, formatCount } from "@/lib/forensic-formatters";

function RangeField({
  controlId,
  displayValue,
  helper,
  max,
  min,
  onCommit,
  onDraftChange,
  step,
  title,
  value,
}) {
  function handleKeyboardCommit(event) {
    if (
      event.key.startsWith("Arrow") ||
      event.key === "Home" ||
      event.key === "End" ||
      event.key === "PageUp" ||
      event.key === "PageDown"
    ) {
      onCommit(Number(event.currentTarget.value));
    }
  }

  return (
    <label className="control-field" htmlFor={controlId}>
      <span className="control-title">{title}</span>
      <span className="control-helper">{helper}</span>
      <div className="control-input-row">
        <input
          id={controlId}
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(event) => onDraftChange(Number(event.target.value))}
          onPointerUp={(event) => onCommit(Number(event.currentTarget.value))}
          onKeyUp={handleKeyboardCommit}
          onBlur={(event) => onCommit(Number(event.currentTarget.value))}
        />
        <output htmlFor={controlId}>{displayValue}</output>
      </div>
    </label>
  );
}

function getStatusLabel(healthStatus) {
  if (healthStatus === "online") {
    return "API connected";
  }

  if (healthStatus === "offline") {
    return "API offline";
  }

  return "Checking API…";
}

export function DemoWorkspace({
  comparisonViews,
  errorMessage,
  healthStatus,
  isAnalyzing,
  previewDataUrl,
  resultData,
  resultsVisible,
  settings,
  visualTab,
  onClearUpload,
  onCommitSetting,
  onSelectFile,
  onUpdateSetting,
  onVisualTabChange,
}) {
  const fileInputRef = useRef(null);
  const [dragActive, setDragActive] = useState(false);

  async function handleFiles(file, source) {
    if (!file) {
      return;
    }

    setDragActive(false);
    await onSelectFile(file, source);
  }

  return (
    <section className="section section-alt" id="demo">
      <div className="container">
        <FadeIn className="section-header">
          <h2 className="section-title">Try It Live</h2>
          <p className="section-subtitle">
            Upload any image to scan for tampering instantly
          </p>
        </FadeIn>

        <FadeIn className="demo-layout">
          <div className="demo-stack">
            <div
              className={`upload-area glass-card ${dragActive ? "drag-active" : ""}`.trim()}
              onDragOver={(event) => {
                event.preventDefault();
                setDragActive(true);
              }}
              onDragLeave={() => setDragActive(false)}
              onDrop={(event) => {
                event.preventDefault();
                const [file] = event.dataTransfer.files;
                handleFiles(file, "drop");
              }}
            >
              {!previewDataUrl ? (
                <div className="upload-content">
                  <div className="upload-icon">
                    <svg
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    >
                      <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4" />
                      <polyline points="17 8 12 3 7 8" />
                      <line x1="12" y1="3" x2="12" y2="15" />
                    </svg>
                  </div>
                  <h3>Drop image here</h3>
                  <p>JPEG, PNG, WebP · Max 20 MB</p>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/jpeg,image/png,image/webp"
                    hidden
                    onChange={(event) => {
                      const [file] = event.target.files || [];
                      handleFiles(file, "browse");
                      event.target.value = "";
                    }}
                  />
                  <button
                    className="btn btn-primary"
                    type="button"
                    onClick={() => fileInputRef.current?.click()}
                  >
                    <svg
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    >
                      <rect x="3" y="3" width="18" height="18" rx="2" />
                      <circle cx="8.5" cy="8.5" r="1.5" />
                      <polyline points="21 15 16 10 5 21" />
                    </svg>
                    Select Image
                  </button>
                </div>
              ) : (
                <div className="upload-preview">
                  <img src={previewDataUrl} alt="Uploaded image preview" />
                  <button
                    className="btn btn-sm btn-secondary"
                    type="button"
                    onClick={onClearUpload}
                  >
                    <svg
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      style={{ width: 14, height: 14 }}
                    >
                      <line x1="18" y1="6" x2="6" y2="18" />
                      <line x1="6" y1="6" x2="18" y2="18" />
                    </svg>
                    Clear
                  </button>
                </div>
              )}

              {isAnalyzing ? (
                <div className="progress-bar">
                  <div className="progress-bar-fill indeterminate" />
                </div>
              ) : null}
            </div>

            <div className="controls-card glass-card">
              <div className="controls-header">
                <h3>Forensic Controls</h3>
                <p>
                  Tune the notebook-style thresholds live. Higher values are
                  stricter.
                </p>
              </div>
              <div className="controls-grid">
                <RangeField
                  controlId="pixelThreshold"
                  title="Pixel threshold"
                  helper="0.05 to 0.95. Higher means fewer pixels survive."
                  min={0.05}
                  max={0.95}
                  step={0.05}
                  value={settings.pixelThreshold}
                  displayValue={Number(settings.pixelThreshold).toFixed(2)}
                  onDraftChange={(value) => onUpdateSetting("pixelThreshold", value)}
                  onCommit={(value) => onCommitSetting("pixelThreshold", value)}
                />
                <RangeField
                  controlId="maskAreaThreshold"
                  title="Image area threshold"
                  helper="0 to 147456. Higher requires a larger suspicious region."
                  min={0}
                  max={147456}
                  step={25}
                  value={settings.maskAreaThreshold}
                  displayValue={`${formatCount(settings.maskAreaThreshold)} px`}
                  onDraftChange={(value) =>
                    onUpdateSetting("maskAreaThreshold", value)
                  }
                  onCommit={(value) => onCommitSetting("maskAreaThreshold", value)}
                />
                <RangeField
                  controlId="minPredictionAreaPixels"
                  title="Minimum predicted area"
                  helper="0 to 147456. Higher removes tiny blobs after thresholding."
                  min={0}
                  max={147456}
                  step={25}
                  value={settings.minPredictionAreaPixels}
                  displayValue={`${formatCount(settings.minPredictionAreaPixels)} px`}
                  onDraftChange={(value) =>
                    onUpdateSetting("minPredictionAreaPixels", value)
                  }
                  onCommit={(value) =>
                    onCommitSetting("minPredictionAreaPixels", value)
                  }
                />
                <RangeField
                  controlId="reviewConfidenceThreshold"
                  title="Review confidence"
                  helper="0.05 to 0.95. Higher flags more borderline cases."
                  min={0.05}
                  max={0.95}
                  step={0.05}
                  value={settings.reviewConfidenceThreshold}
                  displayValue={Number(settings.reviewConfidenceThreshold).toFixed(2)}
                  onDraftChange={(value) =>
                    onUpdateSetting("reviewConfidenceThreshold", value)
                  }
                  onCommit={(value) =>
                    onCommitSetting("reviewConfidenceThreshold", value)
                  }
                />
              </div>
              <label className="control-select" htmlFor="thresholdSensitivityPreset">
                <span className="control-title">Threshold sensitivity preset</span>
                <span className="control-helper">
                  This affects the comparison chart, not the live verdict
                  threshold.
                </span>
                <select
                  id="thresholdSensitivityPreset"
                  value={settings.thresholdSensitivityPreset}
                  onChange={(event) =>
                    onCommitSetting(
                      "thresholdSensitivityPreset",
                      event.target.value
                    )
                  }
                >
                  <option value="lenient">
                    Lenient · {PRESET_LABELS.lenient}
                  </option>
                  <option value="balanced">
                    Balanced · {PRESET_LABELS.balanced}
                  </option>
                  <option value="strict">
                    Strict · {PRESET_LABELS.strict}
                  </option>
                </select>
              </label>
            </div>
          </div>

          {resultsVisible ? (
            <ResultsPanel
              comparisonViews={comparisonViews}
              errorMessage={errorMessage}
              isAnalyzing={isAnalyzing}
              resultData={resultData}
              visualTab={visualTab}
              onVisualTabChange={(nextTab) =>
                onVisualTabChange(nextTab, nextTab !== visualTab)
              }
            />
          ) : null}
        </FadeIn>

        <div className="status-bar">
          <div className={`status-dot ${healthStatus}`.trim()} />
          <span>{getStatusLabel(healthStatus)}</span>
        </div>
      </div>
    </section>
  );
}
