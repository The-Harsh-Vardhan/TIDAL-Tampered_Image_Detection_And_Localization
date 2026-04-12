"use client";
/* eslint-disable @next/next/no-img-element */

import { useRef, useState } from "react";

import { FadeIn } from "@/components/fade-in";
import { ResultsPanel } from "@/components/results-panel";
import {
  ANALYTICS_MODE_ADVANCED,
  ANALYTICS_MODE_SIMPLE,
  PRESET_LABELS,
  formatCount,
} from "@/lib/forensic-formatters";

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

function ModeButton({ isActive, label, mode, onSelect }) {
  return (
    <button
      className={`mode-option ${isActive ? "is-active" : ""}`.trim()}
      type="button"
      aria-pressed={isActive}
      onClick={() => onSelect(mode)}
    >
      {label}
    </button>
  );
}

export function DemoWorkspace({
  analyticsMode,
  comparisonViews,
  errorMessage,
  healthStatus,
  isAnalyzing,
  isDemoLoading,
  previewDataUrl,
  resultData,
  resultsVisible,
  settings,
  visualTab,
  onAnalyticsModeChange,
  onClearUpload,
  onCommitSetting,
  onRunDemo,
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

  const isAdvancedMode = analyticsMode === ANALYTICS_MODE_ADVANCED;
  const statusLabel = getStatusLabel(healthStatus);

  return (
    <section className="section section-alt forensic-workspace-section" id="demo">
      <div className="container">
        <FadeIn className="section-header">
          <div className="section-kicker">Operator Console</div>
          <h2 className="section-title">Live Forensic Workspace</h2>
          <p className="section-subtitle">
            Start with the bundled tampered sample or upload your own image.
            Simple mode keeps the readout visual; Advanced exposes every
            forensic threshold.
          </p>
        </FadeIn>

        <FadeIn className="forensic-console">
          <aside className="forensic-rail glass-card" aria-label="Forensic controls">
            <div className="rail-status">
              <span className={`status-dot ${healthStatus}`.trim()} />
              <div>
                <span className="rail-eyebrow">HF Space</span>
                <strong>{statusLabel}</strong>
              </div>
            </div>

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

            <div className="rail-panel">
              <span className="rail-eyebrow">Input</span>
              <div className="rail-action-grid">
                <button
                  className="rail-button rail-button--primary"
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
                  Upload
                </button>
                <button
                  className="rail-button"
                  type="button"
                  disabled={isDemoLoading}
                  onClick={onRunDemo}
                >
                  <svg
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  >
                    <polygon points="5 3 19 12 5 21 5 3" />
                  </svg>
                  Demo
                </button>
              </div>
              <p className="rail-note">
                {isDemoLoading
                  ? "Running demo image through HF inference..."
                  : "Use the sample if you do not have a CASIA-style tampered image ready."}
              </p>
            </div>

            <button
              className={`rail-dropzone ${dragActive ? "drag-active" : ""}`.trim()}
              type="button"
              onClick={() => fileInputRef.current?.click()}
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
              <span className="upload-icon rail-upload-icon">
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
              </span>
              <strong>Drop image here</strong>
              <span>JPEG, PNG, WebP · Max 20 MB</span>
            </button>

            {previewDataUrl ? (
              <div className="rail-preview">
                <img src={previewDataUrl} alt="Current forensic input preview" />
                <div className="rail-preview-meta">
                  <span>Current input</span>
                  <button className="btn btn-sm btn-secondary" type="button" onClick={onClearUpload}>
                    <svg
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    >
                      <line x1="18" y1="6" x2="6" y2="18" />
                      <line x1="6" y1="6" x2="18" y2="18" />
                    </svg>
                    Clear
                  </button>
                </div>
              </div>
            ) : null}

            {(isAnalyzing || isDemoLoading) ? (
              <div className="progress-bar">
                <div className="progress-bar-fill indeterminate" />
              </div>
            ) : null}

            <div className="rail-panel">
              <span className="rail-eyebrow">Analytics Mode</span>
              <div className="mode-toggle" role="group" aria-label="Analytics mode">
                <ModeButton
                  isActive={analyticsMode === ANALYTICS_MODE_SIMPLE}
                  label="Simple Analytics"
                  mode={ANALYTICS_MODE_SIMPLE}
                  onSelect={onAnalyticsModeChange}
                />
                <ModeButton
                  isActive={isAdvancedMode}
                  label="Advanced Analytics"
                  mode={ANALYTICS_MODE_ADVANCED}
                  onSelect={onAnalyticsModeChange}
                />
              </div>
              <p className="rail-note">
                Simple shows the verdict, comparisons, coverage, and decision
                pressure. Advanced adds controls and diagnostics.
              </p>
            </div>

            {isAdvancedMode ? (
            <div className="controls-card rail-controls">
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
            ) : null}
          </aside>

          <main className="forensic-canvas" aria-live="polite">
            <div className="workspace-strip">
              <span className="workspace-chip">vR.P.30.1</span>
              <span className="workspace-chip">
                {analyticsMode === ANALYTICS_MODE_SIMPLE
                  ? "Simple Analytics"
                  : "Advanced Analytics"}
              </span>
              <span className="workspace-chip workspace-chip--status">
                {isAnalyzing ? "Inference running" : "Ready"}
              </span>
            </div>

            {resultsVisible ? (
            <ResultsPanel
              analyticsMode={analyticsMode}
              comparisonViews={comparisonViews}
              errorMessage={errorMessage}
              isAnalyzing={isAnalyzing}
              resultData={resultData}
              visualTab={visualTab}
              onVisualTabChange={(nextTab) =>
                onVisualTabChange(nextTab, nextTab !== visualTab)
              }
            />
            ) : (
              <div className="workspace-empty glass-card">
                <span className="workspace-empty-tag">No scan loaded</span>
                <h3>Run Demo to see a full forensic readout instantly.</h3>
                <p>
                  The console will render original, black-background detection,
                  red overlay, coverage pixels, and the decision threshold
                  pressure without needing any new backend route.
                </p>
                <button
                  className="btn btn-primary"
                  type="button"
                  disabled={isDemoLoading}
                  onClick={onRunDemo}
                >
                  <svg
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  >
                    <polygon points="5 3 19 12 5 21 5 3" />
                  </svg>
                  Run Demo
                </button>
              </div>
            )}
          </main>
        </FadeIn>
      </div>
    </section>
  );
}
