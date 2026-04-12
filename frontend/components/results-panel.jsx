/* eslint-disable @next/next/no-img-element */

import {
  PRESET_LABELS,
  TAB_COMPARISON,
  clamp,
  formatCount,
  formatRatioPercent,
} from "@/lib/forensic-formatters";

function VerdictIcon({ variant }) {
  if (variant === "error") {
    return (
      <svg
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <circle cx="12" cy="12" r="10" />
        <line x1="12" y1="8" x2="12" y2="12" />
        <line x1="12" y1="16" x2="12.01" y2="16" />
      </svg>
    );
  }

  if (variant === "loading") {
    return (
      <svg
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <line x1="12" y1="2" x2="12" y2="6" />
        <line x1="12" y1="18" x2="12" y2="22" />
        <line x1="4.93" y1="4.93" x2="7.76" y2="7.76" />
        <line x1="16.24" y1="16.24" x2="19.07" y2="19.07" />
        <line x1="2" y1="12" x2="6" y2="12" />
        <line x1="18" y1="12" x2="22" y2="12" />
        <line x1="4.93" y1="19.07" x2="7.76" y2="16.24" />
        <line x1="16.24" y1="7.76" x2="19.07" y2="4.93" />
      </svg>
    );
  }

  if (variant === "tampered") {
    return (
      <svg
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <line x1="18" y1="6" x2="6" y2="18" />
        <line x1="6" y1="6" x2="18" y2="18" />
      </svg>
    );
  }

  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2.5"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <polyline points="20 6 9 17 4 12" />
    </svg>
  );
}

function getVerdictState(resultData, isAnalyzing, errorMessage) {
  if (errorMessage) {
    return {
      className: "",
      detail: errorMessage,
      label: "Error",
      variant: "error",
    };
  }

  if (isAnalyzing) {
    return {
      className: "",
      detail: "Applying forensic controls",
      label: "Analyzing…",
      variant: "loading",
    };
  }

  if (resultData?.is_tampered) {
    return {
      className: "verdict-tampered",
      detail: `${formatRatioPercent(resultData.tampered_ratio)} of image shows tampering`,
      label: "Tampered",
      variant: "tampered",
    };
  }

  return {
    className: "verdict-authentic",
    detail: "No tampering detected",
    label: "Authentic",
    variant: "authentic",
  };
}

function getDiagnosticState(resultData, isAnalyzing, errorMessage) {
  if (errorMessage) {
    return {
      message:
        "The API rejected the request or the forensic model is unavailable.",
      variant: "banner-alert",
    };
  }

  if (isAnalyzing) {
    return {
      message: "Running the vR.P.30.1 forensic pipeline…",
      variant: "",
    };
  }

  if (!resultData) {
    return {
      message: "Waiting for inference…",
      variant: "",
    };
  }

  if (resultData.needs_review) {
    return {
      message:
        "Borderline or unstable evidence detected. Manual review is recommended.",
      variant: "",
    };
  }

  if (resultData.is_tampered) {
    return {
      message:
        "Stable suspicious region detected under the current forensic settings.",
      variant: "banner-alert",
    };
  }

  return {
    message:
      "No stable suspicious region detected under the current forensic settings.",
    variant: "banner-ok",
  };
}

function renderSettingsPills(appliedSettings = {}) {
  if (!Object.keys(appliedSettings).length) {
    return [];
  }

  const preset = appliedSettings.threshold_sensitivity_preset || "balanced";

  return [
    `Pixel ${Number(appliedSettings.pixel_threshold).toFixed(2)}`,
    `Image area ${formatCount(appliedSettings.mask_area_threshold)} px`,
    `Min area ${formatCount(appliedSettings.min_prediction_area_pixels)} px`,
    `Review ${Number(appliedSettings.review_confidence_threshold).toFixed(2)}`,
    `${preset} · ${PRESET_LABELS[preset]}`,
  ];
}

function SummarySection({ resultData }) {
  if (!resultData) {
    return (
      <div className="forensic-summary glass-card">
        <div className="forensic-summary-header">
          <div className="forensic-summary-title">
            <svg
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <path d="M3 3v18h18" />
              <path d="M7 14l3-3 3 2 4-5" />
            </svg>
            Forensic Summary
          </div>
          <p>
            Read the current detection as three quick signals: image coverage,
            decision pressure, and review confidence.
          </p>
        </div>
        <div className="summary-grid">
          {["Coverage", "Decision Threshold", "Confidence vs Review"].map(
            (label) => (
              <article key={label} className="summary-card">
                <span className="summary-eyebrow">{label}</span>
                <div className="summary-value-row">
                  <strong>—</strong>
                  <span>Waiting for inference</span>
                </div>
                <div className="summary-meter">
                  <div className="summary-meter-fill" style={{ width: "0%" }} />
                </div>
                <p className="summary-note">
                  The live summary will populate after inference.
                </p>
              </article>
            )
          )}
        </div>
        <div className="sensitivity-panel sensitivity-panel--full">
          <div className="sensitivity-header">
            <h4>Threshold Sensitivity</h4>
            <p>Predicted pixels after the minimum-area filter at each preset threshold.</p>
          </div>
          <div className="sensitivity-table">
            <div className="sensitivity-empty">
              Sensitivity data will appear after inference.
            </div>
          </div>
        </div>
      </div>
    );
  }

  const appliedSettings = resultData.applied_settings || {};
  const coveragePercent = Number(resultData.tampered_ratio || 0) * 100;
  const rawPixels = Number(resultData.raw_tampered_pixel_count || 0);
  const finalPixels = Number(resultData.tampered_pixel_count || 0);
  const thresholdPixels = Number(appliedSettings.mask_area_threshold || 0);
  const confidence = Number(resultData.confidence || 0);
  const reviewThreshold = Number(
    appliedSettings.review_confidence_threshold || 0.65
  );

  const decisionFill =
    thresholdPixels <= 0
      ? resultData.is_tampered
        ? 100
        : 0
      : clamp(
          finalPixels >= thresholdPixels
            ? 100
            : Math.max((finalPixels / thresholdPixels) * 100, finalPixels > 0 ? 4 : 0),
          0,
          100
        );

  const sensitivityRows = resultData.threshold_sensitivity || [];
  const maxPixels = Math.max(
    ...sensitivityRows.map((row) => Number(row.final_pixels || 0)),
    1
  );

  return (
    <div className="forensic-summary glass-card">
      <div className="forensic-summary-header">
        <div className="forensic-summary-title">
          <svg
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="M3 3v18h18" />
            <path d="M7 14l3-3 3 2 4-5" />
          </svg>
          Forensic Summary
        </div>
        <p>
          Read the current detection as three quick signals: image coverage,
          decision pressure, and review confidence.
        </p>
      </div>
      <div className="summary-grid">
        <article className="summary-card">
          <span className="summary-eyebrow">Coverage</span>
          <div className="summary-value-row">
            <strong>{coveragePercent.toFixed(2)}%</strong>
            <span>
              {formatCount(finalPixels)} px kept from {formatCount(rawPixels)} raw pixels
            </span>
          </div>
          <div className="summary-meter">
            <div
              className="summary-meter-fill"
              style={{ width: `${clamp(coveragePercent, 0, 100)}%` }}
            />
          </div>
          <p className="summary-note">
            Share of the image that remains positive after the minimum-area
            filter.
          </p>
        </article>
        <article className="summary-card">
          <span className="summary-eyebrow">Decision Threshold</span>
          <div className="summary-value-row">
            <strong>
              {formatCount(finalPixels)} / {formatCount(thresholdPixels)} px
            </strong>
            <span>
              {finalPixels >= thresholdPixels
                ? `${formatCount(finalPixels - thresholdPixels)} px above threshold`
                : `${formatCount(thresholdPixels - finalPixels)} px below threshold`}
            </span>
          </div>
          <div className="summary-meter">
            <div
              className="summary-meter-fill summary-meter-fill--danger"
              style={{ width: `${decisionFill}%` }}
            />
          </div>
          <p className="summary-note">
            How close the current prediction is to the image-level tampered
            decision threshold.
          </p>
        </article>
        <article className="summary-card">
          <span className="summary-eyebrow">Confidence vs Review</span>
          <div className="summary-value-row">
            <strong>{(confidence * 100).toFixed(1)}%</strong>
            <span>
              Review line {(reviewThreshold * 100).toFixed(1)}%
            </span>
          </div>
          <div className="summary-meter summary-meter--marker">
            <div
              className="summary-meter-fill summary-meter-fill--accent"
              style={{ width: `${clamp(confidence * 100, 0, 100)}%` }}
            />
            <span
              className="summary-meter-marker"
              style={{ left: `${clamp(reviewThreshold * 100, 0, 100)}%` }}
            />
          </div>
          <p className="summary-note">
            {resultData.needs_review
              ? "Manual review is recommended under the current confidence and area settings."
              : "Confidence clears the current review line and the region looks stable."}
          </p>
        </article>
      </div>
      <div className="sensitivity-panel sensitivity-panel--full">
        <div className="sensitivity-header">
          <h4>Threshold Sensitivity</h4>
          <p>Predicted pixels after the minimum-area filter at each preset threshold.</p>
        </div>
        <div className="sensitivity-table">
          {sensitivityRows.map((row) => (
            <div key={row.threshold} className="sensitivity-row">
              <span className="sensitivity-threshold">
                &gt;{Number(row.threshold).toFixed(2)}
              </span>
              <div className="sensitivity-bar-wrap">
                <div
                  className="sensitivity-bar"
                  style={{
                    width: `${Math.max(
                      (Number(row.final_pixels || 0) / maxPixels) * 100,
                      3
                    )}%`,
                  }}
                />
              </div>
              <span className="sensitivity-count">
                {formatCount(row.final_pixels)} px
              </span>
              <span className="sensitivity-filter">
                {row.area_filtered ? "filtered" : "kept"}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function DiagnosticsSection({ diagnosticState, resultData }) {
  const pills = renderSettingsPills(resultData?.applied_settings || {});

  return (
    <div className="result-diagnostics glass-card">
      <div className="result-diagnostics-title">
        <svg
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <path d="M12 20h9" />
          <path d="M12 4h9" />
          <path d="M4 9h16" />
          <path d="M4 15h16" />
          <path d="M8 4v16" />
        </svg>
        Forensic Diagnostics
      </div>
      <div className={`diagnostic-banner ${diagnosticState.variant}`.trim()}>
        {diagnosticState.message}
      </div>
      <div className="diagnostic-grid">
        <div className="diagnostic-item">
          <span className="diagnostic-label">Model</span>
          <span className="diagnostic-value">
            {resultData?.model_version || "—"}
          </span>
        </div>
        <div className="diagnostic-item">
          <span className="diagnostic-label">Needs review</span>
          <span className="diagnostic-value">
            {resultData ? (resultData.needs_review ? "Yes" : "No") : "—"}
          </span>
        </div>
        <div className="diagnostic-item">
          <span className="diagnostic-label">Raw predicted pixels</span>
          <span className="diagnostic-value">
            {resultData ? formatCount(resultData.raw_tampered_pixel_count) : "—"}
          </span>
        </div>
        <div className="diagnostic-item">
          <span className="diagnostic-label">Final predicted pixels</span>
          <span className="diagnostic-value">
            {resultData ? formatCount(resultData.tampered_pixel_count) : "—"}
          </span>
        </div>
        <div className="diagnostic-item">
          <span className="diagnostic-label">Area filter hit</span>
          <span className="diagnostic-value">
            {resultData ? (resultData.area_filter_triggered ? "Yes" : "No") : "—"}
          </span>
        </div>
        <div className="diagnostic-item">
          <span className="diagnostic-label">Mean probability</span>
          <span className="diagnostic-value">
            {resultData
              ? Number(resultData.confidence_mean_prob || 0).toFixed(4)
              : "—"}
          </span>
        </div>
      </div>
      <div className="settings-summary">
        <h4>Applied Settings</h4>
        <div className="settings-pills">
          {pills.length ? (
            pills.map((pill) => (
              <span key={pill} className="settings-pill">
                {pill}
              </span>
            ))
          ) : (
            <span className="settings-pill">Waiting for inference</span>
          )}
        </div>
      </div>
    </div>
  );
}

export function ResultsPanel({
  comparisonViews,
  errorMessage,
  isAnalyzing,
  resultData,
  visualTab,
  onVisualTabChange,
}) {
  const verdict = getVerdictState(resultData, isAnalyzing, errorMessage);
  const diagnosticState = getDiagnosticState(resultData, isAnalyzing, errorMessage);
  const maskDataUrl = resultData?.mask_base64
    ? `data:image/png;base64,${resultData.mask_base64}`
    : "";

  return (
    <div className="results-panel">
      <div className="result-header glass-card">
        <div className={`result-verdict ${verdict.className}`.trim()}>
          <div className="verdict-icon-wrap">
            <VerdictIcon variant={verdict.variant} />
          </div>
          <div>
            <div className="verdict-text">{verdict.label}</div>
            <div className="verdict-detail">{verdict.detail}</div>
          </div>
        </div>
        <div className="result-metrics">
          <div className="metric">
            <span className="metric-label">Confidence</span>
            <span className="metric-value">
              {resultData ? `${(resultData.confidence * 100).toFixed(1)}%` : "—"}
            </span>
          </div>
          <div className="metric">
            <span className="metric-label">Tampered</span>
            <span className="metric-value">
              {resultData ? formatRatioPercent(resultData.tampered_ratio) : "—"}
            </span>
          </div>
          <div className="metric">
            <span className="metric-label">Time</span>
            <span className="metric-value">
              {resultData ? `${Number(resultData.inference_time_ms).toFixed(0)}ms` : "—"}
            </span>
          </div>
        </div>
      </div>

      <div className="result-visuals glass-card">
        <div className="result-visuals-header">
          <div>
            <div className="result-visuals-title">
              <svg
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <rect x="3" y="3" width="18" height="18" rx="2" />
                <path d="M7 16l3-3 2 2 5-5" />
              </svg>
              Visual Comparison
            </div>
            <p>
              Compare the uploaded image, isolated detected region, and a red
              forensic overlay. Switch to heatmap when you want the raw
              localization mask.
            </p>
          </div>
          <div className="visual-tabs" role="tablist" aria-label="Result visualizations">
            <button
              className={`visual-tab ${visualTab === TAB_COMPARISON ? "is-active" : ""}`.trim()}
              type="button"
              role="tab"
              aria-selected={visualTab === TAB_COMPARISON}
              onClick={() => onVisualTabChange(TAB_COMPARISON)}
            >
              Comparison
            </button>
            <button
              className={`visual-tab ${visualTab !== TAB_COMPARISON ? "is-active" : ""}`.trim()}
              type="button"
              role="tab"
              aria-selected={visualTab !== TAB_COMPARISON}
              onClick={() => onVisualTabChange("heatmap")}
            >
              Heatmap
            </button>
          </div>
        </div>

        <div
          className={`visual-panel ${visualTab === TAB_COMPARISON ? "is-active" : ""}`.trim()}
          hidden={visualTab !== TAB_COMPARISON}
        >
          <div className="comparison-grid">
            <figure className="comparison-card">
              <div className="comparison-frame">
                {comparisonViews.originalSrc ? (
                  <img
                    src={comparisonViews.originalSrc}
                    alt="Original uploaded image"
                  />
                ) : null}
              </div>
              <figcaption>
                <h4>Original Image</h4>
                <p>Uploaded frame used as the source for the forensic pass.</p>
              </figcaption>
            </figure>

            <figure className="comparison-card comparison-card--blackout">
              <div className="comparison-frame comparison-frame--black">
                {comparisonViews.detectedRegionSrc ? (
                  <img
                    src={comparisonViews.detectedRegionSrc}
                    alt="Detected region on black background"
                  />
                ) : null}
                {!comparisonViews.hasMask ? (
                  <div className="comparison-empty">No detected region</div>
                ) : null}
              </div>
              <figcaption>
                <h4>Detected Region</h4>
                <p>
                  Only mask-positive pixels are preserved, everything else is
                  suppressed to black.
                </p>
              </figcaption>
            </figure>

            <figure className="comparison-card">
              <div className="comparison-frame">
                {comparisonViews.overlaySrc ? (
                  <img
                    src={comparisonViews.overlaySrc}
                    alt="Detected region overlaid in red on the original image"
                  />
                ) : null}
              </div>
              <figcaption>
                <h4>Red Overlay</h4>
                <p>
                  The localized region is tinted in red over the original image
                  for quick visual review.
                </p>
              </figcaption>
            </figure>
          </div>
        </div>

        <div
          className={`visual-panel ${visualTab !== TAB_COMPARISON ? "is-active" : ""}`.trim()}
          hidden={visualTab === TAB_COMPARISON}
        >
          <div className="result-mask">
            <div className="result-mask-title">
              <svg
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <rect x="3" y="3" width="18" height="18" rx="2" />
                <line x1="3" y1="9" x2="21" y2="9" />
                <line x1="9" y1="3" x2="9" y2="21" />
              </svg>
              Tamper Heatmap
            </div>
            {maskDataUrl ? (
              <img
                src={maskDataUrl}
                alt="Tamper localization heatmap showing highlighted tampered regions"
              />
            ) : (
              <div className="comparison-empty">No heatmap available yet</div>
            )}
          </div>
        </div>
      </div>

      <SummarySection resultData={resultData} />
      <DiagnosticsSection
        diagnosticState={diagnosticState}
        resultData={resultData}
      />
    </div>
  );
}
