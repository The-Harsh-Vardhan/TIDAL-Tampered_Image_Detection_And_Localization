import { track } from "@vercel/analytics";

const VALID_EVENT_NAMES = new Set([
  "image_upload_selected",
  "image_upload_cleared",
  "inference_requested",
  "inference_completed",
  "inference_failed",
  "forensic_control_changed",
  "results_visual_tab_changed",
  "nav_link_clicked",
  "external_link_clicked",
]);

function sanitizeValue(value) {
  if (value === null || value === undefined || value === "") {
    return undefined;
  }

  if (typeof value === "number") {
    return Number.isFinite(value) ? value : undefined;
  }

  if (typeof value === "boolean" || typeof value === "string") {
    return value;
  }

  return undefined;
}

function sanitizePayload(payload = {}) {
  return Object.fromEntries(
    Object.entries(payload)
      .map(([key, value]) => [key, sanitizeValue(value)])
      .filter(([, value]) => value !== undefined)
  );
}

export function trackTidalEvent(name, payload = {}) {
  if (!VALID_EVENT_NAMES.has(name)) {
    return;
  }

  try {
    track(name, sanitizePayload(payload));
  } catch {
    // Analytics must never break the forensic workflow.
  }
}

export function trackNavLink(destinationId, source) {
  trackTidalEvent("nav_link_clicked", {
    destination_id: destinationId,
    source,
  });
}

export function trackExternalLink(destinationId, source) {
  trackTidalEvent("external_link_clicked", {
    destination_id: destinationId,
    source,
  });
}
