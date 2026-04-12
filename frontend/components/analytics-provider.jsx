"use client";

import { Analytics } from "@vercel/analytics/next";

function sanitizePageView(event) {
  if (!event?.url) {
    return event;
  }

  try {
    const url = new URL(event.url);

    if (url.hostname === "localhost" || url.hostname === "127.0.0.1") {
      return null;
    }

    url.hash = "";
    url.search = "";

    return {
      ...event,
      url: url.toString(),
    };
  } catch {
    return event;
  }
}

export function AnalyticsProvider() {
  return <Analytics beforeSend={sanitizePageView} />;
}
