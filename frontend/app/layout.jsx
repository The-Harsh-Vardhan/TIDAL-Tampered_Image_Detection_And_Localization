import { Inter, Outfit } from "next/font/google";

import "@/app/globals.css";
import { AnalyticsProvider } from "@/components/analytics-provider";

const DEFAULT_SITE_URL = "https://tidal-orpin.vercel.app";

function resolveSiteUrl() {
  const configuredUrl = process.env.NEXT_PUBLIC_SITE_URL?.trim();
  if (configuredUrl) {
    return configuredUrl;
  }

  const vercelHost =
    process.env.VERCEL_PROJECT_PRODUCTION_URL || process.env.VERCEL_URL;
  if (vercelHost) {
    return `https://${vercelHost}`;
  }

  return DEFAULT_SITE_URL;
}

const siteUrl = resolveSiteUrl();

const inter = Inter({
  subsets: ["latin"],
  display: "swap",
  variable: "--font-inter",
});

const outfit = Outfit({
  subsets: ["latin"],
  display: "swap",
  variable: "--font-outfit",
});

export const metadata = {
  metadataBase: new URL(siteUrl),
  title: "TIDAL — Tampered Image Detection & Localization",
  description:
    "Detect and localize tampered regions in images using grayscale multi-quality ELA, CBAM-enhanced UNet segmentation, and notebook-style forensic controls.",
  openGraph: {
    title: "TIDAL — Tampered Image Detection & Localization",
    description:
      "Deep learning forensic analysis using grayscale multi-quality ELA with CBAM and analyst-tunable forensic controls.",
    type: "website",
    url: siteUrl,
  },
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body className={`${inter.variable} ${outfit.variable}`}>
        {children}
        <AnalyticsProvider />
      </body>
    </html>
  );
}
