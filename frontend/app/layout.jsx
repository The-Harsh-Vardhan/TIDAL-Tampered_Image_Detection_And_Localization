import { Inter, Outfit } from "next/font/google";

import "@/app/globals.css";
import { AnalyticsProvider } from "@/components/analytics-provider";

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
  metadataBase: new URL("https://tidal-orpin.vercel.app"),
  title: "TIDAL — Tampered Image Detection & Localization",
  description:
    "Detect and localize tampered regions in images using grayscale multi-quality ELA, CBAM-enhanced UNet segmentation, and notebook-style forensic controls.",
  openGraph: {
    title: "TIDAL — Tampered Image Detection & Localization",
    description:
      "Deep learning forensic analysis using grayscale multi-quality ELA with CBAM and analyst-tunable forensic controls.",
    type: "website",
    url: "https://tidal-orpin.vercel.app",
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
