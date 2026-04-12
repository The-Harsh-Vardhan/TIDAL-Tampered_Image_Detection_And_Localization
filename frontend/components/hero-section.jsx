import { CountUp } from "@/components/count-up";

export function HeroSection({ onNavLinkClick }) {
  return (
    <section className="hero" id="hero">
      <div className="hero-grid" />
      <div className="hero-blob hero-blob--1" />
      <div className="hero-blob hero-blob--2" />
      <div className="hero-blob hero-blob--3" />

      <div className="hero-content">
        <div className="hero-badge">
          <svg
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2" />
          </svg>
          Pixel F1 = 0.7965 on CASIA 2.0
        </div>
        <h1>
          Detect Image
          <br />
          Tampering with AI
        </h1>
        <p className="hero-subtitle">
          Deep learning forensic analysis using grayscale multi-quality Error
          Level Analysis, CBAM-enhanced UNet segmentation, and analyst-tunable
          forensic thresholds.
        </p>
        <div className="hero-actions">
          <a
            href="#demo"
            className="btn btn-primary"
            onClick={() => onNavLinkClick("demo", "hero_primary")}
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
            Try Demo
          </a>
          <a
            href="#pipeline"
            className="btn btn-secondary"
            onClick={() => onNavLinkClick("pipeline", "hero_secondary")}
          >
            <svg
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <circle cx="12" cy="12" r="10" />
              <line x1="12" y1="16" x2="12" y2="12" />
              <line x1="12" y1="8" x2="12.01" y2="8" />
            </svg>
            How It Works
          </a>
        </div>
        <div className="hero-stats">
          <div className="stat">
            <span className="stat-value">
              <CountUp target={60} suffix="+" />
            </span>
            <span className="stat-label">Experiments</span>
          </div>
          <div className="stat">
            <span className="stat-value">3ch</span>
            <span className="stat-label">Gray MQ-ELA</span>
          </div>
          <div className="stat">
            <span className="stat-value">
              <CountUp target={12614} suffix="+" />
            </span>
            <span className="stat-label">Training Images</span>
          </div>
        </div>
      </div>
    </section>
  );
}
