import { CountUp } from "@/components/count-up";
import { FadeIn } from "@/components/fade-in";

export function KeyResultsSection() {
  return (
    <section className="section" id="results">
      <div className="container">
        <FadeIn className="section-header">
          <h2 className="section-title">Key Results</h2>
          <p className="section-subtitle">
            Best run vR.P.19 — Multi-Quality RGB ELA, 25 epochs
          </p>
        </FadeIn>
        <div className="metrics-grid">
          <FadeIn className="metric-card glass-card" delayMs={0}>
            <div className="metric-card-value">
              <CountUp target={0.7965} decimals={4} />
            </div>
            <div className="metric-card-label">Pixel F1</div>
            <p>
              Best result on CASIA 2.0 with Multi-Q RGB ELA input
              representation
            </p>
          </FadeIn>
          <FadeIn className="metric-card glass-card" delayMs={80}>
            <div className="metric-card-value">+34.19pp</div>
            <div className="metric-card-label">ELA Impact</div>
            <p>
              More improvement than all architectural changes combined
            </p>
          </FadeIn>
          <FadeIn className="metric-card glass-card" delayMs={160}>
            <div className="metric-card-value">
              <CountUp target={0.9665} decimals={4} />
            </div>
            <div className="metric-card-label">Pixel AUC</div>
            <p>Area Under ROC Curve across all tampered image pairs</p>
          </FadeIn>
        </div>
      </div>
    </section>
  );
}
