import { CountUp } from "@/components/count-up";
import { FadeIn } from "@/components/fade-in";

export function KeyResultsSection() {
  return (
    <section className="section" id="results">
      <div className="container">
        <FadeIn className="section-header">
          <h2 className="section-title">Key Results</h2>
          <p className="section-subtitle">
            Live inference runs vR.P.30.1 — gray multi-quality ELA with CBAM
            decoder attention
          </p>
        </FadeIn>
        <div className="metrics-grid">
          <FadeIn className="metric-card glass-card" delayMs={0}>
            <div className="metric-card-value">
              <CountUp target={0.7965} decimals={4} />
            </div>
            <div className="metric-card-label">Pixel F1 Benchmark</div>
            <p>
              Project research benchmark retained for context while the public
              console serves the vR.P.30.1 inference bundle
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
            <div className="metric-card-label">Pixel AUC Benchmark</div>
            <p>Reference localization AUC across tampered image pairs</p>
          </FadeIn>
        </div>
      </div>
    </section>
  );
}
