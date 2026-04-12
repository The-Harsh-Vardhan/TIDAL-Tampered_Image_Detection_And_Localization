import { FadeIn } from "@/components/fade-in";

const PIPELINE_STEPS = [
  {
    step: "Step 01",
    title: "Error Level Analysis",
    description:
      "Re-saves image at Q=75, 85, 95 and stacks grayscale ELA maps into a 3-channel forensic signature.",
    icon: (
      <svg
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <circle cx="11" cy="11" r="8" />
        <line x1="21" y1="21" x2="16.65" y2="16.65" />
      </svg>
    ),
  },
  {
    step: "Step 02",
    title: "Neural Segmentation",
    description:
      "UNet with a ResNet-34 encoder and CBAM-enhanced decoder blocks processes the grayscale MQ-ELA stack.",
    icon: (
      <svg
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <rect x="2" y="2" width="6" height="6" rx="1" />
        <rect x="16" y="2" width="6" height="6" rx="1" />
        <rect x="9" y="16" width="6" height="6" rx="1" />
        <path d="M5 8v3a1 1 0 001 1h12a1 1 0 001-1V8" />
        <line x1="12" y1="12" x2="12" y2="16" />
      </svg>
    ),
  },
  {
    step: "Step 03",
    title: "Tamper Localization",
    description:
      "Generates binary mask highlighting tampered regions with per-pixel confidence scoring.",
    icon: (
      <svg
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <circle cx="12" cy="12" r="10" />
        <line x1="22" y1="12" x2="18" y2="12" />
        <line x1="6" y1="12" x2="2" y2="12" />
        <line x1="12" y1="6" x2="12" y2="2" />
        <line x1="12" y1="22" x2="12" y2="18" />
      </svg>
    ),
  },
];

export function PipelineSection() {
  return (
    <section className="section" id="pipeline">
      <div className="container">
        <FadeIn className="section-header">
          <h2 className="section-title">How the Pipeline Works</h2>
          <p className="section-subtitle">
            Three-stage forensic analysis from raw image to tamper localization
          </p>
        </FadeIn>
        <div className="pipeline-steps">
          {PIPELINE_STEPS.map((item, index) => (
            <FadeIn
              key={item.step}
              className="pipeline-card glass-card"
              delayMs={index * 80}
            >
              <div className="pipeline-icon">{item.icon}</div>
              <div className="pipeline-step-num">{item.step}</div>
              <h3>{item.title}</h3>
              <p>{item.description}</p>
            </FadeIn>
          ))}
        </div>
      </div>
    </section>
  );
}
