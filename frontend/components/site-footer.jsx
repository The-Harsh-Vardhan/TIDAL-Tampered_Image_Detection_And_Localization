export function SiteFooter({ onExternalLinkClick }) {
  return (
    <footer className="footer">
      <div className="footer-links">
        <a
          href="https://github.com/The-Harsh-Vardhan/TIDAL-Tampered_Image_Detection_And_Localization"
          target="_blank"
          rel="noopener noreferrer"
          onClick={() => onExternalLinkClick("github_repo", "footer")}
        >
          <svg viewBox="0 0 24 24" fill="currentColor">
            <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z" />
          </svg>
          GitHub
        </a>
        <a
          href="https://wandb.ai/tampered-image-detection-and-localization/Tampered%20Image%20Detection%20%26%20Localization"
          target="_blank"
          rel="noopener noreferrer"
          onClick={() => onExternalLinkClick("wandb_dashboard", "footer")}
        >
          <svg
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <line x1="18" y1="20" x2="18" y2="10" />
            <line x1="12" y1="20" x2="12" y2="4" />
            <line x1="6" y1="20" x2="6" y2="14" />
          </svg>
          W&amp;B Dashboard
        </a>
        <a
          href="https://the-harsh-vardhan-tidal-api.hf.space/docs"
          target="_blank"
          rel="noopener noreferrer"
          onClick={() => onExternalLinkClick("api_docs", "footer")}
        >
          <svg
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z" />
            <polyline points="14 2 14 8 20 8" />
            <line x1="16" y1="13" x2="8" y2="13" />
            <line x1="16" y1="17" x2="8" y2="17" />
          </svg>
          API Docs
        </a>
      </div>
      <div className="footer-meta">
        TIDAL v1.0.0 <span>·</span> UNet + ResNet-34 + Gray MQ-ELA + CBAM{" "}
        <span>·</span> © 2025 Harsh Vardhan — IIIT Nagpur
      </div>
    </footer>
  );
}
