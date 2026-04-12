"use client";

import { useState } from "react";

const NAV_LINKS = [
  { href: "#pipeline", label: "Pipeline", destinationId: "pipeline" },
  { href: "#demo", label: "Demo", destinationId: "demo" },
  { href: "#results", label: "Results", destinationId: "results" },
];

export function SiteNav({ onExternalLinkClick, onNavLinkClick }) {
  const [isOpen, setIsOpen] = useState(false);

  function handleNavClick(destinationId) {
    onNavLinkClick(destinationId, "nav");
    setIsOpen(false);
  }

  return (
    <nav className="nav" role="navigation" aria-label="Main navigation">
      <div className="nav-brand">
        <svg
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
          <circle cx="12" cy="10" r="3" />
        </svg>
        <span className="nav-title">TIDAL</span>
      </div>
      <div className={`nav-links ${isOpen ? "open" : ""}`.trim()}>
        {NAV_LINKS.map((link) => (
          <a
            key={link.destinationId}
            href={link.href}
            onClick={() => handleNavClick(link.destinationId)}
          >
            {link.label}
          </a>
        ))}
        <a
          href="https://github.com/The-Harsh-Vardhan/TIDAL-Tampered_Image_Detection_And_Localization"
          target="_blank"
          rel="noopener noreferrer"
          onClick={() => {
            onExternalLinkClick("github_repo", "nav");
            setIsOpen(false);
          }}
        >
          GitHub
        </a>
      </div>
      <button
        className="nav-toggle"
        type="button"
        aria-label="Toggle menu"
        onClick={() => setIsOpen((current) => !current)}
      >
        <svg
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
        >
          <line x1="3" y1="6" x2="21" y2="6" />
          <line x1="3" y1="12" x2="21" y2="12" />
          <line x1="3" y1="18" x2="21" y2="18" />
        </svg>
      </button>
    </nav>
  );
}
