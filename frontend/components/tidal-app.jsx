"use client";

import { DemoWorkspace } from "@/components/demo-workspace";
import { HeroSection } from "@/components/hero-section";
import { KeyResultsSection } from "@/components/key-results-section";
import { PipelineSection } from "@/components/pipeline-section";
import { SiteFooter } from "@/components/site-footer";
import { SiteNav } from "@/components/site-nav";
import { useTidalForensics } from "@/hooks/use-tidal-forensics";
import { trackExternalLink, trackNavLink } from "@/lib/analytics";

export function TidalApp() {
  const forensic = useTidalForensics();

  return (
    <>
      <SiteNav
        onExternalLinkClick={trackExternalLink}
        onNavLinkClick={trackNavLink}
      />
      <HeroSection onNavLinkClick={trackNavLink} />
      <PipelineSection />
      <DemoWorkspace
        analyticsMode={forensic.analyticsMode}
        comparisonViews={forensic.comparisonViews}
        errorMessage={forensic.errorMessage}
        healthStatus={forensic.healthStatus}
        isAnalyzing={forensic.isAnalyzing}
        isDemoLoading={forensic.isDemoLoading}
        previewDataUrl={forensic.previewDataUrl}
        resultData={forensic.resultData}
        resultsVisible={forensic.resultsVisible}
        settings={forensic.settings}
        visualTab={forensic.visualTab}
        onAnalyticsModeChange={forensic.updateAnalyticsMode}
        onClearUpload={forensic.clearUpload}
        onCommitSetting={forensic.commitSetting}
        onRunDemo={forensic.runDemo}
        onSelectFile={forensic.selectFile}
        onUpdateSetting={forensic.updateSetting}
        onVisualTabChange={forensic.updateVisualTab}
      />
      <KeyResultsSection />
      <SiteFooter onExternalLinkClick={trackExternalLink} />
    </>
  );
}
