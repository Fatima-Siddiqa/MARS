"use client";

import CompanyTechnicalView from "@/components/company/CompanyTechnicalView";
import FundamentalsPanel from "@/components/fundamentals/FundamentalsPanel";
import { useTicker } from "@/lib/ticker-context";

export default function Company() {
  const { ticker } = useTicker();
  // key restarts each view (and its loading state) whenever the selected ticker changes
  return (
    <div className="space-y-6">
      <CompanyTechnicalView key={`tech-${ticker}`} ticker={ticker} />
      <FundamentalsPanel key={`fund-${ticker}`} ticker={ticker} />
    </div>
  );
}
