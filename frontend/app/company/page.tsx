"use client";

import CompanyTechnicalView from "@/components/company/CompanyTechnicalView";
import { useTicker } from "@/lib/ticker-context";

export default function Company() {
  const { ticker } = useTicker();
  // key restarts the view (and its loading state) whenever the selected ticker changes
  return <CompanyTechnicalView key={ticker} ticker={ticker} />;
}
