import { ChartPie, DollarSign, Globe, Newspaper, ShieldAlert, TrendingUp, type LucideIcon } from "lucide-react";
import type { AgentId } from "@/lib/types";

export type AgentConfig = {
  id: AgentId;
  label: string;
  purpose: string;
  /** Weight in the regime-aware synthesis (static for now). */
  weight: number;
  icon: LucideIcon;
  /** Flip to true when that agent's backend is ready. */
  enabled: boolean;
};

export const AGENTS: AgentConfig[] = [
  {
    id: "technical",
    label: "Historical Price & Technical",
    purpose: "Analyzes historical price movements, cyclical patterns, momentum oscillators, and key support/resistance zones.",
    weight: 25,
    icon: TrendingUp,
    enabled: true,
  },
  {
    id: "sentiment",
    label: "Sentiment & News",
    purpose: "Monitors financial press, regulatory announcements, corporate filings, and retail investor discourse across Pakistani financial media.",
    weight: 15,
    icon: Newspaper,
    enabled: true,
  },
  {
    id: "macro",
    label: "Macroeconomic Indicators",
    purpose: "Evaluates State Bank of Pakistan policy rate trajectories, headline/core CPI inflation, sovereign bond yields, and fiscal deficit trends.",
    weight: 15,
    icon: Globe,
    enabled: true,
  },
  {
    id: "fx",
    label: "FX Impact",
    purpose: "Measures how sensitive the stock is to PKR/USD moves, and whether the company is an exporter, importer or neutral.",
    weight: 10,
    icon: DollarSign,
    enabled: true,
  },
  {
    id: "fundamentals",
    label: "Sector & Fundamental Comparison",
    purpose: "Ranks valuation and profitability against sector peers using audited financial statements and ratios.",
    weight: 25,
    icon: ChartPie,
    enabled: true,
  },
  {
    id: "forensics",
    label: "Financial Forensics",
    purpose: "Screens financial statements for accounting red flags such as accrual anomalies, receivable build-ups and bank-specific provisioning risks.",
    weight: 10,
    icon: ShieldAlert,
    enabled: true,
  },
];

export const ENABLED_IDS: AgentId[] = AGENTS.filter((a) => a.enabled).map((a) => a.id);
