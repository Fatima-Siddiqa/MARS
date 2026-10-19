import type { AgentOutput, ChartPoint, Evidence, Indicator, Signal } from "./types";

// Shape returned by GET /agents/technical/{ticker} (backend AgentOutput + metrics).
export type BackendTechnical = {
  ticker: string;
  as_of?: string | null;
  status: "ok" | "degraded" | "unavailable" | "error";
  stance: Signal;
  score: number;
  confidence: number;
  summary?: string;
  key_findings?: string[];
  risks?: string[];
  warnings?: string[];
  evidence?: { claim: string; source_name: string; source_url?: string | null; excerpt?: string; retrieved_at?: string }[];
  metrics?: {
    indicators?: Record<string, number | null>;
    families?: Record<string, number | null>;
    chart?: ChartPoint[];
  };
};

const num = (v: number | null | undefined, d = 1) => (v == null ? "n/a" : v.toFixed(d));
const pctv = (v: number | null | undefined) => (v == null ? "n/a" : `${v >= 0 ? "+" : ""}${(v * 100).toFixed(1)}%`);

function rsiZone(r: number | null | undefined): string {
  if (r == null) return "n/a";
  return r > 70 ? "Overbought" : r < 30 ? "Oversold" : "Neutral";
}

export function adaptTechnical(b: BackendTechnical): AgentOutput {
  const ind = b.metrics?.indicators ?? {};
  const indicators: Indicator[] = [
    { label: "RSI (14)", value: `${num(ind.rsi14)} (${rsiZone(ind.rsi14)})` },
    { label: "MACD histogram", value: `${num(ind.macd_hist, 2)} (${(ind.macd_hist ?? 0) >= 0 ? "Bullish" : "Bearish"})` },
    { label: "ADX (14)", value: `${num(ind.adx14)} (${(ind.adx14 ?? 0) >= 20 ? "Trending" : "Sideways"})` },
    { label: "SMA 50", value: `PKR ${num(ind.sma50, 2)}` },
    { label: "6-month return", value: pctv(ind.ret_6m) },
    { label: "3M vs KSE-100", value: pctv(ind.rs_3m) },
  ];

  const evidence: Evidence[] = (b.evidence ?? []).map((e) => ({
    title: e.claim.charAt(0).toUpperCase() + e.claim.slice(1),
    url: e.source_url ?? undefined,
    source: [e.source_name, e.excerpt].filter(Boolean).join(" · "),
    date: b.as_of?.slice(0, 10),
  }));

  const sentences = (b.summary ?? "").split(/(?<=\.)\s+/);
  const rationale = [
    ...(b.key_findings ?? []),
    ...(b.risks?.length ? [`Risks: ${b.risks.join(" ")}`] : []),
  ].join(" ");

  return {
    agent: "technical",
    ticker: b.ticker,
    signal: b.stance,
    score: b.score,
    confidence: b.confidence,
    summary: sentences.slice(0, 2).join(" ") || undefined,
    rationale: rationale || undefined,
    indicators,
    evidence,
    as_of: b.as_of?.slice(0, 10),
    chart: b.metrics?.chart,
    families: b.metrics?.families,
    risks: b.risks,
    warnings: b.warnings,
  };
}
