// Full analysis from GET /agents/fundamentals/{ticker}/analysis (company page panel).

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export type FundRatio = {
  name: string;
  label: string;
  value: number | null;
  formula: string;
  family: string;
  unit: "x" | "pct";
  reason: string | null;
};

export type PeerStat = { ratio: string; target: number | null; peer_median: number | null; n: number; percentile: number | null };

export type Contribution = { feature: string; label: string; contribution_log: number; effect_pct: number };

export type FundAnalysis = {
  ticker: string;
  as_of: string;
  call: "BUY" | "HOLD" | "AVOID";
  score: number;
  confidence: number;
  gap: number | null;
  gap_source: string;
  threshold: number;
  calibration: { supported: boolean | null; threshold: number; reason: string | null; spearman_ic?: number | null; samples?: number | null };
  company: { type: string; sector: string; sector_code: string; coverage: string | null };
  market: { price: number | null; price_date: string | null; shares: number | null; shares_source: string | null; market_cap: number | null };
  statements: { balance_date: string | null };
  ratios: FundRatio[];
  ratio_coverage: number;
  peers: { peers: string[]; strength: string; stats: Record<string, PeerStat> };
  peer_baseline: { method: string; implied_value_per_share: number | null; gap: number | null; reason: string | null };
  valuation: {
    method: string;
    model_version: string | null;
    implied_value_per_share: number | null;
    range_per_share: [number, number] | null;
    gap: number | null;
    contributions: Contribution[];
    test_metrics: { n?: number; median_abs_pct_error?: number; r2?: number };
    reason: string | null;
  };
  warnings: string[];
};

export async function fetchFundamentalsAnalysis(ticker: string, signal?: AbortSignal): Promise<FundAnalysis> {
  const res = await fetch(`${API_BASE}/agents/fundamentals/${encodeURIComponent(ticker)}/analysis`, { signal });
  if (res.status === 404) throw new Error(`No official PSX fundamentals collected for ${ticker}.`);
  if (!res.ok) throw new Error(`Fundamentals analysis request failed (${res.status})`);
  return (await res.json()) as FundAnalysis;
}

export const fmtRatio = (r: { value: number | null; unit: string }, v: number | null = r.value) =>
  v === null || v === undefined ? "—" : r.unit === "pct" ? `${(v * 100).toFixed(1)}%` : `${v.toFixed(2)}x`;

export const fmtRs = (v: number | null | undefined) =>
  v === null || v === undefined ? "—" : `Rs ${v.toLocaleString("en-PK", { maximumFractionDigits: 2 })}`;
