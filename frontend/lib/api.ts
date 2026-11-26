import { mockOutput } from "@/data/mock";
import { adaptTechnical, type BackendTechnical } from "./technical-adapter";
import type { AgentId, AgentOutput, Indicator, Signal } from "./types";

const USE_MOCK = process.env.NEXT_PUBLIC_USE_MOCK === "true";
const LIVE_SENTIMENT = process.env.NEXT_PUBLIC_LIVE_SENTIMENT !== "false";
const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

const sleep = (ms: number, signal?: AbortSignal) =>
  new Promise<void>((resolve, reject) => {
    const id = setTimeout(resolve, ms);
    signal?.addEventListener("abort", () => {
      clearTimeout(id);
      reject(new DOMException("Aborted", "AbortError"));
    }, { once: true });
  });

// The technical agent is wired to the real backend. Set NEXT_PUBLIC_USE_MOCK_TECHNICAL=true to mock it.
const TECH_MOCK = process.env.NEXT_PUBLIC_USE_MOCK_TECHNICAL === "true";
// Raw price files end in mid-2026; set e.g. 2026-07-14 to analyse as of a past date (optional).
const TECH_AS_OF = process.env.NEXT_PUBLIC_TECH_AS_OF;

async function fetchTechnical(ticker: string, signal?: AbortSignal): Promise<AgentOutput> {
  const q = TECH_AS_OF ? `?as_of=${encodeURIComponent(TECH_AS_OF)}` : "";
  const res = await fetch(`${API_BASE}/agents/technical/${encodeURIComponent(ticker)}${q}`, { signal });
  if (!res.ok) throw new Error(`Request failed (${res.status})`);
  const body = (await res.json()) as BackendTechnical;
  if (body.status === "unavailable" || body.status === "error") {
    throw new Error(body.warnings?.[0] ?? `Technical agent ${body.status}`);
  }
  return adaptTechnical(body);
}

type BackendEvidence = {
  claim: string;
  source_name: string;
  source_url?: string | null;
  source_type?: string | null;
  excerpt?: string | null;
};

type BackendAgentOutput = {
  agent: string;
  ticker: string;
  as_of: string;
  status: "ok" | "degraded" | "unavailable" | "error";
  stance: Signal;
  score: number;
  confidence: number;
  summary?: string;
  key_findings?: string[];
  risks?: string[];
  evidence?: BackendEvidence[];
  metrics?: Record<string, unknown>;
  warnings?: string[];
};

function toIndicators(raw: BackendAgentOutput): Indicator[] {
  const m = raw.metrics ?? {};
  const items: Array<[string, unknown]> = [
    ["Score", raw.score.toFixed(4)],
    ["Confidence", `${Math.round(raw.confidence * 100)}%`],
    ["Articles in window", m.articles_in_window],
    ["Effective articles", m.effective_articles],
    ["Window", typeof m.window_days === "number" ? `${m.window_days} days` : undefined],
    ["Model", m.model_version],
    ["Scored by", Array.isArray(m.scored_with) ? m.scored_with.join(", ").replace("ProsusAI/finbert+lora@T=1.0", "FinBERT + LoRA (fine-tuned, ft1)") : undefined],
    ["Parent", m.parent],
  ];
  return items
    .filter(([, value]) => value !== undefined && value !== null && value !== "")
    .map(([label, value]) => ({ label, value: String(value) }));
}

function adaptSentiment(raw: BackendAgentOutput): AgentOutput {
  if (raw.status !== "ok" && raw.status !== "degraded") {
    throw new Error((raw.warnings ?? []).join("; ") || `Sentiment agent status: ${raw.status}`);
  }
  return {
    agent: "sentiment",
    ticker: raw.ticker,
    signal: raw.stance,
    score: raw.score,
    // FinBERT (LoRA fine-tuned) model confidence, shown as "Confidence" like the other agents.
    confidence: raw.confidence,
    summary: raw.summary ?? "No sentiment summary returned.",
    rationale: (raw.key_findings ?? []).join(" ") || raw.summary || "No additional rationale returned.",
    indicators: toIndicators(raw),
    evidence: (raw.evidence ?? []).map((ev) => ({
      title: ev.claim,
      source: ev.source_name,
      ...(ev.source_url ? { url: ev.source_url } : {}),
    })),
    generated_at: raw.as_of,
  };
}

// Macro adapter (from Mariam's 52fa21e, lost in merge 4341b86). Needs the macro service running
// (`macro-agent serve --port 8010`); otherwise the card shows the backend's "not reachable" warning.
function adaptMacro(raw: BackendAgentOutput): AgentOutput {
  if (raw.status !== "ok" && raw.status !== "degraded") {
    throw new Error(raw.warnings?.[0] ?? `Macro agent ${raw.status}`);
  }
  const rationale = [...(raw.key_findings ?? []), ...(raw.risks ?? []).map((r) => `Risk: ${r}`)].join("\n");
  const indicators = (raw.metrics as { indicators?: Indicator[] } | undefined)?.indicators;
  return {
    agent: "macro",
    ticker: raw.ticker,
    signal: raw.stance,
    score: raw.score,
    confidence: raw.confidence,
    summary: raw.summary,
    rationale: rationale || undefined,
    indicators,
    evidence: (raw.evidence ?? []).map((e) => ({
      title: e.claim,
      source: e.source_name,
      ...(e.source_url ? { url: e.source_url } : {}),
    })),
    generated_at: raw.as_of,
  };
}

const fmtPct = (v: unknown) => (typeof v === "number" ? `${(v * 100).toFixed(1)}%` : undefined);
const fmtRs = (v: unknown) => (typeof v === "number" ? `Rs ${v.toFixed(2)}` : undefined);
const fmtX = (v: unknown) => (typeof v === "number" ? `${v.toFixed(2)}x` : undefined);

// Fundamentals agent (official PSX data + trained valuation model). The BUY / HOLD / AVOID call
// is in metrics.call; stance carries the same decision.
function adaptFundamentals(raw: BackendAgentOutput): AgentOutput {
  if (raw.status !== "ok" && raw.status !== "degraded") {
    throw new Error((raw.warnings ?? []).join("; ") || raw.summary || `Fundamentals agent status: ${raw.status}`);
  }
  const m = (raw.metrics ?? {}) as Record<string, any>; // eslint-disable-line @typescript-eslint/no-explicit-any
  const ratio = (name: string) => (m.ratios ?? []).find((r: { name: string }) => r.name === name);
  const v = m.valuation ?? {};
  const items: Array<[string, string | undefined]> = [
    ["Call", m.call],
    ["Fair value / share", fmtRs(v.implied_value_per_share ?? m.peer_baseline?.implied_value_per_share)],
    ["Price", fmtRs(m.market?.price)],
    ["Valuation gap", fmtPct(m.gap)],
    ["P/E (TTM)", fmtX(ratio("pe")?.value)],
    ["P/B", fmtX(ratio("pb")?.value)],
    ["ROE", fmtPct(ratio("roe")?.value)],
    ["Peers", m.peers ? `${m.peers.peers.length} (${m.peers.strength})` : undefined],
    ["Valued by", m.gap_source],
  ];
  return {
    agent: "fundamentals",
    ticker: raw.ticker,
    signal: raw.stance,
    call: m.call,
    score: raw.score,
    confidence: raw.confidence,
    summary: raw.summary,
    rationale: (raw.key_findings ?? []).join("\n") || raw.summary,
    indicators: items.filter(([, val]) => val !== undefined).map(([label, value]) => ({ label, value: value as string })),
    evidence: (raw.evidence ?? []).map((ev) => ({
      title: ev.claim,
      source: ev.source_name,
      ...(ev.source_url ? { url: ev.source_url } : {}),
    })),
    generated_at: raw.as_of,
    risks: raw.risks,
    warnings: raw.warnings,
  };
}

export async function fetchFundamentals(ticker: string, signal?: AbortSignal): Promise<AgentOutput> {
  const res = await fetch(`${API_BASE}/agents/fundamentals/${encodeURIComponent(ticker)}`, { signal });
  if (!res.ok) throw new Error(`Fundamentals API request failed (${res.status})`);
  return adaptFundamentals((await res.json()) as BackendAgentOutput);
}

export async function fetchAgent(agent: AgentId, ticker: string, signal?: AbortSignal): Promise<AgentOutput> {
  // The technical agent is wired to the real backend.
  if (agent === "technical" && !TECH_MOCK) return fetchTechnical(ticker, signal);
  if (agent === "fundamentals" && !USE_MOCK) return fetchFundamentals(ticker, signal);
  if (agent === "macro" && !USE_MOCK) {
    const res = await fetch(`${API_BASE}/agents/macro/${encodeURIComponent(ticker)}`, { signal });
    if (!res.ok) throw new Error(`Macro API request failed (${res.status})`);
    return adaptMacro((await res.json()) as BackendAgentOutput);
  }
  // The sentiment agent is live; other specialist agents remain on demo data until implemented.
  if (USE_MOCK || agent !== "sentiment" || !LIVE_SENTIMENT) {
    await sleep(800 + Math.random() * 1400, signal);
    // Demo of the error state. Delete this line when real wiring starts.
    if (agent === "macro" && ticker === "SEARL") throw new Error("Macro agent timed out (mock error)");
    return mockOutput(agent, ticker);
  }

  const res = await fetch(`${API_BASE}/agents/sentiment/${encodeURIComponent(ticker)}`, { signal });
  if (!res.ok) {
    const detail = await res.text().catch(() => "");
    throw new Error(`Sentiment API request failed (${res.status})${detail ? `: ${detail}` : ""}`);
  }
  return adaptSentiment((await res.json()) as BackendAgentOutput);
}
