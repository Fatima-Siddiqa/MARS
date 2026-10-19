import { mockOutput } from "@/data/mock";
import type { AgentId, AgentOutput } from "./types";

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
    ["Data quality", `${Math.round(raw.confidence * 100)}%`],
    ["Articles in window", m.articles_in_window],
    ["Effective articles", m.effective_articles],
    ["Window", typeof m.window_days === "number" ? `${m.window_days} days` : undefined],
    ["Model", m.model_version],
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
    // For sentiment, this shared UI field carries the MARS quality index, not
    // a calibrated probability. The UI labels it "Data quality" for this agent.
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

// Shape the backend actually returns (the team's agent contract).
type BackendOutput = {
  agent: AgentId;
  ticker: string;
  as_of?: string;
  status?: "ok" | "degraded" | "unavailable";
  stance: Signal;
  score: number;
  confidence: number;
  summary?: string;
  key_findings?: string[];
  risks?: string[];
  evidence?: {
    claim: string;
    source_name?: string;
    source_url?: string;
    retrieved_at?: string;
  }[];
  metrics?: { indicators?: { label: string; value: string }[] };
  warnings?: string[];
};

function toAgentOutput(raw: BackendOutput): AgentOutput {
  const rationale = [
    ...(raw.key_findings ?? []),
    ...(raw.risks ?? []).map((r) => `Risk: ${r}`),
  ].join("\n");
  return {
    agent: raw.agent,
    ticker: raw.ticker,
    signal: raw.stance,
    score: raw.score,
    confidence: raw.confidence,
    summary: raw.summary,
    rationale: rationale || undefined,
    indicators: raw.metrics?.indicators,
    evidence: (raw.evidence ?? []).map((e) => ({
      title: e.claim,
      url: e.source_url,
      source: e.source_name,
      date: e.retrieved_at?.slice(0, 10),
    })),
    generated_at: raw.as_of,
  };
}

export async function fetchAgent(agent: AgentId, ticker: string, signal?: AbortSignal): Promise<AgentOutput> {
  // The technical agent is wired to the real backend.
  if (agent === "technical" && !TECH_MOCK) return fetchTechnical(ticker, signal);
  // The sentiment agent is live; other specialist agents remain on demo data until implemented.
  if (USE_MOCK || agent !== "sentiment" || !LIVE_SENTIMENT) {
    await sleep(800 + Math.random() * 1400, signal);
    // Demo of the error state. Delete this line when real wiring starts.
    if (agent === "macro" && ticker === "SEARL") throw new Error("Macro agent timed out (mock error)");
    return mockOutput(agent, ticker);
  }

  // Confirm the exact URL/params with Fatima's FastAPI skeleton.
  const res = await fetch(`${API_BASE}/agents/${agent}?ticker=${encodeURIComponent(ticker)}`, { signal });
  if (!res.ok) throw new Error(`Request failed (${res.status})`);
  return (await res.json()) as AgentOutput;
}