import { mockOutput } from "@/data/mock";
import type { AgentId, AgentOutput, Signal } from "./types";

// NEXT_PUBLIC_ variables are baked in at BUILD time (matters for Docker).
const USE_MOCK = process.env.NEXT_PUBLIC_USE_MOCK !== "false";
const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

const sleep = (ms: number, signal?: AbortSignal) =>
  new Promise<void>((resolve, reject) => {
    const id = setTimeout(resolve, ms);
    signal?.addEventListener("abort", () => {
      clearTimeout(id);
      reject(new DOMException("Aborted", "AbortError"));
    });
  });

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
  if (USE_MOCK) {
    await sleep(800 + Math.random() * 1400, signal); // different delays so cards load independently
    // Demo of the error state. Delete this line when real wiring starts.
    if (agent === "macro" && ticker === "SEARL") throw new Error("Macro agent timed out (mock error)");
    return mockOutput(agent, ticker);
  }

  const res = await fetch(`${API_BASE}/agents/${agent}/${encodeURIComponent(ticker)}`, { signal });
  if (!res.ok) throw new Error(`Request failed (${res.status})`);
  const raw = (await res.json()) as BackendOutput;
  if (raw.status === "unavailable") {
    throw new Error(raw.warnings?.[0] ?? "Agent unavailable");
  }
  return toAgentOutput(raw);
}