import { mockOutput } from "@/data/mock";
import { adaptTechnical, type BackendTechnical } from "./technical-adapter";
import type { AgentId, AgentOutput } from "./types";

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

export async function fetchAgent(agent: AgentId, ticker: string, signal?: AbortSignal): Promise<AgentOutput> {
  if (agent === "technical" && !TECH_MOCK) return fetchTechnical(ticker, signal);

  if (USE_MOCK) {
    await sleep(800 + Math.random() * 1400, signal); // different delays so cards load independently
    // Demo of the error state. Delete this line when real wiring starts.
    if (agent === "macro" && ticker === "SEARL") throw new Error("Macro agent timed out (mock error)");
    return mockOutput(agent, ticker);
  }

  // Confirm the exact URL/params with Fatima's FastAPI skeleton.
  const res = await fetch(`${API_BASE}/agents/${agent}?ticker=${encodeURIComponent(ticker)}`, { signal });
  if (!res.ok) throw new Error(`Request failed (${res.status})`);
  return (await res.json()) as AgentOutput;
}
