// SHARED AGENT OUTPUT FORMAT.
// DRAFT: confirm every field name and range with Fatima, then edit ONLY this file.

export type Signal = "bullish" | "bearish" | "neutral";

export type AgentId = "technical" | "sentiment" | "macro" | "fx" | "fundamentals" | "forensics";

export type Evidence = {
  title: string;
  /** News evidence has a URL. Computed or file-based evidence may not. */
  url?: string;
  /** Publisher, dataset or file, e.g. "Dawn", "SBP EasyData". */
  source?: string;
  /** ISO date string, e.g. "2026-10-02". */
  date?: string;
};

/** One tracked metric, shown as a small tile in the detail popup. */
export type Indicator = { label: string; value: string };

export type AgentOutput = {
  agent: AgentId;
  ticker: string;
  signal: Signal;
  /** Assumed -1 (very bearish) to +1 (very bullish). Not shown in the approved UI. */
  score: number;
  /** Assumed 0 to 1. Shown as a percentage. */
  confidence: number;
  /** "Current finding" on the card: one or two sentences. */
  summary?: string;
  /** "Analytical rationale" in the detail popup. */
  rationale?: string;
  /** "Specialist indicators tracked" in the detail popup. */
  indicators?: Indicator[];
  /** "Audited evidence" in the detail popup. */
  evidence: Evidence[];
  generated_at?: string;
};

export type AgentState =
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "success"; data: AgentOutput }
  | { status: "unavailable" }; // agent not built yet
