"use client";

import { useState } from "react";
import AgentGrid, { type SignalFilter } from "@/components/agents/AgentGrid";
import { findTicker } from "@/data/tickers";
import { useTicker } from "@/lib/ticker-context";

const FILTERS: { value: SignalFilter; label: string }[] = [
  { value: "all", label: "ALL" },
  { value: "bullish", label: "Positive" },
  { value: "neutral", label: "Neutral" },
  { value: "bearish", label: "Negative" },
];

export default function SpecialistAgents() {
  const { ticker } = useTicker();
  const selected = findTicker(ticker);
  const [filter, setFilter] = useState<SignalFilter>("all");

  return (
    <div className="space-y-5">
      <div className="flex flex-col gap-4 rounded-2xl border border-slate-200 bg-white p-6 shadow-sm md:flex-row md:items-start md:justify-between">
        <div>
          <div className="flex flex-wrap items-center gap-2 text-xs">
            <span className="rounded border border-blue-200 bg-blue-50 px-2 py-0.5 font-semibold text-blue-700">MARS Multi-Agent Architecture</span>
            {selected && (
              <span className="text-slate-500">
                • Currently analyzing {selected.name} ({selected.symbol})
              </span>
            )}
          </div>
          <h1 className="mt-2 text-3xl font-extrabold">Specialist Agents Overview</h1>
          <p className="mt-1 max-w-2xl text-sm text-slate-600">
            MARS operates six domain-specific agents that independently evaluate technicals, sentiment, macroeconomic factors, FX impacts,
            fundamentals, and forensic governance before reconciling under the prevailing market regime.
          </p>
        </div>

        <div role="group" aria-label="Filter by signal" className="flex shrink-0 rounded-xl bg-slate-100 p-1 text-sm font-semibold">
          {FILTERS.map((f) => (
            <button
              key={f.value}
              onClick={() => setFilter(f.value)}
              aria-pressed={filter === f.value}
              className={`rounded-lg px-3 py-1.5 ${filter === f.value ? "bg-white text-blue-700 shadow-sm" : "text-slate-600 hover:text-slate-900"}`}
            >
              {f.label}
            </button>
          ))}
        </div>
      </div>

      {selected ? (
        <AgentGrid key={selected.symbol} ticker={selected.symbol} filter={filter} />
      ) : (
        <div className="rounded-2xl border border-dashed border-slate-300 p-12 text-center text-slate-500">Select a ticker in the search bar.</div>
      )}
    </div>
  );
}
