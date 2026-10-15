"use client";

import { useState } from "react";
import { AGENTS, ENABLED_IDS } from "@/data/agents";
import { useAgents } from "@/lib/useAgents";
import type { AgentId, Signal } from "@/lib/types";
import AgentCard from "./AgentCard";
import AgentDetailModal from "./AgentDetailModal";

export type SignalFilter = "all" | Signal;

type Props = { ticker: string; filter: SignalFilter };

// Render with key={ticker} so changing the ticker restarts every card in the loading state.
export default function AgentGrid({ ticker, filter }: Props) {
  const { states, retry } = useAgents(ENABLED_IDS, ticker);
  const [openId, setOpenId] = useState<AgentId | null>(null);

  const visible = AGENTS.filter((a) => {
    if (filter === "all") return true;
    const s = states[a.id];
    return s.status === "success" && s.data.signal === filter;
  });

  const openAgent = AGENTS.find((a) => a.id === openId);
  const openState = openId ? states[openId] : null;

  return (
    <>
      {visible.length === 0 ? (
        <div className="rounded-2xl border border-dashed border-slate-300 p-10 text-center text-sm text-slate-500">
          No agents match this filter yet.
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-5 md:grid-cols-2 xl:grid-cols-3">
          {visible.map((agent) => (
            <AgentCard key={agent.id} agent={agent} state={states[agent.id]} onRetry={() => retry(agent.id)} onOpen={() => setOpenId(agent.id)} />
          ))}
        </div>
      )}

      {openAgent && openState?.status === "success" && (
        <AgentDetailModal agent={openAgent} data={openState.data} onClose={() => setOpenId(null)} />
      )}
    </>
  );
}
