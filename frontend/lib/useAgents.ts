"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { fetchAgent } from "./api";
import { AGENTS } from "@/data/agents";
import type { AgentId, AgentState } from "./types";

type StateMap = Record<AgentId, AgentState>;

function initial(ids: AgentId[]): StateMap {
  const map = {} as StateMap;
  for (const a of AGENTS) map[a.id] = ids.includes(a.id) ? { status: "loading" } : { status: "unavailable" };
  return map;
}

/**
 * Runs all enabled agents for one ticker in parallel. Each agent loads and fails on its own.
 * `ids` must be a stable array. Render the consumer with key={ticker} so a new ticker starts fresh.
 */
export function useAgents(ids: AgentId[], ticker: string) {
  const [states, setStates] = useState<StateMap>(() => initial(ids));
  const controllers = useRef(new Map<AgentId, AbortController>());

  const load = useCallback(
    (id: AgentId) => {
      controllers.current.get(id)?.abort();
      const ctrl = new AbortController();
      controllers.current.set(id, ctrl);
      fetchAgent(id, ticker, ctrl.signal)
        .then((data) => setStates((s) => ({ ...s, [id]: { status: "success", data } })))
        .catch((err: unknown) => {
          if (ctrl.signal.aborted) return;
          const message = err instanceof Error ? err.message : "Something went wrong";
          setStates((s) => ({ ...s, [id]: { status: "error", message } }));
        });
    },
    [ticker],
  );

  useEffect(() => {
    const map = controllers.current;
    ids.forEach(load);
    return () => map.forEach((c) => c.abort());
  }, [ids, load]);

  const retry = useCallback(
    (id: AgentId) => {
      setStates((s) => ({ ...s, [id]: { status: "loading" } }));
      load(id);
    },
    [load],
  );

  return { states, retry };
}
