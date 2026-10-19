"use client";

import { ArrowRight, RotateCw } from "lucide-react";
import type { AgentConfig } from "@/data/agents";
import { SIGNAL_BADGE, SIGNAL_LABEL, confidenceBar, pct } from "@/lib/format";
import type { AgentState } from "@/lib/types";

type Props = {
  agent: AgentConfig;
  state: AgentState;
  onRetry?: () => void;
  onOpen?: () => void;
};

const pill = "rounded-md border px-2.5 py-0.5 text-xs font-bold";
const skeleton = "animate-pulse rounded bg-slate-200";

export default function AgentCard({ agent, state, onRetry, onOpen }: Props) {
  const Icon = agent.icon;
  const data = state.status === "success" ? state.data : null;

  return (
    <article
      aria-busy={state.status === "loading"}
      className="flex h-full flex-col rounded-2xl border border-slate-200 bg-white p-6 shadow-sm"
    >
      <div className="mb-4 flex items-start justify-between">
        <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-blue-50 text-blue-600">
          <Icon className="h-5 w-5" />
        </div>
        <div className="flex flex-col items-end gap-1">
          {state.status === "loading" && (
            <div className={`${skeleton} h-6 w-16`} />
          )}
          {state.status === "error" && (
            <span
              className={`${pill} border-rose-200 bg-rose-50 text-rose-800`}
            >
              Error
            </span>
          )}
          {state.status === "unavailable" && (
            <span
              className={`${pill} border-slate-200 bg-slate-100 text-slate-500`}
            >
              Coming soon
            </span>
          )}
          {data && (
            <span className={`${pill} ${SIGNAL_BADGE[data.signal]}`}>
              {SIGNAL_LABEL[data.signal]}
            </span>
          )}
          <span className="font-mono text-[10px] text-slate-400">
            Weight: {agent.weight}%
          </span>
        </div>
      </div>

      <h2 className="break-words text-base font-bold">{agent.label}</h2>

      <div className="mt-3 text-[11px] font-semibold uppercase tracking-wide text-slate-400">
        Purpose:
      </div>
      <p className="mt-1 text-sm text-slate-600">{agent.purpose}</p>

      <div className="mt-auto pt-4">
        <div className="min-h-24 rounded-xl border border-slate-100 bg-slate-50 p-3">
          <div className="text-[11px] font-semibold uppercase tracking-wide text-slate-400">
            Current finding:
          </div>
          {state.status === "loading" && (
            <div
              role="status"
              aria-label="Loading agent output"
              className="mt-2 space-y-2"
            >
              <div className={`${skeleton} h-3 w-full`} />
              <div className={`${skeleton} h-3 w-4/5`} />
            </div>
          )}
          {state.status === "error" && (
            <div role="alert" className="mt-1 text-sm text-rose-700">
              <p className="break-words">{state.message}</p>
              {onRetry && (
                <button
                  onClick={onRetry}
                  className="mt-2 inline-flex items-center gap-1.5 rounded-md bg-rose-600 px-2.5 py-1 text-xs font-semibold text-white hover:bg-rose-700"
                >
                  <RotateCw className="h-3 w-3" /> Retry
                </button>
              )}
            </div>
          )}
          {state.status === "unavailable" && (
            <p className="mt-1 text-sm text-slate-500">
              This agent is connected in Iteration 2.
            </p>
          )}
          {data && (
            <p className="mt-1 text-sm font-semibold">
              {data.summary ?? "No summary returned."}
            </p>
          )}
        </div>

        <div className="mt-4 border-t border-slate-100 pt-4">
          <div className="mb-1.5 flex justify-between text-xs">
            <span className="text-slate-500">{data?.agent === "sentiment" ? "Data quality" : "Confidence"}</span>
            {state.status === "loading" ? (
              <span className={`${skeleton} h-3 w-8`} />
            ) : (
              <span className="font-bold">
                {data ? pct(data.confidence) : "—"}
              </span>
            )}
          </div>
          <div
            className="h-1.5 rounded-full bg-slate-100"
            role="meter"
            aria-label="Confidence"
            aria-valuemin={0}
            aria-valuemax={1}
            aria-valuenow={data?.confidence}
          >
            {data && (
              <div
                className={`h-1.5 rounded-full ${confidenceBar(data.confidence)}`}
                style={{ width: pct(data.confidence) }}
              />
            )}
          </div>
        </div>

        <button
          onClick={onOpen}
          disabled={!data}
          className="mt-4 flex w-full items-center justify-center gap-2 rounded-xl border border-slate-200 bg-slate-50 py-2.5 text-sm font-semibold text-slate-700 hover:bg-slate-100 disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:bg-slate-50"
        >
          View Full Reasoning &amp; Evidence <ArrowRight className="h-4 w-4" />
        </button>
      </div>
    </article>
  );
}
