"use client";

import { useCallback, useEffect, useState } from "react";
import { AlertTriangle, RotateCw } from "lucide-react";
import TechnicalChart from "@/components/charts/TechnicalChart";
import { findTicker } from "@/data/tickers";
import { fetchAgent } from "@/lib/api";
import { SIGNAL_BADGE, SIGNAL_LABEL, confidenceBar, pct } from "@/lib/format";
import type { AgentOutput } from "@/lib/types";

type State = { status: "loading" } | { status: "error"; message: string } | { status: "success"; data: AgentOutput };

const FAMILY_LABEL: Record<string, string> = {
  trend: "Trend",
  momentum: "Momentum",
  relative_strength: "Relative strength vs KSE-100",
  risk: "Risk",
};
const card = "rounded-2xl border border-slate-200 bg-white p-6 shadow-sm";

// Render with key={ticker} so a new ticker restarts in the loading state.
export default function CompanyTechnicalView({ ticker }: { ticker: string }) {
  const [state, setState] = useState<State>({ status: "loading" });
  const [attempt, setAttempt] = useState(0);
  const t = findTicker(ticker);

  const load = useCallback(
    (signal: AbortSignal) => {
      fetchAgent("technical", ticker, signal)
        .then((data) => setState({ status: "success", data }))
        .catch((err: unknown) => {
          if (signal.aborted) return;
          setState({ status: "error", message: err instanceof Error ? err.message : "Something went wrong" });
        });
    },
    [ticker],
  );

  useEffect(() => {
    const ctrl = new AbortController();
    load(ctrl.signal);
    return () => ctrl.abort();
  }, [load, attempt]);

  const data = state.status === "success" ? state.data : null;

  return (
    <div className="space-y-5">
      <div className={card}>
        <div className="flex flex-wrap items-center gap-2 text-xs">
          <span className="rounded border border-blue-200 bg-blue-50 px-2 py-0.5 font-semibold text-blue-700">Deep Dive Analysis</span>
          <span className="text-slate-500">
            • Selected: {t?.name ?? ticker} ({ticker})
          </span>
          {data?.as_of && <span className="text-slate-500">• Data as of {data.as_of}</span>}
        </div>
        <h1 className="mt-2 text-3xl font-extrabold">Company Analysis</h1>
        <p className="mt-1 max-w-2xl text-sm text-slate-600">Price chart and technical indicators from the Historical Price &amp; Technical agent.</p>
      </div>

      {state.status === "loading" && (
        <div role="status" aria-label="Loading technical analysis" className={`${card} space-y-3`}>
          <div className="h-4 w-1/3 animate-pulse rounded bg-slate-200" />
          <div className="h-[460px] animate-pulse rounded-xl bg-slate-100" />
        </div>
      )}

      {state.status === "error" && (
        <div className={`${card} flex items-center justify-between gap-4 border-rose-200 bg-rose-50`}>
          <p className="text-sm text-rose-800">Could not load technical analysis: {state.message}</p>
          <button
            onClick={() => {
              setState({ status: "loading" });
              setAttempt((a) => a + 1);
            }}
            className="flex items-center gap-2 rounded-xl bg-slate-900 px-3 py-2 text-sm font-semibold text-white hover:bg-slate-800"
          >
            <RotateCw className="h-4 w-4" /> Retry
          </button>
        </div>
      )}

      {data && (
        <>
          {data.warnings && data.warnings.length > 0 && (
            <div className="flex items-start gap-2 rounded-xl border border-amber-200 bg-amber-50 p-3 text-sm text-amber-800">
              <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
              <span>{data.warnings.join(" · ")}</span>
            </div>
          )}

          <div className={`${card} flex flex-wrap items-center justify-between gap-4`}>
            <div className="min-w-0 flex-1">
              <div className="text-[11px] font-semibold uppercase tracking-wide text-slate-400">Technical verdict</div>
              <div className="mt-1 flex items-center gap-3">
                <span className={`rounded-md border px-2.5 py-0.5 text-xs font-bold ${SIGNAL_BADGE[data.signal]}`}>{SIGNAL_LABEL[data.signal]}</span>
                <span className="font-mono text-sm text-slate-600">score {data.score >= 0 ? "+" : ""}{data.score.toFixed(2)}</span>
              </div>
              {data.summary && <p className="mt-2 text-sm text-slate-600">{data.summary}</p>}
            </div>
            <div className="w-40 text-right">
              <div className="text-[11px] font-semibold uppercase tracking-wide text-slate-400">Confidence</div>
              <div className="text-xl font-bold">{pct(data.confidence)}</div>
              <div className="mt-1 h-1.5 w-full rounded bg-slate-100">
                <div className={`h-1.5 rounded ${confidenceBar(data.confidence)}`} style={{ width: pct(data.confidence) }} />
              </div>
            </div>
          </div>

          <div className={card}>
            {data.chart && data.chart.length > 0 ? (
              <TechnicalChart data={data.chart} />
            ) : (
              <p className="p-10 text-center text-sm text-slate-500">No chart data (mock mode). Run the backend and unset NEXT_PUBLIC_USE_MOCK_TECHNICAL.</p>
            )}
          </div>

          <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
            {data.families && (
              <div className={card}>
                <h2 className="mb-3 text-xs font-bold uppercase tracking-wide text-slate-800">Signal families</h2>
                <ul className="space-y-3">
                  {Object.entries(data.families).map(([k, v]) => (
                    <li key={k}>
                      <div className="mb-1 flex justify-between text-xs text-slate-600">
                        <span>{FAMILY_LABEL[k] ?? k}</span>
                        <span className="font-mono">{v == null ? "n/a" : `${v >= 0 ? "+" : ""}${v.toFixed(2)}`}</span>
                      </div>
                      <div className="relative h-2 rounded bg-slate-100">
                        <div className="absolute left-1/2 top-0 h-2 w-px bg-slate-400" />
                        {v != null && (
                          <div
                            className={`absolute top-0 h-2 rounded ${v >= 0 ? "bg-emerald-500" : "bg-rose-500"}`}
                            style={{ left: v >= 0 ? "50%" : `${50 + v * 50}%`, width: `${Math.min(Math.abs(v), 1) * 50}%` }}
                          />
                        )}
                      </div>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            <div className={card}>
              <h2 className="mb-3 text-xs font-bold uppercase tracking-wide text-slate-800">Indicators</h2>
              <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
                {data.indicators?.map((ind) => (
                  <div key={ind.label} className="rounded-xl border border-slate-200 p-3">
                    <div className="text-[11px] text-slate-500">{ind.label}</div>
                    <div className="mt-0.5 font-mono text-sm font-semibold">{ind.value}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {data.risks && data.risks.length > 0 && (
            <div className={card}>
              <h2 className="mb-2 text-xs font-bold uppercase tracking-wide text-slate-800">Risks</h2>
              <ul className="list-disc space-y-1 pl-5 text-sm text-slate-600">
                {data.risks.map((r) => (
                  <li key={r}>{r}</li>
                ))}
              </ul>
            </div>
          )}

          <p className="text-xs text-slate-400">MARS is decision-support only. This describes past price behaviour; it is not a forecast or investment advice.</p>
        </>
      )}
    </div>
  );
}
