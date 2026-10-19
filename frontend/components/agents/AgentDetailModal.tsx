"use client";

import { useEffect, type ReactNode } from "react";
import { Activity, CheckCircle2, ExternalLink, Layers, ListChecks, X } from "lucide-react";
import type { AgentConfig } from "@/data/agents";
import { SIGNAL_BADGE, SIGNAL_LABEL, pct } from "@/lib/format";
import type { AgentOutput } from "@/lib/types";

type Props = { agent: AgentConfig; data: AgentOutput; onClose: () => void };

function Section({ icon, title, children }: { icon: ReactNode; title: string; children: ReactNode }) {
  return (
    <section>
      <h3 className="mb-2 flex items-center gap-2 text-xs font-bold uppercase tracking-wide text-slate-800">
        <span className="text-blue-600">{icon}</span>
        {title}
      </h3>
      {children}
    </section>
  );
}

export default function AgentDetailModal({ agent, data, onClose }: Props) {
  const Icon = agent.icon;

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/50 p-4" onMouseDown={onClose}>
      <div
        role="dialog"
        aria-modal="true"
        aria-label={`${agent.label} details`}
        onMouseDown={(e) => e.stopPropagation()}
        className="flex max-h-[90vh] w-full max-w-xl flex-col overflow-hidden rounded-2xl bg-white shadow-2xl"
      >
        <div className="flex items-center gap-3 bg-slate-900 px-5 py-4 text-white">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-blue-500/20 text-blue-300">
            <Icon className="h-5 w-5" />
          </div>
          <div className="min-w-0 flex-1">
            <div className="text-[11px] font-semibold uppercase tracking-wide text-blue-300">
              Specialist Agent <span className="mx-1 text-slate-500">•</span>
              <span className="font-mono text-slate-300">{data.ticker}</span>
            </div>
            <div className="truncate text-lg font-bold">{agent.label}</div>
          </div>
          <button onClick={onClose} aria-label="Close" className="rounded-lg bg-slate-800 p-2 hover:bg-slate-700">
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="space-y-5 overflow-y-auto p-5">
          <div className="flex items-center justify-between rounded-xl border border-slate-200 bg-slate-50 p-3">
            <div>
              <div className="text-[11px] font-semibold uppercase tracking-wide text-slate-400">Current finding</div>
              <div className="mt-1 flex items-center gap-2 text-sm">
                <span className={`rounded-md border px-2.5 py-0.5 text-xs font-bold ${SIGNAL_BADGE[data.signal]}`}>{SIGNAL_LABEL[data.signal]}</span>
                <span className="text-slate-500">
                  Weight in Regime: <b className="text-slate-800">{agent.weight}%</b>
                </span>
              </div>
            </div>
            <div className="text-right">
              <div className="text-[11px] font-semibold uppercase tracking-wide text-slate-400">{data.agent === "sentiment" ? "Data quality" : "Confidence"}</div>
              <div className="text-xl font-bold">{pct(data.confidence)}</div>
            </div>
          </div>

          <Section icon={<Activity className="h-4 w-4" />} title="Agent mandate & purpose">
            <p className="rounded-xl border border-slate-100 bg-slate-50 p-3 text-sm text-slate-600">{agent.purpose}</p>
          </Section>

          {data.rationale && (
            <Section icon={<ListChecks className="h-4 w-4" />} title="Analytical rationale">
              <p className="rounded-xl border border-blue-100 bg-blue-50 p-3 text-sm font-medium text-slate-800">{data.rationale}</p>
            </Section>
          )}

          {data.indicators && data.indicators.length > 0 && (
            <Section icon={<Layers className="h-4 w-4" />} title="Specialist indicators tracked">
              <div className="grid grid-cols-1 gap-2 sm:grid-cols-3">
                {data.indicators.map((ind) => (
                  <div key={ind.label} className="rounded-xl border border-slate-200 p-3">
                    <div className="text-[11px] text-slate-500">{ind.label}</div>
                    <div className="mt-0.5 font-mono text-sm font-semibold">{ind.value}</div>
                  </div>
                ))}
              </div>
            </Section>
          )}

          <Section icon={<CheckCircle2 className="h-4 w-4" />} title="Audited evidence">
            {data.evidence.length === 0 ? (
              <p className="text-sm text-slate-500">No evidence returned.</p>
            ) : (
              <ul className="space-y-2">
                {data.evidence.map((ev, i) => (
                  <li key={`${ev.title}-${i}`} className="flex items-start gap-3 rounded-xl border border-emerald-200 bg-emerald-50/60 p-3 text-sm">
                    <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-emerald-600" />
                    <div className="min-w-0">
                      {ev.url ? (
                        <a href={ev.url} target="_blank" rel="noopener noreferrer" className="inline-flex items-center gap-1 font-medium text-blue-700 hover:underline">
                          {ev.title} <ExternalLink className="h-3 w-3" />
                        </a>
                      ) : (
                        <span className="font-medium text-slate-700">{ev.title}</span>
                      )}
                      {(ev.source || ev.date) && <div className="text-xs text-slate-500">{[ev.source, ev.date].filter(Boolean).join(" · ")}</div>}
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </Section>
        </div>

        <div className="flex justify-end border-t border-slate-100 p-4">
          <button onClick={onClose} className="rounded-xl bg-slate-900 px-4 py-2 text-sm font-semibold text-white hover:bg-slate-800">
            Close Details
          </button>
        </div>
      </div>
    </div>
  );
}