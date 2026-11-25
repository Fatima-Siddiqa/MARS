"use client";

import { useCallback, useEffect, useState } from "react";
import { AlertTriangle, RotateCw } from "lucide-react";
import { confidenceBar, pct } from "@/lib/format";
import { type FundAnalysis, fetchFundamentalsAnalysis, fmtRatio, fmtRs } from "@/lib/fundamentals";

type State = { status: "loading" } | { status: "error"; message: string } | { status: "success"; data: FundAnalysis };

const card = "rounded-2xl border border-slate-200 bg-white p-6 shadow-sm";
const CALL_STYLE: Record<FundAnalysis["call"], string> = {
  BUY: "border-emerald-200 bg-emerald-50 text-emerald-800",
  HOLD: "border-amber-200 bg-amber-50 text-amber-800",
  AVOID: "border-rose-200 bg-rose-50 text-rose-800",
};
const FAMILY_ORDER = ["valuation", "profitability", "growth", "leverage", "liquidity", "efficiency"];

// Render with key={ticker} so a new ticker restarts in the loading state.
export default function FundamentalsPanel({ ticker }: { ticker: string }) {
  const [state, setState] = useState<State>({ status: "loading" });
  const [attempt, setAttempt] = useState(0);

  const load = useCallback(
    (signal: AbortSignal) => {
      fetchFundamentalsAnalysis(ticker, signal)
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

  if (state.status === "loading") {
    return (
      <section className={card} aria-busy="true">
        <h2 className="text-lg font-bold">Sector &amp; Fundamentals</h2>
        <div role="status" aria-label="Loading fundamentals" className="mt-4 space-y-2">
          <div className="h-4 w-1/3 animate-pulse rounded bg-slate-200" />
          <div className="h-24 animate-pulse rounded bg-slate-100" />
        </div>
      </section>
    );
  }
  if (state.status === "error") {
    return (
      <section className={card} role="alert">
        <h2 className="text-lg font-bold">Sector &amp; Fundamentals</h2>
        <p className="mt-2 flex items-center gap-2 text-sm text-rose-700">
          <AlertTriangle className="h-4 w-4" /> {state.message}
        </p>
        <button
          onClick={() => {
            setState({ status: "loading" });
            setAttempt((a) => a + 1);
          }}
          className="mt-3 inline-flex items-center gap-1.5 rounded-md bg-rose-600 px-2.5 py-1 text-xs font-semibold text-white hover:bg-rose-700"
        >
          <RotateCw className="h-3 w-3" /> Retry
        </button>
      </section>
    );
  }

  const a = state.data;
  const v = a.valuation;
  const fair = v.implied_value_per_share ?? a.peer_baseline.implied_value_per_share;
  const families = FAMILY_ORDER.filter((f) => a.ratios.some((r) => r.family === f));

  return (
    <section className={`${card} space-y-6`}>
      <header className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h2 className="text-lg font-bold">Sector &amp; Fundamentals</h2>
          <p className="text-xs text-slate-500">
            {a.company.sector} • official PSX filings and prices • as of {a.as_of}
          </p>
        </div>
        <div className="text-right">
          <span className={`rounded-md border px-3 py-1 text-sm font-extrabold ${CALL_STYLE[a.call]}`}>{a.call}</span>
          <div className="mt-2 w-40">
            <div className="flex justify-between text-xs">
              <span className="text-slate-500">Confidence</span>
              <span className="font-bold">{pct(a.confidence)}</span>
            </div>
            <div className="mt-1 h-1.5 rounded-full bg-slate-100" role="meter" aria-label="Confidence" aria-valuemin={0} aria-valuemax={1} aria-valuenow={a.confidence}>
              <div className={`h-1.5 rounded-full ${confidenceBar(a.confidence)}`} style={{ width: pct(a.confidence) }} />
            </div>
          </div>
        </div>
      </header>

      <div className="grid gap-3 sm:grid-cols-4">
        <Tile label="Fair value / share" value={fmtRs(fair)} sub={v.range_per_share ? `80% range ${fmtRs(v.range_per_share[0])} – ${fmtRs(v.range_per_share[1])}` : a.gap_source} />
        <Tile label="Price" value={fmtRs(a.market.price)} sub={a.market.price_date ? `PSX close ${a.market.price_date}` : "no recent price"} />
        <Tile label="Valuation gap" value={a.gap === null ? "—" : `${a.gap >= 0 ? "+" : ""}${(a.gap * 100).toFixed(0)}%`} sub={`BUY ≥ +${(a.threshold * 100).toFixed(0)}% • valued by ${a.gap_source}`} />
        <Tile label="Peers" value={`${a.peers.peers.length}`} sub={`${a.peers.strength} peer group`} />
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <div>
          <h3 className="mb-2 text-sm font-bold">Ratios vs PSX sector peers</h3>
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-[11px] uppercase tracking-wide text-slate-400">
                <th className="py-1">Ratio</th>
                <th className="py-1 text-right">{a.ticker}</th>
                <th className="py-1 text-right">Peer median</th>
              </tr>
            </thead>
            <tbody>
              {families.flatMap((fam) =>
                a.ratios
                  .filter((r) => r.family === fam)
                  .map((r) => {
                    const st = a.peers.stats[r.name];
                    return (
                      <tr key={r.name} className="border-t border-slate-100" title={r.formula}>
                        <td className="py-1.5">{r.label}</td>
                        <td className="py-1.5 text-right font-mono">{r.value === null ? <span className="text-xs text-slate-400" title={r.reason ?? ""}>n/a</span> : fmtRatio(r)}</td>
                        <td className="py-1.5 text-right font-mono text-slate-500">{st ? fmtRatio(r, st.peer_median) : "—"}</td>
                      </tr>
                    );
                  }),
              )}
            </tbody>
          </table>
          {a.peers.peers.length > 0 && (
            <p className="mt-2 text-xs text-slate-500">Peers: {a.peers.peers.slice(0, 15).join(", ")}{a.peers.peers.length > 15 ? ` +${a.peers.peers.length - 15} more` : ""}</p>
          )}
        </div>

        <div className="space-y-5">
          <div>
            <h3 className="text-sm font-bold">What drives the model&apos;s fair value</h3>
            <p className="mb-2 text-[11px] text-slate-500">Effect of each input compared with a typical PSX company (SHAP).</p>
            {v.contributions.length === 0 ? (
              <p className="text-sm text-slate-500">{v.reason ?? "The peer baseline is a fixed formula (no model drivers)."}</p>
            ) : (
              <ul className="space-y-1.5 text-sm">
                {v.contributions.slice(0, 6).map((c) => (
                  <li key={c.feature} className="flex justify-between">
                    <span>{c.label}</span>
                    <span className={`font-mono ${c.effect_pct >= 0 ? "text-emerald-700" : "text-rose-700"}`}>
                      {c.effect_pct >= 0 ? "+" : ""}
                      {(c.effect_pct * 100).toFixed(0)}%
                    </span>
                  </li>
                ))}
              </ul>
            )}
          </div>
          <div className="rounded-xl border border-slate-100 bg-slate-50 p-3 text-xs text-slate-600">
            <div className="font-semibold text-slate-700">Model ({v.method}{v.model_version ? `, ${v.model_version}` : ""})</div>
            {v.test_metrics.n ? (
              <p className="mt-1">
                Tested on {v.test_metrics.n} unseen company-years: median error {pct(v.test_metrics.median_abs_pct_error ?? 0)}, R² {(v.test_metrics.r2 ?? 0).toFixed(2)}.
              </p>
            ) : (
              <p className="mt-1">No trained model available.</p>
            )}
            <p className="mt-1">Call threshold: {a.calibration.supported ? "confirmed" : "not confirmed"} by later returns. {a.calibration.reason}</p>
          </div>
          {a.warnings.length > 0 && (
            <ul className="space-y-1 text-xs text-amber-800">
              {a.warnings.map((w) => (
                <li key={w} className="flex gap-1.5"><AlertTriangle className="h-3.5 w-3.5 shrink-0" />{w}</li>
              ))}
            </ul>
          )}
          <p className="text-[11px] text-slate-400">
            Balance sheet {a.statements.balance_date ?? "not available"} • shares: {a.market.shares_source ?? "—"} • decision support only, not investment advice.
          </p>
        </div>
      </div>
    </section>
  );
}

function Tile({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="rounded-xl border border-slate-100 bg-slate-50 p-3">
      <div className="text-[11px] font-semibold uppercase tracking-wide text-slate-400">{label}</div>
      <div className="mt-1 text-lg font-bold">{value}</div>
      {sub && <div className="mt-0.5 text-[11px] text-slate-500">{sub}</div>}
    </div>
  );
}
