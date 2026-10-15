"use client";

import { findTicker } from "@/data/tickers";
import { useTicker } from "@/lib/ticker-context";

type Props = { eyebrow: string; title: string; description: string; note: string };

// Header card from the approved design + a marker for content that is built later.
export default function PagePlaceholder({ eyebrow, title, description, note }: Props) {
  const { ticker } = useTicker();
  const t = findTicker(ticker);
  return (
    <div className="space-y-5">
      <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
        <div className="flex flex-wrap items-center gap-2 text-xs">
          <span className="rounded border border-blue-200 bg-blue-50 px-2 py-0.5 font-semibold text-blue-700">{eyebrow}</span>
          {t && (
            <span className="text-slate-500">
              • Selected: {t.name} ({t.symbol})
            </span>
          )}
        </div>
        <h1 className="mt-2 text-3xl font-extrabold">{title}</h1>
        <p className="mt-1 max-w-2xl text-sm text-slate-600">{description}</p>
      </div>
      <div className="rounded-2xl border border-dashed border-slate-300 p-12 text-center text-sm text-slate-500">{note}</div>
    </div>
  );
}
