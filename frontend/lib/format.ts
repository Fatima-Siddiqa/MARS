import type { Signal } from "./types";

// The approved UI says Positive / Neutral / Negative.
export const SIGNAL_LABEL: Record<Signal, string> = {
  bullish: "Positive",
  neutral: "Neutral",
  bearish: "Negative",
};

export const SIGNAL_BADGE: Record<Signal, string> = {
  bullish: "border-emerald-200 bg-emerald-50 text-emerald-800",
  neutral: "border-amber-200 bg-amber-50 text-amber-800",
  bearish: "border-rose-200 bg-rose-50 text-rose-800",
};

export function confidenceBar(c: number): string {
  if (c >= 0.7) return "bg-emerald-500";
  if (c >= 0.4) return "bg-amber-500";
  return "bg-rose-500";
}

export const pct = (n: number) => `${Math.round(n * 100)}%`;
