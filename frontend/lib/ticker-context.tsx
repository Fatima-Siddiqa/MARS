"use client";

import { createContext, useContext, useState, type ReactNode } from "react";

type TickerCtx = { ticker: string; setTicker: (symbol: string) => void };
const Ctx = createContext<TickerCtx | null>(null);

// The selected ticker is shared by the top-bar search and every page.
export function TickerProvider({ children }: { children: ReactNode }) {
  const [ticker, setTicker] = useState("HBL");
  return <Ctx.Provider value={{ ticker, setTicker }}>{children}</Ctx.Provider>;
}

export function useTicker(): TickerCtx {
  const ctx = useContext(Ctx);
  if (!ctx) throw new Error("useTicker must be used inside <TickerProvider>");
  return ctx;
}
