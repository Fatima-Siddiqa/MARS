"use client";

import { Bell, ChevronDown, Menu, Sparkles } from "lucide-react";
import TickerSearch from "@/components/search/TickerSearch";

export default function TopBar({ onMenu }: { onMenu: () => void }) {
  return (
    <header className="flex items-center gap-3 border-b border-slate-200 bg-white px-4 py-3 sm:px-6">
      <button onClick={onMenu} className="rounded-lg p-2 text-slate-600 hover:bg-slate-100 lg:hidden" aria-label="Open navigation">
        <Menu className="h-5 w-5" />
      </button>

      <TickerSearch />

      <div className="ml-auto flex items-center gap-3">
        <button className="hidden items-center gap-2 rounded-lg bg-blue-50 px-3 py-2 text-sm font-semibold text-blue-700 hover:bg-blue-100 sm:flex">
          <Sparkles className="h-4 w-4" />
          Explainability
        </button>
        <button className="relative rounded-lg p-2 text-slate-500 hover:bg-slate-100" aria-label="Notifications">
          <Bell className="h-5 w-5" />
          <span className="absolute right-1.5 top-1.5 h-2 w-2 rounded-full bg-blue-600" />
        </button>
        <div className="hidden items-center gap-3 border-l border-slate-200 pl-4 sm:flex">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-blue-600 text-sm font-bold text-white">A</div>
          <div className="leading-tight">
            <div className="text-sm font-semibold">A</div>
            <div className="text-[11px] text-slate-500">Investor Account</div>
          </div>
          <ChevronDown className="h-4 w-4 text-slate-400" />
        </div>
      </div>
    </header>
  );
}
