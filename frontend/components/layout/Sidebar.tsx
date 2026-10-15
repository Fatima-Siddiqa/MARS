"use client";

import { Bookmark, Building2, Cpu, FileText, LayoutDashboard } from "lucide-react";
import { AGENTS } from "@/data/agents";
import NavigationItem, { type NavItem } from "@/components/navigation/NavigationItem";

const NAV: NavItem[] = [
  { href: "/", label: "My Workspace", icon: LayoutDashboard },
  { href: "/company", label: "Company Analysis", icon: Building2 },
  { href: "/agents", label: "Specialist Agents", icon: Cpu, suffix: ` (${AGENTS.length})` },
  { href: "/reports", label: "Research Reports", icon: FileText },
  { href: "/watchlist", label: "Watchlist", icon: Bookmark },
];

export default function Sidebar({ open, onClose }: { open: boolean; onClose: () => void }) {
  return (
    <>
      {open && <div className="fixed inset-0 z-30 bg-slate-900/40 lg:hidden" onClick={onClose} aria-hidden />}
      <aside
        className={`fixed inset-y-0 left-0 z-40 flex w-60 shrink-0 flex-col border-r border-slate-200 bg-white transition-transform lg:static lg:translate-x-0 ${
          open ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        <div className="flex items-center gap-3 border-b border-slate-100 px-4 py-4">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-blue-600 text-lg font-extrabold text-white">M</div>
          <div>
            <div className="flex items-center gap-1.5">
              <span className="text-lg font-extrabold leading-none">MARS</span>
              <span className="rounded border border-blue-200 bg-blue-50 px-1.5 text-[10px] font-bold text-blue-700">PSX</span>
            </div>
            <div className="mt-0.5 text-[11px] text-slate-500">Stock Analysis Intelligence</div>
          </div>
        </div>

        <nav className="flex-1 px-3 py-4">
          <div className="mb-2 px-2 text-[11px] font-semibold uppercase tracking-wider text-slate-400">Navigation</div>
          <ul className="space-y-1">
            {NAV.map((item) => (
              <li key={item.href}>
                <NavigationItem item={item} onNavigate={onClose} />
              </li>
            ))}
          </ul>
        </nav>

        <div className="border-t border-slate-100 p-3">
          <div className="flex items-center gap-3 rounded-xl border border-slate-200 p-2">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-blue-600 text-sm font-bold text-white">A</div>
            <div className="leading-tight">
              <div className="text-sm font-semibold">A</div>
              <div className="font-mono text-[10px] text-slate-500">CDC-69305-PSX</div>
            </div>
          </div>
        </div>
      </aside>
    </>
  );
}
