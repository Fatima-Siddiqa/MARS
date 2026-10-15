"use client";

import { useEffect, useId, useMemo, useRef, useState } from "react";
import { usePathname, useRouter } from "next/navigation";
import { Check, Search, X } from "lucide-react";
import { TICKERS } from "@/data/tickers";
import { useTicker } from "@/lib/ticker-context";

export default function TickerSearch() {
  const { ticker, setTicker } = useTicker();
  const router = useRouter();
  const pathname = usePathname();
  const listId = useId();
  const rootRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [active, setActive] = useState(0);

  const results = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return TICKERS;
    return TICKERS.filter((t) => t.symbol.toLowerCase().includes(q) || t.name.toLowerCase().includes(q) || t.sector.toLowerCase().includes(q));
  }, [query]);

  // Close when clicking outside.
  useEffect(() => {
    const onDown = (e: MouseEvent) => {
      if (!rootRef.current?.contains(e.target as Node)) {
        setOpen(false);
        setQuery("");
      }
    };
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, []);

  // Press "/" anywhere to focus the search, like the prototype hint.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement;
      const typing = el.tagName === "INPUT" || el.tagName === "TEXTAREA" || el.isContentEditable;
      if (e.key === "/" && !typing) {
        e.preventDefault();
        inputRef.current?.focus();
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, []);

  function choose(symbol: string) {
    setTicker(symbol);
    setOpen(false);
    setQuery("");
    inputRef.current?.blur();
    // Picking a ticker takes you to where the agents run (unless you're already on an analysis page).
    if (pathname !== "/agents" && pathname !== "/company") router.push("/agents");
  }

  function onKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setOpen(true);
      setActive((i) => Math.min(i + 1, results.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setActive((i) => Math.max(i - 1, 0));
    } else if (e.key === "Enter" && open && results[active]) {
      e.preventDefault();
      choose(results[active].symbol);
    } else if (e.key === "Escape") {
      setOpen(false);
      setQuery("");
      inputRef.current?.blur();
    }
  }

  return (
    <div ref={rootRef} className="relative w-full max-w-xl">
      <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />
      <input
        ref={inputRef}
        role="combobox"
        aria-label="Search PSX tickers"
        aria-expanded={open}
        aria-controls={listId}
        aria-autocomplete="list"
        aria-activedescendant={open && results[active] ? `${listId}-${results[active].symbol}` : undefined}
        autoComplete="off"
        value={query}
        placeholder="Search stocks, tickers (e.g. HBL, LUCKY)..."
        onFocus={() => setOpen(true)}
        onChange={(e) => {
          setQuery(e.target.value);
          setActive(0);
          setOpen(true);
        }}
        onKeyDown={onKeyDown}
        className="w-full rounded-xl border border-slate-200 bg-slate-50 py-2 pl-9 pr-10 text-sm outline-none placeholder:text-slate-400 focus:border-blue-500 focus:bg-white focus:ring-2 focus:ring-blue-100"
      />
      {query ? (
        <button
          type="button"
          aria-label="Clear search"
          onMouseDown={(e) => e.preventDefault()}
          onClick={() => {
            setQuery("");
            inputRef.current?.focus();
          }}
          className="absolute right-2 top-1/2 -translate-y-1/2 rounded p-1 text-slate-400 hover:bg-slate-200 hover:text-slate-600"
        >
          <X className="h-4 w-4" />
        </button>
      ) : (
        <kbd className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 rounded bg-slate-200 px-1.5 font-mono text-[11px] text-slate-500">/</kbd>
      )}

      {open && (
        <ul
          id={listId}
          role="listbox"
          className="absolute z-30 mt-2 max-h-80 w-full overflow-auto rounded-xl border border-slate-200 bg-white py-1 shadow-lg"
        >
          {results.length === 0 && <li className="px-4 py-3 text-sm text-slate-500">No matching ticker</li>}
          {results.map((t, i) => (
            <li
              key={t.symbol}
              id={`${listId}-${t.symbol}`}
              role="option"
              aria-selected={t.symbol === ticker}
              onMouseEnter={() => setActive(i)}
              onMouseDown={(e) => {
                e.preventDefault();
                choose(t.symbol);
              }}
              className={`flex cursor-pointer items-center gap-3 px-4 py-2 text-sm ${i === active ? "bg-blue-50" : ""}`}
            >
              <span className="w-16 font-mono font-semibold">{t.symbol}</span>
              <span className="flex-1 truncate text-slate-700">{t.name}</span>
              <span className="hidden text-xs text-slate-400 sm:inline">{t.sector}</span>
              {t.symbol === ticker && <Check className="h-4 w-4 text-blue-600" />}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
