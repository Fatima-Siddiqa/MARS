"use client";

import { useEffect, useRef } from "react";
import {
  CandlestickSeries,
  ColorType,
  HistogramSeries,
  LineSeries,
  LineStyle,
  createChart,
  type Time,
} from "lightweight-charts";
import type { ChartPoint } from "@/lib/types";

const UP = "#059669";
const DOWN = "#e11d48";
const LEGEND = [
  { label: "SMA 20", color: "#2563eb" },
  { label: "SMA 50", color: "#f59e0b" },
  { label: "SMA 200", color: "#7c3aed" },
];

type P<K extends keyof ChartPoint> = { time: Time; value: NonNullable<ChartPoint[K]> };
function series<K extends keyof ChartPoint>(data: ChartPoint[], key: K): P<K>[] {
  return data.filter((d) => d[key] != null).map((d) => ({ time: d.time as Time, value: d[key] as NonNullable<ChartPoint[K]> }));
}

/** Candlesticks + SMA20/50/200 + volume, with RSI and MACD histogram in synced panes below. */
export default function TechnicalChart({ data }: { data: ChartPoint[] }) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el || data.length === 0) return;
    const chart = createChart(el, {
      autoSize: true,
      layout: { background: { type: ColorType.Solid, color: "#ffffff" }, textColor: "#475569", fontFamily: "inherit" },
      grid: { vertLines: { color: "#f1f5f9" }, horzLines: { color: "#f1f5f9" } },
      rightPriceScale: { borderColor: "#e2e8f0" },
      timeScale: { borderColor: "#e2e8f0" },
    });

    const candles = chart.addSeries(CandlestickSeries, {
      upColor: UP, downColor: DOWN, borderUpColor: UP, borderDownColor: DOWN, wickUpColor: UP, wickDownColor: DOWN,
    });
    candles.setData(
      data
        .filter((d) => d.open != null && d.high != null && d.low != null && d.close != null)
        .map((d) => ({ time: d.time as Time, open: d.open!, high: d.high!, low: d.low!, close: d.close! })),
    );
    chart.addSeries(LineSeries, { color: LEGEND[0].color, lineWidth: 1, priceLineVisible: false, lastValueVisible: false }).setData(series(data, "sma20"));
    chart.addSeries(LineSeries, { color: LEGEND[1].color, lineWidth: 2, priceLineVisible: false, lastValueVisible: false }).setData(series(data, "sma50"));
    chart.addSeries(LineSeries, { color: LEGEND[2].color, lineWidth: 2, priceLineVisible: false, lastValueVisible: false }).setData(series(data, "sma200"));

    const vol = chart.addSeries(HistogramSeries, { priceScaleId: "volume", priceFormat: { type: "volume" }, priceLineVisible: false, lastValueVisible: false });
    vol.priceScale().applyOptions({ scaleMargins: { top: 0.8, bottom: 0 } });
    vol.setData(
      data.map((d) => ({
        time: d.time as Time,
        value: d.volume,
        color: (d.close ?? 0) >= (d.open ?? 0) ? "rgba(5,150,105,0.35)" : "rgba(225,29,72,0.35)",
      })),
    );

    // Pane 1: RSI(14) with 30/70 guides
    const rsi = chart.addSeries(LineSeries, { color: "#0ea5e9", lineWidth: 2, priceLineVisible: false }, 1);
    rsi.setData(series(data, "rsi14"));
    for (const price of [70, 30]) {
      rsi.createPriceLine({ price, color: "#94a3b8", lineWidth: 1, lineStyle: LineStyle.Dashed, axisLabelVisible: true, title: "" });
    }

    // Pane 2: MACD histogram
    const macd = chart.addSeries(HistogramSeries, { priceLineVisible: false }, 2);
    macd.setData(series(data, "macd_hist").map((p) => ({ ...p, color: p.value >= 0 ? UP : DOWN })));

    const panes = chart.panes();
    panes[0]?.setStretchFactor(3);
    panes[1]?.setStretchFactor(1);
    panes[2]?.setStretchFactor(1);
    chart.timeScale().fitContent();

    return () => chart.remove();
  }, [data]);

  if (data.length === 0) {
    return <div className="p-10 text-center text-sm text-slate-500">No chart data returned.</div>;
  }
  return (
    <div>
      <div className="mb-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-slate-500">
        <span className="font-semibold text-slate-700">Price · Volume</span>
        {LEGEND.map((l) => (
          <span key={l.label} className="flex items-center gap-1">
            <span className="inline-block h-0.5 w-4" style={{ background: l.color }} /> {l.label}
          </span>
        ))}
        <span className="ml-auto">Lower panes: RSI (14) with 30/70 guides · MACD histogram</span>
      </div>
      <div ref={ref} className="h-[560px] w-full" />
    </div>
  );
}
