import { findTicker } from "./tickers";
import type { AgentId, AgentOutput, Evidence, Indicator, Signal } from "@/lib/types";

function hash(s: string): number {
  let h = 0;
  for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) >>> 0;
  return h;
}

function toSignal(score: number): Signal {
  if (score > 0.2) return "bullish";
  if (score < -0.2) return "bearish";
  return "neutral";
}

type Detail = { summary: string; rationale: string; indicators: Indicator[]; evidence: Evidence[] };

function detail(agent: AgentId, name: string): Detail {
  switch (agent) {
    case "technical":
      return {
        summary: `Mock: ${name} trades above its 50-day average with RSI in neutral territory.`,
        rationale: "Mock rationale: price is holding above both the 50-day and 200-day averages with no bearish divergence on the daily RSI.",
        indicators: [
          { label: "RSI (14)", value: "62.4 (Neutral)" },
          { label: "MACD (12, 26, 9)", value: "1.85 (Bullish)" },
          { label: "SMA 50", value: "PKR 198.40" },
        ],
        evidence: [
          { title: "PSX daily closing prices, last 250 sessions", source: "PSX price files" },
          { title: "50-day and 200-day moving average crossover", source: "Computed from PSX prices", date: "2026-10-02" },
        ],
      };
    case "sentiment":
      return {
        summary: `Mock: recent coverage of ${name} is mixed to mildly positive.`,
        rationale: "Mock rationale: most recent headlines mention earnings and dividends, with few negative stories in the last 30 days.",
        indicators: [
          { label: "Articles scored", value: "24" },
          { label: "Avg. sentiment", value: "+0.31" },
          { label: "Positive / Negative", value: "15 / 4" },
        ],
        evidence: [
          { title: `Mock headline about ${name} earnings`, source: "Business Recorder", url: "https://www.brecorder.com/", date: "2026-10-03" },
          { title: `Mock headline about ${name} sector outlook`, source: "Dawn", url: "https://www.dawn.com/business", date: "2026-10-02" },
          { title: `Mock headline about ${name} dividend`, source: "Express Tribune", url: "https://tribune.com.pk/business", date: "2026-10-01" },
        ],
      };
    case "macro":
      return {
        summary: "Mock: policy rate easing and stable inflation are mildly supportive.",
        rationale: "Mock rationale: lower interest rates reduce gross spreads on government securities, while credit demand is expected to recover.",
        indicators: [
          { label: "SBP Policy Rate", value: "11.00% (Neutral)" },
          { label: "CPI Inflation", value: "6.2% YoY" },
          { label: "PKR / USD", value: "Stable, 30d" },
        ],
        evidence: [
          { title: "SBP Monetary Policy Committee statement", source: "SBP EasyData", date: "2026-09-15" },
          { title: "Monthly CPI inflation series", source: "World Bank API", date: "2026-09-30" },
        ],
      };
    case "fx":
      return {
        summary: `Mock: ${name} shows low sensitivity to PKR moves.`,
        rationale: "Mock rationale: rolling beta of returns against PKR/USD is close to zero.",
        indicators: [
          { label: "PKR beta", value: "0.12" },
          { label: "Exposure", value: "Neutral" },
          { label: "Window", value: "250d" },
        ],
        evidence: [{ title: "PKR/USD history", source: "SBP EasyData" }],
      };
    case "fundamentals":
      return {
        summary: `Mock: ${name} trades at a discount to its sector peers.`,
        rationale: "Mock rationale: P/E and P/B sit in the lower quartile of the peer group while ROE is above the median.",
        indicators: [
          { label: "P/E", value: "4.8x" },
          { label: "P/B", value: "1.35x" },
          { label: "ROE", value: "44.2%" },
        ],
        evidence: [{ title: "Audited financial statements and ratios", source: "Financial statement files" }],
      };
    case "forensics":
      return {
        summary: `Mock: no major accounting red flags found for ${name}.`,
        rationale: "Mock rationale: accruals and receivables grow in line with revenue.",
        indicators: [
          { label: "Red flags", value: "0 high" },
          { label: "Beneish M-score", value: "-2.4" },
          { label: "Accruals", value: "Normal" },
        ],
        evidence: [{ title: "Balance sheet, line item: trade receivables", source: "Financial statement files", date: "FY25" }],
      };
  }
}

export function mockOutput(agent: AgentId, ticker: string): AgentOutput {
  const h = hash(`${agent}:${ticker}`);
  const score = Math.round(((h % 200) - 100)) / 100; // -1.00 .. 0.99
  const confidence = Math.round(50 + ((h >>> 3) % 45)) / 100; // 0.50 .. 0.94
  const d = detail(agent, findTicker(ticker)?.name ?? ticker);
  return { agent, ticker, score, confidence, signal: toSignal(score), generated_at: new Date().toISOString(), ...d };
}
