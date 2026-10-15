export type Ticker = { symbol: string; name: string; sector: string };

// Hardcoded for now. Replace with GET /tickers once the backend exists.
export const TICKERS: Ticker[] = [
  { symbol: "ENGROH", name: "Engro Holdings", sector: "Fertiliser / Conglomerate" },
  { symbol: "HBL", name: "Habib Bank Limited", sector: "Banking" },
  { symbol: "HUBC", name: "Hub Power Company", sector: "Power" },
  { symbol: "LUCKY", name: "Lucky Cement", sector: "Cement" },
  { symbol: "OGDCL", name: "Oil & Gas Development Company", sector: "Oil & Gas" },
  { symbol: "PPL", name: "Pakistan Petroleum Limited", sector: "Oil & Gas" },
  { symbol: "PTCL", name: "Pakistan Telecommunication Company", sector: "Telecom" },
  { symbol: "SEARL", name: "The Searle Company", sector: "Pharma" },
  { symbol: "SYSTEMS", name: "Systems Limited", sector: "Technology" },
  { symbol: "UBL", name: "United Bank Limited", sector: "Banking" },
];

export function findTicker(symbol: string | null): Ticker | undefined {
  return TICKERS.find((t) => t.symbol === symbol);
}
