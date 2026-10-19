# Historical Price & Technical Agent (Level 2)

Owner: Fatima. Code: `backend/app/agents/technical/`. Data: `backend/app/ingestion/prices.py`.

## Data setup (raw data is gitignored)
Copy the 10 xlsx files from the dataset zip (`Pakistan/Stocks Data/`) into `data/raw/stocks/` and the
KSE-100 csv to `data/raw/KSE_100_index.csv`. Symbols: HBL HUBC OGDC UBL PPL PTCL SEARL(SERL.xlsx)
LUCK(Lucky Cement.xlsx) SYS(Systems Limited.xlsx) ENGROH. Env overrides: `MARS_PRICE_DIR`, `MARS_BENCHMARK_PATH`.

## Pipeline
clean (dashes, Adj.Close, O/H/L scaling) -> indicators -> 14 oriented scores (robust expanding z,
clip +-3, /3; RSI zone and vol percentile mapped directly) -> 4 family means (trend, momentum,
relative strength, risk) -> equal-weight score -> stance (+-0.20) -> confidence = agreement x
quality (history x activity x freshness) -> template explanation + evidence.
Liquidity (log volume ratio, log Amihud, no-trade share) only feeds quality / model inputs.

## Optional model
`python -m scripts.technical_eval` runs walk-forward (purge + 20d embargo) comparing Level 1, Level 2
rules, ridge logistic and shallow GBT; writes `data/processed/technical/evaluation_report.json`.
The model is used by the agent only if `adopted=true` (>=3 test years better than Level 1 and
bootstrap P>=0.90). Current result on the 10-stock dataset: NOT adopted, rules ship.

## Known deviations from the methodology doc
- z-score warm-up is 252 observations (not 36 months) so training can start ~2018.
- Sklearn `HistGradientBoostingClassifier` has no row subsampling (0.8 not applied).
- Business-day count approximates trading days for the freshness check.
