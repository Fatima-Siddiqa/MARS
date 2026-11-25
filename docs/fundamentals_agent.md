# Agent 5: Sector & Fundamental Comparison

## As built (November 2026)

The agent turns a PSX ticker into a **BUY / HOLD / AVOID** call with a confidence score, a model
fair value per share, financial ratios compared with PSX sector peers, and a cited source for every
figure. All data is **official PSX data**; no LLM and no third-party data are used. It is decision
support only. The original design specification is kept unchanged at the end of this document.

### Pipeline

```
official PSX data -> point-in-time snapshot -> ratios -> PSX sector peers
                  -> valuation model (trained on all PSX companies) -> calibrated call -> AgentOutput
```

| Step | Code (`backend/app/agents/fundamentals/`) | What it does |
|---|---|---|
| Data | `data/psx_official.py`, `data/auto_extract.py`, `data/validate_official.py` | PSX company pages, filing PDFs and dates; automatic reading of new filings; checks. Details: `docs/fundamentals_data.md` |
| Prices | `app/ingestion/psx_eod.py` | PSX daily market summaries, one file per listed company |
| Snapshot | `data/loader.py` | Only facts published by the valuation date; filed report beats company page; PKR millions; trailing twelve months |
| Ratios | `features.py` | Bank and non-bank ratio sets; each ratio has its formula and sources, or a reason when missing |
| Peers | `peers.py` | Same PSX sector code + company type, priced on the date (all of PSX, ~445 companies) |
| Model | `valuation/` | Training set, walk-forward training, calibration, prediction with SHAP |
| Agent / API | `agent.py`, `api.py` | Call, confidence, evidence; `/agents/fundamentals/{ticker}`, `/{ticker}/analysis`, `/model` |
| Refresh | `pipeline.py`, `backend/scripts/refresh_fundamentals.cmd` | Daily automatic refresh |

### The trained model

- **Training data:** 1,426 company-years from 401 PSX companies (fiscal years 2022-2026), built
  only from the PSX company pages (annual revenue / total income, profit after tax, EPS) and PSX
  daily prices. One sample per company per published fiscal year, observed on the first trading day
  after PSX published the figures.
- **Target:** log(market cap) minus the market's median log P/E on that day
  (`valuation/market_level.py`). The model therefore learns how a company's fundamentals translate
  into value *relative to the market's typical multiple*, so the PSX re-rating of 2024-2026 (median
  P/E 5.9 in mid-2023, 11.3 in mid-2025) does not appear as a year bias.
- **Features:** revenue and profit levels, losses, net margin, revenue and profit growth, bank flag,
  PSX sector. No price-based feature, so the model cannot copy the answer.
- **Candidates:** sector median P/E baseline (no fitting), Ridge regression, gradient-boosted trees.
- **Evaluation:** walk-forward by year (each test year is predicted by a model trained only on
  earlier years). The simplest candidate within 5% of the best test error is chosen, and a fitted
  model must also beat the baseline.

| candidate | test company-years | MAE (log) | median abs. error | R² |
|---|---|---|---|---|
| sector P/E baseline | 1,133 | 0.934 | 81% | 0.53 |
| Ridge | 1,133 | 0.665 | 69% | 0.81 |
| **gradient-boosted trees (chosen)** | 1,133 | **0.590** | **60%** | **0.85** |

The trees beat the baseline in all three test years (2024, 2025, 2026). Explanations use SHAP
(effect of each input versus a typical PSX company). The 80% range shown with each fair value comes
from the model's real test errors and is wide (roughly 0.4x to 2.6x the point estimate): the model
gets the cross-section right but single-company valuations remain uncertain.

### BUY / HOLD / AVOID

`gap = model fair value / price - 1`. With threshold `T`:
**BUY** if gap ≥ T, **AVOID** if gap ≤ 1/(1+T) - 1 (same distance in log terms), otherwise **HOLD**.

`T` is calibrated (`valuation/calibrate.py`) on the out-of-sample predictions: for 850 test
company-years, the return over the next 126 trading days (about six months) minus the median return
of all listed PSX companies over the same days.

| valuation-gap tercile | n | mean 6-month excess return | hit rate |
|---|---|---|---|
| low gap (expensive) | 284 | -2.5% | 44% |
| middle | 283 | +0.7% | 43% |
| high gap (cheap) | 283 | +6.3% | 52% |

Rank correlation between the gap and later excess return: **+0.11 (p = 0.001)**. The threshold with
the widest BUY-AVOID spread is **T = 50%** (BUY +6.7%, AVOID -2.9%, spread +9.6% over six months).
Limitations: PSX closes are not dividend-adjusted; windows with a one-day fall over 40% (probable
bonus issue or split) are excluded; 50% is the top of the tested threshold grid.

### Confidence

`confidence = ratio coverage × model fit × evidence × peer strength × margin`, capped at 0.95:

- ratio coverage: share of the company's ratios that could be computed;
- model fit: exp(-test MAE) of the chosen model (0.3 when only the peer baseline is available);
- evidence: 1.0 if the calibration found a significant link to later returns, otherwise 0.5;
- peer strength: 1.0 (3 or more peers), 0.8 (1-2), 0.6 (none);
- margin: BUY/AVOID 0.5 at the threshold rising to 1.0 at twice the threshold; HOLD 1.0 at a zero
  gap falling to 0.5 at the threshold.

### Output

The shared `AgentOutput`: `stance` bullish / neutral / bearish = BUY / HOLD / AVOID, `score` in
[-1, 1], `confidence`, summary, key findings (ratios vs peer medians, top model drivers), risks,
evidence (price, equity, trailing profit, model, peers, each with its PSX source), and `metrics`
(`call`, `gap`, valuation with range and SHAP, ratios, peers, calibration, data dates). The shared
contract is unchanged: the call is in `metrics["call"]`.

### Running it

```bash
cd backend
python -m app.agents.fundamentals.pipeline              # daily refresh (prices, filings, checks, model)
python -m app.agents.fundamentals.valuation.train       # retrain + write the test report
python -m app.agents.fundamentals.valuation.calibrate   # recalibrate BUY / HOLD / AVOID
uvicorn app.main:app --reload                           # GET /agents/fundamentals/HBL
```

Reports: `data/processed/fundamentals/models/valuation_report.md` and `call_calibration.md`.
Frontend: the Specialist Agents card shows the call; the Company page has the full fundamentals
panel.

### Automation

- Windows Task Scheduler task **"MARS fundamentals refresh"** runs
  `backend/scripts/refresh_fundamentals.cmd` daily at 19:00 (after PSX publishes the day's file):
  prices, company pages, the whole market weekly, automatic reading of new filings, validation, and
  retraining + recalibration when new annual figures arrive or the model is older than 30 days.
  Log: `data/processed/fundamentals/refresh_log.jsonl`. Remove with
  `schtasks /Delete /TN "MARS fundamentals refresh" /F`.
- New filings are read automatically (text PDFs directly, scanned PDFs with Tesseract OCR, installed
  at `C:\Program Files\Tesseract-OCR`). A value is used only if it passes arithmetic checks
  (assets = liabilities + equity on independently read totals, equity = its components, share
  capital vs PSX share count, year-to-date profit = PSX quarters). Everything else goes to
  `data/raw/fundamentals/official/review_queue.csv` for a person to key.

### Limitations

- The model uses page-level fundamentals (no balance sheet) because PSX pages publish only income
  lines for all companies; balance sheets are keyed only for the 14 reference companies.
- Single-company fair values are uncertain (median error 60%); the call should be read with its
  confidence and range.
- Historical share counts are profit / reported EPS (PSX shows only today's count).
- No portfolio backtest with transaction costs or survivorship controls.

---

# Original design specification (kept for reference)

## Agent 5: Sector & Fundamental Comparison

Point-in-time financial ratios, PSX peer comparison, and a model-implied value per share with SHAP
explanations. Deterministic Python only (no LLM). It never issues BUY/HOLD/AVOID: its `AgentOutput` is
neutral with confidence 0, so the synthesis layer gives it zero weight until you decide how to use it.

All backend code is in `backend/app/agents/fundamentals/` (it replaces the placeholder `agent.py`).
The zip has `new_files/` (copy over the project) and `edited_existing_files/` (8 small edits that link
Agent 5 into MARS; listed at the end).

### Run it

From `backend/` (venv active): `pip install shap scipy yfinance pytest`, then

    uvicorn app.main:app --reload --port 8000

From `frontend/` in a second terminal: `npm run dev`, open **http://localhost:3000/company**, and pick a
company in the header search box. With no data you see "Financial statements are not available" and a
**Gather data** button.

### Data gathering (follows the methodology)

| Guideline | How it is handled |
|---|---|
| Prefer official PSX filings | **Official import** (below) is the preferred path; Yahoo is a convenience fallback. |
| `published_at` must be the real publication date | Official rows must carry it (import is strict). Yahoo rows get an **estimated** date (period end + 60 days quarterly / 90 annual), flagged in the data, in `data_coverage.estimated_publication_dates` and in a warning on every result. Official rows **supersede** estimated rows for the same fact. |
| Restatements never overwrite history | A changed figure is stored as a **new version** (`is_restated=True`, dated the day first observed, or its own filing date for official rows). Earlier valuation dates keep seeing the original number. |
| Source URL / identifier, retrieval timestamp | Every row has `source`, `source_ref` (document URL or id) and `retrieved_at`. |
| PSX sector and industry | Sector from `data/ticker_map.csv`; **industry, official listing and delisting dates** come from an optional `universe.csv` (below). |
| Shares, corporate actions | Yahoo share counts and splits only. Bonus/rights issues need official `corporate_actions.csv`; without it the split/bonus consistency check cannot catch them (reported on every collection). |
| Missing values are not zero | Missing lines stay missing and are listed in the collection report. |
| Units normalised, originals kept | Everything is converted to PKR millions; the reported value and unit are kept. |

#### Gather from the app or command line (needs internet on the server)

The page's **Gather data** / **Gather peer data** buttons collect the company and its same-sector peers
and reload the analysis. Same from the command line:

    python -m app.agents.fundamentals.data.collector --tickers HBL UBL --with-sector-peers
    python -m app.agents.fundamentals.data.collector --all-mapped

#### Import official filings (preferred)

Key statements from PSX filings into a CSV with the columns below and import it (strict: rows without a
real `published_at` or a `source_ref`, or marked "estimated", are **rejected and counted**, never defaulted):

    python -m app.agents.fundamentals.data.collector --import-statements official_filings.csv

`ticker, line_item, value, unit, period_start, period_end, published_at, source, source_ref`
(optional: `period_kind, retrieved_at, is_restated, version`). Cumulative year-to-date figures are fine.
Valid `line_item` names and units: `data/normalization.py`. Prices come from the existing price files.

Other files in `data/raw/fundamentals/` (git-ignored) that you can provide yourself:

| file | columns |
|---|---|
| `universe.csv` | symbol, company_name, sector, industry, yahoo_symbol, listed_from, delisted_on. Adds PSX industry, official listing/**delisting** dates and extra peers beyond `ticker_map.csv`. |
| `corporate_actions.csv` | ticker, ex_date, action (`split`/`bonus`/`rights`/`dividend`), share_multiplier, source |
| `shares.csv` | ticker, as_of, shares_outstanding, unit, source, source_ref |

#### Limits of the Yahoo fallback

Few recent periods (not enough to train a model), estimated filing dates, sparse bank-specific lines,
no bonus/rights issues, no delisted companies. Not tested against live Yahoo from the build environment
(no internet there): row labels are in `data/collector.py` (`INCOME_MAP` etc.); a label that does not match
shows as a missing line in the collection report rather than being guessed. Run it once and read the report.

### What you see without a trained model

With too little history there is nothing to train on, so the agent serves ratios, peer comparison and the
**transparent peer baseline** (median peer P/B x book equity), labelled `peer_baseline` / "No trained
model". It is never presented as a fitted model and has no prediction interval or SHAP. To train real
models you need years of quarterly statements for dozens of companies (official import), then:

    python -m app.agents.fundamentals.valuation.evaluate     # trains, evaluates, saves models + report

Report: `fundamentals/reports/evaluation_report.{json,md}` (currently `data_unavailable`, because no
statements exist in the repository; nothing is claimed).

#### Demo data (to see every feature, including a trained model and SHAP)

    python -m app.agents.fundamentals.demo_data
    # then start the API with the two environment variables it prints

Invented numbers, labelled "(synthetic demo)", written to `data/demo`; not PSX data. Delete when finished.

### Endpoints (all accept `?valuation_date=YYYY-MM-DD`; default = latest price date)

    GET  /agents/fundamentals/{ticker}              shared AgentOutput contract
    GET  /agents/fundamentals/{ticker}/analysis     full analysis (used by the page)
    GET  /agents/fundamentals/{ticker}/metrics | /peers | /valuation | /explanation
    GET  /agents/fundamentals/evaluation            saved evaluation report
    POST /agents/fundamentals/{ticker}/collect      gather statements (company + sector peers)
    GET  /agents/fundamentals/{ticker}/collect      collection job status

### Tests

    # backend (from backend/): existing 18 + 151 new
    pytest

    # frontend (from frontend/): one-time install of the test tooling
    npm i -D vitest@3 jsdom @testing-library/react @testing-library/dom @testing-library/jest-dom
    npx vitest run
    # optional, against a running backend (real HTTP):
    FUNDAMENTALS_LIVE_API=1 NEXT_PUBLIC_API_URL=http://127.0.0.1:8000 npx vitest run

### How it works (short)

1. Statements are filtered to `published_at <= valuation date`; restatements are separate versions.
2. Cumulative YTD figures become discrete quarters, then TTM. Ratios are null (with a reason) when
   inputs are missing or denominators are zero/negative. Banks and non-banks use separate pipelines.
3. Peers: same industry, else sector, same company type, listed on the date, PSX only; the target is
   excluded from its own median.
4. Model: predicts log market cap from fundamentals only (no price, multiples, technicals, sentiment,
   macro). Walk-forward selection; the latest 20% of dates is an untouched test block. Candidates: peer
   P/B baseline, Ridge, gradient-boosted trees; the simplest competitive one wins and replaces the
   baseline only if it also beats it on the test block.
5. Value per share = exp(prediction) / shares outstanding; gap = (implied - price) / price.

### Limitations

- No real PSX evaluation exists yet (no statements in the repository, no internet in the build environment).
- No portfolio backtest: it needs survivorship controls (delisted names) the data does not have.
- No neural network: far too few independent PSX companies.
- Net interest margin needs an `earning_assets` line; otherwise it is reported unavailable.

### Edits to existing files (`edited_existing_files/`)

| file | change |
|---|---|
| `backend/app/main.py` | includes the Agent 5 router |
| `backend/pyproject.toml` | adds `scipy`, `shap`, `yfinance`; test path for the new tests |
| `frontend/app/company/page.tsx` | shows the Agent 5 panel under the existing placeholder |
| `frontend/lib/api.ts`, `lib/types.ts`, `components/agents/AgentCard.tsx`, `AgentDetailModal.tsx`, `AgentGrid.tsx` | the Specialist Agents card for fundamentals is live (never mocked) and shows "No directional call" with data coverage instead of a signal/confidence |
