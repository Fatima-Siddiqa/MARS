# Agent 5: Sector & Fundamental Comparison

Point-in-time financial ratios, PSX peer comparison, and a model-implied value per share with SHAP
explanations. Deterministic Python only (no LLM). It never issues BUY/HOLD/AVOID: its `AgentOutput` is
neutral with confidence 0, so the synthesis layer gives it zero weight until you decide how to use it.

All backend code is in `backend/app/agents/fundamentals/` (it replaces the placeholder `agent.py`).
The zip has `new_files/` (copy over the project) and `edited_existing_files/` (8 small edits that link
Agent 5 into MARS; listed at the end).

## Run it

From `backend/` (venv active): `pip install shap scipy yfinance pytest`, then

    uvicorn app.main:app --reload --port 8000

From `frontend/` in a second terminal: `npm run dev`, open **http://localhost:3000/company**, and pick a
company in the header search box. With no data you see "Financial statements are not available" and a
**Gather data** button.

## Data gathering (follows the methodology)

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

### Gather from the app or command line (needs internet on the server)

The page's **Gather data** / **Gather peer data** buttons collect the company and its same-sector peers
and reload the analysis. Same from the command line:

    python -m app.agents.fundamentals.data.collector --tickers HBL UBL --with-sector-peers
    python -m app.agents.fundamentals.data.collector --all-mapped

### Import official filings (preferred)

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

### Limits of the Yahoo fallback

Few recent periods (not enough to train a model), estimated filing dates, sparse bank-specific lines,
no bonus/rights issues, no delisted companies. Not tested against live Yahoo from the build environment
(no internet there): row labels are in `data/collector.py` (`INCOME_MAP` etc.); a label that does not match
shows as a missing line in the collection report rather than being guessed. Run it once and read the report.

## What you see without a trained model

With too little history there is nothing to train on, so the agent serves ratios, peer comparison and the
**transparent peer baseline** (median peer P/B x book equity), labelled `peer_baseline` / "No trained
model". It is never presented as a fitted model and has no prediction interval or SHAP. To train real
models you need years of quarterly statements for dozens of companies (official import), then:

    python -m app.agents.fundamentals.valuation.evaluate     # trains, evaluates, saves models + report

Report: `fundamentals/reports/evaluation_report.{json,md}` (currently `data_unavailable`, because no
statements exist in the repository; nothing is claimed).

### Demo data (to see every feature, including a trained model and SHAP)

    python -m app.agents.fundamentals.demo_data
    # then start the API with the two environment variables it prints

Invented numbers, labelled "(synthetic demo)", written to `data/demo`; not PSX data. Delete when finished.

## Endpoints (all accept `?valuation_date=YYYY-MM-DD`; default = latest price date)

    GET  /agents/fundamentals/{ticker}              shared AgentOutput contract
    GET  /agents/fundamentals/{ticker}/analysis     full analysis (used by the page)
    GET  /agents/fundamentals/{ticker}/metrics | /peers | /valuation | /explanation
    GET  /agents/fundamentals/evaluation            saved evaluation report
    POST /agents/fundamentals/{ticker}/collect      gather statements (company + sector peers)
    GET  /agents/fundamentals/{ticker}/collect      collection job status

## Tests

    # backend (from backend/): existing 18 + 151 new
    pytest

    # frontend (from frontend/): one-time install of the test tooling
    npm i -D vitest@3 jsdom @testing-library/react @testing-library/dom @testing-library/jest-dom
    npx vitest run
    # optional, against a running backend (real HTTP):
    FUNDAMENTALS_LIVE_API=1 NEXT_PUBLIC_API_URL=http://127.0.0.1:8000 npx vitest run

## How it works (short)

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

## Limitations

- No real PSX evaluation exists yet (no statements in the repository, no internet in the build environment).
- No portfolio backtest: it needs survivorship controls (delisted names) the data does not have.
- No neural network: far too few independent PSX companies.
- Net interest margin needs an `earning_assets` line; otherwise it is reported unavailable.

## Edits to existing files (`edited_existing_files/`)

| file | change |
|---|---|
| `backend/app/main.py` | includes the Agent 5 router |
| `backend/pyproject.toml` | adds `scipy`, `shap`, `yfinance`; test path for the new tests |
| `frontend/app/company/page.tsx` | shows the Agent 5 panel under the existing placeholder |
| `frontend/lib/api.ts`, `lib/types.ts`, `components/agents/AgentCard.tsx`, `AgentDetailModal.tsx`, `AgentGrid.tsx` | the Specialist Agents card for fundamentals is live (never mocked) and shows "No directional call" with data coverage instead of a signal/confidence |
