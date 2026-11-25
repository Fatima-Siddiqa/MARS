# Fundamentals agent: official data sources

All data used by the Sector & Fundamentals agent comes from the Pakistan Stock Exchange (PSX).
No third-party (e.g. Yahoo) data is used. Everything below lives in `data/raw/` (git-ignored).

## Sources

| Data | PSX source | How it is collected | Stored in |
|---|---|---|---|
| Daily prices (OHLCV) | PSX Data Portal, daily market summary `dps.psx.com.pk/download/mkt_summary/YYYY-MM-DD.Z` | `python -m app.ingestion.psx_eod` | `data/raw/psx_eod/symbols/{SYMBOL}.csv` (listed companies only) |
| Annual + quarterly revenue / mark-up, total income, profit after tax, EPS | PSX company page, Financials tab `dps.psx.com.pk/company/{SYMBOL}` | `python -m app.agents.fundamentals.data.psx_official` | `data/raw/fundamentals/official/psx_statements.csv` |
| Shares outstanding, face value | PSX company page, Equity Profile | same | `.../psx_shares.csv` |
| Filing dates (publication dates) | PSX company page Announcements + PSX Financials portal `financials.psx.com.pk` | same | `.../psx_filings.csv` |
| Balance sheet, year-to-date income lines | Quarterly / results PDFs filed on PSX | keyed by hand from the PDF, with document id + page | `.../keyed_statements.csv`, PDFs in `.../pdfs/` |

## Rules

- **Standalone (unconsolidated) statements** throughout. The PSX company-page figures are
  standalone, so equity, profit and ratios all use the same basis.
- **Point in time.** Every figure carries `published_at`, the date of the PSX filing that first
  made it public (results announcement, else report transmission, else Financials-portal posting
  date). If no filing for that period exists on PSX, the date of the first PSX filing for a *later*
  period is used: accounts are filed in order, so this can only make a figure visible later than it
  really was, never earlier (labelled "upper bound" in `published_via`). Otherwise it is not used.
- **Restatements never overwrite.** A changed figure is stored as a new version, dated the day it
  was first observed.
- **Missing is not zero.** A "-" on PSX stays missing.
- **Units are kept as printed** (`PKR` or `PKR_thousands`); conversion happens in the agent.
- **Shares for per-share values** come from the filing (share capital / face value), because the
  PSX share count can lag (ENGROH buyback in 2026) or exclude unlisted classes (PTCL Class B).

## Checks

`python -m app.agents.fundamentals.data.validate_official` must report `"ok": true` before the
agent uses new figures. It checks:

1. Assets = liabilities + equity, exactly, for every keyed balance sheet.
2. Every value keyed from a text PDF appears, as printed, on the cited page.
3. Year-to-date profit in the filing = sum of the PSX company-page quarters. A mismatch is reported
   as a source conflict and the filed report is used.
4. Filing-implied share count vs PSX share count (differences reported).
5. No figure is published before its period ends.
6. `needs_keying`: companies with a PSX filing newer than the keyed balance sheet.

### Status on 2026-11-26

- 14 companies, balance sheets as at 2026-09-30 (filed 21-30 Oct 2026), 148 keyed values.
- All balance identities hold; 67 values from text PDFs matched on the cited page; the 81 values
  from scanned PDFs were read from page images and are covered by the balance identity check.
- Known conflict: PTCL Q3 2026 profit on the PSX company page (-1,403,639) disagrees with the filed
  report (-671,157). The company page's own EPS (-0.13) matches the report, and a PTCL results
  announcement for that quarter was revoked, so the report figure is used.
- Dated by the upper-bound rule (no own filing on PSX): HBL annual figures 2022-2025, PPL Q1
  FY2026, SYS Q1 2026, ENGROH 2022-2023.
- Check of the data layer: HBL trailing P/E on 2026-11-26 computes to 6.68, equal to the
  "P/E Ratio (TTM)" shown on the PSX company page.

## Refreshing (automatic)

The Windows scheduled task "MARS fundamentals refresh" runs every day at 19:00:

```bash
cd backend
python -m app.agents.fundamentals.pipeline          # what the task runs; --full re-collects all of PSX
```

It refreshes prices, the reference companies' PSX pages (the whole market weekly), reads any new
filings automatically, validates everything, and retrains/recalibrates the model when new annual
figures arrive. Each run appends a line to `data/processed/fundamentals/refresh_log.jsonl`.

**Automatic reading of new filings** (`data/auto_extract.py`): text PDFs are read directly, scanned
PDFs with Tesseract OCR. A figure is only added to `keyed_statements.csv` (extraction `auto_text` /
`auto_ocr`) if it passes arithmetic checks: assets = liabilities + equity on independently read
totals, equity = sum of its components, share capital consistent with the PSX share count, and
year-to-date profit / revenue = sum of the PSX company-page quarters. Anything else goes to
`review_queue.csv` with the reason; key those by hand (with `doc_id` and `doc_page`) and re-run
`python -m app.agents.fundamentals.data.validate_official`.

Accuracy check against the hand-keyed values: `python -m app.agents.fundamentals.data.auto_extract --check`.
