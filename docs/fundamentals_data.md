# Fundamentals agent: official data sources

All data used by the Sector & Fundamentals agent comes from the Pakistan Stock Exchange (PSX).
No third-party (e.g. Yahoo) data is used. Everything below lives in `data/raw/` (git-ignored).

## Sources

| Data | PSX source | How it is collected | Stored in |
|---|---|---|---|
| Daily prices (OHLCV) | PSX Data Portal, daily market summary `dps.psx.com.pk/download/mkt_summary/YYYY-MM-DD.Z` | `python -m app.ingestion.psx_eod` | `data/raw/psx_eod/` |
| Annual + quarterly revenue / mark-up, total income, profit after tax, EPS | PSX company page, Financials tab `dps.psx.com.pk/company/{SYMBOL}` | `python -m app.agents.fundamentals.data.psx_official` | `data/raw/fundamentals/official/psx_statements.csv` |
| Shares outstanding, face value | PSX company page, Equity Profile | same | `.../psx_shares.csv` |
| Filing dates (publication dates) | PSX company page Announcements + PSX Financials portal `financials.psx.com.pk` | same | `.../psx_filings.csv` |
| Balance sheet, year-to-date income lines | Quarterly / results PDFs filed on PSX | keyed by hand from the PDF, with document id + page | `.../keyed_statements.csv`, PDFs in `.../pdfs/` |

## Rules

- **Standalone (unconsolidated) statements** throughout. The PSX company-page figures are
  standalone, so equity, profit and ratios all use the same basis.
- **Point in time.** Every figure carries `published_at`, the date of the PSX filing that first
  made it public (results announcement, else report transmission, else Financials-portal posting
  date). A figure whose filing date cannot be found on PSX is not used.
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
- Gaps (no PSX publication date found, so not used): HBL annual figures 2022-2025, PPL Q1 FY2026,
  SYS Q1 2026, ENGROH 2022-2023.

## Refreshing

```bash
cd backend
python -m app.ingestion.psx_eod                              # prices up to the latest trading day
python -m app.agents.fundamentals.data.psx_official          # company-page figures, shares, filings
python -m app.agents.fundamentals.data.validate_official     # checks + which filings need keying
```

When `needs_keying` lists a company, download its new report from PSX into
`data/raw/fundamentals/official/pdfs/`, key the balance-sheet and year-to-date lines into
`keyed_statements.csv` (with `doc_id` and `doc_page`), and re-run the validator.
