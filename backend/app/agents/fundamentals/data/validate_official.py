"""Checks on the official fundamentals before the agent may use them.

Inputs (all under data/raw/fundamentals/official/):
- keyed_statements.csv: figures keyed from PSX filing PDFs (balance sheet + year-to-date lines),
  each with its PSX document id and page.
- psx_statements.csv / psx_shares.csv: collected by psx_official.py from PSX company pages.
- pdfs/{TICKER}_{doc_id}.pdf: the cached filings.

Checks (any failure is reported; nothing is "fixed" silently):
1. Balance identity: total_assets == total_liabilities + total_equity (exact).
2. Text PDFs: every keyed value appears, formatted as printed, on the cited page.
3. Cross-source: year-to-date net income from the filing == sum of the PSX company-page
   quarters for the same fiscal year. A mismatch is a source conflict: it is reported and the
   filed report wins (e.g. PTCL Q3 2026: company page -1,403,639 vs filed report -671,157,
   whose 9-month total matches Q1 + Q2 + the filed Q3).
4. Shares: share_capital / face value vs PSX "Shares"; differences above 0.5% are reported
   (e.g. a buyback not yet reflected on PSX, or unlisted share classes).
5. published_at is on or after period_end.

    python -m app.agents.fundamentals.data.validate_official
"""
from __future__ import annotations

import subprocess
from datetime import date

import pandas as pd

from app.agents.fundamentals.data.psx_official import (
    OFFICIAL_DIR,
    SHARES_CSV,
    STATEMENTS_CSV,
    load_companies,
)

KEYED_CSV = OFFICIAL_DIR / "keyed_statements.csv"
PDF_DIR = OFFICIAL_DIR / "pdfs"
DOC_URL = "https://dps.psx.com.pk/download/document/{doc_id}.pdf"
SHARE_TOLERANCE = 0.005


def load_keyed() -> pd.DataFrame:
    k = pd.read_csv(KEYED_CSV, dtype={"doc_id": str})
    k["source_ref"] = k["doc_id"].map(lambda d: DOC_URL.format(doc_id=d)) + "#page=" + \
        k["doc_page"].astype(str)
    return k


def _printed(v: float) -> str:
    s = f"{abs(int(v)):,}"
    return f"({s})" if v < 0 else s


def _page_text(ticker: str, doc_id: str, page: int) -> str | None:
    psx_sym = {c.ticker: c.psx_symbol for c in load_companies()}.get(ticker, ticker)
    pdf = PDF_DIR / f"{psx_sym}_{doc_id}.pdf"
    if not pdf.exists():
        return None
    out = subprocess.run(["pdftotext", "-raw", "-f", str(page), "-l", str(page), str(pdf), "-"],
                         capture_output=True, text=True, errors="ignore")
    return out.stdout


def check_balance_identity(k: pd.DataFrame) -> list[str]:
    errs = []
    inst = k[k["period_kind"] == "instant"]
    for (t, pe), g in inst.groupby(["ticker", "period_end"]):
        v = dict(zip(g["line_item"], g["value"]))
        if {"total_assets", "total_liabilities", "total_equity"} <= v.keys():
            diff = v["total_assets"] - v["total_liabilities"] - v["total_equity"]
            if diff != 0:
                errs.append(f"{t} {pe}: assets - liabilities - equity = {diff:,}")
        else:
            errs.append(f"{t} {pe}: missing one of total_assets/total_liabilities/total_equity")
    return errs


def check_text_pdfs(k: pd.DataFrame) -> tuple[list[str], int]:
    errs, checked = [], 0
    for r in k[k["extraction"] == "pdf_text"].itertuples():
        if "+" in str(r.note):  # a sum of printed components, not printed itself
            continue
        text = _page_text(r.ticker, r.doc_id, int(r.doc_page))
        if text is None:
            errs.append(f"{r.ticker} {r.line_item}: PDF {r.doc_id} not cached")
            continue
        checked += 1
        if _printed(r.value) not in text:
            errs.append(f"{r.ticker} {r.line_item} {r.period_end}: {_printed(r.value)} "
                        f"not found on page {r.doc_page} of {r.doc_id}")
    return errs, checked


def check_against_psx_quarters(k: pd.DataFrame) -> list[str]:
    """YTD net income (filing) == sum of PSX discrete quarters of that fiscal year."""
    errs = []
    s = pd.read_csv(STATEMENTS_CSV)
    q = s[(s["period_kind"] == "quarter") & (s["line_item"] == "net_income")]
    ytd = k[(k["period_kind"] == "ytd") & (k["line_item"] == "net_income")]
    for r in ytd.itertuples():
        scale = 1e-3 if r.unit == "PKR" else 1.0  # PSX page is in Rs '000
        quarters = q[(q["ticker"] == r.ticker) & (q["period_start"] >= r.period_start)
                     & (q["period_end"] <= r.period_end)]
        n_expected = round((pd.Timestamp(r.period_end) - pd.Timestamp(r.period_start)).days / 91.3)
        if len(quarters) != n_expected:
            continue  # PSX page does not show every quarter of this period; nothing to compare
        total = quarters["value"].sum()
        if abs(total - r.value * scale) > max(2, abs(total) * 1e-6):  # rounding of Rs to Rs '000
            errs.append(f"{r.ticker} YTD net income {r.period_end}: filing {r.value * scale:,.0f} "
                        f"vs PSX quarters {total:,.0f}")
    return errs


def check_shares(k: pd.DataFrame) -> list[str]:
    notes = []
    sh = pd.read_csv(SHARES_CSV).sort_values("as_of").groupby("ticker").last()
    cap = k[k["line_item"] == "share_capital"]
    for r in cap.itertuples():
        if r.ticker not in sh.index or pd.isna(sh.loc[r.ticker, "face_value"]):
            notes.append(f"{r.ticker}: no PSX share count / face value to compare")
            continue
        rupees = r.value * (1000 if r.unit == "PKR_thousands" else 1)
        implied = rupees / sh.loc[r.ticker, "face_value"]
        listed = sh.loc[r.ticker, "shares_outstanding"]
        gap = implied / listed - 1
        if abs(gap) > SHARE_TOLERANCE:
            notes.append(f"{r.ticker}: filing implies {implied:,.0f} shares, PSX lists "
                         f"{listed:,.0f} ({gap:+.1%})")
    return notes


def check_new_filings(k: pd.DataFrame) -> list[str]:
    """Companies whose latest PSX filing covers a newer period than the keyed balance sheet."""
    from app.agents.fundamentals.data.psx_official import FILINGS_CSV

    if not FILINGS_CSV.exists():
        return ["psx_filings.csv missing: run psx_official first"]
    f = pd.read_csv(FILINGS_CSV, dtype=str)
    is_date = f["period_end"].str.match(r"\d{4}-\d{2}-\d{2}", na=False)
    dated = is_date & (f["period_end"] <= f["filed"])
    f = f[(f["source"] == "company page") & dated & ~f["title"].str.contains("REVOKED", na=False)]
    latest_filed = f.groupby("ticker")["period_end"].max()
    keyed = k[k["period_kind"] == "instant"].groupby("ticker")["period_end"].max()
    out = []
    for t, filed in latest_filed.items():
        have = keyed.get(t)
        if have is None or filed > have:
            out.append(f"{t}: PSX filing for period ended {filed}; keyed balance sheet is "
                       f"{have or 'missing'}")
    return out


def check_dates(k: pd.DataFrame) -> list[str]:
    bad = k[pd.to_datetime(k["published_at"]) < pd.to_datetime(k["period_end"])]
    return [f"{r.ticker} {r.line_item}: published {r.published_at} before {r.period_end}"
            for r in bad.itertuples()]


def run() -> dict:
    k = load_keyed()
    text_errs, n_text = check_text_pdfs(k)
    result = {
        "rows": len(k),
        "companies": int(k["ticker"].nunique()),
        "balance_identity": check_balance_identity(k),
        "text_pdf_values_checked": n_text,
        "text_pdf_mismatches": text_errs,
        # Two official sources disagree. The filed report is the primary document, so the agent
        # uses the keyed filing figure; the conflict is reported, not hidden.
        "source_conflicts_resolved_to_filing": check_against_psx_quarters(k),
        "share_count_notes": check_shares(k),
        "date_errors": check_dates(k),
        "needs_keying": check_new_filings(k),
        "checked_on": date.today().isoformat(),
    }
    result["ok"] = not (result["balance_identity"] or text_errs or result["date_errors"])
    return result


def main() -> None:
    import json

    print(json.dumps(run(), indent=2))


if __name__ == "__main__":
    main()
