"""Automatic reading of new PSX filings (text PDFs directly, scanned PDFs with Tesseract OCR).

For each company whose newest PSX filing is not keyed yet (validate_official.check_new_filings):
1. download the filing PDF from PSX (cached in data/raw/fundamentals/official/pdfs/);
2. text of each page: pdftotext, or Tesseract OCR (300 dpi) when the page has no text layer;
3. find the standalone (unconsolidated) balance sheet and profit & loss pages;
4. parse "label [note] current previous" lines into the keyed line items;
5. gate every value with arithmetic checks before it may be used:
   - assets = liabilities + equity (exact), with derivation when one total is unreadable;
   - equity = sum of its component lines (share capital + reserves ...);
   - share capital / face value within 2% of the PSX share count;
   - year-to-date profit = sum of the PSX company-page quarters (selects the YTD column).
Values that pass go into keyed_statements.csv (extraction auto_text / auto_ocr); anything else
goes to review_queue.csv with the reason, for a person to key by hand. Nothing unverified is used.

    python -m app.agents.fundamentals.data.auto_extract            # new filings only
    python -m app.agents.fundamentals.data.auto_extract --check    # accuracy vs keyed values
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pandas as pd
import requests

from app.agents.fundamentals.data.psx_official import (
    FILINGS_CSV,
    OFFICIAL_DIR,
    SHARES_CSV,
    STATEMENTS_CSV,
    USER_AGENT,
    load_companies,
)

PDF_DIR = OFFICIAL_DIR / "pdfs"
KEYED_CSV = OFFICIAL_DIR / "keyed_statements.csv"
REVIEW_CSV = OFFICIAL_DIR / "review_queue.csv"
TESSERACT_PATHS = [r"C:\Program Files\Tesseract-OCR\tesseract.exe", "tesseract"]
SHARE_TOLERANCE = 0.02

NUM = r"\(?-?[\d,]{1,20}(?:\.\d+)?\)?|-"
LINE = re.compile(
    rf"^(?P<label>[A-Za-z(][^\d]*?[A-Za-z)'.:])\s+(?:(?P<note>\d{{1,2}}(?:\.\d{{1,2}})?)\s+)?"
    rf"(?P<nums>(?:(?:{NUM})\s+){{1,3}}(?:{NUM}))\s*$"
)
BARE = re.compile(rf"^\s*(?P<nums>(?:(?:{NUM})\s+){{1,3}}(?:{NUM}))\s*$")

BS_ITEMS = {
    "total_assets": r"^total assets$",
    "total_liabilities": r"^total liabilities$",
    "total_equity_and_liabilities": r"^total (equity and liabilities|liabilities and equity)",
    "total_equity": r"^(total equity|net assets|total shareholders'? equity)$",
    "share_capital": r"^(issued, subscribed and paid[- ]up (share )?capital|share capital|"
    r"paid[- ]up capital)$",
    "cash_and_equivalents": r"^cash and (bank balances|balances with treasury banks)$",
    "advances": r"^(advances|islamic financing and related assets)( - net)?$",
    "deposits": r"^deposits and other accounts$",
    "investments": r"^investments$",
}
PL_ITEMS = {
    "net_income": r"^(profit|\(loss\) / profit|profit / \(loss\))( for the (period|quarter)| after "
    r"tax(ation)?)( for the period)?$",
    "revenue": r"^(revenue( from contracts with customers)?( - net)?|turnover( - net)?|"
    r"net (sales|revenue)|sales( - net)?)$",
    "total_income": r"^total income$",
    "markup_earned": r"^(mark-up / return / (profit / )?interest earned|profit / return earned"
    r".*)$",
    "net_interest_income": r"^(net mark-up / (return / )?(profit / )?interest income|"
    r"net profit / return)$",
    "operating_expenses": r"^operating expenses$",
}


@dataclass
class Extraction:
    ticker: str
    doc_id: str
    published_at: str
    period_end: str
    method: str = ""  # pdf_text | ocr
    accepted: list[dict] = field(default_factory=list)
    review: list[dict] = field(default_factory=list)


# --- page text ----------------------------------------------------------------------------------


def _tesseract() -> str | None:
    for p in TESSERACT_PATHS:
        if Path(p).exists() or shutil.which(p):
            return p
    return None


def ocr_page(pdf: Path, page: int, dpi: int = 300) -> str:
    tess = _tesseract()
    if tess is None:
        return ""
    with tempfile.TemporaryDirectory() as d:
        img = Path(d) / "p"
        subprocess.run(
            [
                "pdftoppm",
                "-r",
                str(dpi),
                "-gray",
                "-f",
                str(page),
                "-l",
                str(page),
                "-singlefile",
                "-png",
                str(pdf),
                str(img),
            ],
            capture_output=True,
        )
        return subprocess.run(
            [tess, f"{img}.png", "-", "--psm", "6"], capture_output=True, text=True, errors="ignore"
        ).stdout


def page_texts(pdf: Path, ocr: bool = True) -> tuple[list[str], set[int]]:
    """Text of every page; pages without a text layer are OCR'd (200 dpi, enough to find the
    statements; the chosen pages are re-read at 300 dpi). Returns the OCR'd page indexes."""
    n = int(
        re.search(
            r"Pages:\s+(\d+)",
            subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True).stdout,
        ).group(1)
    )
    texts = [
        subprocess.run(
            ["pdftotext", "-raw", "-f", str(i), "-l", str(i), str(pdf), "-"],
            capture_output=True,
            text=True,
            errors="ignore",
        ).stdout
        for i in range(1, n + 1)
    ]
    ocrd = set()
    if ocr and _tesseract():
        for i, t in enumerate(texts):
            if len(t.strip()) < 300:
                texts[i] = ocr_page(pdf, i + 1, dpi=200)
                ocrd.add(i)
    return texts, ocrd


def _score(text: str, items: dict) -> int:
    labels = [lb for lb, n in parse_lines(text) if lb and len(n) >= 2]
    return sum(any(re.search(p, lb) for lb in labels) for p in items.values())


def find_page(texts: list[str], kind: str) -> int | None:
    """Standalone balance sheet ('bs') or profit and loss ('pl') page: the page matching the most
    statement lines, standalone preferred over consolidated."""
    items = BS_ITEMS if kind == "bs" else PL_ITEMS
    want = (
        r"statement of financial position|balance sheet"
        if kind == "bs"
        else r"profit (and|or) loss|income statement"
    )
    scored = []
    for i, t in enumerate(texts):
        low = " ".join(t.lower().split())
        if not re.search(want, low) or len(re.findall(r"\d{1,3},\d{3}", t)) < 12:
            continue
        s = _score(t, items)
        if kind == "pl" and not re.search(r"profit|loss", low):
            continue
        if s < 2:
            continue
        cons = "consolidated" in low.replace("unconsolidated", "") and "unconsolidated" not in low
        scored.append((1 if cons else 0, -s, i))
    return min(scored)[2] if scored else None


def statement_text(texts: list[str], i: int, items: dict) -> str:
    """The chosen page plus a standalone neighbour that holds the rest of the statement."""
    out = texts[i]
    for j in (i - 1, i + 1):
        if 0 <= j < len(texts):
            low = texts[j].lower()
            cons = "consolidated" in low.replace("unconsolidated", "")
            if not cons and _score(texts[j], items) >= 1 and "notes to" not in low[:200]:
                out += "\n" + texts[j]
    return out


# --- parsing ------------------------------------------------------------------------------------


def _num(s: str) -> float | None:
    s = s.strip()
    if s in {"-", ""}:
        return 0.0 if s == "-" else None
    neg = s.startswith("(") and s.endswith(")")
    try:
        v = float(s.strip("()").replace(",", ""))
    except ValueError:
        return None
    return -v if neg else v


def parse_lines(text: str) -> list[tuple[str | None, list[float]]]:
    """(label or None for an unlabelled subtotal, numbers) for every numeric line.

    A 1-2 digit token right after the label is read as a note reference. If that leaves the line
    one value short of the page's column count, it was a small value, and is put back.
    """
    rows = []
    for raw in text.splitlines():
        line = " ".join(raw.replace("\u2014", "-").replace("\u2013", "-").split())
        m = LINE.match(line)
        if m:
            nums = [x for x in (_num(t) for t in m.group("nums").split()) if x is not None]
            note = _num(m.group("note")) if m.group("note") else None
            rows.append((m.group("label").strip().lower(), nums, note))
            continue
        b = BARE.match(line)
        if b:
            rows.append(
                (
                    None,
                    [x for x in (_num(t) for t in b.group("nums").split()) if x is not None],
                    None,
                )
            )
    width = max((len(n) for _, n, _ in rows), default=0)
    return [
        (lb, [note, *n] if note is not None and len(n) == width - 1 else n) for lb, n, note in rows
    ]


def _first(lines, pattern) -> list[float] | None:
    for label, nums in lines:
        if label and re.search(pattern, label) and len(nums) >= 2:
            return nums
    return None


def balance_sheet(text: str) -> tuple[dict, list[str]]:
    """Current-period balance sheet items, with arithmetic derivations and checks."""
    lines = parse_lines(text)
    got = {k: v[-2] for k, p in BS_ITEMS.items() if (v := _first(lines, p))}
    notes = []
    tel = got.pop("total_equity_and_liabilities", None)
    if "total_assets" not in got and tel:
        got["total_assets"] = tel
        notes.append("total assets = total equity and liabilities")
    read = {k for k in ("total_assets", "total_liabilities", "total_equity") if k in got}
    if "total_assets" not in got and (ta_sum := _block_total(lines, ASSET_START, LIAB_STOP)):
        got["total_assets"] = ta_sum
        read.add("total_assets")
        notes.append("total assets = sum of asset lines (unlabelled subtotal)")
    if "total_liabilities" not in got and (tl_sum := _block_total(lines, LIAB_START, LIAB_END)):
        got["total_liabilities"] = tl_sum
        read.add("total_liabilities")
        notes.append("total liabilities = sum of liability lines (unlabelled subtotal)")
    comp = equity_from_components(lines)
    te = got.get("total_equity")
    if comp is not None:
        if te is None:
            got["total_equity"] = te = comp
            notes.append("equity = sum of equity components (unlabelled subtotal)")
        notes.append(
            "equity components add up"
            if abs(comp - te) <= 1
            else f"FAILED equity components {comp:,.0f} vs equity {te:,.0f}"
        )
    ta, tl = got.get("total_assets"), got.get("total_liabilities")
    if te is None and ta and tl:
        got["total_equity"] = te = ta - tl
        notes.append("equity = total assets - total liabilities")
    if tl is None and ta and te is not None:
        got["total_liabilities"] = tl = ta - te
        notes.append("liabilities = total assets - equity")
    if ta and tl is not None and te is not None:
        if abs(ta - tl - te) > 1:
            notes.append(f"FAILED identity: assets - liabilities - equity = {ta - tl - te:,.0f}")
        elif len(read | ({"total_equity"} if comp is not None else set())) == 3:
            notes.append("identity holds on independently read totals")
    return got, notes


def _block_total(lines, start_re: str, stop_re: str) -> float | None:
    """Largest unlabelled subtotal equal to the running sum of the labelled lines from the first
    line matching start_re up to a line matching stop_re (e.g. a bank's total assets)."""
    start = next(
        (i for i, (lb, n) in enumerate(lines) if lb and len(n) >= 2 and re.search(start_re, lb)),
        None,
    )
    if start is None:
        return None
    run, found = 0.0, None
    for lb, n in lines[start : start + 40]:
        if len(n) < 2:
            continue
        v = n[-2]
        if lb is not None and run and re.search(stop_re, lb):
            break
        if lb is None or lb.startswith("total"):
            if run and abs(v - run) <= 1:
                found = max(found or 0.0, v)
            continue
        run += v
    return found


ASSET_START = r"^(cash|balances with|property|non[- ]current assets)"
LIAB_STOP = r"payable|borrowing|^deposits and other|liabilit|share capital|^lease liab"
LIAB_START = r"^bills payable"
LIAB_END = r"^net assets|share capital|^represented"


def equity_from_components(lines) -> float | None:
    """Total equity as the unlabelled subtotal equal to share capital + the following equity lines.

    Reserve subtotals (e.g. revenue reserves) are skipped because they do not equal the running
    sum from share capital; scanning stops at the first liabilities heading or line.
    """
    start = next(
        (
            i
            for i, (lb, n) in enumerate(lines)
            if lb and len(n) >= 2 and re.search(BS_ITEMS["share_capital"], lb)
        ),
        None,
    )
    if start is None:
        return None
    run, found = 0.0, None
    for lb, n in lines[start : start + 20]:
        if len(n) < 2:
            continue
        v = n[-2]
        if lb is not None and re.search(r"liabilit|borrowing|payable|deferred tax", lb):
            break
        if lb is None or re.search(r"^total equity|^net assets", lb or ""):
            if run and abs(v - run) <= 1:
                found = v
            continue
        run += v
    return found


def ytd_lines(text: str, quarters_sum: float | None, unit_scale: float) -> tuple[dict, str]:
    """Year-to-date P&L items; the YTD column is the one whose profit equals the PSX quarters."""
    lines = parse_lines(text)
    ni = _first(lines, PL_ITEMS["net_income"])
    if ni is None:
        return {}, "profit line not found"
    if quarters_sum is None:
        return {}, "no PSX quarterly figures to confirm the year-to-date column"
    col = next((i for i, v in enumerate(ni) if abs(v * unit_scale - quarters_sum) <= 2), None)
    if col is None:
        return {}, (
            f"no column of the profit line ({ni}) matches the PSX quarters ({quarters_sum:,.0f})"
        )
    prior = col + 1 if col + 1 < len(ni) else None
    got = {}
    for item, pat in PL_ITEMS.items():
        v = _first(lines, pat)
        if v is not None and len(v) == len(ni):
            got[item] = (v[col], v[prior] if prior is not None else None)
    return got, f"year-to-date column {col + 1} confirmed by PSX quarters"


# --- per filing ---------------------------------------------------------------------------------


def _download(psx_symbol: str, doc_url: str) -> Path:
    doc_id = doc_url.rstrip("/").split("/")[-1].replace(".pdf", "").split("=")[-1]
    p = PDF_DIR / f"{psx_symbol}_{doc_id}.pdf"
    if not p.exists():
        r = requests.get(doc_url, headers={"User-Agent": USER_AGENT}, timeout=180)
        r.raise_for_status()
        if not r.content.startswith(b"%PDF"):
            raise ValueError(f"{doc_url} is not a PDF")
        PDF_DIR.mkdir(parents=True, exist_ok=True)
        p.write_bytes(r.content)
        time.sleep(1)
    return p


def _psx_quarters(ticker: str, fy_start: str, end: str, item: str = "net_income") -> float | None:
    """Sum of the PSX company-page discrete quarters of `item` in [fy_start, end] (Rs '000)."""
    s = pd.read_csv(STATEMENTS_CSV)
    q = s[
        (s["ticker"] == ticker)
        & (s["line_item"] == item)
        & (s["period_kind"] == "quarter")
        & (s["period_start"] >= fy_start)
        & (s["period_end"] <= end)
    ]
    months = (pd.Timestamp(end).to_period("M") - pd.Timestamp(fy_start).to_period("M")).n + 1
    q = q.sort_values("version").drop_duplicates("period_end", keep="last")
    if len(q) != months // 3:
        return None
    return float(q["value"].sum())


def extract_filing(
    ticker: str,
    psx_symbol: str,
    fy_month: int,
    doc_url: str,
    filed: str,
    period_end: str,
    face_value: float | None,
    listed: float | None,
    ocr: bool = True,
) -> Extraction:
    pdf = _download(psx_symbol, doc_url)
    doc_id = pdf.stem.split("_", 1)[1]
    ex = Extraction(ticker, doc_id, filed, period_end)
    texts, ocrd = page_texts(pdf, ocr)
    can_ocr = ocr and _tesseract() is not None
    bs_i, pl_i = find_page(texts, "bs"), find_page(texts, "pl")
    if can_ocr and (bs_i is None or pl_i is None):
        # Text layer present but unusable (labels and numbers scrambled): OCR the heading pages.
        heads = r"statement of financial position|balance sheet|profit (and|or) loss"
        for i, t in enumerate(texts):
            if i not in ocrd and re.search(heads, " ".join(t.lower().split())):
                texts[i] = ocr_page(pdf, i + 1, dpi=200)
                ocrd.add(i)
        bs_i, pl_i = find_page(texts, "bs"), find_page(texts, "pl")
    for i in {bs_i, pl_i} - {None}:  # re-read chosen pages (and neighbours) sharply
        for j in (i - 1, i, i + 1):
            if j in ocrd and 0 <= j < len(texts):
                texts[j] = ocr_page(pdf, j + 1, dpi=300)
    ex.method = "ocr" if bs_i in ocrd else "pdf_text"
    src = "auto_ocr" if ex.method == "ocr" else "auto_text"
    if bs_i is None:
        ex.review.append({"item": "balance sheet", "reason": "standalone balance sheet not found"})
    else:
        text = statement_text(texts, bs_i, BS_ITEMS)
        unit = (
            "PKR"
            if re.search(r"\(?rupees\)?\s*$|in rupees\b", text[:600].lower())
            and "'000" not in text[:600]
            and "thousand" not in text[:600].lower()
            else "PKR_thousands"
        )
        got, notes = balance_sheet(text)
        failed = any(n.startswith("FAILED") for n in notes)
        cap_ok = False
        if "share_capital" in got and face_value and listed:
            rupees = got["share_capital"] * (1000 if unit == "PKR_thousands" else 1)
            cap_ok = abs(rupees / face_value / listed - 1) <= SHARE_TOLERANCE
        eq_ok = "equity components add up" in notes
        independent = "identity holds on independently read totals" in notes
        for item, v in got.items():
            core = item in ("total_assets", "total_liabilities", "total_equity", "share_capital")
            if ex.method == "pdf_text":  # printed text: labels are exact, need one cross-check
                trusted = not failed and (eq_ok or cap_ok or independent)
            else:  # OCR: digits may be misread; only arithmetically confirmed totals
                trusted = not failed and core and (independent or (eq_ok and cap_ok))
            row = {
                "ticker": ticker,
                "line_item": item,
                "value": v,
                "unit": unit,
                "period_kind": "instant",
                "period_start": "",
                "period_end": period_end,
                "doc_id": doc_id,
                "doc_page": bs_i + 1,
                "published_at": filed,
                "published_via": "report",
                "extraction": src,
                "note": "; ".join(notes),
            }
            if trusted:
                ex.accepted.append(row)
            else:
                ex.review.append({**row, "reason": "; ".join(notes) or "not confirmed"})
    if pl_i is not None:
        fy_start = (pd.Timestamp(period_end) - pd.offsets.MonthEnd(0)).replace(day=1)
        while fy_start.month != (fy_month % 12) + 1:
            fy_start -= pd.DateOffset(months=1)
        fy_start = fy_start.date().isoformat()
        text = statement_text(texts, pl_i, PL_ITEMS)
        scale = 1e-3 if "rupees" in text[:800].lower() and "'000" not in text[:800] else 1.0
        got, how = ytd_lines(text, _psx_quarters(ticker, fy_start, period_end), scale)
        if not got:
            ex.review.append({"item": "profit and loss", "reason": how})
        prior_start = (pd.Timestamp(fy_start) - pd.DateOffset(years=1)).date().isoformat()
        prior_end = (
            (pd.Timestamp(period_end) - pd.DateOffset(years=1) + pd.offsets.MonthEnd(0))
            .date()
            .isoformat()
        )
        pl_ocr = pl_i in ocrd
        for item, (cur, prev) in got.items():
            check = (
                _psx_quarters(ticker, fy_start, period_end, item) if item != "net_income" else None
            )
            if item != "net_income":
                if check is not None and abs(cur * scale - check) > 2:
                    ex.review.append(
                        {
                            "ticker": ticker,
                            "line_item": item,
                            "value": cur,
                            "reason": f"does not match PSX quarters ({check:,.0f})",
                        }
                    )
                    continue
                if check is None and pl_ocr:
                    ex.review.append(
                        {
                            "ticker": ticker,
                            "line_item": item,
                            "value": cur,
                            "reason": "OCR value without an independent check",
                        }
                    )
                    continue
            for v, (s_, e) in ((cur, (fy_start, period_end)), (prev, (prior_start, prior_end))):
                if v is None:
                    continue
                ex.accepted.append(
                    {
                        "ticker": ticker,
                        "line_item": item,
                        "value": v,
                        "unit": "PKR" if scale != 1.0 else "PKR_thousands",
                        "period_kind": "ytd",
                        "period_start": s_,
                        "period_end": e,
                        "doc_id": doc_id,
                        "doc_page": pl_i + 1,
                        "published_at": filed,
                        "published_via": "report",
                        "extraction": "auto_ocr" if pl_ocr else "auto_text",
                        "note": how
                        + (
                            ""
                            if item == "net_income"
                            else "; matches PSX quarters"
                            if check is not None
                            else ""
                        ),
                    }
                )
    return ex


def _face_and_listed() -> dict:
    sh = pd.read_csv(SHARES_CSV).sort_values("as_of").groupby("ticker").last()
    return {t: (r["face_value"], r["shares_outstanding"]) for t, r in sh.iterrows()}


def pending_filings() -> list[dict]:
    """Newest PSX report / results filing per reference company that is newer than keyed data."""
    keyed = (
        pd.read_csv(KEYED_CSV)
        if KEYED_CSV.exists()
        else pd.DataFrame(columns=["ticker", "period_kind", "period_end"])
    )
    have = keyed[keyed["period_kind"] == "instant"].groupby("ticker")["period_end"].max()
    f = pd.read_csv(FILINGS_CSV, dtype=str)
    f = f[
        (f["source"] == "company page")
        & f["kind"].isin(["report", "results"])
        & f["period_end"].str.match(r"\d{4}-\d{2}-\d{2}", na=False)
        & (f["period_end"] <= f["filed"])
        & ~f["title"].str.contains("REVOKED", na=False)
    ]
    out = []
    for c in load_companies():
        g = f[f["ticker"] == c.ticker]
        if g.empty:
            continue
        latest = g["period_end"].max()
        if c.ticker in have.index and have[c.ticker] >= latest:
            continue
        g = g[g["period_end"] == latest]
        pick = g[g["kind"] == "report"] if (g["kind"] == "report").any() else g
        r = pick.sort_values("filed").iloc[0]
        out.append({"company": c, "url": r["url"], "filed": r["filed"], "period_end": latest})
    return out


def run(ocr: bool = True) -> dict:
    sh = _face_and_listed()
    accepted, review, summary = [], [], {}
    for p in pending_filings():
        c = p["company"]
        face, listed = sh.get(c.ticker, (None, None))
        try:
            ex = extract_filing(
                c.ticker,
                c.psx_symbol,
                c.fiscal_year_end_month,
                p["url"],
                p["filed"],
                p["period_end"],
                face,
                listed,
                ocr,
            )
        except (requests.RequestException, ValueError, subprocess.SubprocessError) as exc:
            review.append(
                {"ticker": c.ticker, "item": "filing", "reason": str(exc), "doc": p["url"]}
            )
            continue
        accepted += ex.accepted
        review += [{"ticker": c.ticker, "doc": p["url"], **r} for r in ex.review]
        summary[c.ticker] = {
            "period_end": p["period_end"],
            "method": ex.method,
            "accepted": len(ex.accepted),
            "for_review": len(ex.review),
        }
    if accepted:
        keyed = pd.read_csv(KEYED_CSV, dtype={"doc_id": str})
        pd.concat([keyed, pd.DataFrame(accepted)], ignore_index=True).to_csv(KEYED_CSV, index=False)
    if review:
        old = pd.read_csv(REVIEW_CSV) if REVIEW_CSV.exists() else pd.DataFrame()
        rv = pd.concat([old, pd.DataFrame(review).assign(queued_on=date.today().isoformat())])
        rv.to_csv(REVIEW_CSV, index=False)
    return summary


def accuracy_check(ocr: bool = True) -> dict:
    """Re-read the already keyed filings automatically and compare with the hand-keyed values."""
    keyed = pd.read_csv(KEYED_CSV, dtype={"doc_id": str})
    keyed = keyed[~keyed["extraction"].str.startswith("auto")]
    sh, comps = _face_and_listed(), {c.ticker: c for c in load_companies()}
    result = {"compared": 0, "matched": 0, "wrong": [], "not_extracted": 0, "companies": {}}
    for (t, doc), g in keyed.groupby(["ticker", "doc_id"]):
        c = comps[t]
        face, listed = sh.get(t, (None, None))
        url = f"https://dps.psx.com.pk/download/document/{doc}.pdf"
        pe = g[g["period_kind"] == "instant"]["period_end"].max()
        ex = extract_filing(
            t,
            c.psx_symbol,
            c.fiscal_year_end_month,
            url,
            g["published_at"].iloc[0],
            pe,
            face,
            listed,
            ocr,
        )
        auto = {
            (r["line_item"], r["period_kind"], r["period_end"]): r["value"] for r in ex.accepted
        }
        ok = wrong = missing = 0
        for r in g.itertuples():
            if "+" in str(r.note):
                continue  # keyed sums (borrowings) are not single printed lines
            key = (r.line_item, r.period_kind, r.period_end)
            if key not in auto:
                missing += 1
            elif abs(auto[key] - r.value) <= 1:
                ok += 1
            else:
                wrong += 1
                result["wrong"].append(
                    f"{t} {r.line_item}: auto {auto[key]:,.0f} vs keyed {r.value:,.0f}"
                )
        result["compared"] += ok + wrong + missing
        result["matched"] += ok
        result["not_extracted"] += missing
        result["companies"][t] = {
            "method": ex.method,
            "matched": ok,
            "wrong": wrong,
            "not_extracted": missing,
        }
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Read new PSX filings automatically")
    ap.add_argument("--check", action="store_true", help="accuracy vs hand-keyed values")
    ap.add_argument("--no-ocr", action="store_true")
    a = ap.parse_args()
    out = accuracy_check(not a.no_ocr) if a.check else run(not a.no_ocr)
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
