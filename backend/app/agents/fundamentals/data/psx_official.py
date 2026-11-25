"""Official fundamentals from the Pakistan Stock Exchange (no third-party data).

Two PSX sources, both public pages that need no login or token:

1. PSX Data Portal company page, `https://dps.psx.com.pk/company/{SYMBOL}`:
   - "Financials": last 4 fiscal years and last 4 quarters of Sales (or Mark-up Earned and
     Total Income for banks), Profit after Taxation and EPS. Standalone (unconsolidated)
     figures in Rs '000; quarters are discrete (not year-to-date). Verified against the
     quarterly reports of HBL and SYS.
   - "Equity Profile": shares outstanding and face value.
   - "Announcements": the dated "Financial Results" / "Transmission of ... Report" filings,
     which give the real publication date of each period.
2. PSX Financials portal, `https://financials.psx.com.pk/` (`annQtrStmts.php`): every annual and
   quarterly report PDF a company has filed, with its PSX posting date.

Every output row carries `source`, `source_ref` (URL), `published_at` (from a dated PSX filing,
never estimated) and `retrieved_at`. A period whose publication date cannot be found on PSX is
not emitted; it is listed in the report instead. Raw pages are cached under
`data/raw/fundamentals/official/` so each figure can be traced to what was downloaded.

    python -m app.agents.fundamentals.data.psx_official            # all companies
    python -m app.agents.fundamentals.data.psx_official HBL LUCK   # some companies
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from dateutil import parser as dateparser

from app.core.config import REPO_ROOT

log = logging.getLogger(__name__)

COMPANY_URL = "https://dps.psx.com.pk/company/{sym}"
DOC_URL = "https://dps.psx.com.pk{href}"
PORTAL_URL = "https://financials.psx.com.pk/annQtrStmts.php"
PORTAL_DOC_URL = "https://financials.psx.com.pk/lib/DownloadPDF.php?id={id}"
USER_AGENT = "MARS-FYP-research/0.1 (academic project; PSX company fundamentals)"
REQUEST_DELAY_S = 0.7

OFFICIAL_DIR = REPO_ROOT / "data" / "raw" / "fundamentals" / "official"
PAGES_DIR = OFFICIAL_DIR / "pages"
STATEMENTS_CSV = OFFICIAL_DIR / "psx_statements.csv"
SHARES_CSV = OFFICIAL_DIR / "psx_shares.csv"
FILINGS_CSV = OFFICIAL_DIR / "psx_filings.csv"
REFERENCE_CSV = Path(__file__).resolve().parents[1] / "reference" / "companies.csv"

SOURCE_PAGE = "PSX Data Portal company page (Financials)"
SOURCE_EQUITY = "PSX Data Portal company page (Equity Profile)"

# PSX row label -> canonical line item (all in Rs '000 except EPS in Rs per share)
LINE_ITEMS = {
    "sales": "revenue",
    "mark-up earned": "markup_earned",
    "total income": "total_income",
    "profit after taxation": "net_income",
    "eps": "eps_reported",
}

STATEMENT_COLUMNS = [
    "ticker", "line_item", "value", "unit", "period_start", "period_end", "period_kind",
    "fiscal_label", "published_at", "published_via", "source", "source_ref", "retrieved_at",
    "is_restated", "version",
]


@dataclass(frozen=True)
class Company:
    ticker: str
    psx_symbol: str
    company_type: str
    fiscal_year_end_month: int


def load_companies() -> list[Company]:
    ref = pd.read_csv(REFERENCE_CSV)
    return [
        Company(r.ticker, r.psx_symbol, r.company_type, int(r.fiscal_year_end_month))
        for r in ref.itertuples()
    ]


# --- fiscal periods ----------------------------------------------------------------------------


def _month_end(y: int, m: int) -> date:
    return (pd.Timestamp(year=y, month=m, day=1) + pd.offsets.MonthEnd(0)).date()


def fiscal_period(label: str, fy_end_month: int) -> tuple[date, date, str]:
    """PSX period label ("2025", "Q3 2026", "Q1 2027") -> (start, end, kind).

    PSX names a fiscal year by the calendar year it ends in: for a June year-end company,
    FY 2027 runs Jul 2026 - Jun 2027, so "Q1 2027" is Jul-Sep 2026.
    """
    m = re.fullmatch(r"(?:Q([1-4])\s+)?(\d{4})", label.strip())
    if not m:
        raise ValueError(f"Unrecognised PSX period label: {label!r}")
    q, fy = (int(m.group(1)) if m.group(1) else None), int(m.group(2))
    fy_end = _month_end(fy, fy_end_month)
    fy_start = (pd.Timestamp(fy_end) - pd.DateOffset(years=1) + pd.Timedelta(days=1)).date()
    if q is None:
        return fy_start, fy_end, "annual"
    start = (pd.Timestamp(fy_start) + pd.DateOffset(months=3 * (q - 1))).date()
    end = (pd.Timestamp(start) + pd.DateOffset(months=3) - pd.Timedelta(days=1)).date()
    return start, end, "quarter"


# --- HTTP + caching ----------------------------------------------------------------------------


def _session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    return s


def _request(session: requests.Session, method: str, url: str, **kw) -> requests.Response:
    """GET/POST with 3 attempts and growing back-off for transient server errors (5xx)."""
    for attempt in range(3):
        try:
            r = session.request(method, url, timeout=60, **kw)
            if r.status_code < 500:
                r.raise_for_status()
                return r
        except requests.ConnectionError:
            if attempt == 2:
                raise
        time.sleep(5 * (attempt + 1))
    r.raise_for_status()
    return r


def fetch_company_page(sym: str, session: requests.Session, today: date) -> tuple[str, Path]:
    """Download (once per day) and cache the PSX company page."""
    path = PAGES_DIR / f"{sym}_{today.isoformat()}.html"
    if not path.exists():
        r = _request(session, "GET", COMPANY_URL.format(sym=sym))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(r.text, encoding="utf-8")
        time.sleep(REQUEST_DELAY_S)
    return path.read_text(encoding="utf-8"), path


_PORTAL = {"failures": 0}
PORTAL_MAX_FAILURES = 3  # consecutive failures before the portal is skipped for this run


def fetch_portal_filings(sym: str, session: requests.Session, today: date) -> list[dict]:
    """Filings on the PSX Financials portal; [] (not an error) if the portal is down.

    After PORTAL_MAX_FAILURES consecutive failures the portal is not called again in this run,
    so an outage on PSX's side does not turn a collection into hours of time-outs.
    """
    path = PAGES_DIR / f"{sym}_{today.isoformat()}_portal.json"
    if not path.exists():
        if _PORTAL["failures"] >= PORTAL_MAX_FAILURES:
            return []
        try:
            r = session.post(PORTAL_URL, data={"name": "get_comp_data", "smbCode": sym},
                             timeout=20)
            r.raise_for_status()
            _PORTAL["failures"] = 0
        except requests.RequestException as exc:
            _PORTAL["failures"] += 1
            log.warning("PSX Financials portal unavailable for %s: %s", sym, exc)
            if _PORTAL["failures"] == PORTAL_MAX_FAILURES:
                log.warning("PSX Financials portal skipped for the rest of this run")
            return []
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(r.text or "[]", encoding="utf-8")
        time.sleep(REQUEST_DELAY_S)
    raw = json.loads(path.read_text(encoding="utf-8") or "[]") or []
    out = []
    for row in raw:
        m = re.search(r"id=([\w-]+)\">(\w+)<", row.get("Reports", ""))
        if not m:
            continue
        out.append({
            "doc_id": m.group(1), "kind": m.group(2).lower(),
            "period_ended": row.get("period_ended"), "posting_date": row.get("posting_date"),
            "url": PORTAL_DOC_URL.format(id=m.group(1)),
        })
    return out


# --- parsing the company page ------------------------------------------------------------------


def _num(text: str) -> float | None:
    t = text.strip().replace(",", "")
    if t in {"", "-", "N/A"}:
        return None
    neg = t.startswith("(") and t.endswith(")")
    try:
        v = float(t.strip("()"))
    except ValueError:
        return None
    return -v if neg else v


def parse_financials(html: str) -> list[tuple[str, str, float]]:
    """(period_label, canonical_line_item, value) from the Financials tab."""
    soup = BeautifulSoup(html, "html.parser")
    sec = soup.find(id="financials")
    out = []
    for table in sec.find_all("table") if sec else []:
        rows = [[c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
                for tr in table.find_all("tr")]
        if not rows:
            continue
        periods = rows[0][1:]
        for r in rows[1:]:
            item = LINE_ITEMS.get(r[0].strip().lower())
            if item is None:
                continue
            for label, cell in zip(periods, r[1:]):
                v = _num(cell)
                if v is not None:
                    out.append((label, item, v))
    return out


def parse_shares(html: str) -> dict | None:
    soup = BeautifulSoup(html, "html.parser")
    eq = soup.find(id="equity")
    text = eq.get_text(" ", strip=True) if eq else ""
    shares = re.search(r"Shares\s+([\d,]+)", text)
    face = re.search(r"Face Value\s+([\d.,]+)", text)
    if not shares:
        return None
    return {"shares_outstanding": float(shares.group(1).replace(",", "")),
            "face_value": float(face.group(1)) if face else None}


_ENDED = re.compile(r"(?:ended|ending)\s*[-:]?\s*(.+)$", re.I)


def _period_end_in(title: str) -> date | None:
    m = _ENDED.search(title)
    if not m:
        return None
    tail = re.split(r"\s+-\s+|\(|\bof\b", m.group(1))[0]
    try:
        return dateparser.parse(tail, fuzzy=True, default=datetime(1900, 1, 1)).date()
    except (ValueError, OverflowError):
        return None


def parse_announcements(html: str) -> list[dict]:
    """Dated filings from the Announcements tab (financial results and report transmissions)."""
    soup = BeautifulSoup(html, "html.parser")
    out = []
    for a in soup.find_all("a", href=lambda h: h and "/download/document/" in h):
        tr = a.find_parent("tr")
        if tr is None:
            continue
        cells = [c.get_text(" ", strip=True) for c in tr.find_all("td")]
        if len(cells) < 2:
            continue
        title = cells[1]
        if re.search(r"board meeting|revoked", title, re.I):
            continue  # not a publication, or withdrawn by the company
        if re.search(r"financial results", title, re.I):
            kind = "results"
        elif re.search(r"transmission of .*(report|accounts)", title, re.I):
            kind = "report"
        else:
            continue
        try:
            filed = dateparser.parse(cells[0]).date()
        except (ValueError, OverflowError):
            continue
        period_end = _period_end_in(title)
        if period_end is not None and period_end > filed:
            period_end = None  # typo in the PSX title (e.g. "3036"): period cannot be trusted
        out.append({"filed": filed, "title": title, "kind": kind,
                    "period_end": period_end, "url": DOC_URL.format(href=a["href"])})
    return out


def publication(period_end: date, kind: str, fy_end: date, anns: list[dict],
                portal: list[dict]) -> tuple[date, str, str] | None:
    """Earliest PSX filing that published `period_end`: (date, how, url), or None.

    Order of preference: the "Financial Results" announcement, then the report transmission on
    the company page, then the posting date on the PSX Financials portal.
    """
    for want in ("results", "report"):
        hits = [a for a in anns if a["kind"] == want and a["period_end"] == period_end]
        if hits:
            first = min(hits, key=lambda a: a["filed"])
            return first["filed"], f"company page announcement ({want})", first["url"]
    for f in portal:
        pe = f["period_ended"] or ""
        match = (pe == period_end.isoformat()) or (
            kind == "annual" and f["kind"] == "annual" and pe == str(fy_end.year)
        )
        if match and f["posting_date"]:
            try:
                posted = date.fromisoformat(f["posting_date"])
            except ValueError:
                continue
            if posted >= period_end:  # sanity: cannot be published before the period ends
                return posted, "PSX Financials portal posting date", f["url"]
    # No filing found for this period. Accounts are filed in order, so the first PSX filing for
    # a LATER period is an upper bound on when this one was public. Using it can only make the
    # figure available later than it really was (never earlier), so there is no look-ahead.
    later = [(a["filed"], a["url"]) for a in anns
             if a["period_end"] and a["period_end"] > period_end and a["filed"] >= period_end]
    for f in portal:
        try:
            pe = date.fromisoformat(f["period_ended"])
            posted = date.fromisoformat(f["posting_date"])
        except (TypeError, ValueError):
            continue
        if period_end < pe <= posted:
            later.append((posted, f["url"]))
    if later:
        filed, url = min(later)
        return filed, "upper bound: first PSX filing for a later period", url
    return None


# --- collection --------------------------------------------------------------------------------


def infer_fy_end_month(anns: list[dict], portal: list[dict],
                       labels: list[str] | None = None) -> tuple[int | None, str]:
    """Fiscal year-end month from a company's own PSX filings.

    1. An annual filing title ("... for the year ended June 30, 2026").
    2. The company page's latest quarter label (e.g. "Q3 2026") must end on a date that a dated
       PSX filing says was reported ("quarter ended ..."): only one year-end month fits.
    3. Otherwise the quarter-end month that never appears among its quarterly portal filings
       since 2020 (the fourth quarter is filed as the annual report, not as a quarterly one).
    """
    for a in anns:
        if a["period_end"] and re.search(r"year ended|annual", a["title"], re.I):
            return a["period_end"].month, "annual filing title"
    quarters = [lb for lb in (labels or []) if lb.startswith("Q")]
    reported = {a["period_end"] for a in anns if a["period_end"]}
    if quarters and reported:
        latest = max(quarters, key=lambda lb: (int(lb.split()[1]), int(lb[1])))
        fits = [m for m in (3, 6, 9, 12) if fiscal_period(latest, m)[1] in reported]
        if len(fits) == 1:
            return fits[0], f"page label {latest} matches a dated filing"
    counts = {3: 0, 6: 0, 9: 0, 12: 0}
    for f in portal:
        if f["kind"] != "quarterly":
            continue
        try:
            pe = date.fromisoformat(f["period_ended"])
            posted = date.fromisoformat(f["posting_date"])
        except (TypeError, ValueError):
            continue
        if pe.year >= 2020 and posted >= pe and pe.month in counts:
            counts[pe.month] += 1
    missing = [m for m, n in counts.items() if n == 0]
    if len(missing) == 1 and sum(counts.values()) >= 6:
        return missing[0], "quarter-end month absent from quarterly filings"
    return None, f"ambiguous quarterly filing months {counts}"


def collect_company(c: Company, session: requests.Session, today: date) -> dict:
    html, page_path = fetch_company_page(c.psx_symbol, session, today)
    portal = fetch_portal_filings(c.psx_symbol, session, today)
    anns = parse_announcements(html)
    fy_how = "reference"
    if not c.fiscal_year_end_month:
        labels = sorted({lb for lb, _, _ in parse_financials(html)})
        month, fy_how = infer_fy_end_month(anns, portal, labels)
        if month is None:
            return {"rows": [], "shares": None, "filings": [], "unpublished": [],
                    "page": str(page_path), "fy": None, "fy_how": fy_how}
        c = Company(c.ticker, c.psx_symbol, c.company_type, month)
    retrieved = datetime.now(UTC).isoformat(timespec="seconds")
    page_url = COMPANY_URL.format(sym=c.psx_symbol)
    rows, unpublished = [], []
    for label, item, value in parse_financials(html):
        start, end, kind = fiscal_period(label, c.fiscal_year_end_month)
        fy_end = end if kind == "annual" else fiscal_period(
            label.split()[-1], c.fiscal_year_end_month)[1]
        pub = publication(end, kind, fy_end, anns, portal)
        if pub is None:
            unpublished.append(f"{label} {item}")
            continue
        published_at, via, doc_url = pub
        rows.append({
            "ticker": c.ticker, "line_item": item, "value": value,
            "unit": "PKR_per_share" if item == "eps_reported" else "PKR_thousands",
            "period_start": start.isoformat(), "period_end": end.isoformat(),
            "period_kind": kind, "fiscal_label": label,
            "published_at": published_at.isoformat(), "published_via": via,
            "source": SOURCE_PAGE, "source_ref": f"{page_url}#financials | filing: {doc_url}",
            "retrieved_at": retrieved, "is_restated": False, "version": 1,
        })
    shares = parse_shares(html)
    share_row = None
    if shares:
        share_row = {"ticker": c.ticker, "as_of": today.isoformat(), **shares,
                     "unit": "shares", "source": SOURCE_EQUITY,
                     "source_ref": f"{page_url}#equity", "retrieved_at": retrieved}
    filings = [{"ticker": c.ticker, "source": "company page", **{k: a[k] for k in
                ("filed", "kind", "period_end", "title", "url")}} for a in anns]
    filings += [{"ticker": c.ticker, "source": "financials portal", "filed": f["posting_date"],
                 "kind": f["kind"], "period_end": f["period_ended"], "title": "",
                 "url": f["url"]} for f in portal]
    return {"rows": rows, "shares": share_row, "filings": filings,
            "unpublished": unpublished, "page": str(page_path),
            "fy": c.fiscal_year_end_month, "fy_how": fy_how}


def merge_versions(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Append new figures; a changed figure for a known fact becomes a new version (restated).

    History is never overwritten: earlier valuation dates keep seeing the original number.
    """
    if old.empty:
        return new
    key = ["ticker", "line_item", "period_end", "period_kind"]
    latest = old.sort_values("version").groupby(key, as_index=False).last()
    merged = new.merge(latest[[*key, "value", "version"]], on=key, how="left",
                       suffixes=("", "_old"))
    known = merged["value_old"].notna()
    unchanged = known & (merged["value"] == merged["value_old"])
    add = merged[~unchanged].copy()
    restated = add["value_old"].notna()
    add.loc[restated, "is_restated"] = True
    add.loc[restated, "version"] = add.loc[restated, "version_old"] + 1
    # A restated figure is only knowable from the day it was first observed.
    add.loc[restated, "published_at"] = add.loc[restated, "retrieved_at"].str[:10]
    return pd.concat([old, add[STATEMENT_COLUMNS]], ignore_index=True)


UNIVERSE_CSV = OFFICIAL_DIR / "universe.csv"
# Not operating companies: closed-end funds, modarabas, REITs, ETFs.
EXCLUDED_SECTORS = {"0806", "0819", "0836", "0837"}


def universe_companies() -> list[Company]:
    """Every listed operating company in the latest official PSX daily file."""
    from app.ingestion import psx_eod

    days = sorted(psx_eod.EOD_DIR.glob("????-??-??.Z"))
    if not days:
        raise FileNotFoundError("No PSX daily files: run python -m app.ingestion.psx_eod first")
    day = psx_eod.parse_day(days[-1])
    known = {c.psx_symbol: c for c in load_companies()}
    out = []
    for r in day.itertuples():
        code = str(r.sector_code).strip()
        psx_sym = psx_eod.PSX_SYMBOL.get(r.symbol, r.symbol)
        if code in EXCLUDED_SECTORS:
            continue
        out.append(known.get(psx_sym) or Company(
            r.symbol, psx_sym, "bank" if code == "0807" else "non_bank", 0))
    return out


def _save(all_rows: list, shares: list, filings: list) -> None:
    OFFICIAL_DIR.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame(all_rows, columns=STATEMENT_COLUMNS)
    old = pd.read_csv(STATEMENTS_CSV) if STATEMENTS_CSV.exists() else new.iloc[0:0]
    merge_versions(old, new).to_csv(STATEMENTS_CSV, index=False)
    sh_old = pd.read_csv(SHARES_CSV) if SHARES_CSV.exists() else pd.DataFrame()
    sh = pd.concat([sh_old, pd.DataFrame(shares)], ignore_index=True)
    sh.drop_duplicates(["ticker", "as_of"], keep="last").to_csv(SHARES_CSV, index=False)
    fl_old = pd.read_csv(FILINGS_CSV) if FILINGS_CSV.exists() else pd.DataFrame()
    fl = pd.concat([fl_old, pd.DataFrame(filings)], ignore_index=True)
    fl.astype(str).drop_duplicates(["ticker", "url"]).to_csv(FILINGS_CSV, index=False)


def collect(tickers: list[str] | None = None, today: date | None = None,
            whole_market: bool = False, batch: int = 50) -> dict:
    """Collect PSX company-page figures for the reference companies (default) or all of PSX."""
    today = today or date.today()
    session = _session()
    pool = universe_companies() if whole_market else load_companies()
    companies = [c for c in pool if not tickers or c.ticker in tickers]
    all_rows, shares, filings, report, uni = [], [], [], {}, []
    for i, c in enumerate(companies, 1):
        try:
            res = collect_company(c, session, today)
        except (requests.RequestException, ValueError, AttributeError) as exc:
            report[c.ticker] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        all_rows += res["rows"]
        filings += res["filings"]
        if res["shares"]:
            shares.append(res["shares"])
        listed = res["shares"]["shares_outstanding"] if res["shares"] else None
        report[c.ticker] = {"rows": len(res["rows"]), "unpublished": res["unpublished"],
                            "shares": listed, "fy": res["fy"], "fy_how": res["fy_how"]}
        uni.append({"ticker": c.ticker, "psx_symbol": c.psx_symbol,
                    "company_type": c.company_type, "fiscal_year_end_month": res["fy"],
                    "fy_how": res["fy_how"], "rows": len(res["rows"])})
        if i % batch == 0:
            _save(all_rows, shares, filings)
            all_rows, shares, filings = [], [], []
            log.info("collected %d / %d companies", i, len(companies))
    _save(all_rows, shares, filings)
    if whole_market:
        pd.DataFrame(uni).to_csv(UNIVERSE_CSV, index=False)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect official PSX fundamentals")
    ap.add_argument("tickers", nargs="*")
    ap.add_argument("--whole-market", action="store_true",
                    help="every listed operating company (for model training and peers)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    report = collect(a.tickers or None, whole_market=a.whole_market)
    if a.whole_market:
        ok = sum(1 for r in report.values() if r.get("rows"))
        print(f"{ok} of {len(report)} companies with figures; see {UNIVERSE_CSV}")
        for t, r in report.items():
            if "error" in r or not r.get("fy"):
                print(t, r.get("error") or r.get("fy_how"))
        return
    for t, r in report.items():
        print(t, r)


if __name__ == "__main__":
    main()
