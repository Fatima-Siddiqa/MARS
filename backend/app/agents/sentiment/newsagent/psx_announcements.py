"""PSX company announcements (results, dividends, board meetings). Structured
and nearly always present, so they are a floor of evidence when media is quiet.

NOTE: the request below follows the public portal's announcements page
(dps.psx.com.pk/announcements). It could not be called from the build
environment, so run `python -m app.check_sources` once on your machine. If PSX
changes its HTML only parse_rows() needs updating. A CSV fallback exists:
`python -m app.import_csv announcements.csv --kind announcement`."""


import json
import re

from app.agents.sentiment.newsagent.http import get, post
import logging
from datetime import date, datetime, timedelta, timezone
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from app.agents.sentiment.newsagent import config
from app.agents.sentiment.newsagent.entities import load_companies
# from app.agents.sentiment.newsagent.http import post
from app.agents.sentiment.newsagent.utils import normalize_url

log = logging.getLogger(__name__)


def _parse_dt(date_text, time_text=""):
    for fmt in ("%b %d, %Y %I:%M %p", "%b %d, %Y", "%d %b %Y", "%Y-%m-%d"):
        try:
            txt = f"{date_text} {time_text}".strip() if "%I" in fmt else date_text
            dt = datetime.strptime(txt, fmt)
            # PSX shows Pakistan time (UTC+5)
            return (dt - timedelta(hours=5)).replace(tzinfo=timezone.utc).isoformat()
        except ValueError:
            continue
    return None


def parse_rows(html, base=config.PSX_ANNOUNCEMENTS_URL):
    """Parse the announcements table. Expected columns: date, time, symbol, name, title, links."""
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    for tr in soup.select("tr"):
        tds = tr.find_all("td")
        if len(tds) < 4:
            continue
        cells = [td.get_text(" ", strip=True) for td in tds]
        links = [urljoin(base, a["href"]) for a in tr.find_all("a", href=True)]
        pdf = next((l for l in links if l.lower().endswith(".pdf") or "/download/" in l), None)
        date_txt, time_txt, symbol = cells[0], cells[1], cells[2].upper()
        title = cells[4] if len(cells) > 4 else cells[3]
        published = _parse_dt(date_txt, time_txt)
        if not (symbol and title and published):
            continue
        rows.append({"symbol": symbol, "title": title, "published_at": published, "doc_url": pdf})
    return rows




def fetch(symbol, date_from, date_to, count=100):
    # Load the page first to obtain its request ID and session cookies.
    page_url = config.PSX_ANNOUNCEMENTS_PAGE_URL
    page_response = get(page_url)

    match = re.search(
        r"window\.__ps\s*=\s*(\{.*?\});",
        page_response.text,
        re.DOTALL,
    )
    if not match:
        raise RuntimeError(
            "Could not find the PSX request ID in the announcements page"
        )

    ps_config = json.loads(match.group(1))
    request_id = ps_config.get("_k")
    if not request_id:
        raise RuntimeError("PSX page did not provide an X-Req-Id value")

    payload = {
        "type": "C",
        "symbol": symbol,
        "query": "",
        "count": count,
        "offset": 0,
        "date_from": date_from.isoformat(),
        "date_to": date_to.isoformat(),
        "page": "annc",
    }

    headers = {
        "X-Req-Id": request_id,
        "X-Requested-With": "XMLHttpRequest",
        "Referer": page_url,
        "Origin": "https://dps.psx.com.pk",
    }

    response = post(
        config.PSX_ANNOUNCEMENTS_URL,
        data=payload,
        headers=headers,
    )

    return parse_rows(
        response.text,
        base=page_url,
    )

def collect(days=30, end=None, tickers=None):
    """Raw records for all tracked companies (same shape as RSS records)."""
    end = end or date.today()
    start = end - timedelta(days=days)
    records, failures = [], {}
    for c in load_companies():
        if tickers and c["ticker"] not in tickers:
            continue
        symbol = (c.get("psx_symbol") or c["ticker"]).upper()
        try:
            for r in fetch(symbol, start, end):
                url = r["doc_url"] or f"psx:{symbol}:{r['published_at']}:{r['title'][:60]}"
                records.append({
                    "source": "psx_announcements", "kind": "announcement", "outlet": "PSX",
                    "title": r["title"], "url": url, "url_key": normalize_url(url) or url,
                    "summary": "", "published_at": r["published_at"],
                    "psx_symbol": symbol, "doc_url": r["doc_url"], "ticker": c["ticker"],
                })
        except Exception as exc:
            failures[symbol] = str(exc)
            log.error("PSX announcements %s failed: %s", symbol, exc)
    return records, failures
