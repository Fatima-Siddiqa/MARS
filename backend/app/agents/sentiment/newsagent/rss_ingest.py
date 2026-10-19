"""RSS and Google News search ingestion. Returns raw records only; cleaning,
linking and storage happen in pipeline.py. GDELT is not used anywhere."""
import logging
from datetime import date, timedelta
from urllib.parse import urlencode

import feedparser

from app.agents.sentiment.newsagent import config
from app.agents.sentiment.newsagent.entities import load_companies
from app.agents.sentiment.newsagent.http import get
from app.agents.sentiment.newsagent.utils import clean_html, normalize_url, to_iso

log = logging.getLogger(__name__)


def search_names(company):
    """Official name and aliases, skipping bare tickers (too ambiguous in global news)."""
    ticker = company["ticker"].upper()
    names = [company["company_name"].strip()]
    for a in (company.get("aliases") or "").split("|"):
        a = a.strip()
        if a and a.upper() != ticker and a.upper() != (company.get("psx_symbol") or "").upper():
            names.append(a)
    out, seen = [], set()
    for n in names:
        if n.casefold() not in seen:
            seen.add(n.casefold())
            out.append(n)
    return out


def date_windows(start, end, chunk_days):
    """Non-overlapping [start, end_exclusive) windows covering start..end inclusive."""
    cursor, stop = start, end + timedelta(days=1)
    while cursor < stop:
        nxt = min(cursor + timedelta(days=max(1, chunk_days)), stop)
        yield cursor, nxt
        cursor = nxt


def google_news_url(company, start_day, end_exclusive):
    names = search_names(company)
    q = " OR ".join(f'"{n}"' for n in names)
    query = f"({q}) after:{start_day.isoformat()} before:{end_exclusive.isoformat()}"
    params = {"q": query, "hl": "en-PK", "gl": "PK", "ceid": "PK:en"}
    return "https://news.google.com/rss/search?" + urlencode(params)


def google_news_feeds(start=None, end=None, chunk_days=None, tickers=None):
    """{source_key: url} for every company and date window."""
    if not config.GOOGLE_NEWS_ENABLED:
        return {}
    end = end or date.today()
    start = start or end - timedelta(days=config.GOOGLE_NEWS_NIGHTLY_LOOKBACK_DAYS)
    chunk_days = chunk_days or config.GOOGLE_NEWS_CHUNK_DAYS
    feeds = {}
    for c in load_companies():
        if tickers and c["ticker"] not in tickers:
            continue
        for s, e in date_windows(start, end, chunk_days):
            feeds[f"google_news_{c['ticker'].lower()}_{s.isoformat()}"] = google_news_url(c, s, e)
    return feeds


def parse_feed(source, feed_url, max_items=None):
    response = get(feed_url)
    feed = feedparser.parse(response.content)
    entries = feed.entries if max_items is None else feed.entries[:max_items]
    records = []
    for item in entries:
        url = (item.get("link") or "").strip()
        title = clean_html(item.get("title", ""))
        if not url or not title:
            continue
        records.append({
            "source": source,
            "title": title,
            "url": url,
            "summary": clean_html(item.get("summary") or item.get("description") or ""),
            "published_at": to_iso(item.get("published") or item.get("updated")),
        })
    return records


def collect(feeds):
    """Fetch every feed in the mapping; a failing feed never stops the others."""
    records, failures = [], {}
    for source, url in feeds.items():
        try:
            max_items = config.GOOGLE_NEWS_MAX_ITEMS if source.startswith("google_news_") else None
            got = parse_feed(source, url, max_items)
            log.info("RSS %s: %d items", source, len(got))
            records.extend(got)
        except Exception as exc:
            failures[source] = str(exc)
            log.error("RSS %s failed: %s", source, exc)
    seen, unique = set(), []
    for r in records:
        key = normalize_url(r["url"])
        if key and key not in seen:
            seen.add(key)
            unique.append({**r, "url_key": key})
    return unique, failures


def collect_publisher_feeds():
    return collect(dict(config.RSS_FEEDS))


def collect_google_news(start=None, end=None, chunk_days=None, tickers=None):
    return collect(google_news_feeds(start, end, chunk_days, tickers))
