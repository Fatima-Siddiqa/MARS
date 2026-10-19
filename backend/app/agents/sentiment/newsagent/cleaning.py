"""Cleaning: outlet extraction, headline normalisation, junk filters,
source credibility and timestamp precision."""
import csv
import hashlib
import re
from datetime import UTC, datetime, timedelta

from app.agents.sentiment.newsagent import config
from app.agents.sentiment.newsagent.utils import clean_html

SPORTS = re.compile(
    r"\b(cricket|football|soccer|hockey|tennis|basketball|scorecard|innings|"
    r"team records|player stats|fixtures|match results|PSL|PCB|T20|ODI)\b", re.I)
# Auto-generated price pages are not news.
PRICE_PAGE = re.compile(
    r"(share price today|stock price today|live price|live prices|price history|"
    r"historical data|share price,? |stock quote)", re.I)

_sources_cache = None


def _load_sources():
    global _sources_cache
    if _sources_cache is None:
        rows = []
        with open(config.SOURCES_CSV, newline="", encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                rows.append(
                    (
                        r["outlet_contains"].lower(),
                        float(r["credibility"]),
                        r["pakistani"] == "1",
                    )
                )
        # longest match first so "Pakistan Today" does not shadow "Profit by Pakistan Today"
        rows.sort(key=lambda t: -len(t[0]))
        _sources_cache = rows
    return _sources_cache


def split_outlet(title):
    """Google News titles end with ' - Outlet'. Return (headline, outlet)."""
    title = clean_html(title or "")
    if " - " in title:
        head, _, tail = title.rpartition(" - ")
        if head and 0 < len(tail) <= 45:
            return head.strip(), tail.strip()
    return title, None


def outlet_from_source(source):
    return {
        "dawn_business": "Dawn", "dawn": "Dawn",
        "business_recorder": "Business Recorder",
        "express_tribune_business": "The Express Tribune", "express_tribune": "The Express Tribune",
        "profit": "Profit by Pakistan Today", "propakistani": "ProPakistani",
        "mettis": "Mettis Global", "the_news_business": "The News",
        "psx_announcements": "PSX",
    }.get(source)


def credibility(outlet):
    """(credibility weight, is_pakistani_outlet). Unknown outlets get 0.6."""
    name = (outlet or "").lower()
    for key, cred, pk in _load_sources():
        if key in name:
            return cred, pk
    return 0.6, False


def is_junk(headline):
    return bool(SPORTS.search(headline or "") or PRICE_PAGE.search(headline or ""))


def norm_headline(text):
    text = re.sub(r"[^a-z0-9% ]+", " ", (text or "").lower())
    return re.sub(r"\s+", " ", text).strip()


def content_hash(headline):
    return hashlib.sha1(norm_headline(headline).encode()).hexdigest()


def effective_time(published_at_iso, delay_days):
    """Return (effective_at_iso, precision).

    Google News and some feeds only carry a date (stamped at a fixed hour). Such
    items are only usable from the next day so a decision on day t never sees
    an item that may have been published after the PSX close.
    """
    if not published_at_iso:
        return None, "date"
    dt = datetime.fromisoformat(published_at_iso)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    dt = dt.astimezone(UTC)
    date_only = dt.minute == 0 and dt.second == 0 and dt.hour in (0, 6, 7, 8)
    if date_only:
        return (dt + timedelta(days=delay_days)).isoformat(), "date"
    return dt.isoformat(), "time"


def prepare_record(raw):
    """Turn a raw feed record into a cleaned article dict, or None if junk."""
    if raw.get("kind") == "announcement":
        # PSX titles carry no outlet suffix
        headline, outlet = clean_html(raw.get("title") or ""), None
    else:
        headline, outlet = split_outlet(raw.get("title"))
    outlet = outlet or raw.get("outlet") or outlet_from_source(raw.get("source", ""))
    if not headline or is_junk(headline):
        return None
    eff, prec = effective_time(raw.get("published_at"), config.DEFAULT_PARAMS.date_only_delay_days)
    if eff is None:
        return None
    return {
        **raw,
        "headline": headline,
        "outlet": outlet,
        "summary": clean_html(raw.get("summary") or ""),
        "effective_at": eff,
        "ts_precision": prec,
        "content_hash": content_hash(headline),
    }
