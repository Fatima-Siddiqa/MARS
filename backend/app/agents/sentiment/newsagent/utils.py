import html
import re
from datetime import UTC
from urllib.parse import urlsplit, urlunsplit

from dateutil import parser as dateparser


def to_iso(value):
    """Turn an RSS date string into an ISO 8601 UTC string, or None."""
    if not value:
        return None
    try:
        dt = dateparser.parse(str(value))
    except (ValueError, OverflowError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC).isoformat()


def clean_html(text):
    """RSS summaries often contain HTML tags. Strip them."""
    text = re.sub(r"<[^>]+>", " ", text or "")
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def normalize_url(url):
    """Light normalisation for deduplication. Query strings are kept on purpose."""
    parts = urlsplit((url or "").strip())
    return urlunsplit(
        (
            parts.scheme.lower(),
            parts.netloc.lower(),
            parts.path.rstrip("/"),
            parts.query,
            "",
        )
    )
