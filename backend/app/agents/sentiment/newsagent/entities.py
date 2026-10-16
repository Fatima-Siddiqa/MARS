"""Entity linking: article -> ticker (relevance 1.0 headline / 0.5 body),
sector (0.3) or market (0.3) bucket. Replaces the old name-only matcher with
extra guards for ambiguous names (Systems Limited, United Bank, HBL ...):
a Pakistan-context test, an India-context rejection and a 'preceded by another
proper noun' test."""
import csv
import re
import unicodedata
from functools import lru_cache

from app.agents.sentiment.newsagent import config

PK_CONTEXT = re.compile(
    r"\b(pakistan|pakistani|psx|kse[- ]?100|kse|karachi|lahore|islamabad|secp|sbp|"
    r"state bank|rs\s?\d|pkr|rupee|ogra|nepra|fbr|propakistani|mettis|brecorder|dawn)\b", re.I)

INDIA_CONTEXT = re.compile(
    r"(\bNSE\s?:|\bBSE\s?:|\u20b9|\bcrores?\b|\blakhs?\b|\bSEBI\b|\bSensex\b|\bNifty\b)", re.I)

# A capitalised word right before the company name usually means it is a
# different company ("Enghouse Systems Limited", "Tata Advanced Systems Limited").
ALLOWED_PREFIX = {"the", "pakistan", "pakistans", "pakistani", "by", "of", "a", "an", "and", "with",
                  "at", "for", "from", "to", "in", "on", "as", "says", "tips", "hbl", "insider"}


def norm(text):
    text = unicodedata.normalize("NFKC", text or "").replace("\u2019", "'").replace("\u2018", "'")
    return re.sub(r"\s+", " ", text).strip()


@lru_cache(maxsize=1)
def load_companies():
    with open(config.COMPANIES_CSV, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


@lru_cache(maxsize=1)
def load_sectors():
    with open(config.SECTORS_CSV, newline="", encoding="utf-8-sig") as fh:
        return {r["sector"]: [k.strip() for k in r["keywords"].split("|") if k.strip()] for r in csv.DictReader(fh)}


@lru_cache(maxsize=1)
def load_market_keywords():
    with open(config.MARKET_CSV, newline="", encoding="utf-8-sig") as fh:
        row = next(csv.DictReader(fh))
    return [k.strip() for k in row["keywords"].split("|") if k.strip()]


def sector_of(ticker):
    for c in load_companies():
        if c["ticker"] == ticker:
            return c.get("sector") or None
    return None


def _names(company):
    names = [company["company_name"].strip()]
    names += [a.strip() for a in (company.get("aliases") or "").split("|") if a.strip()]
    seen, out = set(), []
    for n in names:
        if n.lower() not in seen:
            seen.add(n.lower())
            out.append(n)
    return out


def _find(text, phrase):
    pattern = r"(?<![\w])" + re.escape(phrase) + r"(?![\w])"
    return re.search(pattern, text, flags=re.I)


def _bad_prefix(text, match):
    """True if the match is glued to a preceding proper noun."""
    before = text[: match.start()].rstrip()
    if not before:
        return False

    # Allow the ticker-prefixed insider-news labels used by Investify.
    if re.search(r"\bSYS\s+Insider\s+(Buy|Sell)\s*:\s*$", before, re.I):
        return False

    token = before.split()[-1].strip(chr(34) + chr(39) + "(),:;")
    key = token.lower().replace("'s", "").replace("'", "")
    if not token or key in ALLOWED_PREFIX:
        return False
    return token[0].isupper()


def _link_tickers(headline, summary, body, outlet_is_pk):
    head = norm(headline)
    rest = norm(" ".join([summary or "", body or ""]))
    broad_market = _is_broad_market_headline(headline)
    hits = []

    for c in load_companies():
        needs_ctx = c.get("require_context") == "1"
        best = None

        for name in _names(c):
            # For broad market headlines, only a company name in the
            # headline itself can establish a ticker link.
            texts = [(head, 1.0, True)]
            if not broad_market:
                texts.append((rest, 0.5, False))

            for text, rel, is_headline in texts:
                m = _find(text, name)
                if not m:
                    continue

                if needs_ctx:
                    both = head + " " + rest
                    if _bad_prefix(text, m) or INDIA_CONTEXT.search(both):
                        continue

                    extra = [
                        t for t in (c.get("context_terms") or "").split("|")
                        if t
                    ]
                    if not (
                        outlet_is_pk
                        or PK_CONTEXT.search(both)
                        or any(_find(both, t) for t in extra)
                    ):
                        continue

                if best is None or rel > best[0]:
                    best = (rel, name)

        if best:
            hits.append({
                "entity_type": "ticker",
                "entity_id": c["ticker"],
                "relevance": best[0],
                "matched_name": best[1],
            })

    return hits



def _is_broad_market_headline(headline):
    """Detect market-wide movement headlines, not company-specific news."""
    head = norm(headline)

    market_subject = re.search(
        r"^\s*(?:PSX|KSE[- ]?100|KSE100|"
        r"Pakistan Stock Exchange|stock market|equities|shares|"
        r"stocks|market|benchmark index|bourse)\b",
        head,
        re.I,
    )
    market_movement = re.search(
        r"\b(?:fall|falls|fell|drop|drops|dropped|decline|declines|"
        r"declined|lose|loses|lost|losses|retreat|retreats|retreated|"
        r"slide|slides|slid|plunge|plunges|plunged|"
        r"rise|rises|rose|gain|gains|gained|rally|rallies|rallied)\b",
        head,
        re.I,
    )
    return bool(market_subject and market_movement)



def link_entities(headline, summary="", body="", outlet_is_pk=False):
    """Return a list of entity dicts for one article ([] means not relevant)."""
    ents = _link_tickers(headline, summary, body, outlet_is_pk)
    if ents:
        return ents
    # No company named: sector / market buckets, but only with Pakistan context.
    text = norm(" ".join([headline or "", summary or ""]))
    if not (outlet_is_pk or PK_CONTEXT.search(text)):
        return []
    out = []
    for sector, kws in load_sectors().items():
        for kw in kws:
            if _find(text, kw):
                out.append({"entity_type": "sector", "entity_id": sector, "relevance": 0.3, "matched_name": kw})
                break
    for kw in load_market_keywords():
        if _find(text, kw):
            out.append({"entity_type": "market", "entity_id": "PSX", "relevance": 0.3, "matched_name": kw})
            break
    return out



