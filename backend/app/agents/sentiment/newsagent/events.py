"""Rule-based event layer. FinBERT cannot see magnitude, so tagged events
(results beat/miss, dividend change, default, regulatory action, rate decision,
circular debt) add a bounded adjustment of at most +/-event_cap to the tone."""
import re

from app.agents.sentiment.newsagent.config import DEFAULT_PARAMS

PCT = re.compile(r"(\d{1,3}(?:\.\d+)?)\s?(?:%|pc|percent|per cent)", re.I)

# (tag, regex, direction). Starting rule set: extend as you read more headlines.
RULES = [
    ("profit_up", re.compile(r"\b(profit|earnings|net income|pat)\b.*\b(up|rise|rises|rose|jump|jumps|surge|surges|soar|soars|grow|grows|record|doubles|triples|increase|increases)\b|\brecord (profit|earnings)\b|\bbeats? (estimates|expectations)\b", re.I), +1),
    ("profit_down", re.compile(r"\b(profit|earnings|net income|pat)\b.*\b(down|fall|falls|fell|drop|drops|plunge|plunges|slump|slumps|decline|declines|shrink|shrinks|decrease)\b|\b(net )?loss(es)?\b|\bmisses? (estimates|expectations)\b", re.I), -1),
    ("dividend_up", re.compile(r"\b(declares?|announces?|approves?|recommends?|payout)\b.*\b(cash )?dividend\b|\b(interim|final) (cash )?dividend\b|\bbonus shares?\b", re.I), +1),
    ("dividend_cut", re.compile(r"\b(no|skips?|omits?|suspends?|cuts?|slashes) (the )?(cash )?dividend\b|\bdividend (cut|suspended|omitted)\b", re.I), -1),
    ("default", re.compile(r"\b(default|defaults|defaulted|downgrade|downgrades|downgraded|insolven\w+|bankrupt\w*|fraud|raid|raided|shut ?down|shuts down|suspends? (operations|production))\b", re.I), -1),
    ("upgrade", re.compile(r"\b(upgrade|upgrades|upgraded|outperform|top pick|target price (raised|hiked))\b", re.I), +1),
    ("regulatory_neg", re.compile(r"\b(SECP|OGRA|NEPRA|PTA|NAB|FBR|CCP|competition commission)\b.*\b(penalt\w+|fine[sd]?|show[- ]cause|ban|bans|probe|investigation|notice)\b", re.I), -1),
    ("regulatory_pos", re.compile(r"\b(SECP|OGRA|NEPRA|PTA|ECC|cabinet)\b.*\b(approves?|approved|grants?|allows?|clears?|raises tariff|tariff hike)\b", re.I), +1),
    ("rate_cut", re.compile(r"\b(SBP|policy rate|monetary policy)\b.*\b(cuts?|slash\w*|lowers?|reduces?|eases)\b", re.I), +1),
    ("rate_hike", re.compile(r"\b(SBP|policy rate|monetary policy)\b.*\b(raises?|hikes?|increases?|tightens?)\b", re.I), -1),
    ("circular_debt_pay", re.compile(r"\bcircular debt\b.*\b(payment|payments|settle\w*|clear\w*|receives?|released?)\b|\b(receives?|released?)\b.*\bcircular debt\b", re.I), +1),
    ("circular_debt_neg", re.compile(r"\bcircular debt\b.*\b(rises?|swells?|piles?|balloons?|mounts?|crosses?)\b", re.I), -1),
]

OPPOSITES = [("profit_up", "profit_down"), ("dividend_up", "dividend_cut"),
             ("rate_cut", "rate_hike"), ("circular_debt_pay", "circular_debt_neg"),
             ("regulatory_pos", "regulatory_neg")]

def detect(text):
    """Return (tags, adjustment). Adjustment is in [-cap, +cap]."""
    text = text or ""
    pct = PCT.search(text)
    magnitude = min(1.0, float(pct.group(1)) / 100.0) if pct else 0.0
    hits = [(tag, sign) for tag, rx, sign in RULES if rx.search(text)]
    names = {t for t, _ in hits}
    for a, b in OPPOSITES:
        if a in names and b in names:          # ambiguous: drop both
            hits = [(t, s) for t, s in hits if t not in (a, b)]
    tags = [t for t, _ in hits]
    adj = sum(s * (0.10 + 0.10 * magnitude) for _, s in hits)
    cap = DEFAULT_PARAMS.event_cap
    return tags, max(-cap, min(cap, adj))
