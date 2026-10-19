"""Explanations. Numbers come from the output JSON; the optional local LLM
(Ollama) may only reword, and any rewrite that introduces or changes a number
is rejected in favour of the template."""
import json
import re

import requests

from app.agents.sentiment.newsagent import config

NUM = re.compile(
    r"[-+]?\d+(?:\\.\d+)?"
)


def _fmt_drivers(drivers):
    names = {
        "net_tone": "overall news tone",
        "tone_change": "change in tone versus the last 90 days",
        "negative_share": "share of negative news",
    }
    parts = []
    for d in drivers:
        parts.append(
            f"{names.get(d['name'], d['name'])} "
            f"({d['direction']}, z={d['z']:+.2f})"
        )
    return "; ".join(parts)


def template_explanation(out):
    ctx = out["context"]
    if "no_recent_news" in out["flags"] and not ctx.get("articles_in_window"):
                return (
            f"No recent news was found for {out['ticker']} in the last "
            f"{ctx['window_days']} days, "
            f"so sentiment is "
            + (
                "neutral"
                if not ctx.get("parent")
                else f"taken from the {ctx['parent']} level"
            )
            + f" ({out['score']:+.2f}) with quality {out['quality']:.2f}."
        )
    text = (
        f"Sentiment for {out['ticker']} is {out['state']} "
        f"(score {out['score']:+.2f}, "
        f"quality {out['quality']:.2f}) based on "
        f"{ctx['articles_in_window']} distinct news items "
        f"from {len(ctx['outlets'])} outlets in the last "
        f"{ctx['window_days']} days."
    )
    if out["drivers"]:
        text += " Main drivers: " + _fmt_drivers(out["drivers"]) + "."
    if "insufficient_history" in out["flags"]:
        text += " There is not yet enough history to standardise the "
        "tone, so the score stays near neutral."
    elif any(f.startswith("shrunk_to_") for f in out["flags"]):
        text += (
            " There is little company-specific news, so the score leans toward the "
            f"{ctx['parent'] or 'neutral'} level."
        )
    if out["evidence"]:
        e = out["evidence"][0]
        text += (
            f" Most influential item: \"{e['title']}\" "
            f"({e['outlet'] or 'unknown outlet'}, "
            f"{(e['published_at'] or '')[:10]})."
        )
    return text


def facts_numbers(out):
    """Every number a rewrite is allowed to contain."""
    allowed = set()
    for n in NUM.findall(json.dumps(out)):
        allowed.add(n.lstrip("+"))
    for n in NUM.findall(template_explanation(out)):
        allowed.add(n.lstrip("+"))
    return allowed


def numbers_ok(text, out):
    allowed_f = set()
    for a in facts_numbers(out):
        try:
            allowed_f.add(round(float(a), 2))
        except ValueError:
            pass
    for n in NUM.findall(text):
        try:
            if round(float(n.lstrip("+")), 2) not in allowed_f:
                return False
        except ValueError:
            return False
    return True


def llm_rewrite(out, level="beginner", timeout=30):
    """Return a plain-language rewrite or None (unavailable / rejected)."""
    base = template_explanation(out)
    prompt = (
        f"Rewrite the following for a {level} investor in at most 3 short sentences. "
        "Do not add, change or round any number. Do not give buy or sell advice.\n\n"
        + base
    )
    try:
        r = requests.post(
            f"{config.OLLAMA_URL}/api/generate",
            json={
                "model": config.OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
            },
            timeout=timeout,
        )
        r.raise_for_status()
        text = r.json().get("response", "").strip()
    except Exception:
        return None
    return text if text and numbers_ok(text, out) else None


def explain(out, use_llm=False, level="beginner"):
    result = {"text": template_explanation(out), "source": "template"}
    if use_llm:
        rewritten = llm_rewrite(out, level)
        if rewritten:
            result = {"text": rewritten, "source": "llm_checked"}
    return result
