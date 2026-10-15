"""Optional local LLM (Ollama) that may only REWRITE wording. Every number in the rewrite is checked against the
facts JSON; any mismatch falls back to the deterministic template. The model never produces or changes numbers."""
from __future__ import annotations

import logging
import re

import requests

log = logging.getLogger(__name__)
NUM_RE = re.compile(r"(?<![\w.])[-+−]?\d[\d,]*(?:\.\d+)?")


def _tokens(text: str) -> list[float]:
    out = []
    for m in NUM_RE.findall(text.replace("−", "-")):
        try:
            out.append(abs(float(m.replace(",", ""))))
        except ValueError:
            pass
    return out


def allowed_numbers(facts) -> set[float]:
    """Every number that appears in the facts (numeric values at 0-3 decimals, plus digits inside any string,
    e.g. dates and '6-month' in labels)."""
    allowed: set[float] = set()

    def walk(x):
        if isinstance(x, bool) or x is None:
            return
        if isinstance(x, (int, float)):
            for d in range(4):
                allowed.add(round(abs(float(x)), d))
        elif isinstance(x, str):
            allowed.update(_tokens(x))
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                walk(v)

    walk(facts)
    return allowed


def validate_numbers(text: str, facts) -> tuple[bool, list[float]]:
    allowed = allowed_numbers(facts)
    bad = [n for n in _tokens(text) if n not in allowed]
    return (not bad), bad


def rewrite(template_text: str, facts: dict, llm_cfg: dict) -> tuple[str, bool]:
    """Returns (text, used_llm). Falls back to the template on any error or number mismatch."""
    if not llm_cfg.get("enabled"):
        return template_text, False
    prompt = (
        "Rewrite the following explanation in clear, friendly plain English for a non-expert investor. "
        "Keep every number exactly as written. Do not add, remove or change any number, date or fact, "
        "and do not give investment advice.\n\n" + template_text)
    try:
        r = requests.post(llm_cfg["url"], json={"model": llm_cfg["model"], "prompt": prompt, "stream": False},
                          timeout=llm_cfg.get("timeout_s", 30))
        r.raise_for_status()
        text = (r.json().get("response") or "").strip()
    except Exception as e:                       # noqa: BLE001 - any failure -> template
        log.warning("LLM rewrite failed (%s); using template", e)
        return template_text, False
    ok, bad = validate_numbers(text, facts)
    if not text or not ok:
        log.warning("LLM rewrite rejected, numbers not in facts: %s", bad)
        return template_text, False
    return text, True
