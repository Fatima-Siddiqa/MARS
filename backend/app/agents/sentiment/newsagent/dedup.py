"""Near-duplicate clustering: syndicated copies and re-reports are counted once
(methodology 4.3). Token-count cosine >= 0.9 within +/- dup_window_days."""
import math
import re
from collections import Counter
from datetime import datetime, timedelta

from app.agents.sentiment.newsagent.config import DEFAULT_PARAMS

STOP = set(
    "a an the of to in on for and or at by with from as is are was be its it this that said says "
    "after over amid into new".split()
)


def tokens(text):
    return [t for t in re.findall(r"[a-z0-9%]+", (text or "").lower()) if t not in STOP]


def cosine(a: Counter, b: Counter):
    if not a or not b:
        return 0.0
    dot = sum(v * b.get(k, 0) for k, v in a.items())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def assign_clusters(conn, params=DEFAULT_PARAMS):
    """Give every article without a cluster a cluster_id (the id of the earliest
    member). Returns the number of articles clustered."""
    rows = conn.execute(
        "SELECT id, headline, effective_at, cluster_id FROM articles ORDER BY effective_at, id"
    ).fetchall()
    window = timedelta(days=params.dup_window_days)
    recent = []   # (effective_dt, id, cluster_id, Counter)
    n = 0
    for r in rows:
        eff = datetime.fromisoformat(r["effective_at"])
        vec = Counter(tokens(r["headline"]))
        recent = [x for x in recent if eff - x[0] <= window]
        if r["cluster_id"] is not None:
            recent.append((eff, r["id"], r["cluster_id"], vec))
            continue
        best_cluster, best_sim = None, 0.0
        for _, _, cid, v in recent:
            sim = cosine(vec, v)
            if sim >= params.dup_cosine and sim > best_sim:
                best_cluster, best_sim = cid, sim
        cid = best_cluster if best_cluster is not None else r["id"]
        conn.execute("UPDATE articles SET cluster_id = ? WHERE id = ?", (cid, r["id"]))
        recent.append((eff, r["id"], cid, vec))
        n += 1
    conn.commit()
    return n
