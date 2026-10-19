"""Nightly / on-demand pipeline:
collect -> clean -> link entities -> store -> cluster -> score (FinBERT) -> aggregate -> output."""
import argparse
import json
import logging
from datetime import date, timedelta

from app.agents.sentiment.newsagent import (
    cleaning,
    config,
    dedup,
    events,
    polarity,
    psx_announcements,
    rss_ingest,
)
from app.agents.sentiment.newsagent.database import (
    create_database,
    get_processing,
    insert_article,
    insert_entities,
    record_processing,
)
from app.agents.sentiment.newsagent.engine import Engine
from app.agents.sentiment.newsagent.entities import link_entities, load_companies

log = logging.getLogger(__name__)


def store_records(conn, records):
    """Clean, link and store raw records. Returns a stats dict."""
    stats = {"seen": 0, "stored": 0, "filtered": 0, "no_match": 0, "already": 0}
    for rec in records:
        stats["seen"] += 1
        key = rec["url_key"]
        prev = get_processing(conn, key)
        if prev and prev["status"] in ("stored", "no_match", "filtered"):
            stats["already"] += 1
            continue
        prepared = cleaning.prepare_record(rec)
        if prepared is None:
            record_processing(conn, key, "filtered")
            stats["filtered"] += 1
            continue
        _, is_pk = cleaning.credibility(prepared["outlet"])
        if prepared.get("kind") == "announcement" and prepared.get("ticker"):
            ents = [
                {
                    "entity_type": "ticker",
                    "entity_id": prepared["ticker"],
                    "relevance": 1.0,
                    "matched_name": prepared.get("psx_symbol"),
                }
            ]
        else:
            ents = link_entities(prepared["headline"], prepared["summary"],
                                 prepared.get("text", ""), is_pk)
        if not ents:
            record_processing(conn, key, "no_match")
            stats["no_match"] += 1
            continue
        aid = insert_article(conn, prepared)
        if aid is None:
            stats["already"] += 1
            continue
        insert_entities(conn, aid, ents)
        record_processing(conn, key, "stored")
        stats["stored"] += 1
    conn.commit()
    return stats


def score_pending(conn, scorer, batch=256):
    """Score every stored article that has no polarity yet (inference only)."""
    n = 0
    while True:
        rows = conn.execute(
            """
            SELECT a.id, a.headline, a.summary FROM articles a
            LEFT JOIN article_scores s ON s.article_id = a.id
            WHERE s.article_id IS NULL ORDER BY a.id LIMIT ?
            """,
            (batch,),
        ).fetchall()
        if not rows:
            break
        probs = scorer.score([r["headline"] for r in rows])
        for r, (pp, pn, pu) in zip(rows, probs):
            tone = polarity.tone_from_probs(pp, pn)
            tags, adj = events.detect(f"{r['headline']}. {r['summary'] or ''}")
            conn.execute(
                """INSERT OR REPLACE INTO article_scores
                   (article_id, p_pos, p_neg, p_neu, tone, confidence, event_tags, event_adj,
                    tone_adj, scorer, model_version)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    r["id"],
                    pp,
                    pn,
                    pu,
                    tone,
                    polarity.confidence_from_probs(pu),
                    ",".join(tags),
                    adj,
                    polarity.clip(tone + adj),
                    scorer.name,
                    getattr(scorer, "version", scorer.name),
                ),
            )
        conn.commit()
        n += len(rows)
    return n


def store_outputs(conn, outputs):
    for o in outputs:
        conn.execute(
            """INSERT OR REPLACE INTO agent_outputs
               (ticker, as_of, agent, score, state, quality, output_json, model_version, params_key)
               VALUES (?,?,?,?,?,?,?,?,?)""",
                        (
                o["ticker"],
                o["as_of"],
                config.AGENT_NAME,
                o["score"],
                o["state"],
                o["quality"],
                json.dumps(o),
                o["model_version"],
                o["params_key"],
            ),
        )
    conn.commit()


def compute_outputs(conn, as_of, tickers=None, params=None):
    engine = Engine(conn, params or config.DEFAULT_PARAMS)
    outs = []
    for c in load_companies():
        if tickers and c["ticker"] not in tickers:
            continue
        outs.append(engine.output(c["ticker"], as_of))
    store_outputs(conn, outs)
    return outs


def run(db_path=None, as_of=None, publisher=True, google_days=None, psx_days=30,
        limit=None, allow_fallback=False, score=True, tickers=None):
    """tickers limits Google News, PSX announcements and outputs (publisher feeds are shared)."""
    conn = create_database(db_path)
    as_of = as_of or date.today()
    report = {"as_of": as_of.isoformat(), "failures": {}}
    try:
        records = []
        if publisher:
            r, f = rss_ingest.collect_publisher_feeds()
            records += r
            report["failures"].update(f)
        if google_days is None:
            google_days = config.GOOGLE_NEWS_NIGHTLY_LOOKBACK_DAYS
        if google_days:
            r, f = rss_ingest.collect_google_news(
                as_of - timedelta(days=google_days),
                as_of,
                tickers=tickers,
            )
            records += r
            report["failures"].update(f)
        if psx_days:
            r, f = psx_announcements.collect(
                days=psx_days,
                end=as_of,
                tickers=tickers,
            )
            records += r
            report["failures"].update(f)
        if limit:
            records = records[:limit]
        report["ingest"] = store_records(conn, records)
        report["clustered"] = dedup.assign_clusters(conn)
        if score:
            scorer = polarity.get_scorer(allow_fallback)
            report["scored"] = score_pending(conn, scorer)
            report["scorer"] = scorer.name
            report["outputs"] = len(compute_outputs(conn, as_of, tickers))
    finally:
        conn.close()
    log.info("Run complete: %s", report)
    return report


def main():
    ap = argparse.ArgumentParser(
        description="MARS Sentiment & News agent: nightly pipeline"
    )
    ap.add_argument("--db")
    ap.add_argument("--as-of", help="YYYY-MM-DD (default today)")
    ap.add_argument("--no-publisher-feeds", action="store_true")
    ap.add_argument("--google-days", type=int, help="Google News lookback in days (0 disables)")
    ap.add_argument("--psx-days", type=int, default=30,
                    help="PSX announcements lookback (0 disables)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--tickers", nargs="*")
    ap.add_argument("--no-score", action="store_true", help="ingest only")
    ap.add_argument("--allow-fallback", action="store_true",
                    help="use the lexicon fallback if FinBERT is missing (TEST RUNS ONLY)")
    a = ap.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    rep = run(
        a.db,
        date.fromisoformat(a.as_of) if a.as_of else None,
        not a.no_publisher_feeds,
        a.google_days,
        a.psx_days,
        a.limit,
        a.allow_fallback,
        not a.no_score,
        a.tickers,
    )
    print(json.dumps(rep, indent=2, default=str))


if __name__ == "__main__":
    main()
