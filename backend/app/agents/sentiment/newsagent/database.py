import sqlite3
from pathlib import Path

from app.agents.sentiment.newsagent import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS articles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    url_key TEXT NOT NULL UNIQUE,
    url TEXT NOT NULL,
    source TEXT NOT NULL,              -- feed key, e.g. dawn_business, google_news_ubl_2026-09-10
    kind TEXT NOT NULL DEFAULT 'news', -- news | announcement
    outlet TEXT,
    title TEXT,                        -- as published
    headline TEXT,                     -- cleaned, outlet suffix removed (model input)
    summary TEXT,
    text TEXT,                         -- optional; headlines + links are the licensed minimum
    published_at TEXT,                 -- ISO UTC as given by the source
    effective_at TEXT,                 -- when the item may first be used (no look-ahead)
    ts_precision TEXT DEFAULT 'time',  -- time | date
    fetched_at TEXT DEFAULT CURRENT_TIMESTAMP,
    content_hash TEXT,
    cluster_id INTEGER,
    psx_symbol TEXT,
    doc_url TEXT
);
CREATE INDEX IF NOT EXISTS idx_articles_effective ON articles(effective_at);
CREATE INDEX IF NOT EXISTS idx_articles_cluster ON articles(cluster_id);
CREATE INDEX IF NOT EXISTS idx_articles_hash ON articles(content_hash);

CREATE TABLE IF NOT EXISTS article_entities (
    article_id INTEGER NOT NULL REFERENCES articles(id) ON DELETE CASCADE,
    entity_type TEXT NOT NULL,         -- ticker | sector | market
    entity_id TEXT NOT NULL,
    relevance REAL NOT NULL,           -- 1.0 headline, 0.5 body only, 0.3 sector/market
    matched_name TEXT,
    PRIMARY KEY (article_id, entity_type, entity_id)
);
CREATE INDEX IF NOT EXISTS idx_entities_entity ON article_entities(entity_type, entity_id);

CREATE TABLE IF NOT EXISTS article_scores (
    article_id INTEGER PRIMARY KEY REFERENCES articles(id) ON DELETE CASCADE,
    p_pos REAL, p_neg REAL, p_neu REAL,
    tone REAL,                         -- P(pos) - P(neg)
    confidence REAL,                   -- 1 - P(neu)
    event_tags TEXT,                   -- comma separated
    event_adj REAL DEFAULT 0,
    tone_adj REAL,                     -- tone + event adjustment, clipped to [-1, 1]
    scorer TEXT,                       -- finbert | lexicon-fallback
    model_version TEXT,
    scored_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS url_processing (
    url TEXT PRIMARY KEY,
    status TEXT NOT NULL,              -- stored | no_match | filtered | failed
    attempts INTEGER NOT NULL DEFAULT 1,
    last_error TEXT,
    last_processed_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS sentiment_daily (
    entity_type TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    as_of TEXT NOT NULL,
    params_key TEXT NOT NULL,
    n_articles INTEGER, n_eff REAL,
    net_tone REAL, tone_change REAL, neg_share REAL,
    agreement REAL, attention REAL,
    z_net_tone REAL, z_tone_change REAL, z_neg_share REAL, z_attention REAL,
    raw_score REAL, last_article_at TEXT,
    PRIMARY KEY (entity_type, entity_id, as_of, params_key)
);

CREATE TABLE IF NOT EXISTS agent_outputs (
    ticker TEXT NOT NULL,
    as_of TEXT NOT NULL,
    agent TEXT NOT NULL DEFAULT 'sentiment_news',
    score REAL, state TEXT, quality REAL,
    output_json TEXT NOT NULL,
    model_version TEXT,
    params_key TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ticker, as_of, agent)
);

CREATE TABLE IF NOT EXISTS labelled_headlines (
    article_id INTEGER PRIMARY KEY REFERENCES articles(id) ON DELETE CASCADE,
    annotator_1 TEXT, annotator_2 TEXT, consensus TEXT
);

CREATE TABLE IF NOT EXISTS backfill_progress (
    key TEXT PRIMARY KEY,
    done_at TEXT DEFAULT CURRENT_TIMESTAMP,
    n_items INTEGER DEFAULT 0
);
"""


def create_database(path=None):
    path = Path(path or config.DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def get_processing(conn, url):
    return conn.execute(
        "SELECT status, attempts, last_error FROM url_processing WHERE url = ?", (url,)
    ).fetchone()


def record_processing(conn, url, status, error=None):
    conn.execute(
        """
        INSERT INTO url_processing (url, status, attempts, last_error, last_processed_at)
        VALUES (?, ?, 1, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(url) DO UPDATE SET
            status = excluded.status,
            attempts = url_processing.attempts + 1,
            last_error = excluded.last_error,
            last_processed_at = CURRENT_TIMESTAMP
        """,
        (url, status, (str(error)[:1000] if error else None)),
    )


def insert_article(conn, rec):
    """Insert one article; return its id, or None if the url already exists."""
    cur = conn.execute(
        """
        INSERT OR IGNORE INTO articles
            (url_key, url, source, kind, outlet, title, headline, summary, text,
             published_at, effective_at, ts_precision, content_hash, psx_symbol, doc_url)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            rec["url_key"], rec["url"], rec["source"], rec.get("kind", "news"),
            rec.get("outlet"), rec.get("title"), rec.get("headline"),
            rec.get("summary"), rec.get("text"), rec.get("published_at"),
            rec.get("effective_at"), rec.get("ts_precision", "time"),
            rec.get("content_hash"), rec.get("psx_symbol"), rec.get("doc_url"),
        ),
    )
    if cur.rowcount == 0:
        return None
    return cur.lastrowid


def insert_entities(conn, article_id, entities):
    for e in entities:
        conn.execute(
            """
            INSERT OR REPLACE INTO article_entities
                (article_id, entity_type, entity_id, relevance, matched_name)
            VALUES (?, ?, ?, ?, ?)
            """,
            (article_id, e["entity_type"], e["entity_id"], e["relevance"], e.get("matched_name")),
        )
