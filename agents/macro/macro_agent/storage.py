"""PostgreSQL / SQLite storage. macro_raw is append-only and versioned: a re-download adds
a new version, never overwrites (methodology 3.1)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy import (Column, DateTime, Float, Integer, MetaData, String, Table, Text, UniqueConstraint,
                        create_engine, func, select, text)
from sqlalchemy.engine import Engine

MARKET_KEY = "_MARKET_"
md = MetaData()

macro_raw = Table(
    "macro_raw", md,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("series_key", String(64), nullable=False, index=True),
    Column("obs_date", String(10), nullable=False),
    Column("value", Float, nullable=False),
    Column("unit", String(64)),
    Column("status", String(32)),
    Column("source_url", Text),
    Column("download_ts", DateTime(timezone=True), nullable=False),
    Column("version", Integer, nullable=False),
    UniqueConstraint("series_key", "obs_date", "version", name="uq_raw_version"),
)
dq_log = Table(
    "dq_log", md,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("run_ts", DateTime(timezone=True), nullable=False),
    Column("series_key", String(64), index=True),
    Column("obs_date", String(10)),
    Column("check", String(48)),
    Column("severity", String(16)),
    Column("detail", Text),
)
features_tbl = Table(
    "features", md,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("as_of", String(10), index=True),
    Column("feature", String(48)),
    Column("raw_value", Float),
    Column("score", Float),
    Column("age_days", Float),
    Column("model_version", String(64)),
    UniqueConstraint("as_of", "feature", "model_version", name="uq_feature"),
)
pillar_scores = Table(
    "pillar_scores", md,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("ticker_key", String(16), nullable=False),
    Column("as_of", String(10), nullable=False, index=True),
    Column("pillar", String(24)),
    Column("score", Float),
    Column("model_version", String(64)),
    UniqueConstraint("ticker_key", "as_of", "pillar", "model_version", name="uq_pillar"),
)
agent_outputs = Table(
    "agent_outputs", md,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("agent", String(24), nullable=False),
    Column("ticker_key", String(16), nullable=False),      # '_MARKET_' for the market-level score
    Column("as_of", String(10), nullable=False, index=True),
    Column("score", Float),
    Column("state", String(12)),
    Column("confidence", Float),
    Column("payload", Text),                                # full JSON contract
    Column("model_version", String(64), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint("agent", "ticker_key", "as_of", "model_version", name="uq_agent_output"),
)
run_log = Table(
    "run_log", md,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("run_ts", DateTime(timezone=True), nullable=False),
    Column("kind", String(24)),
    Column("status", String(16)),
    Column("detail", Text),
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Store:
    def __init__(self, url: str, root: Path | None = None):
        if url.startswith("sqlite:///") and not url.startswith("sqlite:////") and root is not None:
            rel = url.replace("sqlite:///", "")
            p = Path(root) / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            url = f"sqlite:///{p}"
        self.engine: Engine = create_engine(url, future=True)
        md.create_all(self.engine)

    # ---- raw ------------------------------------------------------------------------------
    def append_raw(self, series_key: str, df: pd.DataFrame, unit: str, source_url: str,
                   download_ts: datetime | None = None) -> int:
        """Append observations as a new version. Returns rows written (0 if nothing changed)."""
        if df.empty:
            return 0
        download_ts = download_ts or _now()
        df = df.sort_values("obs_date").drop_duplicates("obs_date", keep="last")
        with self.engine.begin() as cx:
            cur = cx.execute(select(macro_raw.c.obs_date, macro_raw.c.value, macro_raw.c.version)
                             .where(macro_raw.c.series_key == series_key)).fetchall()
            latest: dict[str, tuple[float, int]] = {}
            for d, v, ver in cur:
                if d not in latest or ver > latest[d][1]:
                    latest[d] = (v, ver)
            rows = []
            for r in df.itertuples():
                d = pd.Timestamp(r.obs_date).strftime("%Y-%m-%d")
                prev = latest.get(d)
                if prev is not None and abs(prev[0] - float(r.value)) < 1e-12:
                    continue                      # unchanged -> do not add a version
                rows.append(dict(series_key=series_key, obs_date=d, value=float(r.value), unit=unit,
                                 status=str(getattr(r, "status", "final")), source_url=source_url,
                                 download_ts=download_ts, version=(prev[1] + 1) if prev else 1))
            if rows:
                cx.execute(macro_raw.insert(), rows)
            return len(rows)

    def read_raw_latest(self, series_key: str) -> pd.DataFrame:
        """Latest version per observation date."""
        q = select(macro_raw.c.obs_date, macro_raw.c.value, macro_raw.c.unit, macro_raw.c.status,
                   macro_raw.c.source_url, macro_raw.c.download_ts, macro_raw.c.version
                   ).where(macro_raw.c.series_key == series_key)
        df = pd.read_sql(q, self.engine)
        if df.empty:
            return df
        df = df.sort_values(["obs_date", "version"]).drop_duplicates("obs_date", keep="last")
        df["obs_date"] = pd.to_datetime(df["obs_date"])
        return df.reset_index(drop=True)

    def raw_version_count(self, series_key: str) -> int:
        with self.engine.connect() as cx:
            return cx.execute(select(func.count()).select_from(macro_raw)
                              .where(macro_raw.c.series_key == series_key)).scalar_one()

    def data_fingerprint(self) -> str:
        with self.engine.connect() as cx:
            n, mx = cx.execute(select(func.count(), func.max(macro_raw.c.id)).select_from(macro_raw)).one()
        return f"{n}:{mx}"

    # ---- logs -----------------------------------------------------------------------------
    def log_dq(self, issues: list[dict]) -> None:
        if not issues:
            return
        ts = _now()
        with self.engine.begin() as cx:
            cx.execute(dq_log.insert(), [dict(run_ts=ts, **i) for i in issues])

    def read_dq(self, series_key: str | None = None, limit: int = 200) -> pd.DataFrame:
        q = select(dq_log).order_by(dq_log.c.id.desc()).limit(limit)
        if series_key:
            q = q.where(dq_log.c.series_key == series_key)
        return pd.read_sql(q, self.engine)

    def log_run(self, kind: str, status: str, detail: str = "") -> None:
        with self.engine.begin() as cx:
            cx.execute(run_log.insert().values(run_ts=_now(), kind=kind, status=status, detail=detail))

    # ---- outputs --------------------------------------------------------------------------
    def save_output(self, out: dict, features: pd.DataFrame | None = None) -> None:
        tk = out["ticker"] or MARKET_KEY
        with self.engine.begin() as cx:
            for t, key in ((agent_outputs, ("agent", "ticker_key", "as_of", "model_version")),):
                cx.execute(t.delete().where(
                    (t.c.agent == out["agent"]) & (t.c.ticker_key == tk) & (t.c.as_of == out["as_of"])
                    & (t.c.model_version == out["model_version"])))
            cx.execute(agent_outputs.insert().values(
                agent=out["agent"], ticker_key=tk, as_of=out["as_of"], score=out["score"], state=out["state"],
                confidence=out["confidence"], payload=json.dumps(out), model_version=out["model_version"],
                created_at=_now()))
            cx.execute(pillar_scores.delete().where(
                (pillar_scores.c.ticker_key == tk) & (pillar_scores.c.as_of == out["as_of"])
                & (pillar_scores.c.model_version == out["model_version"])))
            rows = [dict(ticker_key=tk, as_of=out["as_of"], pillar=p, score=s, model_version=out["model_version"])
                    for p, s in out["pillars"].items() if s is not None]
            if rows:
                cx.execute(pillar_scores.insert(), rows)
            if features is not None and tk == MARKET_KEY and not features.empty:
                cx.execute(features_tbl.delete().where(
                    (features_tbl.c.as_of == out["as_of"]) & (features_tbl.c.model_version == out["model_version"])))
                cx.execute(features_tbl.insert(), [
                    dict(as_of=out["as_of"], feature=f, raw_value=_f(r.raw), score=_f(r.score), age_days=_f(r.age),
                         model_version=out["model_version"]) for f, r in features.iterrows()])

    def load_output(self, ticker: str | None, as_of: str, model_version: str) -> dict | None:
        with self.engine.connect() as cx:
            row = cx.execute(select(agent_outputs.c.payload).where(
                (agent_outputs.c.agent == "macro") & (agent_outputs.c.ticker_key == (ticker or MARKET_KEY))
                & (agent_outputs.c.as_of == as_of) & (agent_outputs.c.model_version == model_version))).first()
        return json.loads(row[0]) if row else None

    def source_status(self) -> pd.DataFrame:
        q = text("SELECT series_key, MAX(obs_date) AS last_obs, MAX(download_ts) AS last_download, "
                 "COUNT(*) AS rows FROM macro_raw GROUP BY series_key")
        return pd.read_sql(q, self.engine)


def _f(x) -> float | None:
    return None if x is None or (isinstance(x, float) and pd.isna(x)) else float(x)
