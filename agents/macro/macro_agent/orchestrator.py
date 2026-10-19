"""Agent Orchestrator: FastAPI app + nightly scheduler.
  GET /agents/macro?ticker=&as_of=   On-Demand Handler: latest stored score, or compute a ticker score
  Nightly Scheduler                  refresh sources, precompute and store the market-level score
  Pipeline Runner                    macro_agent.pipeline.MacroPipeline (shared by live and backtest)"""
from __future__ import annotations

import logging
import os
import re
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, Header, HTTPException, Query

from .monitoring import check_alerts, setup_logging
from .pipeline import MacroPipeline, NoDataError
from .schemas import MacroOutput

log = logging.getLogger(__name__)
TICKER_RE = re.compile(r"^[A-Za-z0-9._-]{1,12}$")


def run_nightly(pipe: MacroPipeline) -> dict:
    """Refresh data, precompute the market-level score, store it, raise alerts."""
    status = pipe.refresh()
    try:
        out = pipe.score(persist=True)
        pipe.store.log_run("nightly", "ok", f"as_of={out['as_of']} state={out['state']} score={out['score']}")
    except NoDataError as e:
        pipe.store.log_run("nightly", "failed", str(e))
        out = None
    alerts = check_alerts(pipe)
    return dict(refresh=status, output=out, alerts=alerts)


def _today(pipe: MacroPipeline) -> str:
    return pd.Timestamp.now(tz=pipe.cfg.timezone).strftime("%Y-%m-%d")


def create_app(pipe: MacroPipeline | None = None, scheduler: bool | None = None) -> FastAPI:
    setup_logging()
    pipe = pipe or MacroPipeline()
    if scheduler is None:
        scheduler = os.environ.get("MACRO_SCHEDULER", "1") == "1"

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        sched = None
        if scheduler:
            from apscheduler.schedulers.background import BackgroundScheduler
            sched = BackgroundScheduler(timezone=pipe.cfg.timezone)
            sched.add_job(lambda: run_nightly(pipe), "cron", hour=int(os.environ.get("MACRO_NIGHTLY_HOUR", "2")), minute=0,
                          id="nightly_macro", max_instances=1, coalesce=True)
            sched.start()
            log.info("nightly scheduler started")
        yield
        if sched:
            sched.shutdown(wait=False)

    app = FastAPI(title="MARS Macro Agent", version=pipe.cfg.version, lifespan=lifespan)
    app.state.pipe = pipe

    @app.get("/agents/macro", response_model=MacroOutput)
    def macro(ticker: str | None = Query(None, description="PSX ticker; omit for the market-level score"),
              as_of: str | None = Query(None, description="YYYY-MM-DD; default today (PKT)")):
        if ticker is not None and not TICKER_RE.match(ticker):
            raise HTTPException(400, "invalid ticker")
        today = _today(pipe)
        if as_of is not None:
            try:
                d = pd.Timestamp(as_of)
            except ValueError:
                raise HTTPException(400, "as_of must be YYYY-MM-DD")
            if d > pd.Timestamp(today):
                raise HTTPException(400, "as_of is in the future")
            as_of = d.strftime("%Y-%m-%d")
        date = as_of or today
        key = ticker.upper() if ticker else None
        cached = pipe.store.load_output(key, date, pipe.cfg.model_version)
        if cached and key is None and date == today:                 # nightly precompute hit
            return cached
        try:
            return pipe.score(date, key, persist=True)
        except NoDataError as e:
            raise HTTPException(503, str(e))

    @app.get("/agents/macro/regime-inputs")
    def regime_inputs(as_of: str | None = None):
        try:
            return pipe.regime_inputs(as_of)
        except NoDataError as e:
            raise HTTPException(503, str(e))

    @app.get("/sources")
    def sources():
        return pipe.source_freshness()

    @app.get("/dq")
    def dq(series: str | None = None, limit: int = 100):
        return pipe.store.read_dq(series, min(limit, 1000)).to_dict("records")

    @app.get("/health")
    def health():
        rows = pipe.source_freshness()
        return dict(status="ok", model_version=pipe.cfg.model_version, stale_sources=[r["series"] for r in rows if r["stale"]])

    @app.post("/admin/refresh")
    def admin_refresh(x_admin_token: str | None = Header(None)):
        token = os.environ.get("MACRO_ADMIN_TOKEN")
        if not token or x_admin_token != token:
            raise HTTPException(403, "admin refresh disabled or bad token")
        return run_nightly(pipe)

    return app
