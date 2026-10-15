"""Monitoring & logging: data-quality log, source freshness, alerts, run logs."""
from __future__ import annotations

import logging
import os

from .pipeline import MacroPipeline


def setup_logging(level: str | None = None) -> None:
    logging.basicConfig(level=(level or os.environ.get("LOG_LEVEL", "INFO")).upper(),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def check_alerts(pipe: MacroPipeline, as_of=None) -> list[str]:
    """Stale sources, missing sources and recent data-quality errors. Logged at WARNING; the nightly job also stores them."""
    alerts: list[str] = []
    critical = {s for g in pipe.cfg.quality["critical_groups"].values() for f in g for s in pipe.cfg.features[f].series}
    for r in pipe.source_freshness(as_of):
        if r["status"] == "no data":
            alerts.append(f"{r['series']}: no data" + (" (CRITICAL)" if r["series"] in critical else ""))
        elif r["stale"]:
            alerts.append(f"{r['series']}: {r['status']} - {r['age_days']} days since release (limit {r['limit_days']:.0f})"
                          + (" (CRITICAL)" if r["series"] in critical else ""))
    dq = pipe.store.read_dq(limit=500)
    errs = dq[dq["severity"] == "error"] if not dq.empty else dq
    for r in errs.head(10).itertuples():
        alerts.append(f"data-quality error [{r.series_key}/{r.check}]: {r.detail}")
    for a in alerts:
        logging.getLogger("macro_agent.alerts").warning(a)
    return alerts
