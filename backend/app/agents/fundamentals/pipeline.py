"""Daily refresh of everything the fundamentals agent uses (official PSX data only).

Steps (each is logged; a failing step does not stop the others):
1. prices: official PSX daily market summaries up to the latest trading day;
2. PSX company pages of the reference companies (figures, shares, filing dates);
3. once a week (or --full): the whole market, for peers and model training;
4. new filings read automatically (auto_extract); anything unverified -> review_queue.csv;
5. validation of all keyed figures (validate_official);
6. retrain + recalibrate the valuation model when new annual figures arrived since the last
   training, or the model is older than RETRAIN_DAYS.
A summary line is appended to data/processed/fundamentals/refresh_log.jsonl.

    python -m app.agents.fundamentals.pipeline          # daily
    python -m app.agents.fundamentals.pipeline --full   # also re-collect the whole market
Scheduled daily by scripts/refresh_fundamentals.cmd (Windows Task Scheduler, see docs).
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import traceback
from datetime import UTC, date, datetime

import pandas as pd

from app.agents.fundamentals.data import auto_extract, psx_official, validate_official
from app.agents.fundamentals.valuation import calibrate, train
from app.agents.fundamentals.valuation.dataset import PROCESSED
from app.ingestion import psx_eod

log = logging.getLogger("fundamentals.pipeline")
LOG_FILE = PROCESSED / "refresh_log.jsonl"
WHOLE_MARKET_DAYS = 7
RETRAIN_DAYS = 30


def _age_days(path) -> float | None:
    return (time.time() - path.stat().st_mtime) / 86400 if path.exists() else None


def _new_annuals_since_training() -> int:
    """Annual figures published after the model was trained."""
    if not train.MODEL_PATH.exists() or not psx_official.STATEMENTS_CSV.exists():
        return 0
    trained = datetime.fromtimestamp(train.MODEL_PATH.stat().st_mtime).date()
    s = pd.read_csv(psx_official.STATEMENTS_CSV)
    s = s[(s["period_kind"] == "annual") & (s["line_item"] == "net_income")]
    return int((pd.to_datetime(s["published_at"]).dt.date > trained).sum())


def run(full: bool = False) -> dict:
    out: dict = {"started": datetime.now(UTC).isoformat(timespec="seconds"), "steps": {}}

    def step(name, fn):
        t0 = time.time()
        try:
            res = fn()
            out["steps"][name] = {"ok": True, "result": res, "seconds": round(time.time() - t0)}
        except Exception as exc:  # noqa: BLE001 - one failing step must not stop the refresh
            log.exception("step %s failed", name)
            out["steps"][name] = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "trace": traceback.format_exc(limit=3),
            }

    step("prices", psx_eod.refresh)
    step(
        "reference_companies", lambda: {t: r.get("rows") for t, r in psx_official.collect().items()}
    )
    age = _age_days(psx_official.UNIVERSE_CSV)
    if full or age is None or age > WHOLE_MARKET_DAYS:
        step("whole_market", lambda: {"companies": len(psx_official.collect(whole_market=True))})
    step("new_filings", auto_extract.run)
    step(
        "validation",
        lambda: {
            k: v
            for k, v in validate_official.run().items()
            if k in ("ok", "balance_identity", "text_pdf_mismatches", "needs_keying", "date_errors")
        },
    )
    model_age = _age_days(train.MODEL_PATH)
    new_annuals = _new_annuals_since_training()
    if model_age is None or model_age > RETRAIN_DAYS or new_annuals > 0:
        step("retrain", lambda: {k: train.train()[k] for k in ("chosen", "why", "pooled_test")})
        step(
            "calibrate",
            lambda: {
                k: calibrate.calibrate().get(k)
                for k in ("threshold", "supported", "spearman_ic", "samples")
            },
        )
    else:
        out["steps"]["retrain"] = {
            "ok": True,
            "result": f"skipped: model {model_age:.0f} days old, no new annual figures",
        }
    out["finished"] = datetime.now(UTC).isoformat(timespec="seconds")
    out["ok"] = all(s.get("ok") for s in out["steps"].values())
    PROCESSED.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(out, default=str) + "\n")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Refresh fundamentals data, model and calibration")
    ap.add_argument("--full", action="store_true", help="also re-collect the whole market")
    a = ap.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    res = run(a.full)
    print(
        json.dumps(
            {
                k: {"ok": v.get("ok"), **({"error": v["error"]} if "error" in v else {})}
                for k, v in res["steps"].items()
            },
            indent=2,
        )
    )
    print(f"run {date.today()} ok={res['ok']}; details in {LOG_FILE}")


if __name__ == "__main__":
    main()
