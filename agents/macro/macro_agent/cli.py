"""macro-agent CLI."""
from __future__ import annotations

import argparse
import json
import sys

from .config import load_config
from .monitoring import setup_logging


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="macro-agent")
    ap.add_argument("--config", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("refresh", help="fetch all sources into macro_raw (versioned) and log data quality")
    sub.add_parser("nightly", help="refresh + precompute market-level score + alerts")
    fo = sub.add_parser("fetch-open", help="download real series from open sources (World Bank, BIS, IMF, FRED) into data/drop")
    fo.add_argument("--only", nargs="*")
    ml = sub.add_parser("ml-eval", help="research: monthly walk-forward of ML challengers vs the rules, with ablations")
    ml.add_argument("--out", default=None)
    sub.add_parser("demo-data", help="write SYNTHETIC csvs to data/drop and load them (offline demo/tests only)")
    s = sub.add_parser("score", help="print the output JSON")
    s.add_argument("--ticker"); s.add_argument("--as-of"); s.add_argument("--llm", action="store_true")
    sv = sub.add_parser("serve", help="run the FastAPI service")
    sv.add_argument("--host", default="0.0.0.0"); sv.add_argument("--port", type=int, default=8000)
    e = sub.add_parser("eval", help="offline walk-forward evaluation")
    e.add_argument("--lockbox", action="store_true", help="also run the untouched final period (once)")
    e.add_argument("--force", action="store_true"); e.add_argument("--ticker")
    a = ap.parse_args(argv)

    setup_logging()
    from .pipeline import MacroPipeline
    cfg = load_config(a.config)
    pipe = MacroPipeline(cfg)

    if a.cmd == "refresh":
        print(json.dumps(pipe.refresh(), indent=2))
    elif a.cmd == "nightly":
        from .orchestrator import run_nightly
        r = run_nightly(pipe)
        print(json.dumps({"alerts": r["alerts"], "output": r["output"]}, indent=2))
    elif a.cmd == "fetch-open":
        from .fetch_open import fetch_all
        print(json.dumps(fetch_all(cfg, a.only), indent=2))
    elif a.cmd == "ml-eval":
        from .ml_research import run_ml_research
        res = run_ml_research(pipe, out_dir=a.out)
        print(res.path_md.read_text())
        print(f"written: {res.path_json}")
    elif a.cmd == "demo-data":
        import os

        from .devdata import write_drop_files
        os.environ["MACRO_AGENT_OFFLINE"] = "1"
        write_drop_files(cfg)
        print(json.dumps(pipe.refresh(), indent=2))
        print("\n*** SYNTHETIC DATA loaded - do not use for research results ***")
    elif a.cmd == "score":
        print(json.dumps(pipe.score(a.as_of, a.ticker, use_llm=True if a.llm else None), indent=2))
    elif a.cmd == "serve":
        import uvicorn

        from .orchestrator import create_app
        uvicorn.run(create_app(pipe), host=a.host, port=a.port)
    elif a.cmd == "eval":
        from .evaluation import run_evaluation
        res = run_evaluation(pipe, lockbox=a.lockbox, force=a.force, ticker=a.ticker)
        print(res.path_md.read_text())
        print(f"written: {res.path_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
