"""Run the technical agent: python -m scripts.run_technical HBL [--as-of 2026-07-14] [--chart]"""
import argparse
import asyncio
import json

from app.agents.technical.agent import TechnicalAgent

p = argparse.ArgumentParser()
p.add_argument("ticker")
p.add_argument("--as-of", default=None)
p.add_argument("--chart", action="store_true", help="include chart series in the printout")
a = p.parse_args()
out = asyncio.run(TechnicalAgent().run(a.ticker, a.as_of)).model_dump(mode="json")
if not a.chart:
    out["metrics"].pop("chart", None)
print(json.dumps(out, indent=2))
