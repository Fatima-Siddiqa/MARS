"""SYNTHETIC demo data. Generates plausible-looking but entirely made-up series so the pipeline, API and tests can
run without network access or downloaded files. Every row carries status='synthetic' and the agent adds a visible
warning to every output built on it. NEVER use these numbers for research results."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config


def generate(seed: int = 7, start: str = "2013-07-01", end: str = "2026-10-10") -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    bd = pd.bdate_range(start, end)
    n = len(bd)
    me = pd.date_range(start, end, freq="ME")
    out: dict[str, pd.DataFrame] = {}

    def frame(dates, vals):
        return pd.DataFrame({"obs_date": pd.DatetimeIndex(dates), "value": np.asarray(vals, float), "status": "synthetic"})

    # policy rate: step function, a decision every 6-10 weeks (rows before 2015-05-25 exercise the use_from rule)
    d, rate, dates, vals = pd.Timestamp(start), 9.0, [], []
    while d < pd.Timestamp(end):
        rate = float(np.clip(rate + rng.choice([-1.0, -0.5, 0, 0, 0.5, 1.0, 2.0], p=[.08, .10, .22, .22, .14, .14, .10]), 5.5, 22))
        dates.append(d), vals.append(rate)
        d += pd.Timedelta(days=int(rng.integers(42, 70)))
    out["policy_rate"] = frame(dates, vals)
    pol = pd.Series(vals, index=pd.DatetimeIndex(dates))

    wk = pd.date_range(start, end, freq="W-FRI")
    pol_wk = pol.reindex(pol.index.union(wk)).ffill().reindex(wk)
    out["tbill_3m"] = frame(wk, pol_wk.to_numpy() + rng.normal(-0.2, 0.35, len(wk)))

    cpi = np.zeros(len(me)); cpi[0] = 8.0
    for i in range(1, len(me)):
        cpi[i] = 0.93 * cpi[i - 1] + 0.07 * 8.5 + rng.normal(0, 0.8) + (rng.random() < .03) * rng.normal(2, 1)
    out["cpi_yoy"] = frame(me, np.clip(cpi, 0.5, 38))

    ret = rng.normal(0.00025, 0.0022, n) + (rng.random(n) < .004) * rng.normal(0.03, 0.01, n)
    out["pkr_usd"] = frame(bd, 100 * np.exp(np.cumsum(ret)))

    out["fx_reserves"] = frame(wk, np.clip(9000 + np.cumsum(rng.normal(10, 300, len(wk))), 2500, 24000))
    out["current_account"] = frame(me, rng.normal(-350, 550, len(me)) + 250 * np.sin(np.arange(len(me)) / 6))
    out["remittances"] = frame(me, 1700 * np.exp(np.arange(len(me)) * 0.0045) * (1 + 0.08 * np.sin(np.arange(len(me)) * 2 * np.pi / 12))
                               * (1 + rng.normal(0, 0.04, len(me))))
    out["brent"] = frame(bd, np.clip(65 * np.exp(np.cumsum(rng.normal(0.0001, 0.019, n))), 25, 160))
    out["gold"] = frame(bd, 1250 * np.exp(np.cumsum(rng.normal(0.0003, 0.009, n))))
    yrs = range(pd.Timestamp(start).year, pd.Timestamp(end).year)
    out["gdp_growth"] = frame([pd.Timestamp(f"{y}-06-30") for y in yrs], np.clip(rng.normal(3.8, 2.0, len(yrs)), -1.5, 8))
    # KSE-100 is independent noise by construction: a correct evaluation of this data must find ~no signal
    out["kse100"] = frame(bd, 28000 * np.exp(np.cumsum(rng.normal(0.0004, 0.011, n))))
    return out


def write_drop_files(cfg: Config, seed: int = 7) -> list[Path]:
    paths = []
    for key, df in generate(seed).items():
        sc = cfg.series[key]
        p = cfg.resolve(sc.file)
        p.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"Observation Date": df["obs_date"].dt.strftime("%Y-%m-%d"), "Observation Value": df["value"].round(4),
                      "Observation Status": df["status"]}).to_csv(p, index=False)
        paths.append(p)
    return paths
