"""Calibrate BUY / HOLD / AVOID thresholds on what prices actually did afterwards.

Uses the walk-forward out-of-sample predictions written by train.py: for every test
company-year, `gap_log` = model value - market value (log), from a model that never saw that
year. For each one we measure the stock's log return over the next HORIZON trading days minus
the median log return of all listed PSX companies over the same days (excess return).

Questions answered (all on out-of-sample data):
- Do higher gaps (cheaper vs fundamentals) come before higher excess returns? (Spearman IC)
- Which symmetric threshold T (BUY: gap >= T, AVOID: gap <= -T in log terms) separates
  future winners from losers best, with at least MIN_BUCKET samples on each side?

If the evidence is not significant the default threshold is kept and the agent lowers its
confidence: the call is then a valuation statement, not a tested trading signal.

Limitations (stated in the report): PSX daily closes are not adjusted for cash dividends
(slight bias against high-dividend stocks); windows with a single-day fall over 40% (likely a
bonus issue or split) are dropped.

    python -m app.agents.fundamentals.valuation.calibrate
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from app.agents.fundamentals.valuation.market_index import market_return
from app.agents.fundamentals.valuation.train import MODEL_DIR, OOS_CSV
from app.ingestion import psx_eod

HORIZON = 126  # trading days, about six months
CANDIDATE_T = (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)
DEFAULT_T = 0.25
MIN_BUCKET = 20
SIGNIFICANCE = 0.05
CALIBRATION_JSON = MODEL_DIR / "call_calibration.json"
CALIBRATION_MD = MODEL_DIR / "call_calibration.md"


def forward_excess(ticker: str, obs_date: str) -> float | None:
    df = psx_eod.load_symbol(ticker)
    if df is None:
        return None
    window = df[df.index >= pd.Timestamp(obs_date)]["close"]
    if len(window) <= HORIZON:
        return None  # not enough history after the observation yet
    window = window.iloc[: HORIZON + 1]
    if (window.pct_change() < -0.40).any():
        return None  # probable bonus issue / split inside the window
    t0, t1 = window.index[0], window.index[-1]
    mkt = market_return(t0, t1)
    if mkt is None:
        return None
    return float(np.log(window.iloc[-1] / window.iloc[0]) - mkt)


def _bucket(x: pd.Series) -> dict:
    return {
        "n": int(len(x)),
        "mean_excess": float(x.mean()) if len(x) else None,
        "hit_rate": float((x > 0).mean()) if len(x) else None,
    }


def calibrate() -> dict:
    oos = pd.read_csv(OOS_CSV)
    oos["excess"] = [forward_excess(t, d) for t, d in zip(oos["ticker"], oos["obs_date"])]
    data = oos.dropna(subset=["excess", "gap_log"])
    out: dict = {
        "calibrated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "horizon_trading_days": HORIZON,
        "samples": int(len(data)),
        "samples_without_forward_window": int(oos["excess"].isna().sum()),
    }
    if len(data) < 3 * MIN_BUCKET:
        out.update(
            threshold=DEFAULT_T,
            supported=False,
            reason=f"only {len(data)} out-of-sample rows with a full forward window",
        )
        return _save(out)
    ic, p = spearmanr(data["gap_log"], data["excess"])
    terciles = pd.qcut(
        data["gap_log"], 3, labels=["low gap (expensive)", "middle", "high gap (cheap)"]
    )
    out["spearman_ic"], out["ic_p_value"] = float(ic), float(p)
    out["terciles"] = {
        str(k): _bucket(g["excess"]) for k, g in data.groupby(terciles, observed=True)
    }
    grid = []
    for t in CANDIDATE_T:
        lt = np.log1p(t)
        buy, avoid = data[data["gap_log"] >= lt], data[data["gap_log"] <= -lt]
        hold = data[(data["gap_log"] < lt) & (data["gap_log"] > -lt)]
        row = {
            "threshold": t,
            "buy": _bucket(buy["excess"]),
            "hold": _bucket(hold["excess"]),
            "avoid": _bucket(avoid["excess"]),
        }
        if len(buy) >= MIN_BUCKET and len(avoid) >= MIN_BUCKET:
            row["spread"] = float(buy["excess"].mean() - avoid["excess"].mean())
        grid.append(row)
    out["grid"] = grid
    valid = [g for g in grid if "spread" in g]
    best = max(valid, key=lambda g: g["spread"]) if valid else None
    supported = bool(ic > 0 and p < SIGNIFICANCE and best is not None and best["spread"] > 0)
    out["supported"] = supported
    out["threshold"] = best["threshold"] if supported else DEFAULT_T
    out["reason"] = (
        f"cheaper-vs-fundamentals stocks outperformed (IC {ic:+.2f}, p={p:.3f}); threshold "
        f"{out['threshold']:.0%} gives the widest BUY-AVOID spread"
        if supported
        else "no significant link between the valuation gap and later excess returns "
        f"(IC {ic:+.2f},"
        f" p={p:.3f}); default threshold {DEFAULT_T:.0%} kept and confidence reduced"
    )
    return _save(out)


def _save(out: dict) -> dict:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    CALIBRATION_JSON.write_text(json.dumps(out, indent=2))
    lines = [
        "# BUY / HOLD / AVOID calibration",
        "",
        f"Out-of-sample samples with a {out['horizon_trading_days']}-day forward window: "
        f"{out['samples']}.",
        "",
        f"**Threshold: {out['threshold']:.0%}**; supported by the "
        f"evidence: **{out['supported']}**. {out['reason']}.",
        "",
    ]
    if "terciles" in out:
        lines += [
            "| gap tercile | n | mean 6-month excess return | hit rate |",
            "|---|---|---|---|",
        ]
        for k, b in out["terciles"].items():
            lines.append(f"| {k} | {b['n']} | {b['mean_excess']:+.1%} | {b['hit_rate']:.0%} |")
        lines += [
            "",
            "| threshold | BUY n | BUY excess | AVOID n | AVOID excess | spread |",
            "|---|---|---|---|---|---|",
        ]
        for g in out["grid"]:
            fmt = lambda v: "-" if v is None else f"{v:+.1%}"  # noqa: E731
            lines.append(
                f"| {g['threshold']:.0%} | {g['buy']['n']} | "
                f"{fmt(g['buy']['mean_excess'])} | {g['avoid']['n']} | "
                f"{fmt(g['avoid']['mean_excess'])} | {fmt(g.get('spread'))} |"
            )
    lines += [
        "",
        "Limitations: PSX closes are not dividend-adjusted; windows with a one-day fall "
        "over 40% (probable bonus issue / split) are excluded.",
        "",
    ]
    CALIBRATION_MD.write_text("\n".join(lines))
    return out


def load_calibration() -> dict:
    if CALIBRATION_JSON.exists():
        return json.loads(CALIBRATION_JSON.read_text())
    return {"threshold": DEFAULT_T, "supported": False, "reason": "not calibrated yet"}


def main() -> None:
    r = calibrate()
    print(
        json.dumps(
            {
                k: r.get(k)
                for k in (
                    "samples",
                    "spearman_ic",
                    "ic_p_value",
                    "threshold",
                    "supported",
                    "reason",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
