"""Peer comparison and the transparent peer-baseline valuation.

Peers: same PSX sector code, same company type (bank / non-bank), with an official price on the
valuation date, the target itself excluded. Fewer than MIN_PEERS peers is reported as "weak".

Peer baseline (no fitted model): implied value per share = peer median multiple x the target's
own per-share fundamental. Banks use P/B (book value drives bank valuation); other companies use
P/E when earnings are positive, otherwise P/B.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from statistics import median

from app.agents.fundamentals.data.loader import Snapshot, build_snapshot, companies
from app.agents.fundamentals.features import Ratio, compute_ratios

MIN_PEERS = 3
COMPARED = ("pe", "pb", "ps", "roe", "roa", "net_margin", "debt_to_equity", "earnings_growth",
            "revenue_growth", "advances_to_deposits", "cost_to_income", "equity_to_assets")


@dataclass
class PeerStat:
    ratio: str
    target: float | None
    peer_median: float | None
    n: int
    percentile: float | None  # share of peers below the target (0..1)


@dataclass
class PeerResult:
    peers: list[str]
    strength: str  # adequate | weak | none
    stats: dict[str, PeerStat] = field(default_factory=dict)
    peer_values: dict[str, dict[str, float]] = field(default_factory=dict)  # peer -> ratio -> v


@dataclass
class Baseline:
    method: str  # peer_pb | peer_pe
    multiple: float | None  # peer median used
    implied_value_per_share: float | None
    price: float | None
    gap: float | None  # implied / price - 1
    n_peers: int
    reason: str | None = None


def universe(as_of: date, sector_code: str | None = None,
             company_type: str | None = None) -> dict[str, tuple[Snapshot, list[Ratio]]]:
    """Snapshots + ratios for every known company (optionally one sector / type only)."""
    meta = companies()
    if sector_code is not None:
        meta = meta[meta["psx_sector_code"] == sector_code]
    if company_type is not None:
        meta = meta[meta["company_type"] == company_type]
    out = {}
    for t in meta.index:
        s = build_snapshot(t, as_of)
        out[t] = (s, compute_ratios(s))
    return out


def peer_universe(target: Snapshot) -> dict[str, tuple[Snapshot, list[Ratio]]]:
    """The target's own sector and company type (all of PSX if the universe was collected)."""
    return universe(target.as_of, target.sector_code, target.company_type)


def compare(target: Snapshot, uni: dict[str, tuple[Snapshot, list[Ratio]]]) -> PeerResult:
    tr = {r.name: r.value for r in uni[target.ticker][1]} if target.ticker in uni else {}
    peers = [t for t, (s, _) in uni.items()
             if t != target.ticker and s.sector_code == target.sector_code
             and s.company_type == target.company_type and s.price is not None]
    strength = "none" if not peers else "weak" if len(peers) < MIN_PEERS else "adequate"
    res = PeerResult(sorted(peers), strength)
    for p in res.peers:
        res.peer_values[p] = {r.name: r.value for r in uni[p][1] if r.value is not None}
    for name in COMPARED:
        if name not in tr:
            continue
        vals = [res.peer_values[p][name] for p in res.peers if name in res.peer_values[p]]
        t = tr[name]
        pct = None if t is None or not vals else sum(v < t for v in vals) / len(vals)
        res.stats[name] = PeerStat(name, t, median(vals) if vals else None, len(vals), pct)
    return res


def baseline(target: Snapshot, ratios: list[Ratio], peers: PeerResult) -> Baseline:
    r = {x.name: x for x in ratios}
    use_pe = target.company_type != "bank" and r.get("pe") and r["pe"].value is not None
    method = "peer_pe" if use_pe else "peer_pb"
    key, item = ("pe", "net_income") if use_pe else ("pb", "total_equity")
    fundamental = target.ttm.get(item) if use_pe else target.balance.get(item)
    stat = peers.stats.get(key)
    if target.price is None or not target.shares:
        return Baseline(method, None, None, target.price, None, 0, "no current price or shares")
    if fundamental is None or fundamental.value <= 0:
        return Baseline(method, None, None, target.price, None, 0,
                        f"{item} missing or not positive")
    if stat is None or stat.peer_median is None:
        return Baseline(method, None, None, target.price, None, 0,
                        "no peer with a valid multiple")
    per_share = fundamental.value * 1e6 / target.shares
    implied = stat.peer_median * per_share
    return Baseline(method, stat.peer_median, implied, target.price, implied / target.price - 1,
                    stat.n)
