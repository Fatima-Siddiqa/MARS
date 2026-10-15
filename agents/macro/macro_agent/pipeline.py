"""Pipeline Runner: the single code path used by the live API, the nightly job and the offline backtest.
Layers 1-7 in order: acquire -> store raw -> validate -> as-of align -> features -> standardise -> pillars/composite
-> quality/gates -> explanation -> output JSON."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .alignment import build_calendar, snapshot, truncate_as_of, with_release_dates
from .config import Config, get_config
from .explain import build_facts, display_value, pick_drivers, plain_summary
from .features import FeatureFrame, build_features
from .llm import rewrite
from .loaders import LoaderUnavailable, get_loader
from .quality import QualityFrame, assess, series_limit
from .schemas import MacroOutput
from .scoring import ScoreFrame, score_features
from .standardise import Standardised, standardise_features
from .storage import Store
from .validation import validate_series

log = logging.getLogger(__name__)
NON_FEATURE_SERIES = {"kse100"}


class NoDataError(RuntimeError):
    pass


@dataclass
class Computation:
    cal: pd.DatetimeIndex
    frames: dict[str, pd.DataFrame]
    ff: FeatureFrame
    std: Standardised


class MacroPipeline:
    def __init__(self, cfg: Config | None = None, store: Store | None = None):
        self.cfg = cfg or get_config()
        self.store = store or Store(self.cfg.database_url, self.cfg.root)
        self._frames_cache: tuple[str, dict[str, pd.DataFrame]] | None = None
        self._comp_cache: dict[tuple[str, str], Computation] = {}

    # ---------------------------------------------------------------- Layer 1: acquire / store / validate
    def refresh(self, only: list[str] | None = None) -> dict[str, str]:
        """Fetch every source, validate (writes the data-quality log), append to macro_raw as a new version."""
        status: dict[str, str] = {}
        for key, sc in self.cfg.series.items():
            if only and key not in only:
                continue
            try:
                df = get_loader(sc, self.cfg.root).fetch()
                _, issues = validate_series(df, sc)
                self.store.log_dq(issues)
                n = self.store.append_raw(key, df, sc.unit, sc.source_url)
                status[key] = f"ok: {len(df)} rows fetched, {n} new/changed versions written"
            except LoaderUnavailable as e:
                status[key] = f"unavailable: {e}"
                self.store.log_dq([dict(series_key=key, obs_date=None, check="source_unavailable", severity="error", detail=str(e))])
                log.warning("%s: %s", key, e)
        self.store.log_run("refresh", "ok" if any(v.startswith("ok") for v in status.values()) else "failed", str(status))
        self._frames_cache, self._comp_cache = None, {}
        return status

    def frames(self) -> dict[str, pd.DataFrame]:
        """Clean observations with release dates (latest version of each observation)."""
        fp = self.store.data_fingerprint()
        if self._frames_cache and self._frames_cache[0] == fp:
            return self._frames_cache[1]
        out = {}
        for key, sc in self.cfg.series.items():
            raw = self.store.read_raw_latest(key)
            if raw.empty:
                continue
            clean, _ = validate_series(raw[["obs_date", "value", "status"]], sc)
            if clean.empty:
                continue
            out[key] = with_release_dates(clean, sc)
        self._frames_cache, self._comp_cache = (fp, out), {}
        return out

    def kse100(self) -> pd.Series:
        f = self.frames().get("kse100")
        return pd.Series(dtype=float) if f is None else f.set_index("obs_date")["value"]

    def last_data_date(self) -> pd.Timestamp:
        fr = {k: v for k, v in self.frames().items() if k not in NON_FEATURE_SERIES}
        if not fr:
            raise NoDataError("macro_raw is empty - run `macro-agent refresh` (or `macro-agent demo-data`) first")
        return max(v["release_date"].max() for v in fr.values())

    # ---------------------------------------------------------------- Layers 1.4-3
    def compute(self, end: pd.Timestamp | None = None) -> Computation:
        end = pd.Timestamp(end).normalize() if end is not None else self.last_data_date()
        fp = self.store.data_fingerprint()
        ck = (fp, end.strftime("%Y-%m-%d"))
        if ck in self._comp_cache:
            return self._comp_cache[ck]
        frames = {k: truncate_as_of(v, end) for k, v in self.frames().items() if k not in NON_FEATURE_SERIES}
        frames = {k: v for k, v in frames.items() if not v.empty}
        if not frames:
            raise NoDataError(f"no data released on or before {end.date()}")
        start = min(v["release_date"].min() for v in frames.values())
        cal = build_calendar(start, end)
        ff = build_features(frames, cal, self.cfg)
        std = standardise_features(ff.raw, self.cfg)
        comp = Computation(cal, frames, ff, std)
        if len(self._comp_cache) > 8:
            self._comp_cache.pop(next(iter(self._comp_cache)))
        self._comp_cache[ck] = comp
        return comp

    # ---------------------------------------------------------------- Layers 4-5 over a whole range (backtest)
    def score_series(self, ticker: str | None = None, end: pd.Timestamp | None = None
                     ) -> tuple[pd.DataFrame, ScoreFrame, QualityFrame, Computation]:
        comp = self.compute(end)
        mult = self.cfg.multipliers(ticker)
        sf = score_features(comp.std, self.cfg, mult)
        qf = assess(comp.ff, comp.std, sf, self.cfg, mult)
        df = pd.DataFrame({"score": sf.composite, "state": qf.state, "confidence": qf.confidence,
                           "coverage": qf.coverage, "freshness": qf.freshness, "agreement": qf.agreement,
                           "n_pillars": sf.n_pillars, "gated": qf.gated})
        df = pd.concat([df, sf.pillars.add_prefix("pillar_")], axis=1)
        return df, sf, qf, comp

    # ---------------------------------------------------------------- single date -> output contract
    def score(self, as_of: str | pd.Timestamp | None = None, ticker: str | None = None, use_llm: bool | None = None,
              persist: bool = False) -> dict:
        cfg = self.cfg
        t = pd.Timestamp(as_of).normalize() if as_of is not None else pd.Timestamp.now(tz=cfg.timezone).tz_localize(None).normalize()
        ticker = ticker.upper() if ticker else None
        _, sf, qf, comp = self.score_series(ticker, end=t)
        mult = cfg.multipliers(ticker)
        sector = cfg.sector_of(ticker)

        comp_t = sf.composite.iloc[-1]
        score = 0.0 if pd.isna(comp_t) else float(comp_t)
        conf = float(qf.confidence.iloc[-1])
        state = str(qf.state.iloc[-1])
        pillars = {p: (None if pd.isna(v) else round(float(v), 4)) for p, v in sf.pillars.iloc[-1].items()}

        used = [f for f in cfg.active_features() if pd.notna(sf.adj_score[f].iloc[-1])]
        raw_t = comp.ff.raw.iloc[-1]
        contrib = {f: float(sf.contrib[f].iloc[-1]) for f in used}
        drivers = pick_drivers(cfg, contrib, raw_t.to_dict())
        for d in drivers:
            d["score"] = round(float(sf.adj_score[d["feature"]].iloc[-1]), 4)
            d["display_value"] = display_value(d["value"], d["unit"])

        # ---- evidence: every series behind a used feature, with value, observation date, unit, source URL
        panel_t = {k: p.iloc[-1] for k, p in comp.ff.panel.items()}
        ev: dict[str, dict] = {}
        for f in used:
            for s in cfg.features[f].series:
                row = panel_t.get(s)
                if row is None or pd.isna(row["value"]):
                    continue
                e = ev.setdefault(s, dict(series_key=s, value=float(row["value"]), date=pd.Timestamp(row["obs_date"]).strftime("%Y-%m-%d"),
                                          unit=cfg.series[s].unit, source_url=cfg.series[s].source_url, used_by=[]))
                e["used_by"].append(f)
        evidence = list(ev.values())

        # ---- warnings
        warnings: list[str] = []
        if not used:
            warnings.append("No usable macro data was available on this date.")
        if bool(qf.gate_pillars.iloc[-1]):
            warnings.append(f"Only {int(sf.n_pillars.iloc[-1])} of {len(cfg.pillars)} pillars have data; state forced to Neutral with low confidence.")
        for g, miss in qf.gate_critical.iloc[-1].items():
            if bool(miss):
                warnings.append(f"Critical input missing ({g.replace('_', ' ')}); state forced to Neutral with low confidence.")
        used_series = {s for f in used for s in cfg.features[f].series}
        for s in sorted(used_series):
            fac = qf.series_factor[s].iloc[-1] if s in qf.series_factor else np.nan
            age = panel_t[s]["age_days"] if s in panel_t else np.nan
            if pd.notna(fac) and fac < 1.0:
                lvl = "stale" if fac <= 0.2 else "ageing"
                warnings.append(f"{s} is {lvl}: last released {int(age)} days ago (expected every {cfg.series[s].expected_interval_days} days).")
        missing = sorted(s for s in cfg.series if s not in NON_FEATURE_SERIES and s not in used_series
                         and any(s in cfg.features[f].series and cfg.features[f].enabled and mult.get(f, 1.0) != 0 for f in cfg.features))
        if missing:
            warnings.append("No data available for: " + ", ".join(missing) + ".")
        fb = [f for f in used if comp.std.method[f].iloc[-1] in ("percentile", "fixed") and cfg.features[f].method == "robust_z"]
        fb += [f for f in used if cfg.features[f].method == "percentile"]
        if fb:
            warnings.append("Annual or short history: " + ", ".join(sorted(set(fb))) + " scored with a rank-percentile or fixed rule and given low weight.")
        if ticker and sector is None:
            warnings.append(f"No sector mapping for {ticker}; the market-level score is returned.")
        for s in sorted(used_series):
            fr = comp.frames.get(s)
            if fr is not None and "status" in fr and fr["status"].astype(str).str.contains("synthetic").any():
                warnings.append("DEMO DATA: synthetic series in use - these are not real market or economic values.")
                break
        if used and sf.n_pillars.iloc[-1] >= 1 and not pd.isna(comp_t) and qf.agreement.iloc[-1] < 0.5:
            warnings.append("Pillars disagree with the composite direction; treat the signal with caution.")

        facts = build_facts(state=state, score=score, confidence=conf, as_of=t.strftime("%Y-%m-%d"), drivers=drivers,
                            pillars=pillars, warnings=warnings, ticker=ticker, sector=sector)
        facts["evidence"] = evidence
        summary = plain_summary(facts)
        llm_cfg = dict(cfg.llm)
        if use_llm is not None:
            llm_cfg["enabled"] = use_llm
        summary, _ = rewrite(summary, facts, llm_cfg)

        out = MacroOutput(
            ticker=ticker, score=round(score, 4), state=state, confidence=round(conf, 4), pillars=pillars,
            drivers=[{k: v for k, v in d.items() if k != "display_value"} for d in drivers], evidence=evidence,
            warnings=warnings, plain_summary=summary, as_of=t.strftime("%Y-%m-%d"), horizon_days=cfg.horizon_days,
            model_version=cfg.model_version).model_dump()
        if persist:
            tbl = pd.DataFrame({"raw": raw_t, "score": sf.adj_score.iloc[-1].reindex(raw_t.index),
                                "age": comp.ff.age.iloc[-1]})
            self.store.save_output(out, tbl)
        return out

    # ---------------------------------------------------------------- Regime layer inputs (flags, not labels)
    def regime_inputs(self, as_of: str | pd.Timestamp | None = None) -> dict:
        t = pd.Timestamp(as_of).normalize() if as_of is not None else pd.Timestamp.now(tz=self.cfg.timezone).tz_localize(None).normalize()
        _, sf, _, comp = self.score_series(None, end=t)
        thr = float(self.cfg.regime_flags["z_threshold"])
        flags = {}
        for name, feat in self.cfg.regime_flags["flags"].items():
            z = comp.std.z[feat].iloc[-1] if feat in comp.std.z else np.nan
            flags[name] = dict(feature=feat, z=None if pd.isna(z) else round(float(z), 3),
                               raised=bool(pd.notna(z) and abs(z) >= thr),
                               direction=None if pd.isna(z) or abs(z) < thr else ("up" if z > 0 else "down"))
        return dict(as_of=t.strftime("%Y-%m-%d"), z_threshold=thr,
                    pillars={p: (None if pd.isna(v) else round(float(v), 4)) for p, v in sf.pillars.iloc[-1].items()},
                    flags=flags)

    def source_freshness(self, as_of: str | pd.Timestamp | None = None) -> list[dict]:
        t = pd.Timestamp(as_of).normalize() if as_of is not None else pd.Timestamp.now(tz=self.cfg.timezone).tz_localize(None).normalize()
        snap = snapshot(self.frames(), t)
        rows = []
        for k, sc in self.cfg.series.items():
            r = snap.get(k)
            if r is None:
                rows.append(dict(series=k, last_obs=None, age_days=None, limit_days=series_limit(self.cfg, k), stale=True, status="no data"))
                continue
            age = int((t - r["release_date"]).days)
            lim = series_limit(self.cfg, k)
            rows.append(dict(series=k, last_obs=r["obs_date"].strftime("%Y-%m-%d"), age_days=age, limit_days=lim,
                             stale=age > lim, status="stale" if age > 2 * lim else ("ageing" if age > lim else "fresh")))
        return rows
