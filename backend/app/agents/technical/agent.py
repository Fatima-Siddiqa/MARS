"""Historical Price & Technical Agent (Level 2). Owner: Fatima.

Pipeline: load+clean -> indicators -> robust expanding z-scores -> 4 family scores (+ relative
strength needs KSE-100) -> equal-weight score -> confidence = agreement x quality -> template
explanation + evidence. Optional model blend only if evaluation adopted it.
"""
import pandas as pd

from app.agents.base import BaseAgent
from app.agents.technical import explain, model, scoring
from app.agents.technical.features import (
    FAMILIES,
    compute_raw,
    family_scores,
    feature_scores,
    rule_score,
)
from app.ingestion.prices import fetch_prices, load_benchmark, sanity_warnings
from app.schemas.agent_output import AgentName, AgentOutput, AgentStatus, Stance

CHART_DAYS = 250
RAW_KEYS = [
    "close", "sma20", "sma50", "sma200", "rsi14", "macd_hist", "adx14", "atr_pct", "bb_pctb",
    "vol20", "vol_pctile", "drawdown_252", "mom_12_1", "ret_6m", "rs_3m", "rs_6m",
]


def _r(x, n=4):
    return None if pd.isna(x) else round(float(x), n)


class TechnicalAgent(BaseAgent):
    name = AgentName.TECHNICAL

    async def run(self, ticker: str, as_of: str | None = None) -> AgentOutput:  # type: ignore[override]
        if as_of is None:
            return await super().run(ticker)
        try:
            return await self.analyze(ticker, as_of)
        except Exception as exc:  # noqa: BLE001
            return AgentOutput(agent=self.name, ticker=ticker, status=AgentStatus.ERROR,
                               warnings=[f"{type(exc).__name__}: {exc}"])

    async def analyze(self, ticker: str, as_of: str | None = None) -> AgentOutput:
        ticker = ticker.upper()
        req = pd.Timestamp(as_of).normalize() if as_of else pd.Timestamp.today().normalize()
        try:
            df = await fetch_prices(ticker)
        except FileNotFoundError as exc:
            return AgentOutput(agent=self.name, ticker=ticker, status=AgentStatus.UNAVAILABLE,
                               warnings=[str(exc)])
        df = df[df.index <= req]
        if len(df) < 2:
            return AgentOutput(agent=self.name, ticker=ticker, status=AgentStatus.UNAVAILABLE,
                               warnings=["No price history on or before the requested date"])
        bench = load_benchmark()
        warnings = sanity_warnings(df)
        if bench is not None:
            bench = bench[bench.index <= req]
        if bench is None or bench.empty:
            warnings.append("KSE-100 benchmark missing: relative-strength family skipped")
            bench = None

        raw = compute_raw(df, bench)
        scores = feature_scores(raw, last_only=True).iloc[[-1]]
        fam_s = family_scores(scores).iloc[-1]
        rule = rule_score(family_scores(scores)).iloc[-1]
        rule = 0.0 if pd.isna(rule) else float(rule)
        last = raw.iloc[-1]
        n_days = len(df)

        score, p = rule, None
        mdl = model.load_adopted_model()
        if mdl is not None and scores.notna().sum(axis=1).iloc[0] >= 8:
            p = model.predict_prob(mdl, scores)
            score = model.blend(rule, p)

        fresh = scoring.freshness(df.index[-1], req)
        q = scoring.quality(n_days, last["no_trade_20"], fresh)
        fam = {k: (None if pd.isna(v) else float(v)) for k, v in fam_s.items()}
        conf = scoring.agreement(list(fam.values())) * q["quality"]
        stance = scoring.stance_of(score)

        n_feat = int(scores.notna().sum(axis=1).iloc[0])
        status = AgentStatus.OK
        if n_days < 252 or n_feat < 12:
            status = AgentStatus.DEGRADED
            warnings.append(f"Limited history ({n_days} days): only {n_feat}/14 features available")
        if fresh < 1:
            warnings.append(f"Stale data: last price {df.index[-1].date()}, requested {req.date()}")
        if n_feat == 0:
            stance, score, conf = Stance.NEUTRAL, 0.0, 0.0

        drivers = explain.top_drivers(scores.iloc[0]) if n_feat else []
        fam_clean = {k: v for k, v in fam.items() if v is not None}
        summary, findings = explain.build_text(ticker, stance, score, fam_clean, drivers, last)
        ev = explain.build_evidence(ticker, last, drivers, df.index[-1]) if n_feat else []
        chart = raw.tail(CHART_DAYS).join(df[["Open", "High", "Low", "Volume"]])
        chart_rows = [{"time": str(i.date()), "open": _r(r.Open, 2), "high": _r(r.High, 2),
                       "low": _r(r.Low, 2), "close": _r(r.close, 2), "volume": float(r.Volume),
                       "sma20": _r(r.sma20, 2), "sma50": _r(r.sma50, 2), "sma200": _r(r.sma200, 2),
                       "rsi14": _r(r.rsi14, 1), "macd_hist": _r(r.macd_hist, 4)}
                      for i, r in chart.iterrows()]
        return AgentOutput(
            agent=self.name, ticker=ticker, as_of=df.index[-1].to_pydatetime(), status=status,
            stance=stance, score=round(score, 4), confidence=round(conf, 4), summary=summary,
            key_findings=findings, risks=explain.build_risks(last, fresh, n_days, fam_clean),
            evidence=ev, warnings=warnings,
            metrics={
                "version": "level2-rules" + ("+model" if p is not None else ""),
                "rule_score": round(rule, 4), "model_probability": p,
                "families": {k: _r(v) for k, v in fam.items()},
                "feature_scores": {k: _r(v) for k, v in scores.iloc[0].items()},
                "indicators": {k: _r(last[k]) for k in RAW_KEYS},
                "quality": {k: round(v, 3) for k, v in q.items()},
                "agreement": round(scoring.agreement(list(fam.values())), 3),
                "no_trade_days_20": round(float(last["no_trade_20"]) * 20),
                "families_used": [f for f in FAMILIES if fam.get(f) is not None],
                "history_days": n_days, "requested_date": str(req.date()),
                "chart": chart_rows,
            },
        )
