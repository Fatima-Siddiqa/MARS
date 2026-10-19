# MARS Macro Agent

Rule-based, explainable macroeconomic agent for PSX equities. It turns SBP / PBS / FRED / World Bank series into one
score in [-1, +1], a state (Bullish / Neutral / Bearish), a confidence, and an evidence list with a source for every
number. It never recommends BUY / HOLD / AVOID. Implements `MARS - Macro Agent Methodology` and the architecture diagram.

> **Status:** built and tested end-to-end on **synthetic** data only (`macro-agent demo-data`; every output built on it
> carries a "DEMO DATA" warning). No real series have been loaded. Items marked `[TO CONFIRM]` in
> `config/macro_agent.yaml` (release lags, SBP series codes, yield-spread sign, gold direction, source terms of use)
> must be checked before any real result is reported.

## Quick start
```bash
pip install -e ".[dev]"
pytest                                   # 32 tests
macro-agent demo-data                    # SYNTHETIC data -> data/drop -> macro_raw (offline)
macro-agent score --ticker OGDC          # prints the output JSON
macro-agent eval                         # walk-forward evaluation -> reports/
macro-agent serve                        # GET /agents/macro?ticker=OGDC&as_of=2026-10-10
```
Real data: put CSV exports in `data/drop/` (see the README there) and run `macro-agent refresh`; Brent (FRED) and GDP
growth (World Bank) are fetched online, with the drop folder as offline fallback. Set `MACRO_AGENT_OFFLINE=1` to skip
network calls. Use PostgreSQL with `DATABASE_URL=postgresql+psycopg://user:pw@host/db`.

## Diagram -> code
| Diagram block | Module |
|---|---|
| Orchestrator: nightly scheduler, on-demand handler, pipeline runner | `orchestrator.py` (FastAPI + APScheduler), `pipeline.py` (shared live/backtest) |
| L1 acquire / store raw / validate / release-date align / as-of snapshot | `loaders/`, `storage.py` (versioned `macro_raw`), `validation.py` (+ `dq_log`), `alignment.py` |
| L2 features by pillar | `features.py` |
| L3 expanding robust z, clip, orient, fallback | `standardise.py` |
| L4 pillars, composite, state, sector sensitivity | `scoring.py` (+ `sectors` / `tickers` in config) |
| L5 confidence = coverage x freshness x agreement, gates | `quality.py` |
| L6 drivers, evidence, templates, optional Ollama rewrite | `explain.py`, `llm.py` |
| L7 JSON contract, regime inputs, dashboard/Postgres rows | `schemas.py`, `pipeline.regime_inputs`, `storage.py` (`agent_outputs`, `pillar_scores`, `features`) |
| Offline research pipeline | `evaluation.py` |
| Monitoring and logging | `monitoring.py`, `dq_log`, `run_log`, `/health`, `/sources`, `/dq` |

## Design guarantees (covered by tests)
- **No look-ahead:** a value is usable only from `obs_date + release_lag`; scoring date D gives identical results with or without later data.
- **Raw is append-only:** a re-download adds a version only when a value changed; the original row stays.
- **Rules first:** equal weights by default; fitted pillar weights are chosen only if they beat the rules out-of-sample with a positive signal.
- **LLM never makes numbers:** Ollama text is rejected (template used) if any number is not in the facts JSON.
- **Honest gaps:** stale/missing series are named in `warnings`; fewer than 3 pillars or a missing policy rate / PKR-USD / CPI forces Neutral with confidence <= 0.20.
- **Reproducible:** all defaults live in `config/macro_agent.yaml`; `model_version` embeds its hash.
- **Lock-box:** `macro-agent eval --lockbox` runs the final period once; a second run is refused.

## Interpretation choices to review with your supervisor
1. Standardisation statistics use month-end history (>= 36 month-ends) applied to daily values; this avoids over-counting carried-forward step series.
2. Percentile fallback for features with 12-35 months of history; annual GDP uses distinct releases (>= 5), then a fixed rule. Fallback features get weight 0.5, so they also lower coverage.
3. Pillar = weighted mean of its features (weights are 1 except low-weight features); composite = equal-weight mean of available pillars; each driver's contribution sums exactly to the composite.
4. Sector multipliers: oil reversed for oil & gas producers, rupee effect halved for textile exporters (hypotheses, editable in config). The ticker-to-sector map is an initial guess; check it.
5. Gold is computed but excluded from scoring (`enabled: false`) until its direction is tested.
6. Evaluation regime tags (rate-hike / PKR-shock / oil-shock / calm) exist only inside `evaluation.py`; the agent exposes z-score flags to the Regime layer and never labels regimes.
7. JSON field names match the methodology; `drivers` and `evidence` carry a few additive keys (`label`, `value`, `unit`, `score`, `direction`, `used_by`). Align with Fatima and Aneeqa.
