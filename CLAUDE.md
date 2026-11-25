# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

MARS turns a PSX (Pakistan Stock Exchange) ticker into an explainable BUY / HOLD / AVOID recommendation.
Six specialist agents run in parallel (fan-out); a synthesis step fuses their outputs (fan-in). It is a
decision-support tool only: it never places trades. Final Year Project with three members; each owns
specific agents (see the Ownership table in `README.md`) and builds the backend + frontend pieces for them.

## Commands

Backend (run from `backend/`, venv at repo-root `.venv/` or `backend/.venv/`):

```bash
pip install -e ".[dev]"                      # add ".[dev,sentiment-model]" for torch/transformers (FinBERT)
uvicorn app.main:app --reload                # API on :8000, docs at /docs
pytest                                       # all tests (testpaths set in pyproject.toml)
pytest tests/unit/test_technical_agent.py::test_name   # single test
ruff check .                                 # lint (CI runs exactly ruff + pytest)
alembic revision --autogenerate -m "msg" && alembic upgrade head
python -m scripts.run_technical HBL [--as-of 2026-07-14] [--chart]   # run one agent from CLI
python -m scripts.technical_eval             # technical walk-forward evaluation
python -m app.agents.sentiment.newsagent.pipeline --help   # news collect -> FinBERT score pipeline
python -m app.ingestion.psx_eod                            # refresh official PSX daily prices
python -m app.agents.fundamentals.data.psx_official        # official PSX fundamentals
python -m app.agents.fundamentals.data.validate_official   # must report "ok": true
```

Frontend (run from `frontend/`): `npm run dev` (:3000), `npm run lint`, `npm run build`. CI runs
`npm ci && npm run lint && npm run build`. There is no frontend test runner configured.

Macro service (separate package, run from `agents/macro/`): `pip install -e ".[dev]"`, `pytest`,
`macro-agent serve --port 8010` (the backend's macro adapter calls it), `macro-agent score --ticker OGDC`.

Full stack: `docker compose up --build` (Postgres + backend); `docker compose --profile ollama up` adds a local LLM.

## Architecture

**The shared contract is the core of the design.** Every agent returns `AgentOutput`
(`backend/app/schemas/agent_output.py`): stance, score in [-1, 1], confidence in [0, 1], summary,
key_findings, risks, `evidence` (every claim needs an `Evidence` source) and a free-form `metrics` dict
for agent-specific numbers / chart data. Synthesis and the frontend only consume this shape. Changing it
requires bumping `SCHEMA_VERSION`, updating `docs/agent_contract.md`, and approval from all three
members, so put new agent-specific data in `metrics` instead.

**Agent lifecycle.** Agents subclass `BaseAgent` (`app/agents/base.py`) and implement
`async analyze(ticker)`. Callers always use `run()`, which times the call and converts
`NotImplementedError` into `status=unavailable` and any other exception into `status=error`. This is the
graceful-degradation mechanism: one broken agent never breaks `/recommendation`. All agent instances
live in `app/agents/registry.py` (`AGENTS` dict keyed by `AgentName`); routes look agents up there.

**Request flow.** `GET /agents/{name}/{ticker}` (routes in `app/api/routes/agents/`) runs one agent.
`GET /recommendation/{ticker}` runs all agents with `asyncio.gather`, then
`app/synthesis/baseline_average.py` computes a confidence-weighted average (BUY >= 0.25, AVOID <= -0.25).
The baseline is kept permanently as the fallback and as the evaluation baseline for the planned
regime-aware synthesis. LangGraph orchestration (`app/orchestration/`) is planned but not built.

**Agent status:**
- `technical/`: built. Reads PSX OHLCV via `app/ingestion/prices.py`: the FYP xlsx files in
  `data/raw/stocks/` (adjusted, end mid-2026), extended with official unadjusted PSX daily closes from
  `app/ingestion/psx_eod.py` (`data/raw/psx_eod/eod.csv`). Supports `as_of`.
- `sentiment/`: built. `sentiment/agent.py` is the `AgentOutput` adapter; `sentiment/newsagent/` is a
  self-contained package (RSS/PSX ingestion, entity linking, dedup, FinBERT scoring, aggregation
  engine). The API request only *reads* pre-scored articles from SQLite `sentiment/data/news.db`
  (`MARS_SENTIMENT_DB_PATH`); it never loads FinBERT. Scoring happens offline via `newsagent/pipeline.py`.
- `macro/`: built as a thin HTTP adapter. The real logic is the standalone `agents/macro/` package
  (repo root, its own pyproject/tests), served on `MACRO_AGENT_URL` (default `http://localhost:8010`).
  If that service is not running, the macro agent degrades instead of failing.
- `fundamentals/`: being rebuilt from scratch on `aneeqa/fundamentals-agent`. The target design is in
  `docs/fundamentals_agent.md`; data comes only from PSX (`docs/fundamentals_data.md`): standalone
  statements, `published_at` from dated PSX filings, balance sheets keyed from filing PDFs and checked by
  `data/validate_official.py`. Company metadata (PSX symbol, bank/non-bank, fiscal year-end) is in
  `fundamentals/reference/companies.csv`. An earlier, unverified draft is kept for reference on the
  `wip/fundamentals-agent` branch (`git show wip/fundamentals-agent:<path>`). Do not merge that branch.
- `fx/`, `forensics/`: stubs that raise `NotImplementedError`.

**Backend services:**
- LLM calls go only through `app/llm/client.py`, which switches between Ollama and Groq via `LLM_PROVIDER`.
- Settings come from `app/core/config.py`, which reads the repo-root `.env`.
- The DB engine is lazy, so tests never need Postgres. The `analysis_runs` / `agent_results` tables exist but nothing writes to them yet.

**Data:** `data/raw/` and `data/processed/` are gitignored (only `data/ticker_map.csv` is committed),
so the agents that read local files need the files copied in by hand.

**Frontend** (Next.js 16 App Router, React 19, Tailwind 4):
- `lib/useAgents.ts` fetches each agent independently; every card has its own loading, error and retry state.
- `lib/api.ts` `fetchAgent()` decides, per agent, whether to call the live backend or `data/mock.ts`.
  Technical, sentiment and macro are live (macro and sentiment are mocked when `NEXT_PUBLIC_USE_MOCK=true`);
  the rest are mocked.
- Each live agent needs an adapter that maps the backend `AgentOutput` to the UI `AgentOutput` in
  `lib/types.ts` (see `lib/technical-adapter.ts`).
- `NEXT_PUBLIC_*` flags (`USE_MOCK`, `USE_MOCK_TECHNICAL`, `TECH_AS_OF`) are documented in
  `frontend/.env.example`; copy it to `.env.local`.
- The company page (`components/company/CompanyTechnicalView.tsx`) shows per-agent deep-dive views.
- The selected ticker is shared through `lib/ticker-context.tsx`.
- Read `frontend/AGENTS.md`: this Next.js version has breaking changes, so check `node_modules/next/dist/docs/` before writing Next.js code.

## Conventions

- Adding an agent:
  - Implement `analyze()` in `backend/app/agents/<name>/agent.py`.
  - Put data fetching in `app/ingestion/` and route LLM calls through `llm/client.py`.
  - Add tests in `backend/tests/unit/`, with fixtures in `tests/fixtures/`.
  - Add or extend a route in `app/api/routes/agents/` and include its router in `app/main.py`.
  - Add a frontend adapter in `lib/api.ts`.
- Git:
  - `main` is protected (PR + 1 approval + passing CI); `dev` is the integration branch.
  - Branch names: README says `feat/<area>-<short>`, but in practice members prefix branches with their own
    name (`aneeqa/...`, `fatima/...`, `mariam/...`).
  - New branches must not track `origin/dev`, or a plain `git push` goes to `dev`. Push with
    `git push -u origin <branch>`.
  - `frontend/.venv/` is a stray local Python venv; run `npm run lint` on project dirs if it reports errors there.
  - Never commit `.env` or raw data.
- Ruff: line length 100, target py311, rules E/F/I/UP.
