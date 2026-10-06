# MARS: Regime-Aware Multi-Agent System for Explainable Stock Market Analysis in Emerging Markets

FAST School of Computing, NUCES, Final Year Project 2026.
Team: Mariam Shaiq, Aneeqa Habib, Fatima Siddiqa. Supervisor: Muhammad Atif Saeed.

MARS turns a **PSX-listed ticker** into an explainable, source-grounded **BUY / HOLD / AVOID**
recommendation with a confidence score, investment thesis, key risks, and a citable source per claim.
Six specialist agents run in parallel; a Synthesis agent fuses their outputs using a
regime-aware conflict-resolution protocol (Bull / Bear / Neutral, driven by PKR depreciation,
inflation and SBP policy). MARS is a **decision-support tool, not an automated trading platform**.

## Architecture (fan-out / fan-in)

```
ticker -> [Technical | Sentiment | Macro | FX | Fundamentals | Forensics] -> Synthesis -> recommendation
                                (parallel)                  (+ regime detection, conflict resolution)
```

See `docs/architecture.md` and the shared agent contract in `docs/agent_contract.md`.

## Repository layout

| Path | Purpose |
|---|---|
| `backend/` | FastAPI + LangGraph backend (agents, synthesis, ingestion, LLM wrapper, DB) |
| `backend/app/schemas/agent_output.py` | **Shared contract** all agents must return |
| `frontend/` | Next.js + Tailwind dashboard |
| `data/` | `ticker_map.csv` (committed); `raw/` and `processed/` are gitignored |
| `docs/` | Architecture, agent contract, decision records (ADRs) |
| `evaluation/`, `notebooks/` | Baselines, backtests, experiments |
| `infra/` | Deployment configs |
| `docker-compose.yml` | Postgres, backend, optional Ollama |

## Ownership

| Member | Agents / areas |
|---|---|
| **Fatima** | Technical agent, FX agent, Synthesis agent, backend testing/optimisation |
| **Aneeqa** | Sentiment agent, Fundamentals agent, Regime-aware conflict resolution, frontend/UX, usability study |
| **Mariam** | Macro agent, Forensics agent, Automated reporting, docker-compose, integration & deployment |

Each member also builds the backend + frontend pieces for their own agents.

## Roadmap

| Iteration | Goal | Fatima | Aneeqa | Mariam |
|---|---|---|---|---|
| 1 (Mid) | 3 working agents | Technical | Sentiment | Macro |
| 2 (Final-I) | Full specialist layer | FX | Fundamentals | Forensics |
| 3 (Pre-Job Fair) | Deployed MARS | Synthesis | Regime-aware conflict resolution | Automated reporting + integration |
| 4 (Final) | Testing, refinement, delivery | Backend testing | Frontend/UX, usability | Integration, deployment, validation |

## Quick start

```bash
cp .env.example .env              # then edit
docker compose up --build         # postgres + backend on :8000
# local LLM (optional):  docker compose --profile ollama up
# API docs: http://localhost:8000/docs
```

Without Docker (backend only):

```bash
cd backend
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
uvicorn app.main:app --reload
pytest
ruff check .
```

Database migrations: `cd backend && alembic revision --autogenerate -m "msg" && alembic upgrade head`.

## Adding / implementing an agent

1. Open `backend/app/agents/<name>/agent.py` and implement `analyze(ticker) -> AgentOutput`.
2. Put data fetching in `backend/app/ingestion/`, LLM calls through `backend/app/llm/client.py`.
3. Every claim needs an `Evidence` entry (source name/URL/excerpt).
4. Add unit tests in `backend/tests/unit/` (use `backend/tests/fixtures/` for sample data).
5. Do not change `schemas/agent_output.py` without a PR approved by all three members.

## Git workflow

- `main` is protected: PR + 1 approval + passing CI. No direct pushes.
- Branch names: `feat/<area>-<short>`, `fix/...`, `docs/...` (e.g. `feat/technical-rsi-macd`).
- Small PRs, rebase/squash merge. Never commit `.env` or raw data.

## Disclaimer

MARS is a decision-support system, not financial advice. It never places trades.
