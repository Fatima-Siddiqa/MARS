# 0001. Single monorepo and core stack

- Status: accepted

## Context
Three members build six agents plus a shared synthesis layer and a web app.

## Decision
One GitHub repo. Python + FastAPI + LangGraph backend, PostgreSQL, Next.js + Tailwind frontend,
Docker for environments. LLM access through a single wrapper (`backend/app/llm/client.py`) switching
between Ollama (local) and Groq via the `LLM_PROVIDER` env var. All agents return the shared
`AgentOutput` contract.

## Consequences
Simple integration and CI; contract changes need all-member review.
