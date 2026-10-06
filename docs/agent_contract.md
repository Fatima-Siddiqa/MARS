# Agent contract

Source of truth: `backend/app/schemas/agent_output.py` (`AgentOutput`, `Evidence`).

Every agent implements `async analyze(ticker: str) -> AgentOutput` (subclass of `BaseAgent`) and is
invoked through `BaseAgent.run()`, which times the call and converts failures into
`status=unavailable|error` outputs.

| Field | Meaning |
|---|---|
| `agent` | One of technical, sentiment, macro, fx, fundamentals, forensics |
| `status` | ok / degraded / unavailable / error |
| `stance`, `score` | bullish/neutral/bearish; score in [-1, +1] |
| `confidence` | [0, 1], the agent's own confidence in its score |
| `summary`, `key_findings`, `risks` | Human-readable explanation |
| `evidence` | List of `Evidence` (claim, source_name, source_url, excerpt). **Every claim needs one.** |
| `metrics` | Agent-specific numbers (indicator values, ratios, chart series) for the UI |
| `warnings` | Data gaps, fallbacks used |

## Change policy
Changing the contract affects all agents, synthesis and the frontend. Bump `SCHEMA_VERSION`,
update this file, and get approval from all three members.

## Endpoints
`GET /agents/{technical|sentiment|macro}/{ticker}`, `GET /recommendation/{ticker}`, `GET /tickers`.
Add new routes for fx / fundamentals / forensics in `api/routes/agents/` in Iteration 2.
