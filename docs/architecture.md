# Architecture

```
User -> Web Frontend (Next.js) -> MARS Backend (FastAPI)
          [User request handler | Data router | Task orchestrator (LangGraph)]
                       |
     six specialist agents in parallel (fan-out)
     Technical | Sentiment & News | Macro | FX | Sector & Fundamentals | Forensics
                       |
     Synthesis agent (fan-in)
       regime detection (Bull/Bear/Neutral) -> dynamic weighting x regime fit
       -> conflict detection -> weighted aggregation -> confidence score
       -> regime-aware conflict-resolution protocol
                       |
     Output: BUY/HOLD/AVOID + confidence + thesis + key risks + citable evidence + automated report
```

Supporting services: PostgreSQL, LLM/NLP models (Ollama or Groq via one wrapper, FinBERT),
monitoring, logging/audit trail.

## Principles
- Agents are independent and parallel; they only communicate through `AgentOutput`.
- Graceful degradation: a failed agent yields `status != ok`; synthesis continues without it.
- Baseline (`synthesis/baseline_average.py`) is kept as fallback and as the evaluation baseline.
- GenAI boundary: pretrained LLMs only as reasoning backbone; no model training; never executes trades.
