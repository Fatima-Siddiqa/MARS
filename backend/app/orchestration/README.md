# orchestration/
LangGraph graph (Iteration 2): fan-out to the six agents in parallel -> fan-in to synthesis.
Until then, `api/routes/recommendation.py` runs agents with `asyncio.gather`.
