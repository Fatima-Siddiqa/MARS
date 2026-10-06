# evaluation/
Baselines and experiments. Planned comparisons:
- Single monolithic LLM given all raw data
- Fixed voting/averaging of the same six agents (`backend/app/synthesis/baseline_average.py`)
- Regime-aware synthesis (MARS)

Metrics: prediction accuracy, returns and risk, regime-wise performance, confidence calibration,
source groundedness, usability/trust (user study). Use historical PSX data across differing regimes.
