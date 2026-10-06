# synthesis/
- `baseline_average.py` - fixed averaging baseline (exists now; keep as fallback/eval baseline).
- Later (Iteration 3): regime detection (Bull/Bear/Neutral from PKR depreciation, inflation,
  SBP policy), conflict detection, dynamic weighting, and the regime-aware conflict-resolution
  protocol. Add as new modules here; keep the `synthesize(ticker, outputs) -> Recommendation` shape.
