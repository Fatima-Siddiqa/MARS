# ML prototype: do machine-learning challengers beat the rule-based macro score?

Status: **research prototype** (2026-10-20). Nothing here changes the live agent; the rules stay the production
score. Reproduce with:

```bash
pip install -e ".[dev,ml]"
macro-agent fetch-open     # real open data -> data/drop (see data/drop/SOURCES.md)
macro-agent refresh        # into a fresh macro_raw (do not mix with demo-data rows)
macro-agent ml-eval        # -> reports/ml/ml_research.{md,json}, ml_predictions.csv
```

## Data (all real, open sources)
| series | source | frequency | coverage |
|---|---|---|---|
| policy_rate | SBP EasyData SBPOL0030 (manual export) | decision | 2015-05 to 2026-04 |
| pkr_usd | BIS WS_XRU | monthly average | 1957 to 2026-08 |
| cpi_yoy | World Bank GEM CPTOTSAXNZGY | monthly | 1987 to 2026-07 |
| fx_reserves | World Bank GEM TOTRESV | monthly | 1990 to 2026-06 |
| kse100 | World Bank GEM DSTKMKTXN (rebased PSX index) | **monthly average** | 2006-09 to 2026-08 |
| gold | World Bank Pink Sheet | monthly | 1960 to 2026-09 |
| tbill_3m | IMF MFS_IR T-bill yield | monthly | 1996-07 to **2021-12** |
| brent | FRED DCOILBRENTEU | daily | 1987 to 2026-10 |
| gdp_growth | World Bank WDI | annual | 1961 to 2025 |

Missing: **current_account, remittances** (no open monthly source; export from SBP EasyData), and a current
T-bill series. `pkr_vol_30d` is disabled until daily PKR/USD is available (on monthly data it would be a 30-month vol).

## Design
- One row per month-end *t*; inputs are the agent's own as-of outputs (rule composite, 5 pillar scores, 13 oriented
  feature scores), so the no-look-ahead guarantees of the pipeline carry over.
- **Primary target = skip-month return** log K[t+2]/K[t+1]. The GEM index is a *monthly average*, so the plain
  next-month return overlaps month *t* and is mechanically autocorrelated (Working 1960). The skip-month return shares
  no price with anything known at *t*. The next-month target is reported only to show this artefact.
- Expanding walk-forward: 60 initial months, refit every 12, purge = label horizon. Rows whose label reaches the
  lock-box (from 2025-07-01) are excluded, so the agent's final hold-out stays untouched.
- Metrics: Spearman IC with 90% block-bootstrap CI (6-month blocks), directional hit rate, out-of-sample R^2 vs the
  historical mean (Campbell & Thompson 2008), Sharpe of a long-KSE-when-forecast>0 / flat-otherwise rule (no costs).

## Results (skip-month target, 163 out-of-sample months, 2011-10 to 2025-04)
| model | IC | IC 90% CI | hit | R^2 oos | Sharpe long/flat |
|---|---|---|---|---|---|
| historical mean (benchmark) | n/a | | 0.49 | 0.000 | 0.34 |
| AR(1) on KSE returns | -0.136 | [-0.25, -0.01] | 0.48 | -0.003 | 0.32 |
| AR(3) | -0.072 | [-0.19, +0.04] | 0.46 | -0.073 | 0.13 |
| **rules (current agent)** | **+0.140** | **[+0.01, +0.29]** | 0.56 | n/a | 0.55 |
| rules, linearly calibrated | +0.082 | [-0.05, +0.23] | 0.58 | -0.094 | 0.60 |
| ridge on pillars | +0.069 | [-0.06, +0.19] | 0.56 | -0.318 | 0.71 |
| ridge on features | +0.052 | [-0.07, +0.17] | 0.53 | -0.984 | 0.44 |
| XGBoost, monotone, features | +0.100 | [-0.02, +0.23] | 0.58 | -1.415 | 0.53 |
| XGBoost, unconstrained | +0.044 | [-0.10, +0.18] | 0.56 | -1.160 | 0.66 |
| XGBoost, monotone, pillars | +0.122 | [-0.02, +0.24] | 0.56 | -1.048 | 0.46 |
| XGBoost + momentum | +0.055 | [-0.07, +0.18] | 0.56 | -1.315 | 0.57 |
| XGBoost + rule score | +0.110 | [-0.01, +0.24] | 0.60 | -1.333 | 0.60 |

Buy-and-hold Sharpe over the same months: 0.52.

### Ablations
**Leave one pillar out** (IC; full model in the first column):

| model | full | -monetary | -inflation | -external | -commodity | -growth |
|---|---|---|---|---|---|---|
| ridge features | +0.052 | +0.072 | **-0.017** | +0.107 | +0.060 | +0.060 |
| XGBoost monotone | +0.100 | +0.137 | **+0.005** | +0.176 | +0.125 | +0.115 |

**Sub-periods** (IC): every macro model is negative up to 2018 and clearly positive from 2019 (rules -0.11 -> +0.36,
XGBoost -0.13 -> +0.33, ridge -0.10 -> +0.25), while AR(1) is negative in both. From 2019 the monetary pillar is
live (policy-rate history starts 2015 and needs 36 months to standardise) and the sample covers the 2019-2024
inflation, PKR and rate-hike cycle.

**Label choice:** on the next-month (averaged) target AR(1) jumps to IC +0.19 and positive R^2 (+0.066), while it is
negative on the clean target. This confirms the averaging artefact and why the skip-month target is the primary one.

**XGBoost mean |SHAP|** (skip target): cpi_trend > reserves_chg_3m > pkr_dep_30d > yield_spread > rate_vs_12m_avg >
brent_pct_252 > cpi_level > gdp_growth > brent_3m > rate_change_6m.

## Interpretation
1. **The rules hold up.** The rule score has the only clear positive IC among the main models (CI excludes zero), and
   no ML model beats it by more than the noise band (~0.1 IC with 163 months). Per the selection rule already used for
   pillar weights, the rules stay.
2. **No model beats the historical mean on R^2.** Ridge and XGBoost rank months somewhat well but their forecast
   *magnitudes* overfit (R^2 down to -1.4). This is the Welch & Goyal (2008) pattern for macro predictors; if ML is
   used later, only its ranking/sign, or a forecast shrunk toward the mean, should be trusted.
3. **Inflation is the signal.** Removing the inflation pillar removes almost all predictive power in both learners,
   and cpi_trend is the top SHAP feature. This supports the economic hypothesis behind the inflation pillar.
4. **The external pillar currently hurts the learners** (IC rises without it). Likely cause: PKR/USD is now monthly,
   current account and remittances are missing, and reserves come from a monthly secondary source. Re-test after
   exporting daily/weekly SBP series before changing anything.
5. **Monotone constraints help** (XGBoost +0.100 vs +0.044 unconstrained): forcing economically sensible directions is
   a useful regulariser on ~100-200 training rows.
6. **Instability over time** (negative before 2019) is the main caveat for any claim; combine with the regime layer.

Caveats: 21 variants were tested, so about one would clear a 90% one-sided interval by chance (the "-external" XGBoost
result needs confirmation on new data). The Sharpe column ignores costs and is descriptive only.

## Next steps
1. Export from SBP EasyData: daily PKR/USD, weekly reserves, current account, remittances, current 3-month T-bill;
   re-run `refresh` and `ml-eval`.
2. Get the team's daily KSE-100 file (`data/raw/KSE_100_index.csv`, already used by the backend); `fetch-open` picks
   it up automatically and gives true month-end returns.
3. Pool across tickers/sectors (PSX prices) to learn sector sensitivities, which multiplies the sample size.
4. Only then consider promoting an ML component, and only through the existing out-of-sample selection rule and a
   single lock-box run.
