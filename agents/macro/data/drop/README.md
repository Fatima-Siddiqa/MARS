# Drop folder for downloaded series

Save CSV exports here (SBP EasyData exports work as downloaded: columns `Observation Date`, `Observation Value`,
`Observation Status`; simple `date,value` files also work). File names come from `config/macro_agent.yaml`:

policy_rate.csv, tbill_3m.csv, cpi_yoy.csv, pkr_usd.csv, fx_reserves.csv, current_account.csv, remittances.csv,
gold.csv, kse100.csv - and optionally brent.csv / gdp_growth.csv as offline fallbacks for FRED / World Bank.

Then run `macro-agent refresh`. Raw files are never edited; each refresh appends a new version to `macro_raw`.

`macro-agent fetch-open` writes the open-source series here (pkr_usd, cpi_yoy, fx_reserves, kse100, gold, tbill_3m,
brent, gdp_growth) and records each source in `SOURCES.md`. An SBP EasyData export with the same file name replaces
the open-source version (re-run `refresh`). Synthetic demo files for series without an open source are moved to
`_synthetic/` so they never mix with real data.
