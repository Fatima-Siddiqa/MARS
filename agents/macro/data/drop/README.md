# Drop folder for downloaded series

Save CSV exports here (SBP EasyData exports work as downloaded: columns `Observation Date`, `Observation Value`,
`Observation Status`; simple `date,value` files also work). File names come from `config/macro_agent.yaml`:

policy_rate.csv, tbill_3m.csv, cpi_yoy.csv, pkr_usd.csv, fx_reserves.csv, current_account.csv, remittances.csv,
gold.csv, kse100.csv - and optionally brent.csv / gdp_growth.csv as offline fallbacks for FRED / World Bank.

Then run `macro-agent refresh`. Raw files are never edited; each refresh appends a new version to `macro_raw`.
