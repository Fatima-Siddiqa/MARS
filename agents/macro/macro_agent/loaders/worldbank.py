"""World Bank indicator API (annual). Year Y is stored with obs_date 30 June of Y (Pakistan's fiscal year end);
usable only after release_lag_days (a conservative lag, [TO CONFIRM])."""
from __future__ import annotations

import pandas as pd

from .base import Loader, LoaderUnavailable, http_get


class WorldBankLoader(Loader):
    def _fetch(self) -> pd.DataFrame:
        if not (self.cfg.wb_indicator and self.cfg.wb_country):
            raise LoaderUnavailable(f"{self.cfg.key}: wb_indicator / wb_country not set")
        url = f"https://api.worldbank.org/v2/country/{self.cfg.wb_country}/indicator/{self.cfg.wb_indicator}"
        payload = http_get(url, params={"format": "json", "per_page": 500}).json()
        if not isinstance(payload, list) or len(payload) < 2 or not payload[1]:
            raise LoaderUnavailable(f"{self.cfg.key}: unexpected World Bank response")
        rows = [(f"{r['date']}-06-30", r["value"]) for r in payload[1] if r.get("value") is not None]
        return pd.DataFrame(rows, columns=["obs_date", "value"]).assign(status="final")
