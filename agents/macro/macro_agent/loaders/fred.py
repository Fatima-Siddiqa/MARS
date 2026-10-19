"""FRED public CSV endpoint (e.g. DCOILBRENTEU). Missing values are '.' and are dropped."""
from __future__ import annotations

import io

import pandas as pd

from .base import Loader, LoaderUnavailable, http_get


class FREDLoader(Loader):
    def _fetch(self) -> pd.DataFrame:
        if not self.cfg.fred_id:
            raise LoaderUnavailable(f"{self.cfg.key}: fred_id not set")
        r = http_get("https://fred.stlouisfed.org/graph/fredgraph.csv", params={"id": self.cfg.fred_id})
        df = pd.read_csv(io.StringIO(r.text))
        df.columns = ["obs_date", "value"]
        df["status"] = "final"
        return df[df["value"].astype(str) != "."]
