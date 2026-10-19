"""PSX Data Portal end-of-day index series (KSE-100) - regime support and evaluation only.
Endpoint shape is [TO CONFIRM]; terms of use / robots.txt are not yet checked, so the CSV drop-folder is the
safe default and the web call is only attempted when the drop file is missing."""
from __future__ import annotations

import pandas as pd

from .base import LoaderUnavailable, http_get
from .csvfile import CSVFileLoader


class PSXLoader(CSVFileLoader):
    def _fetch(self) -> pd.DataFrame:
        try:
            return super()._fetch()
        except LoaderUnavailable:
            sym = self.cfg.psx_symbol or "KSE100"
            data = http_get(f"https://dps.psx.com.pk/timeseries/eod/{sym}").json().get("data") or []
            if not data:
                raise LoaderUnavailable("PSX portal returned no data")
            # assumed row shape: [epoch_seconds, close, volume, open]
            df = pd.DataFrame(data).iloc[:, :2]
            df.columns = ["obs_date", "value"]
            df["obs_date"] = pd.to_datetime(df["obs_date"], unit="s").dt.normalize()
            return df.assign(status="final")
