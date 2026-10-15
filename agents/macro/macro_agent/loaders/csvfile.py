"""CSV drop-folder loader. Handles SBP EasyData exports ('Observation Date' / 'Observation Value' /
'Observation Status') and simple two-column files. Use this for any manually downloaded series
(PBS CPI, SBP series, World Bank Pink Sheet gold, Yahoo GC=F)."""
from __future__ import annotations

import pandas as pd

from .base import Loader, LoaderUnavailable

DATE_COLS = ("observation date", "obs_date", "date", "period", "month")
VALUE_COLS = ("observation value", "value", "close", "adj close", "price", "rate")


def read_flexible_csv(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    low = {c.lower().strip(): c for c in df.columns}
    dcol = next((low[c] for c in DATE_COLS if c in low), df.columns[0])
    vcol = next((low[c] for c in VALUE_COLS if c in low), None)
    if vcol is None:
        nums = [c for c in df.columns if c != dcol and pd.api.types.is_numeric_dtype(df[c])]
        if not nums:
            raise LoaderUnavailable(f"{path}: no numeric value column found")
        vcol = nums[0]
    scol = next((low[c] for c in ("observation status", "status") if c in low), None)
    out = pd.DataFrame({"obs_date": df[dcol], "value": df[vcol],
                        "status": df[scol].astype(str) if scol else "final"})
    return out


class CSVFileLoader(Loader):
    def _fetch(self) -> pd.DataFrame:
        if not self.cfg.file:
            raise LoaderUnavailable(f"{self.cfg.key}: no file configured")
        path = self.root / self.cfg.file
        if not path.exists():
            raise LoaderUnavailable(f"{self.cfg.key}: drop file not found: {path}")
        return read_flexible_csv(path)
