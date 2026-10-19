"""Loader interface. A loader returns obs_date, value, status; the pipeline adds unit, source URL
and download timestamp when it appends to macro_raw. Loaders never clean data."""
from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod

import pandas as pd
import requests

from ..config import SeriesCfg

log = logging.getLogger(__name__)
UA = {"User-Agent": "MARS-macro-agent/0.1 (academic research; contact: see repository)"}


class LoaderUnavailable(RuntimeError):
    """The source cannot be reached or is not configured; the pipeline logs it and carries on."""


def parse_dates(col: pd.Series) -> pd.Series:
    """ISO (YYYY-MM-DD) strings are parsed strictly; anything else (e.g. '25-May-2015', '05/06/2015') is parsed
    day-first, the convention of SBP/PBS exports. Never apply dayfirst to ISO dates: 2013-07-01 would become 2013-01-07."""
    if pd.api.types.is_datetime64_any_dtype(col):
        return col
    s = col.astype(str).str.strip()
    out = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
    rest = out.isna()
    if rest.any():
        out[rest] = pd.to_datetime(s[rest], format="mixed", dayfirst=True, errors="coerce")
    if out.isna().any():
        raise ValueError(f"unparseable dates, e.g. {s[out.isna()].iloc[0]!r}")
    return out


def normalise_obs_dates(df: pd.DataFrame, frequency: str) -> pd.DataFrame:
    """Daily/weekly: calendar day. Monthly: month end. Annual: kept as supplied."""
    df = df.copy()
    df["obs_date"] = parse_dates(df["obs_date"]).dt.normalize()
    if frequency == "monthly":
        df["obs_date"] = df["obs_date"] + pd.offsets.MonthEnd(0)
    return df


def http_get(url: str, params: dict | None = None, retries: int = 3, timeout: int = 30) -> requests.Response:
    if os.environ.get("MACRO_AGENT_OFFLINE") == "1":
        raise LoaderUnavailable("offline mode (MACRO_AGENT_OFFLINE=1)")
    last: Exception | None = None
    for i in range(retries):
        try:
            r = requests.get(url, params=params, headers=UA, timeout=timeout)
            r.raise_for_status()
            return r
        except requests.RequestException as e:      # noqa: PERF203
            last = e
            time.sleep(1.5 * (i + 1))
    raise LoaderUnavailable(f"GET {url} failed: {last}")


class Loader(ABC):
    def __init__(self, cfg: SeriesCfg, root):
        self.cfg, self.root = cfg, root

    @abstractmethod
    def _fetch(self) -> pd.DataFrame: ...

    def fetch(self) -> pd.DataFrame:
        try:
            df = self._fetch()
        except LoaderUnavailable:
            # network sources fall back to a manually downloaded CSV in the drop folder, if one is configured
            from .csvfile import CSVFileLoader
            if self.cfg.file and not isinstance(self, CSVFileLoader) and (self.root / self.cfg.file).exists():
                log.warning("%s: network source unavailable, using drop file %s", self.cfg.key, self.cfg.file)
                df = CSVFileLoader(self.cfg, self.root)._fetch()
            else:
                raise
        if df.empty:
            raise LoaderUnavailable(f"{self.cfg.key}: source returned no rows")
        df = df[["obs_date", "value", "status"]].copy()
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["obs_date", "value"])
        return normalise_obs_dates(df, self.cfg.frequency).sort_values("obs_date").reset_index(drop=True)
