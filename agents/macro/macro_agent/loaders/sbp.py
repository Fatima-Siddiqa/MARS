"""SBP EasyData. Primary route: CSV exported from easydata.sbp.org.pk into data/drop/ (works with no key).
Optional API route: set SBP_API_KEY and `sbp_series_code` for the series. The API URL template below is
[TO CONFIRM] against the EasyData API documentation, as are the series codes."""
from __future__ import annotations

import io
import os

import pandas as pd

from .base import LoaderUnavailable, http_get
from .csvfile import CSVFileLoader, read_flexible_csv

API_TEMPLATE = os.environ.get(
    "SBP_API_TEMPLATE", "https://easydata.sbp.org.pk/api/v1/series/{code}/data?api_key={key}&format=csv")


class SBPEasyDataLoader(CSVFileLoader):
    def _fetch(self) -> pd.DataFrame:
        try:
            return super()._fetch()
        except LoaderUnavailable as file_err:
            key = os.environ.get("SBP_API_KEY")
            if not (key and self.cfg.sbp_series_code):
                raise LoaderUnavailable(f"{file_err}; and no SBP_API_KEY/sbp_series_code for the API route")
            r = http_get(API_TEMPLATE.format(code=self.cfg.sbp_series_code, key=key))
            return read_flexible_csv(io.StringIO(r.text))
