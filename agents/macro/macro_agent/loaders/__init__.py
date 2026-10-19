from __future__ import annotations

from ..config import SeriesCfg
from .base import Loader, LoaderUnavailable
from .csvfile import CSVFileLoader
from .fred import FREDLoader
from .psx import PSXLoader
from .sbp import SBPEasyDataLoader
from .worldbank import WorldBankLoader

REGISTRY: dict[str, type[Loader]] = {
    "sbp_easydata": SBPEasyDataLoader,
    "pbs": CSVFileLoader,
    "csv_file": CSVFileLoader,
    "fred": FREDLoader,
    "worldbank": WorldBankLoader,
    "psx": PSXLoader,
}


def get_loader(cfg: SeriesCfg, root) -> Loader:
    if cfg.source not in REGISTRY:
        raise KeyError(f"no loader for source '{cfg.source}'")
    return REGISTRY[cfg.source](cfg, root)


__all__ = ["get_loader", "Loader", "LoaderUnavailable"]
