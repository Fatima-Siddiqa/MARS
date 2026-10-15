"""Typed access to the pinned YAML config. A hash of the file goes into model_version."""
from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config" / "macro_agent.yaml"


class SeriesCfg(BaseModel):
    key: str = ""
    source: str
    unit: str
    frequency: str
    release_lag_days: int
    expected_interval_days: int
    tolerance_days: int
    plausible: tuple[float, float]
    source_url: str
    use_from: str | None = None
    breaks: list[str] = Field(default_factory=list)
    file: str | None = None
    fred_id: str | None = None
    wb_indicator: str | None = None
    wb_country: str | None = None
    sbp_series_code: str | None = None     # EasyData series key [TO CONFIRM]
    psx_symbol: str | None = None


class FeatureCfg(BaseModel):
    name: str = ""
    pillar: str
    orient: int
    series: list[str]
    unit: str
    label: str
    enabled: bool = True
    method: str = "robust_z"          # robust_z | percentile
    weight: float = 1.0
    fixed_rule: dict[str, float] | None = None


class Config(BaseModel):
    version: str
    horizon_days: int
    database_url: str
    timezone: str
    standardise: dict[str, float]
    state: dict[str, float]
    quality: dict[str, Any]
    regime_flags: dict[str, Any]
    pillars: list[str]
    pillar_weights: dict[str, float] = Field(default_factory=dict)
    series: dict[str, SeriesCfg]
    features: dict[str, FeatureCfg]
    feature_params: dict[str, float]
    tickers: dict[str, str]
    sectors: dict[str, dict[str, float]]
    llm: dict[str, Any]
    evaluation: dict[str, Any]
    config_hash: str = ""
    root: Path = ROOT

    @property
    def model_version(self) -> str:
        return f"macro-rules-{self.version}+{self.config_hash[:8]}"

    def active_features(self) -> dict[str, FeatureCfg]:
        return {k: f for k, f in self.features.items() if f.enabled}

    def sector_of(self, ticker: str | None) -> str | None:
        return self.tickers.get(ticker.upper()) if ticker else None

    def multipliers(self, ticker: str | None) -> dict[str, float]:
        sec = self.sector_of(ticker)
        return dict(self.sectors.get(sec, {})) if sec else {}

    def resolve(self, rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else self.root / p


def load_config(path: str | os.PathLike | None = None, overrides: dict | None = None) -> Config:
    path = Path(path or os.environ.get("MACRO_AGENT_CONFIG", DEFAULT_CONFIG))
    raw_text = path.read_text()
    raw = yaml.safe_load(raw_text)
    if overrides:
        raw.update(overrides)
    if os.environ.get("DATABASE_URL"):
        raw["database_url"] = os.environ["DATABASE_URL"]
    for k, s in raw["series"].items():
        s["key"] = k
    for k, f in raw["features"].items():
        f["name"] = k
    cfg = Config(**raw, config_hash=hashlib.sha256(raw_text.encode()).hexdigest())
    cfg.root = path.resolve().parent.parent
    return cfg


@lru_cache(maxsize=1)
def get_config() -> Config:
    return load_config()
