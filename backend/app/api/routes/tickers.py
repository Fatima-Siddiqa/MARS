import csv
from functools import lru_cache

from fastapi import APIRouter, HTTPException

from app.core.config import get_settings

router = APIRouter(prefix="/tickers", tags=["tickers"])


@lru_cache
def load_tickers() -> list[dict]:
    with open(get_settings().ticker_map_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@router.get("")
def list_tickers() -> list[dict]:
    return load_tickers()


@router.get("/{symbol}")
def get_ticker(symbol: str) -> dict:
    for row in load_tickers():
        if row["symbol"].upper() == symbol.upper():
            return row
    raise HTTPException(404, f"Unknown ticker {symbol}")
