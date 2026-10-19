"""Central configuration. Every design parameter from the MARS methodology
(Section 4 and Section 13) lives here so it can be tuned on the validation
period only and recorded with each run."""
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = Path(os.getenv("MARS_SENTIMENT_DB_PATH", str(DATA_DIR / "news.db")))
if not DB_PATH.is_absolute():
    DB_PATH = BASE_DIR / DB_PATH
COMPANIES_CSV = DATA_DIR / "companies.csv"
SECTORS_CSV = DATA_DIR / "sectors.csv"
MARKET_CSV = DATA_DIR / "market_keywords.csv"
SOURCES_CSV = DATA_DIR / "sources.csv"
TICKER_MAP_CSV = DATA_DIR / "ticker_map.csv"

# Publisher feeds. Run `python -m app.check_sources` to see which are alive:
# feed URLs change and a few below could not be verified when this was built.
RSS_FEEDS = {
    "dawn_business": "https://www.dawn.com/feeds/business",
    "business_recorder": "https://www.brecorder.com/feeds/latest-news",
    "express_tribune_business": "https://news.google.com/rss/search?q=site%3Atribune.com.pk%20(business%20OR%20economy%20OR%20stocks%20OR%20PSX)%20-author&hl=en-PK&gl=PK&ceid=PK%3Aen",
    # "profit": "https://profit.pakistantoday.com.pk/feed/",
    "propakistani": "https://propakistani.pk/feed/",
    "mettis": "https://news.google.com/rss/search?q=site%3Amettisglobal.news&hl=en-PK&gl=PK&ceid=PK%3Aen",
    "the_news_business": "https://www.thenews.com.pk/rss/1/8",
}

# Google News RSS search, one query per company and date window. This is also
# the historical back-fill source now that GDELT has been removed.
GOOGLE_NEWS_ENABLED = True
GOOGLE_NEWS_NIGHTLY_LOOKBACK_DAYS = 14
GOOGLE_NEWS_CHUNK_DAYS = 30
GOOGLE_NEWS_MAX_ITEMS = 100
BACKFILL_DEFAULT_START = "2018-01-01"
BACKFILL_DELAY_SECONDS = 2.0

# PSX_ANNOUNCEMENTS_URL = "https://dps.psx.com.pk/announcements/companies"

PSX_ANNOUNCEMENTS_PAGE_URL = (
    "https://dps.psx.com.pk/announcements/companies"
)

PSX_ANNOUNCEMENTS_URL = (
    "https://dps.psx.com.pk/announcements"
)

REQUEST_TIMEOUT = 20
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
)

# Polarity model (inference only, no fine-tuning)
FINBERT_MODEL = os.getenv("FINBERT_MODEL", "ProsusAI/finbert")
FINBERT_TEMPERATURE = float(os.getenv("FINBERT_TEMPERATURE", "1.0"))
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")

AGENT_NAME = "sentiment_news"
MODEL_VERSION = "sentiment-news-1.0"


@dataclass(frozen=True)
class Params:
    """Design parameters (methodology Section 13 initial values)."""
    half_life_days: float = 3.0          # recency decay, validate {1, 3, 7}
    window_days: int = 14                # effective-article window
    shrink_k: float = 3.0                # shrinkage constant k
    neg_multiplier: float = 1.5          # negative-news weight, tuned on validation
    # PSX filings count this much in tone (news = 1.0); tune on 2021 only
    announcement_tone_weight: float = 0.3
    neg_threshold: float = 0.6           # P(neg) above this counts as negative
    dup_cosine: float = 0.9              # near-duplicate threshold
    dup_window_days: int = 7
    event_cap: float = 0.2               # max event-layer tone adjustment
    baseline_days: int = 90              # baseline for tone change
    min_baseline_obs: int = 5
    min_z_obs: int = 36                  # expanding z needs this many points
    z_clip: float = 3.0
    state_threshold: float = 0.20        # Bullish > +0.20, Bearish < -0.20
    direction_band: float = 0.10         # |tone| above this counts for agreement
    horizon_days: int = 20
    decision_hour_utc: int = 11          # after the PSX close (15:30 PKT = 10:30 UTC)
    date_only_delay_days: int = 1        # date-only timestamps count from next day

    def key(self) -> str:
        blob = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha1(blob).hexdigest()[:10]

    def to_dict(self) -> dict:
        return asdict(self)


TUNED_PARAMS_JSON = DATA_DIR / "tuned_params.json"   # written by `python -m app.tune --commit`


def _load_default_params():
    if TUNED_PARAMS_JSON.exists():
        try:
            return Params(**json.loads(TUNED_PARAMS_JSON.read_text()))
        except Exception:
            pass
    return Params()


DEFAULT_PARAMS = _load_default_params()
