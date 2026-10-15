"""Plain-English explanation, risks and evidence (template based: no LLM, no hallucination)."""
import pandas as pd

from app.ingestion.prices import PRICE_SOURCE
from app.schemas.agent_output import Evidence, Stance

# feature -> (bullish phrase, bearish phrase) written for non-experts
PHRASES = {
    "px_sma50": ("the price is above its 2-3 month average", "the price is below its 2-3 month average"),  # noqa: E501
    "px_sma200": ("the price is above its 10-month average", "the price is below its 10-month average"),  # noqa: E501
    "sma50_sma200": ("the medium-term trend is rising versus the long-term one", "the medium-term trend is falling versus the long-term one"),  # noqa: E501
    "macd_hist": ("short-term momentum is strengthening", "short-term momentum is fading"),
    "adx_dir": ("trend strength points upward", "trend strength points downward"),
    "mom_12_1": ("the stock rose over the past year (excluding last month)", "the stock fell over the past year (excluding last month)"),  # noqa: E501
    "ret_6m": ("the 6-month return is positive", "the 6-month return is negative"),
    "rsi_zone": ("recent selling looks stretched (RSI oversold)", "recent buying looks stretched (RSI overbought)"),  # noqa: E501
    "rs_3m": ("it beat the KSE-100 over 3 months", "it lagged the KSE-100 over 3 months"),
    "rs_6m": ("it beat the KSE-100 over 6 months", "it lagged the KSE-100 over 6 months"),
    "vol_pctile": ("price swings are calmer than usual", "price swings are wilder than usual"),
    "atr_pct": ("daily trading range is narrower than usual", "daily trading range is wider than usual"),  # noqa: E501
    "drawdown": ("it is close to its 1-year high", "it is well below its 1-year high"),
    "bollinger": ("it is not stretched above its recent band", "it is stretched near the top of its recent band"),  # noqa: E501
}
FAMILY_LABEL = {"trend": "trend", "momentum": "momentum", "relative_strength": "relative strength vs KSE-100", "risk": "risk"}  # noqa: E501


def top_drivers(scores: pd.Series, k: int = 3) -> list[tuple[str, float]]:
    s = scores[list(PHRASES)].dropna()
    return [(n, float(s[n])) for n in s.abs().sort_values(ascending=False).index[:k]]


def build_text(ticker: str, stance: Stance, score: float, fam: dict[str, float],
               drivers: list[tuple[str, float]], raw: pd.Series) -> tuple[str, list[str]]:
    tone = {Stance.BULLISH: "mildly positive" if score < 0.4 else "positive",
            Stance.BEARISH: "mildly negative" if score > -0.4 else "negative",
            Stance.NEUTRAL: "mixed / neutral"}[stance]
    findings = [f"{PHRASES[n][0 if v > 0 else 1].capitalize()} (score {v:+.2f})."
                for n, v in drivers]
    why = "; ".join(PHRASES[n][0 if v > 0 else 1] for n, v in drivers)
    fams = ", ".join(f"{FAMILY_LABEL[k]} {v:+.2f}" for k, v in fam.items() if v == v)
    summary = (f"{ticker}'s recent price behaviour is {tone} (score {score:+.2f}). "
               + (f"Main drivers: {why}. " if why else "Not enough history for a reading. ")
               + f"Family scores: {fams}. "
               + "This describes past price behaviour; it is not a forecast.")
    return summary.replace("  ", " "), findings


def build_risks(raw: pd.Series, fresh: float, n_days: int, fam: dict[str, float]) -> list[str]:
    r = []
    if raw.get("adx14", 99) < 20:
        r.append("Market is moving sideways (ADX < 20): trend signals are less reliable.")
    if raw.get("vol_pctile", 0) > 0.8:
        r.append("Volatility is in the top 20% of its past year: larger swings are likely.")
    if raw.get("drawdown_252", 0) < -0.2:
        r.append(f"Price is {raw['drawdown_252']:.0%} below its 1-year high.")
    if raw.get("no_trade_20", 0) > 0.25:
        r.append("Thin trading: many no-trade days recently, quoted prices may be unreliable.")
    if fresh < 1:
        r.append("Price data is not up to date; the reading may not reflect current conditions.")
    if n_days < 252:
        r.append("Less than one year of history: several indicators are unavailable.")
    if fam.get("momentum", 0) > 0.3:
        r.append("Momentum can reverse suddenly, especially after market panics.")
    return r


def build_evidence(ticker: str, raw: pd.Series, drivers: list[tuple[str, float]],
                   as_of: pd.Timestamp) -> list[Evidence]:
    val = {
        "px_sma50": f"Close {raw['close']:.2f} vs SMA50 {raw['sma50']:.2f}",
        "px_sma200": f"Close {raw['close']:.2f} vs SMA200 {raw['sma200']:.2f}",
        "sma50_sma200": f"SMA50 {raw['sma50']:.2f} vs SMA200 {raw['sma200']:.2f}",
        "macd_hist": f"MACD histogram {raw['macd_hist']:.3f}",
        "adx_dir": f"ADX(14) {raw['adx14']:.1f}",
        "mom_12_1": f"12-1 momentum {raw['mom_12_1']:+.1%}",
        "ret_6m": f"6-month return {raw['ret_6m']:+.1%}",
        "rsi_zone": f"RSI(14) {raw['rsi14']:.1f}",
        "rs_3m": f"3m excess vs KSE-100 {raw['rs_3m']:+.1%}",
        "rs_6m": f"6m excess vs KSE-100 {raw['rs_6m']:+.1%}",
        "vol_pctile": f"20d volatility {raw['vol20']:.1%} (percentile {raw['vol_pctile']:.0%})",
        "atr_pct": f"ATR(14) {raw['atr_pct']:.2%} of price",
        "drawdown": f"Drawdown vs 252d high {raw['drawdown_252']:.1%}",
        "bollinger": f"Bollinger %B {raw['bb_pctb']:.2f}",
    }
    ev = []
    for n, sc in drivers:
        src = "KSE-100 index history + " + PRICE_SOURCE if n.startswith("rs_") else PRICE_SOURCE
        ev.append(Evidence(claim=PHRASES[n][0 if sc > 0 else 1], source_name=src,
                           source_type="price_data", excerpt=f"{val[n]} (as of {as_of.date()})"))
    return ev
