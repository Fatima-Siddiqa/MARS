"""Indicators, robust expanding z-scores and oriented feature scores (Level 2).

Every value on date t uses data up to t only (no look-ahead). Inputs are the cleaned frame from
`app.ingestion.prices.clean_prices` (Close == Adj. Close; O/H/L already adjusted).
"""
import numpy as np
import pandas as pd

TRADING_DAYS = 245
Z_MIN_OBS = 252  # min observations before a z-score exists (guide says 36m; see README)

# oriented scores in [-1, +1] (positive == bullish), grouped in families
FAMILIES: dict[str, list[str]] = {
    "trend": ["px_sma50", "px_sma200", "sma50_sma200", "macd_hist", "adx_dir"],
    "momentum": ["mom_12_1", "ret_6m", "rsi_zone"],
    "relative_strength": ["rs_3m", "rs_6m"],
    "risk": ["vol_pctile", "atr_pct", "drawdown", "bollinger"],
}
SCORE_COLS = [c for cols in FAMILIES.values() for c in cols]
LIQ_COLS = ["log_vol_ratio", "log_amihud"]  # model inputs only; drive confidence, not score
FEATURE_COLS = SCORE_COLS + LIQ_COLS


def _wilder(s: pd.Series, n: int = 14) -> pd.Series:
    return s.ewm(alpha=1 / n, adjust=False).mean()


def compute_raw(df: pd.DataFrame, bench: pd.Series | None = None) -> pd.DataFrame:
    """Raw (un-standardised) indicators, one row per trading day."""
    c, h, lo, v = df["Close"], df["High"], df["Low"], df["Volume"]
    out = pd.DataFrame(index=df.index)
    out["close"] = c
    for n in (20, 50, 200):
        out[f"sma{n}"] = c.rolling(n).mean()
    d = c.diff()
    avg_g, avg_l = _wilder(d.clip(lower=0)), _wilder((-d).clip(lower=0))
    rsi = 100 - 100 / (1 + avg_g / avg_l.replace(0, np.nan))
    rsi = rsi.fillna(100.0)  # no losses in window -> 100
    rsi.iloc[:14] = np.nan  # warm-up
    out["rsi14"] = rsi
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    out["macd_hist"] = macd - macd.ewm(span=9, adjust=False).mean()
    # ADX / ATR
    pc = c.shift()
    tr = pd.concat([h - lo, (h - pc).abs(), (lo - pc).abs()], axis=1).max(axis=1)
    up, dn = h.diff(), -lo.diff()
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=c.index)
    mdm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=c.index)
    atr = _wilder(tr)
    pdi, mdi = 100 * _wilder(pdm) / atr, 100 * _wilder(mdm) / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    out["adx14"] = _wilder(dx.fillna(0.0))
    out["di_sign"] = np.sign(pdi - mdi)
    out["atr_pct"] = atr / c
    sd20 = c.rolling(20).std()
    out["bb_pctb"] = (c - (out["sma20"] - 2 * sd20)) / (4 * sd20).replace(0, np.nan)
    lr = np.log(c / pc)
    out["vol20"] = lr.rolling(20).std() * np.sqrt(TRADING_DAYS)
    out["vol_pctile"] = out["vol20"].rolling(252).apply(lambda x: (x <= x[-1]).mean(), raw=True)
    out["drawdown_252"] = c / c.rolling(252).max() - 1
    out["mom_12_1"] = c.shift(21) / c.shift(252) - 1
    out["ret_6m"] = c / c.shift(126) - 1
    out["ret_3m"] = c / c.shift(63) - 1
    if bench is not None:
        b = bench.reindex(c.index.union(bench.index)).ffill().reindex(c.index)
        out["rs_3m"] = out["ret_3m"] - (b / b.shift(63) - 1)
        out["rs_6m"] = out["ret_6m"] - (b / b.shift(126) - 1)
    else:
        out["rs_3m"] = out["rs_6m"] = np.nan
    # liquidity
    out["no_trade"] = df["no_trade"].astype(float)
    out["no_trade_20"] = out["no_trade"].rolling(20, min_periods=1).mean()
    out["log_vol_ratio"] = np.log((v + 1) / (v.rolling(20).mean() + 1))
    ami = (c.pct_change().abs() / (c * v).where(v > 0)).rolling(60, min_periods=20).mean()
    out["log_amihud"] = np.log(ami.where(ami > 0))
    return out


def expanding_robust_z(
    s: pd.Series, min_obs: int = Z_MIN_OBS, last_only: bool = False
) -> pd.Series:
    """(x - expanding median) / (1.4826 * expanding MAD), clipped at +-3 and divided by 3."""
    v = s.to_numpy(float)
    ok = np.flatnonzero(~np.isnan(v))
    out = np.full(len(v), np.nan)
    if len(ok) >= min_obs:
        vv = v[ok]
        ks = [len(vv) - 1] if last_only else range(min_obs - 1, len(vv))
        for k in ks:
            w = vv[: k + 1]
            med = np.median(w)
            mad = 1.4826 * np.median(np.abs(w - med))
            z = 0.0 if mad < 1e-12 else (vv[k] - med) / mad
            out[ok[k]] = np.clip(z, -3, 3) / 3
    return pd.Series(out, index=s.index)


def _rsi_zone(r: pd.Series) -> pd.Series:
    s = np.where(r < 30, np.minimum(1, (30 - r) / 15), np.where(r > 70, -np.minimum(1, (r - 70) / 15), 0.0))  # noqa: E501
    return pd.Series(s, index=r.index).where(r.notna())


def feature_scores(raw: pd.DataFrame, last_only: bool = False) -> pd.DataFrame:
    """Oriented scores in [-1, 1] for every column in FEATURE_COLS."""
    c = raw["close"]
    base = {
        "px_sma50": c / raw["sma50"] - 1,
        "px_sma200": c / raw["sma200"] - 1,
        "sma50_sma200": raw["sma50"] / raw["sma200"] - 1,
        "macd_hist": raw["macd_hist"] / c,
        "adx_dir": raw["adx14"] / 100 * raw["di_sign"],  # trend strength signed by direction
        "mom_12_1": raw["mom_12_1"],
        "ret_6m": raw["ret_6m"],
        "rs_3m": raw["rs_3m"],
        "rs_6m": raw["rs_6m"],
        "atr_pct": -raw["atr_pct"],  # higher range = riskier
        "drawdown": raw["drawdown_252"],  # deeper drawdown = more bearish
        "bollinger": -raw["bb_pctb"],  # stretched upward = mild bearish
        "log_vol_ratio": raw["log_vol_ratio"],
        "log_amihud": -raw["log_amihud"],  # thinner trading = lower
    }
    out = pd.DataFrame(index=raw.index)
    for k, s in base.items():
        out[k] = expanding_robust_z(s, last_only=last_only)
    out["rsi_zone"] = _rsi_zone(raw["rsi14"])
    out["vol_pctile"] = -(2 * raw["vol_pctile"] - 1)  # rank percentile -> [-1, 1], high vol bearish
    return out[FEATURE_COLS]


def family_scores(scores: pd.DataFrame) -> pd.DataFrame:
    """Mean of available features per family (NaN if none available)."""
    return pd.DataFrame({f: scores[cols].mean(axis=1, skipna=True) for f, cols in FAMILIES.items()})


def rule_score(fams: pd.DataFrame) -> pd.Series:
    """Equal-weight mean of the available families (liquidity does not push the score)."""
    return fams.mean(axis=1, skipna=True)
