"""Fetch real series from open, keyless sources into data/drop/ (CSV in the drop-folder format).

Sources (all public, attribution required, no account or API key):
  pkr_usd      BIS WS_XRU, monthly PKR per USD (average of observations; the BIS daily series ends 2019-03)
  cpi_yoy      World Bank Global Economic Monitor (GEM) CPTOTSAXNZGY, monthly % y-o-y
  fx_reserves  World Bank GEM TOTRESV, monthly total reserves (USD million)
  kse100       World Bank GEM DSTKMKTXN, monthly stock market index in PKR (KSE-100 based).
               A daily KSE-100 file at <repo>/data/raw/KSE_100_index.csv (team FYP dataset) is preferred if present.
  gold         World Bank Pink Sheet (CMO historical monthly), gold USD/troy oz
  tbill_3m     IMF MFS_IR GSTBILY (T-bill yield), monthly. Ends 2021-12: history only, stale for live scoring.
  brent        FRED DCOILBRENTEU (daily)
  gdp_growth   World Bank WDI NY.GDP.MKTP.KD.ZG (annual, stored at 30 June)

Not available from open sources (export from SBP EasyData into data/drop/ instead): current_account, remittances,
and a current tbill_3m. Downloads are staged under data/downloads/<date>/ (gitignored) and never edited; the
converted CSVs carry status 'actual'. A manifest (data/drop/SOURCES.md) records URL, retrieval time and coverage.
"""
from __future__ import annotations

import io
import json
import logging
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import Config
from .loaders.base import http_get

log = logging.getLogger(__name__)

WB_GEM = "https://api.worldbank.org/v2/country/PAK/indicator/{ind}?source=15&format=json&per_page=5000&date=1980M01:2035M12"
WB_WDI = "https://api.worldbank.org/v2/country/PAK/indicator/{ind}?format=json&per_page=500"
BIS_XRU = "https://stats.bis.org/api/v1/data/WS_XRU/M.PK.PKR.A?format=csv"  # daily (D.) stops in 2019-03
IMF_IR = "https://api.imf.org/external/sdmx/3.0/data/dataflow/IMF.STA/MFS_IR/+/PAK.GSTBILY_RT_PT_A_PT.M"
FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={id}"
PINK_PAGE = "https://www.worldbank.org/en/research/commodity-markets"

# series that open sources cannot supply; their synthetic demo files are moved out of data/drop
NO_OPEN_SOURCE = ("current_account", "remittances")


def _gem_monthly(ind: str, raw_dir: Path) -> pd.DataFrame:
    r = http_get(WB_GEM.format(ind=ind), timeout=60)
    (raw_dir / f"wb_gem_{ind}.json").write_text(r.text, encoding="utf-8")
    return parse_gem(r.json())


def parse_gem(payload) -> pd.DataFrame:
    """World Bank v2 JSON with monthly dates like '2024M07' -> month-end rows."""
    if not isinstance(payload, list) or len(payload) < 2 or not payload[1]:
        raise ValueError("unexpected World Bank response")
    rows = [(r["date"], r["value"]) for r in payload[1] if r.get("value") is not None and "M" in str(r["date"])]
    df = pd.DataFrame(rows, columns=["period", "value"])
    df["obs_date"] = pd.to_datetime(df["period"].str.replace("M", "-"), format="%Y-%m") + pd.offsets.MonthEnd(0)
    return df[["obs_date", "value"]].sort_values("obs_date").reset_index(drop=True)


def parse_bis_xru(text: str) -> pd.DataFrame:
    d = pd.read_csv(io.StringIO(text), usecols=["TIME_PERIOD", "OBS_VALUE"]).dropna()
    df = pd.DataFrame({"obs_date": pd.to_datetime(d["TIME_PERIOD"]) + pd.offsets.MonthEnd(0),
                       "value": pd.to_numeric(d["OBS_VALUE"], errors="coerce")})
    return df.dropna().sort_values("obs_date").reset_index(drop=True)


def parse_imf_monthly(text: str) -> pd.DataFrame:
    d = pd.read_csv(io.StringIO(text))
    d = d[d["FREQUENCY"] == "M"]
    df = pd.DataFrame({"obs_date": pd.to_datetime(d["TIME_PERIOD"].str.replace("-M", "-"), format="%Y-%m")
                       + pd.offsets.MonthEnd(0), "value": pd.to_numeric(d["OBS_VALUE"], errors="coerce")})
    return df.dropna().sort_values("obs_date").reset_index(drop=True)


def parse_pink_sheet(xlsx, column: str = "Gold") -> pd.DataFrame:
    p = pd.read_excel(xlsx, sheet_name="Monthly Prices", header=None)
    hdr = next(i for i in range(15) if p.iloc[i].astype(str).str.strip().eq(column).any())
    col = int(p.iloc[hdr].astype(str).str.strip().eq(column).to_numpy().nonzero()[0][0])
    body = p.iloc[hdr + 2:, [0, col]].dropna()
    body = body[body.iloc[:, 0].astype(str).str.fullmatch(r"\d{4}M\d{2}")]
    df = pd.DataFrame({"obs_date": pd.to_datetime(body.iloc[:, 0].str.replace("M", "-"), format="%Y-%m") + pd.offsets.MonthEnd(0),
                       "value": pd.to_numeric(body.iloc[:, 1], errors="coerce")})
    return df.dropna().reset_index(drop=True)


def _write(df: pd.DataFrame, path: Path) -> None:
    out = pd.DataFrame({"Observation Date": df["obs_date"].dt.strftime("%Y-%m-%d"),
                        "Observation Value": df["value"].round(6), "Observation Status": "actual"})
    out.to_csv(path, index=False)


def fetch_all(cfg: Config, only: list[str] | None = None) -> dict[str, dict]:
    root = cfg.root
    drop = root / "data" / "drop"
    raw_dir = root / "data" / "downloads" / datetime.now(timezone.utc).strftime("%Y-%m-%d")
    raw_dir.mkdir(parents=True, exist_ok=True)
    repo_kse = root.parent.parent / "data" / "raw" / "KSE_100_index.csv"

    def kse():
        if repo_kse.exists():  # team FYP daily dataset (same file the backend technical agent uses)
            d = pd.read_csv(repo_kse)
            low = {c.lower().strip(): c for c in d.columns}
            dc = low.get("date") or d.columns[0]
            vc = next((low[c] for c in ("adj close", "close", "price") if c in low), d.columns[-1])
            df = pd.DataFrame({"obs_date": pd.to_datetime(d[dc], format="mixed", dayfirst=False),
                               "value": pd.to_numeric(d[vc].astype(str).str.replace(",", ""), errors="coerce")})
            return df.dropna().sort_values("obs_date"), str(repo_kse.relative_to(root.parent.parent)), "daily"
        return _gem_monthly("DSTKMKTXN", raw_dir), WB_GEM.format(ind="DSTKMKTXN"), "monthly"

    def pkr():
        r = http_get(BIS_XRU, timeout=90)
        (raw_dir / "bis_xru_pk.csv").write_text(r.text, encoding="utf-8")
        return parse_bis_xru(r.text), BIS_XRU, "monthly"

    def tbill():
        r = http_get(IMF_IR, timeout=90, headers={"Accept": "application/vnd.sdmx.data+csv;version=2.0.0"})
        (raw_dir / "imf_mfs_ir_tbill.csv").write_text(r.text, encoding="utf-8")
        return parse_imf_monthly(r.text), IMF_IR, "monthly"

    def gold():
        page = http_get(PINK_PAGE, timeout=60).text
        m = re.search(r"https://thedocs\.worldbank\.org/[^\"']*CMO-Historical-Data-Monthly\.xlsx", page)
        if not m:
            raise ValueError("Pink Sheet monthly xlsx link not found on the commodity-markets page")
        r = http_get(m.group(0), timeout=90)
        f = raw_dir / "CMO-Historical-Data-Monthly.xlsx"
        f.write_bytes(r.content)
        return parse_pink_sheet(f), m.group(0), "monthly"

    def brent():
        url = FRED.format(id="DCOILBRENTEU")
        r = http_get(url, timeout=60)
        (raw_dir / "fred_DCOILBRENTEU.csv").write_text(r.text, encoding="utf-8")
        d = pd.read_csv(io.StringIO(r.text))
        d.columns = ["obs_date", "value"]
        d = d[d["value"].astype(str) != "."]
        df = pd.DataFrame({"obs_date": pd.to_datetime(d["obs_date"]), "value": pd.to_numeric(d["value"], errors="coerce")})
        return df.dropna(), url, "daily"  # holidays are blank (or '.') in the FRED CSV

    def gdp():
        url = WB_WDI.format(ind="NY.GDP.MKTP.KD.ZG")
        r = http_get(url, timeout=60)
        (raw_dir / "wb_wdi_gdp_growth.json").write_text(r.text, encoding="utf-8")
        rows = [(f"{x['date']}-06-30", x["value"]) for x in r.json()[1] if x.get("value") is not None]
        df = pd.DataFrame(rows, columns=["obs_date", "value"])
        return df.assign(obs_date=pd.to_datetime(df["obs_date"])).sort_values("obs_date"), url, "annual"

    jobs = {
        "pkr_usd": pkr,
        "cpi_yoy": lambda: (_gem_monthly("CPTOTSAXNZGY", raw_dir), WB_GEM.format(ind="CPTOTSAXNZGY"), "monthly"),
        "fx_reserves": lambda: (_gem_monthly("TOTRESV", raw_dir), WB_GEM.format(ind="TOTRESV"), "monthly"),
        "kse100": kse,
        "gold": gold,
        "tbill_3m": tbill,
        "brent": brent,
        "gdp_growth": gdp,
    }
    report: dict[str, dict] = {}
    for key, job in jobs.items():
        if only and key not in only:
            continue
        try:
            df, url, freq = job()
            _write(df, drop / f"{key}.csv")
            report[key] = {"status": "ok", "rows": len(df), "first": str(df["obs_date"].min().date()),
                           "last": str(df["obs_date"].max().date()), "frequency": freq, "url": url}
        except Exception as e:  # noqa: BLE001 - one failed source must not stop the others
            log.warning("%s: %s", key, e)
            report[key] = {"status": f"failed: {type(e).__name__}: {e}"}

    # synthetic demo files for series with no open source would silently mix fake and real data
    moved = []
    bak = drop / "_synthetic"
    for key in NO_OPEN_SOURCE:
        f = drop / f"{key}.csv"
        if f.exists() and "synthetic" in f.read_text(encoding="utf-8", errors="ignore")[:2000]:
            bak.mkdir(exist_ok=True)
            shutil.move(str(f), bak / f.name)
            moved.append(key)
    for key in NO_OPEN_SOURCE:
        report.setdefault(key, {"status": "no open source: export from SBP EasyData into data/drop/"
                                + (" (synthetic file moved to data/drop/_synthetic/)" if key in moved else "")})
    side = drop / "sources.json"  # merged across runs so `--only` keeps the other entries
    merged = json.loads(side.read_text(encoding="utf-8")) if side.exists() else {}
    merged.update(report)
    side.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    _write_manifest(drop / "SOURCES.md", merged)
    (raw_dir / "fetch_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _write_manifest(path: Path, report: dict[str, dict]) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    keep = {"policy_rate": {"status": "ok (manual)", "frequency": "decision",
                            "url": "https://easydata.sbp.org.pk/ (SBPOL0030, exported manually)"}}
    lines = ["# Data sources (generated by `macro-agent fetch-open`)", "", f"Last fetch: {now}", "",
             "| series | status | frequency | coverage | source |", "|---|---|---|---|---|"]
    for k, r in {**keep, **report}.items():
        cov = f"{r['first']} to {r['last']} ({r['rows']} rows)" if "rows" in r else ""
        lines.append(f"| {k} | {r['status'].split(':')[0] if r['status'].startswith('ok') else r['status']} | "
                     f"{r.get('frequency', '')} | {cov} | {r.get('url', '')} |")
    lines += ["", "Attribution: World Bank (CC BY 4.0), BIS, IMF, FRED / U.S. EIA, State Bank of Pakistan.",
              "tbill_3m (IMF) ends 2021-12 and is only usable for history; export the current series from SBP EasyData."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
