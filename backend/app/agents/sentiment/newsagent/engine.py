"""Point-in-time scoring engine (methodology 4.2 - 4.5).

For a decision date t it only uses items whose effective_at is before the
decision time (default 11:00 UTC, after the PSX close). The same code serves
the nightly run, the on-demand request, the back-test and parameter tuning."""
import math
from bisect import bisect_left, bisect_right
from collections import defaultdict
from datetime import datetime, time, timedelta, timezone

import numpy as np

from app.agents.sentiment.newsagent import config
from app.agents.sentiment.newsagent.cleaning import credibility
from app.agents.sentiment.newsagent.config import DEFAULT_PARAMS
from app.agents.sentiment.newsagent.entities import load_companies, sector_of

NaN = float("nan")


def _isnan(x):
    return x is None or (isinstance(x, float) and math.isnan(x))


def cutoff_for(d, params):
    return datetime.combine(d, time(params.decision_hour_utc), tzinfo=timezone.utc)


class Engine:
    def __init__(self, conn, params=DEFAULT_PARAMS):
        self.params = params
        self.rows = defaultdict(list)     # (etype, eid) -> rows sorted by time
        self.times = {}
        self.scorers = set()
        self._load(conn)

    # ------------------------------------------------------------------ load
    def _load(self, conn):
        cur = conn.execute(
            """
            SELECT a.id, a.effective_at, a.cluster_id, a.headline, a.outlet, a.url, a.doc_url,
                   a.kind, a.published_at, e.entity_type, e.entity_id, e.relevance,
                   s.tone_adj, s.p_neg, s.event_tags, s.scorer
            FROM articles a
            JOIN article_entities e ON e.article_id = a.id
            JOIN article_scores s ON s.article_id = a.id
            ORDER BY a.effective_at, a.id
            """)
        seen_cluster = set()
        for r in cur:
            ts = datetime.fromisoformat(r["effective_at"])
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            key = (r["entity_type"], r["entity_id"])
            cid = r["cluster_id"] if r["cluster_id"] is not None else r["id"]
            # The earliest member of a cluster (per entity) is the primary item.
            primary = (key, cid) not in seen_cluster
            seen_cluster.add((key, cid))
            cred, _ = credibility(r["outlet"])
            self.scorers.add(r["scorer"])
            self.rows[key].append({
                "id": r["id"], "ts": ts, "primary": primary, "cred": cred, "rel": r["relevance"],
                "tone": r["tone_adj"], "p_neg": r["p_neg"], "title": r["headline"],
                "outlet": r["outlet"], "url": r["url"], "doc_url": r["doc_url"], "kind": r["kind"],
                "published_at": r["published_at"], "tags": r["event_tags"] or "",
            })
        for key, lst in self.rows.items():
            self.times[key] = [x["ts"] for x in lst]

    # -------------------------------------------------------------- features
    def _window(self, key, cutoff):
        times = self.times.get(key)
        if not times:
            return []
        hi = bisect_right(times, cutoff)
        lo = bisect_left(times, cutoff - timedelta(days=self.params.window_days))
        return [r for r in self.rows[key][lo:hi] if r["primary"]]

    def features(self, key, cutoff):
        p = self.params
        window = self._window(key, cutoff)
        if not window:
            return None
        ln2 = math.log(2)
        sw = swt = nw = nd = 0.0
        bull = bear = 0.0
        for r in window:
            age = max(0.0, (cutoff - r["ts"]).total_seconds() / 86400.0)
            w = r["cred"] * r["rel"] * math.exp(-ln2 * age / p.half_life_days)
            if r["kind"] == "announcement":
                w *= p.announcement_tone_weight
            is_neg = r["p_neg"] is not None and r["p_neg"] > p.neg_threshold
            m = p.neg_multiplier if is_neg else 1.0
            sw += w
            swt += w * r["tone"]
            nw += w * m * (1.0 if is_neg else 0.0)
            nd += w * m
            if r["tone"] > p.direction_band:
                bull += w
            elif r["tone"] < -p.direction_band:
                bear += w
        if sw <= 0:
            return None
        if bull + bear > 0:
            ratio = (bull - bear) / (bull + bear)
            agreement = 1.0 - math.sqrt(max(0.0, 1.0 - ratio * ratio))   # Antweiler-Frank
        else:
            agreement = 1.0
        return {
            "n_articles": len(window),
            "n_eff": float(sum(r["rel"] for r in window)),
            "net_tone": swt / sw,
            "neg_share": nw / nd if nd > 0 else 0.0,
            "agreement": agreement,
            "attention": float(len(window)),
            "last_article_at": max(r["ts"] for r in window),
        }

    # ---------------------------------------------------------------- series
    def _z(self, hist, v):
        if _isnan(v) or len(hist) < self.params.min_z_obs:
            return NaN
        arr = np.asarray(hist, dtype=float)
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med)))
        scale = 1.4826 * mad
        if scale <= 0:
            scale = 1.2533 * float(np.mean(np.abs(arr - med)))   # MAD-equivalent fallback
        if scale <= 0:
            return 0.0
        z = (v - med) / scale
        c = self.params.z_clip
        return max(-c, min(c, z))

    def entity_series(self, key, dates, eval_dates=None):
        """Per-date feature rows for one entity. `dates` must be ascending calendar days."""
        p = self.params
        eval_set = set(eval_dates) if eval_dates is not None else set(dates)
        hist = {"net_tone": [], "tone_change": [], "neg_share": [], "attention": []}
        past = []   # (date, net_tone)
        out = {}
        for d in dates:
            f = self.features(key, cutoff_for(d, p))
            row = None
            if f is not None:
                base = [t for (pd_, t) in past if (d - pd_).days <= p.baseline_days]
                tc = f["net_tone"] - float(np.mean(base)) if len(base) >= p.min_baseline_obs else NaN
                row = {**f, "tone_change": tc}
                past.append((d, f["net_tone"]))
                if d in eval_set:
                    row["z_net_tone"] = self._z(hist["net_tone"] + [f["net_tone"]], f["net_tone"])
                    row["z_tone_change"] = self._z(hist["tone_change"] + ([tc] if not _isnan(tc) else []), tc)
                    row["z_neg_share"] = self._z(hist["neg_share"] + [f["neg_share"]], f["neg_share"])
                    row["z_attention"] = self._z(hist["attention"] + [f["attention"]], f["attention"])
                    subs = []
                    if not _isnan(row["z_net_tone"]):
                        subs.append(row["z_net_tone"] / p.z_clip)
                    if not _isnan(row["z_tone_change"]):
                        subs.append(row["z_tone_change"] / p.z_clip)
                    if not _isnan(row["z_neg_share"]):
                        subs.append(-row["z_neg_share"] / p.z_clip)
                    row["n_sub"] = len(subs)
                    row["raw_score"] = max(-1.0, min(1.0, sum(subs) / len(subs))) if subs else None
                hist["net_tone"].append(f["net_tone"])
                if not _isnan(tc):
                    hist["tone_change"].append(tc)
                hist["neg_share"].append(f["neg_share"])
                hist["attention"].append(f["attention"])
            out[d] = row
        return out

    def _first_date(self, key):
        times = self.times.get(key)
        return times[0].date() if times else None

    # -------------------------------------------------------------- finalise
    def _freshness(self, last, cutoff):
        if last is None:
            return 0.0
        days = (cutoff - last).total_seconds() / 86400.0
        return 1.0 if days <= 2 else 0.7 if days <= 7 else 0.4 if days <= 14 else 0.1

    def finalize(self, ticker, d, srow, sector_row, market_row):
        p = self.params
        cutoff = cutoff_for(d, p)
        k = p.shrink_k
        s_stock = srow.get("raw_score") if srow else None
        n = srow["n_eff"] if srow else 0.0
        flags = []
        parent_name, parent_val = None, None
        if sector_row and sector_row.get("raw_score") is not None:
            parent_name, parent_val = "sector", sector_row["raw_score"]
        elif market_row and market_row.get("raw_score") is not None:
            parent_name, parent_val = "market", market_row["raw_score"]

        if s_stock is not None and n >= k:
            score = s_stock
        elif not srow and parent_val is None:
            score = 0.0                                   # nothing at any level (methodology 4.5.3)
            flags.append("no_recent_news")
        else:
            w = n / (n + k) if (s_stock is not None and n > 0) else 0.0
            pv = parent_val if parent_val is not None else 0.0
            score = w * (s_stock if s_stock is not None else 0.0) + (1 - w) * pv
            flags.append(f"shrunk_to_{parent_name or 'neutral'}")
            if not srow:
                flags.append("no_recent_news")
            else:
                if s_stock is None:
                    flags.append("insufficient_history")  # fewer than min_z_obs days of baseline
                if n < k:
                    flags.append("low_coverage")
        score = max(-1.0, min(1.0, score))

        if srow:
            coverage = min(1.0, n / (2 * k))
            fresh = self._freshness(srow["last_article_at"], cutoff)
            complete = srow.get("n_sub", 0) / 3.0
            q = coverage * fresh * complete * (1.0 - 0.5 * (1.0 - srow["agreement"]))
        else:
            q = 0.0
        q = max(0.0, min(1.0, q))
        t = p.state_threshold
        state = "Bullish" if score > t else "Bearish" if score < -t else "Neutral"
        return {"score": score, "quality": q, "state": state, "flags": flags,
                "parent": parent_name, "parent_score": parent_val}

    # ----------------------------------------------------------- public APIs
    def _dates(self, key, end):
        first = self._first_date(key)
        if first is None or first > end:
            return []
        n = (end - first).days + 1
        return [first + timedelta(days=i) for i in range(n)]

    def score_table(self, start, end, tickers=None):
        """Final scores for each ticker and every calendar date in [start, end]."""
        companies = [c for c in load_companies() if tickers is None or c["ticker"] in tickers]
        eval_dates = [start + timedelta(days=i) for i in range((end - start).days + 1)]
        cache = {}

        def series(key):
            if key not in cache:
                cache[key] = self.entity_series(key, self._dates(key, end), eval_dates)
            return cache[key]

        rows = []
        for c in companies:
            t = c["ticker"]
            ser = series(("ticker", t))
            sec = series(("sector", c["sector"])) if c.get("sector") else {}
            mkt = series(("market", "PSX"))
            for d in eval_dates:
                srow = ser.get(d)
                fin = self.finalize(t, d, srow, sec.get(d), mkt.get(d))
                rows.append({"ticker": t, "as_of": d.isoformat(), **fin,
                             "n_eff": srow["n_eff"] if srow else 0.0,
                             "net_tone": srow["net_tone"] if srow else None})
        return rows

    def output(self, ticker, as_of):
        """Full agent output contract for one ticker and date."""
        p = self.params
        sector = sector_of(ticker)
        key = ("ticker", ticker)
        cache = {}

        def one(k):
            if k not in cache:
                cache[k] = self.entity_series(k, self._dates(k, as_of), [as_of]).get(as_of)
            return cache[k]

        srow = one(key)
        sec = one(("sector", sector)) if sector else None
        mkt = one(("market", "PSX"))
        fin = self.finalize(ticker, as_of, srow, sec, mkt)
        cutoff = cutoff_for(as_of, p)

        drivers = []
        if srow and srow.get("n_sub", 0):
            for name, rawk, zk, sign in (("net_tone", "net_tone", "z_net_tone", 1),
                                         ("tone_change", "tone_change", "z_tone_change", 1),
                                         ("negative_share", "neg_share", "z_neg_share", -1)):
                z = srow.get(zk)
                if _isnan(z):
                    continue
                contrib = sign * z / p.z_clip
                drivers.append({"name": name, "value": round(float(srow[rawk]), 4),
                                "z": round(float(z), 3), "contribution": round(contrib, 3),
                                "direction": "bullish" if contrib > 0.02 else "bearish" if contrib < -0.02 else "neutral"})
            drivers.sort(key=lambda x: -abs(x["contribution"]))
            drivers = drivers[:3]

        evidence = []
        window = self._window(key, cutoff)
        ln2 = math.log(2)
        ranked = sorted(window, key=lambda r: -(r["cred"] * r["rel"] * abs(r["tone"]) *
                        math.exp(-ln2 * max(0.0, (cutoff - r["ts"]).total_seconds() / 86400.0) / p.half_life_days)))
        for r in ranked[:5]:
            evidence.append({"article_id": r["id"], "title": r["title"], "outlet": r["outlet"],
                             "url": r["doc_url"] or r["url"], "published_at": r["published_at"],
                             "tone": round(float(r["tone"]), 3),
                             "event_tags": [t for t in r["tags"].split(",") if t], "kind": r["kind"]})
        outlets = sorted({r["outlet"] for r in window if r["outlet"]})
        return {
            "agent": config.AGENT_NAME,
            "ticker": ticker,
            "as_of": as_of.isoformat(),
            "horizon_days": p.horizon_days,
            "score": round(fin["score"], 4),
            "state": fin["state"],
            "quality": round(fin["quality"], 4),
            "drivers": drivers,
            "evidence": evidence,
            "context": {
                "articles_in_window": srow["n_articles"] if srow else 0,
                "effective_articles": round(srow["n_eff"], 2) if srow else 0.0,
                "window_days": p.window_days,
                "outlets": outlets,
                "agreement": round(srow["agreement"], 3) if srow else None,
                "attention_z": None if (not srow or _isnan(srow.get("z_attention"))) else round(float(srow["z_attention"]), 3),
                "parent": fin["parent"],
                "parent_score": None if fin["parent_score"] is None else round(fin["parent_score"], 4),
            },
            "flags": fin["flags"],
            "scorer": sorted(s for s in self.scorers if s),
            "model_version": config.MODEL_VERSION,
            "params_key": p.key(),
            "params": p.to_dict(),
        }
