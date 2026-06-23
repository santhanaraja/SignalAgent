#!/usr/bin/env python3
"""
GICS Classifier
===============
Resolves a GICS sub-industry string for a ticker, with a 30-day on-disk cache.

Resolution order (per spec — INDUSTRY_GROUPS first to avoid re-classifying
tickers we already know, then cache, then a live yfinance lookup):
  1. signal_engine.INDUSTRY_GROUPS base map  (read-only reuse; authoritative
     custom group names, which are themselves GICS sub-industry labels)
  2. on-disk cache (also seeded for free from the S&P 500 CSV's GICS column)
  3. yfinance Ticker(t).info["industryDisp"]  (Yahoo industry ~ GICS sub-industry)

Does NOT modify INDUSTRY_GROUPS or signal_engine — it only imports the
constant to read it.
"""

import datetime
import json
import os
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _now_utc():
    return datetime.datetime.now(datetime.timezone.utc)


def _norm(sym):
    """Upper/strip; US class-share dots (BRK.B -> BRK-B) to dashes; keep foreign
    exchange suffixes intact (000660.KS, 2454.TW, IFX.DE)."""
    if sym is None:
        return None
    s = str(sym).strip().upper()
    if "." in s:
        head, _, tail = s.rpartition(".")
        if head and len(tail) == 1 and tail.isalpha():
            s = head + "-" + tail
    return s or None


def _build_industry_groups_map():
    """ticker -> custom group name from signal_engine.INDUSTRY_GROUPS (read-only)."""
    mapping = {}
    try:
        from signal_engine import INDUSTRY_GROUPS
        for group_name, info in INDUSTRY_GROUPS.items():
            for t in info.get("tickers", []):
                key = _norm(t)
                if key:
                    mapping[key] = group_name
    except Exception as e:
        print(f"[gics] could not import INDUSTRY_GROUPS base map: {e}")
    return mapping


class GICSClassifier:
    """Classify tickers by GICS sub-industry with caching + INDUSTRY_GROUPS reuse."""

    GICS_TTL_DAYS = 30

    def __init__(self, cache_dir, ttl_days=None):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_path = os.path.join(self.cache_dir, "gics_cache.json")
        self.ttl_days = ttl_days or self.GICS_TTL_DAYS
        self.cache = self._load_cache()
        self.industry_groups_map = _build_industry_groups_map()
        self._dirty = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def classify(self, ticker, allow_remote=True):
        """Return the GICS sub-industry for `ticker` (None if unresolved)."""
        t = _norm(ticker)
        if not t:
            return None

        # 1. INDUSTRY_GROUPS base mapping (already-classified tickers)
        if t in self.industry_groups_map:
            return self.industry_groups_map[t]

        # 2. cache (incl. S&P 500 seed)
        cached = self._cache_get(t)
        if cached is not None:
            return cached

        # 3. live yfinance lookup
        if allow_remote:
            sub = self._fetch_industry(t)
            if sub:
                self._cache_put(t, sub, "yfinance")
                return sub
        return None

    def classify_batch(self, tickers, allow_remote=True, sleep=0.3):
        """Classify many tickers; rate-limits ONLY the live yfinance calls."""
        out = {}
        for t in tickers:
            tn = _norm(t)
            if tn is None:
                continue
            # Will this need a live call? (used purely for rate-limiting)
            needs_remote = (tn not in self.industry_groups_map
                            and self._cache_get(tn) is None)
            out[tn] = self.classify(tn, allow_remote=allow_remote)
            if needs_remote and allow_remote and sleep:
                time.sleep(sleep)
        if self._dirty:
            self._save_cache()
        return out

    def get_universe_by_gics(self, tickers, allow_remote=True):
        """Return {gics_sub_industry: [tickers]}; unresolved go to '_unclassified'."""
        result = {}
        for t, sub in self.classify_batch(tickers, allow_remote=allow_remote).items():
            result.setdefault(sub or "_unclassified", []).append(t)
        return {k: sorted(v) for k, v in result.items()}

    def seed(self, mapping, source="seed"):
        """Pre-populate cache from a {ticker: sub_industry} map (e.g. S&P 500 GICS).

        Only fills gaps (never overwrites a fresh cached/known value), and skips
        tickers already covered by the INDUSTRY_GROUPS base map — so that map and
        the existing cache stay the single authoritative source per ticker (no
        orphaned/conflicting cache entries).
        """
        for t, sub in (mapping or {}).items():
            tn = _norm(t)
            if (tn and sub and tn not in self.industry_groups_map
                    and self._cache_get(tn) is None):
                self._cache_put(tn, sub, source)
        if self._dirty:
            self._save_cache()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _fetch_industry(self, ticker):
        try:
            import yfinance as yf
            info = yf.Ticker(ticker).info or {}
            sub = (info.get("industryDisp") or info.get("industry") or "").strip()
            return sub or None
        except Exception as e:
            print(f"[gics] yfinance classify failed for {ticker}: {e}")
            return None

    def _load_cache(self):
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path) as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_cache(self):
        try:
            with open(self.cache_path, "w") as f:
                json.dump(self.cache, f, indent=2)
            self._dirty = False
        except Exception as e:
            print(f"[gics] cache save failed: {e}")

    def _cache_get(self, ticker):
        entry = self.cache.get(ticker)
        if not entry:
            return None
        try:
            fetched = datetime.datetime.fromisoformat(entry["fetched_at"])
            if self.ttl_days is not None and (_now_utc() - fetched).total_seconds() > self.ttl_days * 86400:
                return None
        except Exception:
            return None
        return entry.get("sub_industry")

    def _cache_put(self, ticker, sub_industry, source):
        self.cache[ticker] = {
            "sub_industry": sub_industry,
            "source": source,
            "fetched_at": _now_utc().isoformat(),
        }
        self._dirty = True


if __name__ == "__main__":
    import sys
    clf = GICSClassifier(os.path.join(BASE_DIR, "data", "universe_cache"))
    for tk in (sys.argv[1:] or ["LRCX", "NVDA", "COIN", "AAPL"]):
        print(f"{tk:8} -> {clf.classify(tk)!r}")
