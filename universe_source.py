#!/usr/bin/env python3
"""
Universe Source Layer
=====================
Assembles a *candidate* ticker universe from multiple feeds:
  1. Top holdings of major sector + thematic ETFs (yfinance funds_data)
  2. S&P 500 constituents (datahub CSV primary; Wikipedia fallback)
  3. Manual additions
  4. Manual exclusions

Then (via GICSClassifier) classifies each candidate by GICS sub-industry and
writes an inspectable snapshot to data/ + public/universe_candidates.json.

This is the FOUNDATION layer only — it does NOT touch INDUSTRY_GROUPS,
signal_engine.py, history_manager.py, the framework, or any UI. Next weekend
wires the candidate universe into signal_engine to replace INDUSTRY_GROUPS.

Run directly to build + print the universe:
    python universe_source.py
"""

import csv
import datetime
import json
import os
from io import StringIO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE_DIR = os.path.join(BASE_DIR, "data", "universe_cache")
CONFIG_PATH = os.path.join(BASE_DIR, "framework", "config.yaml")

_HEADERS = {"User-Agent": "Mozilla/5.0 SignalAgent/1.0 (universe-source)"}
SP500_CSV_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def _now_utc():
    return datetime.datetime.now(datetime.timezone.utc)


def _norm(sym):
    """Normalize a ticker to yfinance form.

    Upper/strip; convert US class-share dots (BRK.B -> BRK-B) to dashes, but
    keep foreign exchange suffixes intact (000660.KS, 2454.TW, IFX.DE) — those
    have multi-char suffixes and must not be dash-mangled.
    """
    if sym is None:
        return None
    s = str(sym).strip().upper()
    if "." in s:
        head, _, tail = s.rpartition(".")
        if head and len(tail) == 1 and tail.isalpha():
            s = head + "-" + tail
    return s or None


class UniverseSource:
    """Assemble a deduplicated candidate universe from multiple feeds."""

    def __init__(self, config: dict, cache_dir: str = None):
        """
        Args:
            config: the `universe` config subtree, i.e. config["universe"], with
                    keys `source` (etf_holdings/include_sp500/manual_*) and
                    optional `cache` (universe_ttl_days/gics_ttl_days).
            cache_dir: where to cache feed responses (default data/universe_cache/).
        """
        src = config["source"]
        self.etf_holdings_tickers = src["etf_holdings"]
        self.include_sp500 = src["include_sp500"]
        self.manual_additions = src.get("manual_additions", []) or []
        self.manual_exclusions = src.get("manual_exclusions", []) or []

        cache_cfg = config.get("cache", {}) or {}
        self.universe_ttl_days = cache_cfg.get("universe_ttl_days", 7)

        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)

        # Captured during the S&P 500 fetch: {ticker: GICS sub-industry}.
        # Lets the classifier reuse the free CSV classification (no .info calls).
        self.sp500_gics_map = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_candidate_universe(self):
        """Return the deduplicated, sorted list of candidate tickers."""
        return sorted(self.get_candidate_universe_detailed()["tickers"])

    def get_candidate_universe_detailed(self):
        """Like get_candidate_universe() but also returns the per-source breakdown."""
        etf_set = set()
        for etf in self.etf_holdings_tickers:
            etf_set.update(self._fetch_etf_holdings(etf, top_n=25))

        sp_set = set()
        if self.include_sp500:
            sp_set.update(self._fetch_sp500_constituents())

        add_set = {t for t in (_norm(x) for x in self.manual_additions) if t}
        excl_set = {t for t in (_norm(x) for x in self.manual_exclusions) if t}

        tickers = (etf_set | sp_set | add_set) - excl_set

        return {
            "tickers": tickers,
            "by_source": {
                "etf_holdings": len(etf_set - excl_set),
                "sp500": len(sp_set - excl_set),
                "manual_additions": len(add_set - excl_set),
                "overlap_count": len((etf_set & sp_set) - excl_set),
            },
            "sets": {"etf": etf_set, "sp500": sp_set,
                     "additions": add_set, "exclusions": excl_set},
        }

    # ------------------------------------------------------------------
    # Sources
    # ------------------------------------------------------------------
    def _fetch_etf_holdings(self, etf_ticker, top_n=25):
        """Top-N holdings of an ETF via yfinance funds_data (Yahoo caps at ~10).

        Cached for `universe_ttl_days`. Falls back to stale cache (then []) on
        any failure. A legitimately-empty result (e.g. GLD/IBIT) is cached as [].
        """
        cache_path = os.path.join(self.cache_dir, f"etf_{_norm(etf_ticker)}.json")
        cached = self._read_cache(cache_path, self.universe_ttl_days)
        if cached is not None:
            return cached
        try:
            import yfinance as yf
            th = yf.Ticker(etf_ticker).funds_data.top_holdings
            tickers = ([t for t in (_norm(s) for s in list(th.index)[:top_n]) if t]
                       if th is not None else [])
            self._write_cache(cache_path, tickers)   # cache only on a successful fetch
            return tickers
        except Exception as e:
            print(f"[universe] ETF holdings fetch failed for {etf_ticker}: {e}")
            stale = self._read_cache(cache_path, ttl_days=None)
            return stale if stale is not None else []

    def _fetch_sp500_constituents(self):
        """Current S&P 500 list (and its GICS sub-industry map).

        Cached for `universe_ttl_days`. Primary source is the datahub CSV (which
        carries GICS Sub-Industry, parseable with stdlib csv); Wikipedia is the
        documented fallback (needs lxml). Falls back to stale cache on failure.
        """
        cache_path = os.path.join(self.cache_dir, "sp500.json")
        cached = self._read_cache_obj(cache_path, self.universe_ttl_days)
        if cached is not None:
            self.sp500_gics_map = cached.get("gics", {})
            return list(cached.get("tickers", []))

        tickers, gics = self._fetch_sp500_from_csv()
        if not tickers:
            tickers, gics = self._fetch_sp500_from_wiki()
        if not tickers:
            stale = self._read_cache_obj(cache_path, ttl_days=None)
            if stale is not None:
                self.sp500_gics_map = stale.get("gics", {})
                return list(stale.get("tickers", []))
            return []

        self.sp500_gics_map = gics
        self._write_cache_obj(cache_path, {"tickers": tickers, "gics": gics})
        return tickers

    def _fetch_sp500_from_csv(self):
        """Primary S&P 500 source: datahub constituents.csv (Symbol + GICS Sub-Industry)."""
        try:
            import requests
            r = requests.get(SP500_CSV_URL, headers=_HEADERS, timeout=30)
            r.raise_for_status()
            tickers, gics = [], {}
            for row in csv.DictReader(StringIO(r.text)):
                t = _norm(row.get("Symbol"))
                if not t:
                    continue
                tickers.append(t)
                sub = (row.get("GICS Sub-Industry") or "").strip()
                if sub:
                    gics[t] = sub
            return tickers, gics
        except Exception as e:
            print(f"[universe] S&P500 CSV fetch failed: {e}")
            return [], {}

    def _fetch_sp500_from_wiki(self):
        """Fallback S&P 500 source: Wikipedia (requires lxml/bs4 for read_html)."""
        try:
            import requests
            import pandas as pd
            r = requests.get(SP500_WIKI_URL, headers=_HEADERS, timeout=30)
            r.raise_for_status()
            df = pd.read_html(StringIO(r.text))[0]
            tickers, gics = [], {}
            for _, row in df.iterrows():
                t = _norm(row.get("Symbol"))
                if not t:
                    continue
                tickers.append(t)
                sub = str(row.get("GICS Sub-Industry") or "").strip()
                if sub and sub.lower() != "nan":
                    gics[t] = sub
            return tickers, gics
        except Exception as e:
            print(f"[universe] S&P500 Wikipedia fetch failed: {e}")
            return [], {}

    # ------------------------------------------------------------------
    # Cache helpers (TTL in days; ttl_days=None means "ignore age")
    # ------------------------------------------------------------------
    def _read_cache(self, path, ttl_days):
        obj = self._read_cache_obj(path, ttl_days)
        return None if obj is None else obj.get("tickers")

    def _read_cache_obj(self, path, ttl_days):
        if not os.path.exists(path):
            return None
        try:
            with open(path) as f:
                obj = json.load(f)
            if ttl_days is not None:
                fetched = datetime.datetime.fromisoformat(obj["fetched_at"])
                if (_now_utc() - fetched).total_seconds() > ttl_days * 86400:
                    return None
            return obj
        except Exception:
            return None

    def _write_cache(self, path, tickers):
        self._write_cache_obj(path, {"tickers": list(tickers)})

    def _write_cache_obj(self, path, obj):
        obj = dict(obj)
        obj["fetched_at"] = _now_utc().isoformat()
        try:
            with open(path, "w") as f:
                json.dump(obj, f, indent=2)
        except Exception as e:
            print(f"[universe] cache write failed {path}: {e}")


# ----------------------------------------------------------------------
# Build orchestrator (assemble + classify + write inspectable snapshot)
# ----------------------------------------------------------------------
def load_universe_config(config_path=None):
    import yaml
    with open(config_path or CONFIG_PATH) as f:
        full = yaml.safe_load(f) or {}
    return full.get("universe", {})


def build_universe_candidates(config_path=None, write=True, allow_remote=True):
    """Assemble the candidate universe, classify by GICS, and (optionally) write
    data/public universe_candidates.json. Returns (response_dict, tickers, by_gics)."""
    from gics_classifier import GICSClassifier

    uni_cfg = load_universe_config(config_path)
    if not uni_cfg:
        raise RuntimeError("config.yaml has no 'universe' section")

    src = UniverseSource(uni_cfg)
    detailed = src.get_candidate_universe_detailed()
    tickers = sorted(detailed["tickers"])

    cache_cfg = uni_cfg.get("cache", {}) or {}
    gics = GICSClassifier(src.cache_dir, ttl_days=cache_cfg.get("gics_ttl_days", 30))
    # Seed authoritative GICS from the S&P 500 CSV (free; avoids .info calls).
    gics.seed(src.sp500_gics_map, source="sp500_csv")
    by_gics = gics.get_universe_by_gics(tickers, allow_remote=allow_remote)

    unclassified = by_gics.pop("_unclassified", [])
    by_counts = {k: len(v) for k, v in sorted(by_gics.items(),
                                              key=lambda kv: (-len(kv[1]), kv[0]))}

    out = {
        "generated_at": _now_utc().isoformat(),
        "total_count": len(tickers),
        "by_source": detailed["by_source"],
        "by_gics_sub_industry": by_counts,
        "unclassified": len(unclassified),
        "unclassified_tickers": sorted(unclassified),
    }

    if write:
        for d in (os.path.join(BASE_DIR, "data"), os.path.join(BASE_DIR, "public")):
            try:
                os.makedirs(d, exist_ok=True)
                with open(os.path.join(d, "universe_candidates.json"), "w") as f:
                    json.dump(out, f, indent=2)
            except Exception as e:
                print(f"[universe] write failed for {d}: {e}")
    return out, tickers, by_gics


if __name__ == "__main__":
    out, tickers, by_gics = build_universe_candidates()
    print(json.dumps({k: out[k] for k in
                      ("generated_at", "total_count", "by_source", "unclassified")},
                     indent=2))
    print(f"\nby_gics_sub_industry ({len(out['by_gics_sub_industry'])} sub-industries):")
    for name, count in out["by_gics_sub_industry"].items():
        print(f"  {count:>4}  {name}")
    if out["unclassified"]:
        print(f"\nunclassified ({out['unclassified']}): {', '.join(out['unclassified_tickers'][:40])}"
              + (" ..." if out["unclassified"] > 40 else ""))
