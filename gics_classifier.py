#!/usr/bin/env python3
"""
GICS Classifier
===============
Resolves a GICS sub-industry string for a ticker, with a 30-day on-disk cache.

Resolution order (static base map first to avoid re-classifying tickers we
already know, then cache, then a live yfinance lookup):
  1. signal_engine.FALLBACK_INDUSTRY_GROUPS base map (read-only reuse of the
     STATIC legacy groups — deliberately not the dynamic universe, which is
     itself derived from this classifier and would be circular)
  2. on-disk cache (also seeded for free from the S&P 500 CSV's GICS column)
  3. yfinance Ticker(t).info["industryDisp"]  (Yahoo industry ~ GICS sub-industry)

All three paths are canonicalized through the config alias map
(universe.gics_aliases) at read time; the cache stores raw labels.

Does NOT modify FALLBACK_INDUSTRY_GROUPS or signal_engine — it only imports
the constant to read it.

A label the alias map does not cover is not an inert cosmetic difference:
universe_builder GROUPS BY THIS STRING, so an uncanonicalised vendor label
becomes its own group, sits below rotation.min_candidates, and can never be
eligible at any composite. `unaliased_labels()` below is how that condition
is detected rather than discovered months later (D-023).
"""

import datetime
import json
import os
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# The canonical GICS sub-industry vocabulary — committed, read, never fetched
# (same doctrine as data/pool/nasdaq100.json). See that file's own _doc for
# what it is, where it comes from, and why it is deliberately incomplete.
VOCABULARY_PATH = os.path.join(BASE_DIR, "data", "pool", "gics_vocabulary.json")


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
    """ticker -> group name from signal_engine.FALLBACK_INDUSTRY_GROUPS
    (read-only; the static legacy constant, NOT the dynamic universe —
    classifications must not depend on the groups derived from them)."""
    mapping = {}
    try:
        from signal_engine import FALLBACK_INDUSTRY_GROUPS
        for group_name, info in FALLBACK_INDUSTRY_GROUPS.items():
            for t in info.get("tickers", []):
                key = _norm(t)
                if key:
                    mapping[key] = group_name
    except Exception as e:
        print(f"[gics] could not import FALLBACK_INDUSTRY_GROUPS base map: {e}")
    return mapping


def load_vocabulary(path=None):
    """Canonical GICS sub-industry names from the committed vocabulary file.

    Raises rather than degrading to an empty set. An empty vocabulary makes
    every group name look non-canonical, and a caller that reads "empty" as
    "no opinion" makes every group name look fine — a missing input reading
    as a clean result is the exact failure this whole check exists to end.
    """
    with open(path or VOCABULARY_PATH) as f:
        doc = json.load(f)
    names = {n for n in (doc.get("sub_industries") or []) if n}
    if not names:
        raise ValueError(
            f"{path or VOCABULARY_PATH} carries no sub_industries — the "
            "canonical vocabulary is committed and has no fetch fallback; "
            "refusing to report every group name as canonical by default.")
    return names


def unaliased_labels(group_names, aliases=None, unmappable=None, vocabulary=None):
    """Group names that are a vendor's own label, not a GICS sub-industry.

    Returns {"new": [...], "known": [...]}:
      · **new** — flagged and UNDECIDED. Each one is a group that exists only
        because a label was not canonicalised, holding names that can never
        be eligible and thinning the canonical group where their peers sit.
      · **known** — labels config records as deliberately unmapped, with a
        reason (universe.gics_unmappable_labels). A vendor label whose GICS
        image is many-to-one CANNOT be repaired by a label→label alias; any
        target it is given misfiles some other member. Listing them keeps a
        decided problem from drowning the signal for an undecided one.

    Canonical means: in the committed vocabulary, or the TARGET of a
    configured alias (an alias target is a name a human reviewed). That
    second clause is self-certifying by construction — an alias pointing at
    a label nobody verified would make that label canonical here — so the
    pin, not this function, is what checks targets against the vocabulary
    (test_gics_aliases.test_alias_and_unmappable_maps_are_coherent).
    """
    vocab = set(vocabulary) if vocabulary is not None else load_vocabulary()
    vocab |= set((aliases or {}).values())
    known_set = set(unmappable or ())
    flagged = [n for n in sorted(set(group_names)) if n and n not in vocab]
    return {"new": [n for n in flagged if n not in known_set],
            "known": [n for n in flagged if n in known_set]}


class GICSClassifier:
    """Classify tickers by GICS sub-industry with caching + static base-map reuse."""

    GICS_TTL_DAYS = 30

    def __init__(self, cache_dir, ttl_days=None, aliases=None):
        """
        Args:
            aliases: optional {label: canonical_gics_sub_industry} map
                     (config `universe.gics_aliases`). Applied as a read-time
                     view on every resolution path — the on-disk cache keeps
                     raw values so the map stays reversible/extensible.
        """
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_path = os.path.join(self.cache_dir, "gics_cache.json")
        self.ttl_days = ttl_days or self.GICS_TTL_DAYS
        self.cache = self._load_cache()
        self.industry_groups_map = _build_industry_groups_map()
        self.aliases = dict(aliases or {})
        self._dirty = False

    def canon(self, sub_industry):
        """Canonical GICS name for a raw classification label."""
        if sub_industry is None:
            return None
        return self.aliases.get(sub_industry, sub_industry)

    def unaliased_group_names(self, group_names, unmappable=None):
        """`unaliased_labels` against THIS classifier's alias map.

        Never raises. A vocabulary that cannot be read is reported as
        `error` — not as an empty `new` list. The whole point of the check
        is that a missing input must not read as a clean result, and that
        applies to the check's own input first.
        """
        try:
            out = unaliased_labels(group_names, aliases=self.aliases,
                                   unmappable=unmappable)
            out["error"] = None
            return out
        except Exception as e:
            print(f"[gics] canonical vocabulary unreadable ({e}) — group names "
                  f"NOT checked this run")
            return {"new": [], "known": [], "error": str(e)}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def classify(self, ticker, allow_remote=True):
        """Return the canonical GICS sub-industry for `ticker` (None if unresolved)."""
        t = _norm(ticker)
        if not t:
            return None

        # 1. FALLBACK_INDUSTRY_GROUPS base mapping (already-classified tickers)
        if t in self.industry_groups_map:
            return self.canon(self.industry_groups_map[t])

        # 2. cache (incl. S&P 500 seed)
        cached = self._cache_get(t)
        if cached is not None:
            return self.canon(cached)

        # 3. live yfinance lookup
        if allow_remote:
            sub = self._fetch_industry(t)
            if sub:
                self._cache_put(t, sub, "yfinance")
                return self.canon(sub)
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
        tickers already covered by the FALLBACK_INDUSTRY_GROUPS base map — so that
        map and the existing cache stay the single authoritative source per ticker
        (no orphaned/conflicting cache entries).
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
