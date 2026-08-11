#!/usr/bin/env python3
"""
Universe Source Layer
=====================
Assembles a *candidate* ticker universe from multiple feeds:

    pool = ETF top-holdings  ∪  S&P 500  ∪  Nasdaq-100  ∪  inclusions
           −  exclusions

  1. Top holdings of major sector + thematic ETFs (yfinance funds_data)
  2. S&P 500 constituents (datahub CSV primary; Wikipedia fallback)
  3. Nasdaq-100 constituents — READ FROM A COMMITTED FILE, never fetched
  4. Inclusions — committed, hand-maintained, for listings the indices lag
  5. Exclusions — applied LAST, to the union, so exclusion always wins

Then (via GICSClassifier) classifies each candidate by GICS sub-industry and
writes an inspectable snapshot to data/ + public/universe_candidates.json.

This is the candidate SOURCE layer. universe_builder.py consumes it weekly
(Friday-evening rotation) to build the active universe that signal_engine
serves via get_industry_groups(); this module touches nothing downstream.

WHY NASDAQ-100 IS A FILE AND NOT A FETCH
----------------------------------------
Index membership changes a handful of times a year, on announced dates. A
live scrape would hand a third party the ability to change what the system
can trade, silently, between two rotations — and would do it through a
parser that has no way to tell a reconstitution from an outage. The list is
committed with a dated provenance header and moved deliberately by
scripts/refresh_nasdaq100.py, whose diff a human reads before committing.
The S&P 500 feed predates this rule and is still fetched (with a TTL cache
and a stale-cache fallback); the Nasdaq-100 path must not copy it.

POOL VERSIONING
---------------
POOL_VERSION names the pool DEFINITION; pool_definition_hash() fingerprints
its actual contents. Both are stamped into every pool-derived artifact, so a
stored ranking can always be read against the pool that produced it — and a
list edited without a version bump still moves the hash.

Run directly to build + print the universe:
    python universe_source.py
"""

import csv
import datetime
import hashlib
import json
import os
from io import StringIO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE_DIR = os.path.join(BASE_DIR, "data", "universe_cache")
CONFIG_PATH = os.path.join(BASE_DIR, "framework", "config.yaml")

# Committed pool inputs — read, never fetched.
POOL_DIR = os.path.join(BASE_DIR, "data", "pool")
NASDAQ100_PATH = os.path.join(POOL_DIR, "nasdaq100.json")
INCLUSIONS_PATH = os.path.join(POOL_DIR, "inclusions.json")

# The pool DEFINITION version. Bump when the set of sources, the membership
# of a committed list, or the exclusion/inclusion lists change.
#   v1  (through 2026-08-09)  ETF holdings ∪ S&P 500 ∪ manual_additions
#                             − manual_exclusions
#   v2  (2026-08-10)          + Nasdaq-100 (committed list), + inclusions
#                             list, exclusion precedence made explicit
#   v3  (2026-08-11)          first inclusion-list entry: ARWR. The pool's
#                             MEMBERSHIP changed by hand, which is exactly
#                             the case this counter exists to mark — the
#                             hash moves regardless, but a reader scanning
#                             versions should see that a human added a name.
POOL_VERSION = "v3-2026-08-11"

_HEADERS = {"User-Agent": "Mozilla/5.0 SignalAgent/1.0 (universe-source)"}
SP500_CSV_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


class UniverseSourceError(RuntimeError):
    """A committed pool input is missing or unusable.

    Raised rather than degraded-to-empty on purpose: a source that silently
    contributes nothing shrinks what the system can trade without anyone
    seeing it happen. The rotation fails, and the previous universe — which
    is committed and known-good — stays authoritative.
    """


def _now_utc():
    return datetime.datetime.now(datetime.timezone.utc)


def _rel(path):
    return os.path.relpath(path, BASE_DIR)


def _read_committed_json(path, label, required=True):
    """Read a committed pool input. Never fetches; raises if unusable.

    `required=False` tolerates ABSENCE (returning None) while still raising on
    a file that exists and is broken. Used for the inclusion list: an empty
    inclusion list is the normal state and contributes nothing, so letting a
    missing file abort the weekly rotation would buy a new single point of
    failure for no benefit. Absence is reported in the artifact instead —
    recorded, not fatal.
    """
    if not os.path.exists(path):
        if not required:
            return None
        raise UniverseSourceError(
            f"{label} missing at {_rel(path)}. This file is committed and is "
            "the only source for it — there is no fetch fallback by design.")
    try:
        with open(path) as f:
            obj = json.load(f)
    except Exception as e:
        raise UniverseSourceError(
            f"{label} at {_rel(path)} is unreadable: {e}") from e
    if not isinstance(obj, dict):
        raise UniverseSourceError(
            f"{label} at {_rel(path)} is not a JSON object")
    return obj


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
        self.include_nasdaq100 = src.get("include_nasdaq100", False)
        self.manual_additions = src.get("manual_additions", []) or []
        self.manual_exclusions = src.get("manual_exclusions", []) or []

        # Provenance of the committed Nasdaq-100 list, captured at load so the
        # candidates artifact can PRINT where the list came from and its as-of
        # date rather than asserting that a list exists.
        self.nasdaq100_provenance = {}
        # Whether the inclusion file was found. Absence is tolerated (an empty
        # list is the normal state) but must be visible, not inferred.
        self.inclusions_file_present = None
        # Per-ETF fetch coverage: {ticker: {outcome, count, ...}}. See
        # _fetch_etf_holdings — this is what keeps an outage from reading as a
        # healthy build now that the caches are committed.
        self.etf_coverage = {}

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

        ndx_set = set()
        if self.include_nasdaq100:
            ndx_set.update(self._load_nasdaq100())

        add_set = {t for t in (_norm(x) for x in self.manual_additions) if t}
        inc_set = self._load_inclusions()
        excl_set = {t for t in (_norm(x) for x in self.manual_exclusions) if t}

        # Exclusions are subtracted from the UNION, once, at the end. That is
        # what makes "exclusion wins" structural rather than a rule someone
        # has to remember: there is no ordering of the sources by which an
        # inclusion can re-admit an excluded ticker.
        tickers = (etf_set | sp_set | ndx_set | add_set | inc_set) - excl_set

        index_union = etf_set | sp_set
        return {
            "tickers": tickers,
            "by_source": {
                "etf_holdings": len(etf_set - excl_set),
                "sp500": len(sp_set - excl_set),
                "nasdaq100": len(ndx_set - excl_set),
                "manual_additions": len(add_set - excl_set),
                "inclusions": len(inc_set - excl_set),
                # What each hand/index source CONTRIBUTES that nothing else
                # already supplied — the count that answers "what did adding
                # this source actually buy us".
                # Net-new must subtract EVERY other source, not just the two
                # index feeds — otherwise a name hand-added to the inclusion
                # list, then later admitted by the index, gets counted as
                # newly contributed by the index when the pool did not move.
                "nasdaq100_net_new": len(
                    (ndx_set - index_union - add_set - inc_set) - excl_set),
                "inclusions_net_new": len(
                    (inc_set - index_union - ndx_set - add_set) - excl_set),
                "overlap_count": len((etf_set & sp_set) - excl_set),
                "excluded_applied": sorted(
                    excl_set & (etf_set | sp_set | ndx_set | add_set | inc_set)),
            },
            "sets": {"etf": etf_set, "sp500": sp_set, "nasdaq100": ndx_set,
                     "additions": add_set, "inclusions": inc_set,
                     "exclusions": excl_set},
        }

    # ------------------------------------------------------------------
    # Sources
    # ------------------------------------------------------------------
    def _fetch_etf_holdings(self, etf_ticker, top_n=25):
        """Top-N holdings of an ETF via yfinance funds_data (Yahoo caps at ~10).

        Cached for `universe_ttl_days`, with a stale-cache fallback on failure.

        RECORDS COVERAGE, NOT JUST THE OUTCOME (D-019). Every call writes an
        entry to self.etf_coverage saying HOW the answer was obtained: fresh
        fetch, TTL cache hit, stale fallback, or nothing at all.

        This is not bookkeeping. Before the caches were committed, a fresh CI
        checkout had no cache, so a Yahoo outage returned [] and collapsed
        `by_source.etf_holdings` from ~150 to 0 — loud, and visible in the
        published artifact. Committing the caches removes that signal, because
        a stale copy is now always present: the build would sail through on
        last week's holdings and look identical to a healthy one. The signal
        has to be carried explicitly instead, or the outage impersonates
        safety — exactly what D-019 forbids.
        """
        key = _norm(etf_ticker)
        cache_path = os.path.join(self.cache_dir, f"etf_{key}.json")

        fresh_cache = self._read_cache_obj(cache_path, self.universe_ttl_days)
        if fresh_cache is not None:
            tickers = fresh_cache.get("tickers") or []
            self.etf_coverage[key] = {
                "outcome": "cache_hit", "count": len(tickers),
                "fetched_at": fresh_cache.get("fetched_at")}
            return tickers

        prev = self._read_cache_obj(cache_path, ttl_days=None)
        prev_tickers = (prev or {}).get("tickers") or []

        # The PARSE stays inside the try alongside the fetch. A malformed
        # payload is a source failure, not a programming error: if `.index`
        # raises out here the exception escapes this method, escapes the ETF
        # loop (which has no handler), and aborts a rotation that the stale
        # cache could have carried — while recording no coverage entry at all.
        try:
            import yfinance as yf
            th = yf.Ticker(etf_ticker).funds_data.top_holdings
            tickers = ([t for t in (_norm(s) for s in list(th.index)[:top_n]) if t]
                       if th is not None else [])
        except Exception as e:
            print(f"[universe] ETF holdings fetch failed for {etf_ticker}: {e}")
            return self._etf_fallback(key, prev, prev_tickers, str(e))

        # An empty result is NOT the same claim as "this fund holds no
        # equities". GLD and IBIT legitimately return nothing; a fund with
        # holdings on record returning nothing is a broken scrape. Writing []
        # over real holdings would stamp a fresh timestamp on a false
        # statement — and now that these files are committed, would enter that
        # statement into the repo as evidence.
        if not tickers and prev_tickers:
            return self._etf_fallback(
                key, prev, prev_tickers,
                "source returned no holdings for a fund that has them on record")

        if tickers:
            outcome = "fresh"
        elif prev is not None:
            # Empty now, empty last time too — GLD/IBIT and friends. The
            # history corroborates it, so this is a real answer.
            outcome = "empty_expected"
        else:
            # Empty with NOTHING to corroborate it. Counted as degraded: were
            # this written as a plain success, the committed [] would read as
            # a healthy TTL hit on every later run and the guard above would
            # be permanently disarmed for this fund — an empty source that
            # reports itself perfectly healthy forever.
            outcome = "empty_unverified"

        self._write_cache(cache_path, tickers)   # cache only on a real answer
        self.etf_coverage[key] = {"outcome": outcome, "count": len(tickers)}
        return tickers

    def _etf_fallback(self, key, prev, prev_tickers, reason):
        """Serve the stale cache, and say so loudly enough to be auditable."""
        if prev is None:
            self.etf_coverage[key] = {"outcome": "unavailable", "count": 0,
                                      "reason": reason}
            return []
        age = None
        try:
            fetched = datetime.datetime.fromisoformat(prev["fetched_at"])
            age = round((_now_utc() - fetched).total_seconds() / 86400.0, 2)
        except Exception:
            pass
        self.etf_coverage[key] = {
            "outcome": "stale_fallback", "count": len(prev_tickers),
            "fetched_at": prev.get("fetched_at"), "age_days": age,
            "reason": reason}
        return prev_tickers

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

    def _load_nasdaq100(self):
        """Nasdaq-100 membership, READ FROM THE COMMITTED FILE.

        There is deliberately no fetch and no network fallback in this method.
        The list moves only when a human runs scripts/refresh_nasdaq100.py,
        reads the membership diff, and commits it. Reintroducing a fetch here
        would restore exactly the failure this design exists to prevent: what
        the system can trade changing between two rotations, unreviewed,
        because a third party's endpoint changed or broke.
        """
        obj = _read_committed_json(NASDAQ100_PATH, "Nasdaq-100 membership list")
        symbols = obj.get("symbols")
        if not isinstance(symbols, list) or not symbols:
            raise UniverseSourceError(
                f"{_rel(NASDAQ100_PATH)} has no usable 'symbols' list. Refusing "
                "to build a pool from an index source that silently contributes "
                "nothing — rebuild it with scripts/refresh_nasdaq100.py.")
        prov = dict(obj.get("_provenance") or {})
        prov["count"] = len(symbols)
        self.nasdaq100_provenance = prov
        return {t for t in (_norm(s) for s in symbols) if t}

    def _load_inclusions(self):
        """Hand-maintained candidate additions, read from the committed file.

        Entries enter the CANDIDATE set only. Every gate still applies; an
        entry whose history is too short simply produces no ranking row, which
        is the gate working, not a fault. Exclusions are subtracted from the
        union afterwards, so an entry that is also excluded stays excluded.
        """
        obj = _read_committed_json(INCLUSIONS_PATH, "pool inclusion list",
                                   required=False)
        if obj is None:
            # Absent, not empty. Recorded rather than fatal — but recorded,
            # because the two must not look alike to a reader.
            print(f"[universe] NOTE: {_rel(INCLUSIONS_PATH)} is absent; "
                  "treating the inclusion list as empty")
            self.inclusions_file_present = False
            return set()
        self.inclusions_file_present = True
        entries = obj.get("inclusions")
        if not isinstance(entries, list):
            raise UniverseSourceError(
                f"{_rel(INCLUSIONS_PATH)} has no 'inclusions' list (use [] for "
                "none). A missing list and an empty one must not look alike: "
                "the first is a broken file, the second is a deliberate state.")
        out = set()
        for e in entries:
            t = _norm(e.get("ticker") if isinstance(e, dict) else e)
            if t:
                out.add(t)
        return out

    # ------------------------------------------------------------------
    # Cache helpers (TTL in days; ttl_days=None means "ignore age")
    # ------------------------------------------------------------------
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


def _coverage_summary(coverage):
    """{outcome: count} plus the count that should make a reader stop.

    `degraded` is the number of ETFs whose holdings did NOT come from a live
    fetch or a within-TTL cache this run. Zero is the healthy state. Anything
    else means the pool was partly assembled from history, and the artifact
    says so rather than leaving a reader to infer it from a ticker count.
    """
    by_outcome = {}
    for rec in coverage.values():
        by_outcome[rec["outcome"]] = by_outcome.get(rec["outcome"], 0) + 1
    return {
        "by_outcome": by_outcome,
        "sources": len(coverage),
        "degraded": sum(by_outcome.get(k, 0) for k in
                        ("stale_fallback", "unavailable", "empty_unverified")),
        # Any source contributing nothing, whatever the reason. GLD and IBIT
        # sit here permanently and legitimately, so this is a number to watch
        # for CHANGE rather than a fault on its own — but a fund that quietly
        # stops contributing is otherwise invisible.
        "zero_holding_sources": sorted(
            k for k, rec in coverage.items() if not rec.get("count")),
    }


def _inclusions_coverage(inc_live, by_gics, unclassified):
    """Did each hand-added entry actually reach a group?

    `degraded` is zero in the healthy state, the same contract as
    etf_coverage_summary. A non-zero value means somebody deliberately added
    a name and the pool silently did nothing with it.
    """
    inc_live = sorted(inc_live)
    unc = set(unclassified or [])
    missed = [t for t in inc_live if t in unc]
    # A name can also vanish by classifying nowhere at all — absent from
    # by_gics AND absent from the unclassified list. Treat that as degraded
    # too rather than reporting a group of None as if it were placed.
    placed = {}
    for t in inc_live:
        if t in unc:
            continue
        g = next((name for name, m in by_gics.items() if t in m), None)
        if g is None:
            missed.append(t)
        else:
            placed[t] = g
    return {
        "declared": inc_live,
        "classified": placed,
        "unclassified": sorted(set(missed)),
        "degraded": len(set(missed)),
    }


def pool_definition(uni_cfg=None):
    """The canonical description of what the pool IS, for fingerprinting.

    Deliberately describes the DEFINITION, not the outcome: the ETF *tickers*
    we source from, not the holdings they returned this week. Holdings churn
    weekly; if they were in here the fingerprint would move every rotation and
    would tell you nothing about whether the pool was redefined.

    Gates (min_market_cap and friends) are NOT here either. They filter the
    pool, they do not constitute it, and rotation_config already records them
    verbatim in the same artifact. Pool version answers "which names could
    have been considered"; rotation_config answers "what did they have to
    clear". Together they fully determine a ranking.
    """
    if uni_cfg is None:
        uni_cfg = load_universe_config()
    src = uni_cfg.get("source", {}) or {}

    ndx_symbols, ndx_as_of = [], None
    if src.get("include_nasdaq100", False):
        obj = _read_committed_json(NASDAQ100_PATH, "Nasdaq-100 membership list")
        # sorted(set(...)) — the loader returns a SET, so a duplicate entry in
        # the file changes no pool. A fingerprint that moved on it would report
        # a redefinition that did not happen.
        ndx_symbols = sorted({t for t in (_norm(s) for s in obj.get("symbols") or []) if t})
        ndx_as_of = (obj.get("_provenance") or {}).get("as_of")

    inc = _read_committed_json(INCLUSIONS_PATH, "pool inclusion list", required=False)
    inc_tickers = sorted(
        {t for t in (_norm(e.get("ticker") if isinstance(e, dict) else e)
                     for e in ((inc or {}).get("inclusions") or [])) if t})

    # NOTE: nasdaq100_as_of is deliberately NOT in here. The refresh script
    # stamps today's date on every write, so a no-change refresh would move
    # the fingerprint and hash equality would stop meaning pool equality. The
    # as-of date is carried in pool_stamp() for display; MEMBERSHIP is what
    # identifies the pool.
    return {
        "pool_version": POOL_VERSION,
        "etf_holdings": sorted(set(src.get("etf_holdings") or [])),
        "include_sp500": bool(src.get("include_sp500")),
        "include_nasdaq100": bool(src.get("include_nasdaq100", False)),
        "nasdaq100_symbols": ndx_symbols,
        "manual_additions": sorted(
            {t for t in (_norm(x) for x in (src.get("manual_additions") or [])) if t}),
        "inclusions": inc_tickers,
        "manual_exclusions": sorted(
            {t for t in (_norm(x) for x in (src.get("manual_exclusions") or [])) if t}),
        "_as_of_display_only": ndx_as_of,
    }


def _definition_blob(d):
    """Canonical bytes for hashing: underscore-prefixed keys are display-only
    and excluded, so a field that changes without the pool changing (an as-of
    date) cannot move the fingerprint."""
    return json.dumps({k: v for k, v in d.items() if not k.startswith("_")},
                      sort_keys=True, separators=(",", ":"))


def pool_definition_hash(uni_cfg=None):
    """16-hex fingerprint of pool_definition(). Moves on any deliberate edit
    to a source list — including one made without bumping POOL_VERSION."""
    return hashlib.sha256(
        _definition_blob(pool_definition(uni_cfg)).encode()).hexdigest()[:16]


# The exact keys pool_stamp() emits. Downstream artifacts copy the stamp by
# THESE keys rather than recomputing it — recomputing at read time would brand
# an old ranking with today's pool definition, which is precisely the
# confusion the versioning exists to prevent.
POOL_STAMP_KEYS = ("pool_version", "pool_definition_hash", "pool_sources",
                   "nasdaq100_as_of", "pool_exclusions", "pool_inclusions")


def pool_stamp(uni_cfg=None):
    """The pool identity block stamped into every pool-derived artifact."""
    d = pool_definition(uni_cfg)
    sources = ["etf_holdings"]
    if d["include_sp500"]:
        sources.append("sp500")
    if d["include_nasdaq100"]:
        sources.append("nasdaq100")
    if d["inclusions"]:
        sources.append("inclusions")
    if d["manual_additions"]:
        sources.append("manual_additions")
    return {
        "pool_version": POOL_VERSION,
        "pool_definition_hash": hashlib.sha256(
            _definition_blob(d).encode()).hexdigest()[:16],
        "pool_sources": sources,
        "nasdaq100_as_of": d["_as_of_display_only"],
        "pool_exclusions": d["manual_exclusions"],
        "pool_inclusions": d["inclusions"],
    }


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
    gics = GICSClassifier(src.cache_dir, ttl_days=cache_cfg.get("gics_ttl_days", 30),
                          aliases=uni_cfg.get("gics_aliases"))
    # Seed authoritative GICS from the S&P 500 CSV (free; avoids .info calls).
    gics.seed(src.sp500_gics_map, source="sp500_csv")
    by_gics = gics.get_universe_by_gics(tickers, allow_remote=allow_remote)

    unclassified = by_gics.pop("_unclassified", [])

    # CLASSIFICATION COVERAGE FOR THE HAND-MAINTAINED ENTRIES (D-019).
    #
    # An inclusion is a deliberate act: somebody decided this name should be
    # considered. If it fails to classify it lands in _unclassified, is
    # dropped from by_gics by the builder, and contributes NOTHING — with
    # the pool count one higher and no other trace. That is the same shape
    # as an absent input reading as a normal result.
    #
    # The exposure is real, not theoretical: a name absent from the S&P 500
    # CSV seed has no free classification and falls through to a live
    # yfinance .info call cached in data/universe_cache/gics_cache.json,
    # which is gitignored — so on a fresh CI checkout the call must succeed
    # or the entry silently does nothing.
    inclusions_coverage = _inclusions_coverage(
        detailed["sets"]["inclusions"] - detailed["sets"]["exclusions"],
        by_gics, unclassified)
    if inclusions_coverage["degraded"]:
        print(f"[universe] *** INCLUSION NOT CLASSIFIED: "
              f"{', '.join(inclusions_coverage['unclassified'])} — these names "
              f"are in the pool but belong to no group, so they contribute "
              f"nothing to any ranking. Deliberate entries must not fail "
              f"silently; check data/universe_cache/gics_cache.json and the "
              f"GICS lookup.")
    # CANONICALISATION COVERAGE FOR THE GROUP NAMES THEMSELVES (D-023).
    #
    # The builder groups by the classification STRING. A vendor label the
    # alias map does not cover therefore becomes its own group — below
    # rotation.min_candidates, permanently ineligible at any composite — and
    # its members are simultaneously missing from the canonical group where
    # their peers compete, so both groups' composites are computed over the
    # wrong population. Nothing in the artifact distinguished that from a
    # genuinely small sub-industry; the eight found by the 2026-08-11 sweep
    # had been accumulating unnoticed for as long as the pool has reached
    # beyond the S&P 500 CSV seed.
    #
    # Recorded, never fatal. A rotation that refuses on a classification
    # nobody has reviewed yet would trade a silent mistake for an outage.
    unaliased = gics.unaliased_group_names(
        list(by_gics), uni_cfg.get("gics_unmappable_labels"))
    if unaliased["new"]:
        print(f"[universe] *** UNCANONICALISED GROUP NAME(S): "
              f"{', '.join(unaliased['new'])} — each is a group of its own "
              f"that can never reach min_candidates, holding names that are "
              f"missing from the canonical group where their peers sit. Add "
              f"an entry to universe.gics_aliases, or — if the label's GICS "
              f"image is many-to-one — to universe.gics_unmappable_labels "
              f"with the reason.")
    by_counts = {k: len(v) for k, v in sorted(by_gics.items(),
                                              key=lambda kv: (-len(kv[1]), kv[0]))}

    out = {
        "generated_at": _now_utc().isoformat(),
        **pool_stamp(uni_cfg),
        "total_count": len(tickers),
        "by_source": detailed["by_source"],
        # Where the committed index list came from and when — carried into the
        # artifact so source attribution is evidence a reader can check, not a
        # claim they have to take on faith.
        "nasdaq100_provenance": src.nasdaq100_provenance,
        # HOW each ETF's holdings were obtained this run. A build that served
        # stale holdings must not be indistinguishable from one that fetched
        # them (D-019).
        "etf_coverage": src.etf_coverage,
        "etf_coverage_summary": _coverage_summary(src.etf_coverage),
        "inclusions_file_present": src.inclusions_file_present,
        "by_gics_sub_industry": by_counts,
        # Which hand-added entries actually reached a group. `degraded` is
        # zero in the healthy state, exactly like etf_coverage_summary.
        "inclusions_coverage": inclusions_coverage,
        # Group names that are a vendor's own label rather than a GICS
        # sub-industry. `new` is empty in the healthy state; `known` carries
        # the labels config records as unmappable-by-alias, with reasons.
        "unaliased_labels": unaliased,
        "unclassified": len(unclassified),
        "unclassified_tickers": sorted(unclassified),
    }

    if write:
        try:
            write_candidates_artifact(out)
        except Exception as e:
            print(f"[universe] candidates write failed: {e}")
    return out, tickers, by_gics


def write_candidates_artifact(out):
    """Atomically write data|public/universe_candidates.json. Raises on failure
    (universe_builder calls this only after its viability floor passes, so a
    degraded candidate list is never published)."""
    for d in (os.path.join(BASE_DIR, "data"), os.path.join(BASE_DIR, "public")):
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, "universe_candidates.json")
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(out, f, indent=2)
        os.replace(tmp, path)


if __name__ == "__main__":
    out, tickers, by_gics = build_universe_candidates()
    print(json.dumps({k: out[k] for k in
                      ("generated_at", "pool_version", "pool_definition_hash",
                       "pool_sources", "total_count", "by_source", "unclassified")},
                     indent=2))
    prov = out.get("nasdaq100_provenance") or {}
    if prov:
        print(f"\nNasdaq-100 source: {prov.get('source_url')}"
              f"\n  as of      : {prov.get('as_of')}  ({prov.get('count')} symbols)"
              f"\n  retrieved  : {prov.get('retrieved_at')} by {prov.get('retrieved_by')}"
              f"\n  runtime fetches this list: "
              f"{prov.get('runtime_fetches_this')}")
    print(f"\nby_gics_sub_industry ({len(out['by_gics_sub_industry'])} sub-industries):")
    for name, count in out["by_gics_sub_industry"].items():
        print(f"  {count:>4}  {name}")
    if out["unclassified"]:
        print(f"\nunclassified ({out['unclassified']}): {', '.join(out['unclassified_tickers'][:40])}"
              + (" ..." if out["unclassified"] > 40 else ""))
