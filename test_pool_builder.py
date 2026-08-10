#!/usr/bin/env python3
"""Pins for the pool-builder change (2026-08-10, pool v2).

    pool = ETF top-holdings ∪ S&P 500 ∪ Nasdaq-100 ∪ inclusions − exclusions

Five invariants, each written the way docs/testing.md requires: drive the
real entry point, assert the invariant rather than the implementation, and
demonstrate the failure the pin exists to catch.

  1. The Nasdaq-100 list is READ, never fetched — proven with the network cut.
  2. Inclusion ∧ exclusion resolves to EXCLUDED, whatever the source order.
  3. An included name with insufficient history yields no qualifier, no
     selection, and no error.
  4. The pool version reaches every ranking artifact, and is carried by the
     artifact rather than recomputed when it is read.
  5. data/universe_ranking.json cannot be both tracked and ignored.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

import universe_builder as ub
import universe_source as us


# ----------------------------------------------------------------------
# 1. The Nasdaq-100 list is read from the committed file, never fetched.
# ----------------------------------------------------------------------
class _NetworkTripwire(BaseException):
    """Deliberately NOT an Exception.

    Every fetch in universe_source is wrapped in `except Exception` with a
    fallback — that is the house style. A tripwire raised as an Exception is
    therefore swallowed by the most likely regression, and the pin goes green
    while the code fetches. Deriving from BaseException makes the tripwire
    survive those handlers.
    """


class _NetworkCut:
    """Best-effort egress block. A SUPPORTING check, never the load-bearing one.

    It cannot be complete: curl_cffi and libcurl-backed clients bypass Python
    sockets entirely, and yfinance — already imported by this module — is one
    import away. So the decisive test is the SENTINEL below, which does not
    depend on blocking anything. This block only widens the net.
    """

    def __enter__(self):
        import http.client
        import socket
        import urllib.request
        self._saved = []

        def boom(*a, **k):
            raise _NetworkTripwire(
                "NETWORK REACHED. The Nasdaq-100 path must read "
                "data/pool/nasdaq100.json and nothing else — a fetch here "
                "would let index membership, and so what the system can "
                "trade, change between two rotations unreviewed.")

        import requests
        targets = [(requests, "get"), (requests, "post"),
                   (requests.Session, "request"), (requests.Session, "send"),
                   (urllib.request, "urlopen"), (urllib.request, "build_opener"),
                   (http.client.HTTPConnection, "request"),
                   (http.client.HTTPSConnection, "request"),
                   (socket, "create_connection"), (socket.socket, "connect")]
        try:
            import curl_cffi.requests as cc
            targets += [(cc, "get"), (cc, "request")]
        except Exception:
            pass
        for mod, name in targets:
            try:
                self._saved.append((mod, name, getattr(mod, name)))
                setattr(mod, name, boom)
            except Exception:
                pass
        return self

    def __exit__(self, *exc):
        for mod, name, orig in self._saved:
            try:
                setattr(mod, name, orig)
            except Exception:
                pass
        return False


def test_nasdaq100_is_read_not_fetched():
    cfg = us.load_universe_config()

    # (1a) THE DECISIVE TEST — a sentinel file, not an egress block.
    #
    # Point the loader at a committed-file location holding a membership list
    # that CANNOT come off the wire, and require the loader to return exactly
    # it. No amount of network blocking is needed and no bypass route helps:
    # an implementation that fetches returns the real index and fails, an
    # implementation that reads returns the sentinel and passes. This is the
    # invariant ("the file is the source of truth") rather than a proxy for it.
    tmpdir = tempfile.mkdtemp()
    saved = us.NASDAQ100_PATH
    try:
        sentinel = {"ZZSENTINELA", "ZZSENTINELB", "ZZSENTINELC"}
        p = os.path.join(tmpdir, "nasdaq100.json")
        with open(p, "w") as f:
            json.dump({"_provenance": {"as_of": "1999-01-01",
                                       "source_url": "sentinel",
                                       "retrieved_at": "1999-01-01T00:00:00",
                                       "runtime_fetches_this": False},
                       "symbols": sorted(sentinel)}, f)
        us.NASDAQ100_PATH = p
        got = us.UniverseSource(cfg)._load_nasdaq100()
        assert got == sentinel, (
            f"loader returned {sorted(got)[:6]}... instead of the sentinel — "
            "it is not reading the committed file")
        assert "AAPL" not in got, "real index membership leaked in — this is a fetch"
        print("  (1a) the loader returns the SENTINEL list, not real index "
              "membership — the committed file is the only source: OK")

        # THE FAILURE, DEMONSTRATED, against the same sentinel. A fetching
        # implementation returns the real index and is caught — with no
        # reliance on blocking egress, so no bypass route rescues it.
        class Fetching(us.UniverseSource):
            def _load_nasdaq100(self):
                import requests
                r = requests.get(
                    "https://api.nasdaq.com/api/quote/list-type/nasdaq100",
                    headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
                rows = r.json()["data"]["data"]["rows"]
                return {row["symbol"].strip().upper() for row in rows}

        try:
            live = Fetching(cfg)._load_nasdaq100()
        except Exception as e:            # offline CI: the fetch simply fails
            print(f"  (1b) a fetching variant could not satisfy the sentinel "
                  f"(fetch errored: {type(e).__name__}): OK")
        else:
            assert live != sentinel, "a live fetch returned the sentinel?!"
            assert "AAPL" in live, "precondition: the live source has real names"
            print(f"  (1b) a fetching variant returns {len(live)} real symbols "
                  "instead of the sentinel and is caught: OK")
    finally:
        us.NASDAQ100_PATH = saved
        shutil.rmtree(tmpdir, ignore_errors=True)

    # (1c) The real committed file, with egress blocked as a supporting check.
    src = us.UniverseSource(cfg)
    with open(us.NASDAQ100_PATH) as f:
        committed = json.load(f)
    expected = {t for t in (us._norm(s) for s in committed["symbols"]) if t}
    with _NetworkCut():
        got = src._load_nasdaq100()
    assert got == expected and len(got) >= 95, "committed list did not load"
    prov = src.nasdaq100_provenance
    for key in ("as_of", "source_url", "retrieved_at"):
        assert prov.get(key), f"provenance missing {key}"
    assert prov["runtime_fetches_this"] is False
    print(f"  (1c) the real committed list loads {len(got)} symbols with "
          f"egress blocked; as_of {prov['as_of']}: OK")

    # (1d) A missing committed file must stop the build, not quietly shrink
    # the pool to two sources.
    saved = us.NASDAQ100_PATH
    try:
        us.NASDAQ100_PATH = os.path.join(REPO, "data", "pool", "__absent__.json")
        try:
            us.UniverseSource(cfg)._load_nasdaq100()
        except us.UniverseSourceError:
            print("  (1d) a missing committed list raises rather than "
                  "silently contributing nothing: OK")
        else:
            raise AssertionError("missing list did not raise")
    finally:
        us.NASDAQ100_PATH = saved


# ----------------------------------------------------------------------
# 2. Exclusion beats inclusion.
# ----------------------------------------------------------------------
def test_exclusion_beats_inclusion():
    cfg = us.load_universe_config()
    src = us.UniverseSource(cfg)
    ndx = src._load_nasdaq100()
    excl = {t for t in (us._norm(x) for x in src.manual_exclusions) if t}

    # The live case is an OBSERVATION, not the mechanism test. SPCX's index
    # membership is outside our control: Nasdaq drops it at some
    # reconstitution and this stops being the live case, which must not turn
    # a healthy pin red. The mechanism is pinned on a synthetic ticker below.
    if "SPCX" in ndx and "SPCX" in excl:
        print("  (2-obs) live case present: SPCX is a Nasdaq-100 member and "
              "an exclusion")
    else:
        print("  (2-obs) live SPCX case has retired (index membership or the "
              "exclusion changed) — the mechanism pin below is unaffected")

    # Synthetic case, driven through the REAL assembly entry point. The two
    # fetching sources are configured off so the whole thing runs with the
    # network cut; data is injected, no logic is stubbed.
    tmpdir = tempfile.mkdtemp()
    saved = us.INCLUSIONS_PATH
    try:
        path = os.path.join(tmpdir, "inclusions.json")
        with open(path, "w") as f:
            json.dump({"inclusions": [
                {"ticker": "ZZBOTH", "reason": "test", "added": "2026-08-10"},
                {"ticker": "ZZINCONLY", "reason": "test", "added": "2026-08-10"},
            ]}, f)
        us.INCLUSIONS_PATH = path

        # Synthetic tickers, so nothing here depends on live index membership.
        cfg2 = json.loads(json.dumps(cfg))
        cfg2["source"]["etf_holdings"] = []
        cfg2["source"]["include_sp500"] = False
        cfg2["source"]["include_nasdaq100"] = True
        cfg2["source"]["manual_additions"] = ["ZZBOTH"]
        cfg2["source"]["manual_exclusions"] = ["ZZBOTH"]

        def assemble():
            with _NetworkCut():
                return us.UniverseSource(cfg2).get_candidate_universe_detailed()

        detailed = assemble()
        pool = detailed["tickers"]
        assert "ZZBOTH" in detailed["sets"]["inclusions"], \
            "precondition: the ticker really is on the inclusion list"
        assert "ZZBOTH" in detailed["sets"]["exclusions"], \
            "precondition: the ticker really is on the exclusion list"
        assert "ZZBOTH" not in pool, "an inclusion re-admitted an excluded name"
        assert "ZZINCONLY" in pool, "a non-excluded inclusion did not arrive"
        assert "ZZBOTH" in detailed["by_source"]["excluded_applied"]
        print("  (2a) through the real assembly: a ticker on both lists "
              "resolves to EXCLUDED, while a non-excluded inclusion still "
              "arrives: OK")

        # THE FAILURE, DEMONSTRATED — by mutating PRODUCTION, not by doing set
        # algebra in the test. Subtracting exclusions per-source and then
        # unioning the inclusions is a plausible refactor; under it the same
        # assertion above must fail. That is what proves the assertion is
        # capable of failing at all.
        orig = us.UniverseSource.get_candidate_universe_detailed

        def wrong_order(self):
            d = orig(self)
            S = d["sets"]
            d["tickers"] = ((S["etf"] | S["sp500"] | S["nasdaq100"]
                             | S["additions"]) - S["exclusions"]) | S["inclusions"]
            return d

        us.UniverseSource.get_candidate_universe_detailed = wrong_order
        try:
            broken_pool = assemble()["tickers"]
        finally:
            us.UniverseSource.get_candidate_universe_detailed = orig
        assert "ZZBOTH" in broken_pool, (
            "the wrong ordering did NOT re-admit the excluded ticker — this "
            "pin cannot distinguish correct assembly from broken")
        print("  (2b) with production mutated to apply exclusions before "
              "unioning inclusions, the excluded ticker returns and (2a)'s "
              "assertion fails: OK")
    finally:
        us.INCLUSIONS_PATH = saved
        shutil.rmtree(tmpdir, ignore_errors=True)


# ----------------------------------------------------------------------
# 3. An included name with too little history is silent, not fatal.
# ----------------------------------------------------------------------
def _metric(t, score=70, hist=400, vol=5_000_000, ytd=20.0):
    return {"ticker": t, "score": score, "history_days": hist,
            "avg_volume": vol, "ytd": ytd, "r3m": 5.0, "r1m": 2.0,
            "score_components": {}, "above_50dma": True}


def test_included_name_without_history_is_silent():
    rot = dict(ub.DEFAULT_ROTATION)
    rot["composite_weights"] = dict(ub.DEFAULT_ROTATION["composite_weights"])
    rot.update({"top_n": 15, "min_candidates": 3, "max_tickers_per_group": 7,
                "min_composite_score": 50, "min_history_days": 90,
                "min_avg_volume": 500_000, "min_market_cap": 5_000_000_000})

    incumbents = ["AAA", "BBB", "CCC"]
    caps = {t: {"market_cap_usd": 50e9} for t in incumbents + ["NEWCO"]}

    # (a) A brand-new listing with too few bars never reaches metrics at all.
    by_gics = {"Test Industry": incumbents + ["NEWCO"]}
    metrics = {t: _metric(t) for t in incumbents}
    ranking, selected = ub.rank_and_select(by_gics, metrics, caps, rot)
    g = ranking[0]
    assert "NEWCO" in g["members"], "the included name left the candidate set"
    assert "NEWCO" not in [q["ticker"] for q in g["qualifiers"]]
    assert g["qualifier_count"] == 3 and g["candidates"] == 4
    assert selected and selected[0]["name"] == "Test Industry", \
        "an unrankable member broke its group's selection"
    print("  (3a) <63 bars: no metrics, no qualifier, group still selects, "
          "no error: OK")

    # (b) Enough bars to be measured, not enough to clear the 90-day gate.
    metrics_b = dict(metrics)
    metrics_b["NEWCO"] = _metric("NEWCO", hist=40)
    ranking, selected = ub.rank_and_select(by_gics, metrics_b, caps, rot)
    g = ranking[0]
    assert "NEWCO" not in [q["ticker"] for q in g["qualifiers"]]
    assert "NEWCO" in g["disqualified"]
    assert ub._gate_status(g["disqualified"]["NEWCO"]) == "failed_history_gate"
    assert selected and selected[0]["name"] == "Test Industry"
    print("  (3b) 40 bars against a 90-day gate: recorded as "
          "failed_history_gate, never a qualifier, group unaffected: OK")

    # (c) The same name, once it has the bars, becomes a qualifier. The gate
    #     is a wait, not a rejection — which is why (a) and (b) are correct
    #     outcomes rather than defects to work around.
    metrics_c = dict(metrics)
    metrics_c["NEWCO"] = _metric("NEWCO", hist=120)
    ranking, _ = ub.rank_and_select(by_gics, metrics_c, caps, rot)
    assert "NEWCO" in [q["ticker"] for q in ranking[0]["qualifiers"]]
    print("  (3c) the same name qualifies once it has 120 days — the gate "
          "delays, it does not reject: OK")

    # (d) The AUDIT path — where "no error" would actually break. A name with
    #     too little history is absent from `metrics` entirely, and the audit
    #     builder is the only place that absence is dereferenced. Pinning
    #     rank_and_select alone leaves this uncovered: `metrics[t]` instead of
    #     `metrics.get(t)` raises KeyError right here and aborts the rotation.
    ranking, _ = ub.rank_and_select(by_gics, metrics, caps, rot)
    ub.audit_ticker_rows(ranking, metrics, rot)      # must not raise
    rows = {r["ticker"]: r for r in ranking[0]["tickers"]}
    assert rows["NEWCO"]["status"] == "no_valid_data", rows["NEWCO"]
    assert rows["NEWCO"]["score"] is None and rows["NEWCO"]["fails"] == []
    assert len(rows) == 4 and all(rows[t]["status"] == "selected"
                                 for t in incumbents)
    print("  (3d) the audit table carries the unrankable name as "
          "no_valid_data without raising, and the group's other rows are "
          "unaffected: OK")

    # A row in this table means "was considered", NOT "passed" — the
    # distinction that explains a sub-floor market cap appearing in the
    # published artifact without the gate having leaked.
    ranking, _ = ub.rank_and_select(by_gics, metrics_b, caps, rot)
    ub.audit_ticker_rows(ranking, metrics_b, rot)
    row = {r["ticker"]: r for r in ranking[0]["tickers"]}["NEWCO"]
    assert row["status"] == "failed_history_gate" and row["score"] is not None
    print("  (3e) a gate-failing candidate still gets an audit row, carrying "
          "the gate it failed — presence is not admission: OK")


# ----------------------------------------------------------------------
# 4. The pool version reaches every artifact, carried not recomputed.
# ----------------------------------------------------------------------
def test_pool_version_in_every_artifact():
    stamp = us.pool_stamp()
    for k in us.POOL_STAMP_KEYS:
        assert k in stamp, f"pool_stamp missing {k}"
    assert stamp["pool_version"] == us.POOL_VERSION
    assert len(stamp["pool_definition_hash"]) == 16
    assert "nasdaq100" in stamp["pool_sources"]
    assert stamp["nasdaq100_as_of"], "as-of date absent from the stamp"

    # The ranking payload — the artifact served at /api/universe/ranking.json
    # and read by every study.
    active = {"built_at": "2026-08-10T00:00:00+00:00", "week_key": "2026-08-14",
              "rotation_config": {}, "candidates_total": 548, "ranking": [],
              "groups": {}, "scorer_version": "score_stock_v2", **stamp}
    payload = ub.build_ranking_payload(active)
    for k in us.POOL_STAMP_KEYS:
        assert payload.get(k) == stamp[k], f"{k} did not reach the payload"
    print(f"  (4a) pool_version {stamp['pool_version']} and hash "
          f"{stamp['pool_definition_hash']} reach the ranking payload: OK")

    # THE FAILURE, DEMONSTRATED. A payload built from an OLD artifact must
    # report the OLD pool. Recomputing the stamp at read time would brand a
    # historical ranking with today's pool, which is exactly the confusion
    # the version exists to prevent.
    old = dict(active)
    old.update({"pool_version": "v1-legacy", "pool_definition_hash": "0" * 16,
                "pool_sources": ["etf_holdings", "sp500"]})
    old_payload = ub.build_ranking_payload(old)
    assert old_payload["pool_version"] == "v1-legacy", \
        "the payload recomputed the stamp instead of carrying the artifact's"
    assert old_payload["pool_definition_hash"] == "0" * 16
    print("  (4b) a pre-v2 artifact keeps its own pool stamp when re-read; "
          "the current config does not overwrite it: OK")

    # A pre-stamp artifact yields None, never a KeyError and never a
    # fabricated version.
    bare = {k: v for k, v in active.items() if k not in us.POOL_STAMP_KEYS}
    assert ub.build_ranking_payload(bare)["pool_version"] is None
    print("  (4c) an artifact built before pool versioning reports None "
          "rather than being retro-labelled: OK")

    # The hash moves on a list edit made WITHOUT a version bump — the case a
    # hand-maintained version string cannot catch.
    tmpdir = tempfile.mkdtemp()
    saved = us.INCLUSIONS_PATH
    try:
        path = os.path.join(tmpdir, "inclusions.json")
        with open(path, "w") as f:
            json.dump({"inclusions": [{"ticker": "ZZTESTONLY",
                                       "reason": "t", "added": "2026-08-10"}]}, f)
        us.INCLUSIONS_PATH = path
        moved = us.pool_definition_hash()
        assert moved != stamp["pool_definition_hash"], \
            "editing the inclusion list did not move the fingerprint"
        print("  (4d) editing a source list moves the fingerprint even with "
              "POOL_VERSION unchanged: OK")
    finally:
        us.INCLUSIONS_PATH = saved
        shutil.rmtree(tmpdir, ignore_errors=True)


# ----------------------------------------------------------------------
# 5. Nothing is both tracked and ignored.
# ----------------------------------------------------------------------
def _git(*args, index=None):
    env = dict(os.environ)
    if index:
        env["GIT_INDEX_FILE"] = index
    return subprocess.run(["git"] + list(args), cwd=REPO, env=env,
                          capture_output=True, text=True)


def _is_ignored(path, index=None):
    # --no-index is load-bearing. WITHOUT it, git check-ignore never reports a
    # TRACKED path as ignored, so `ignored and tracked` is unsatisfiable and
    # the contradiction this pin exists to detect can never fire.
    return _git("check-ignore", "--no-index", "-q", path,
                index=index).returncode == 0


def _is_tracked(path, index=None):
    return _git("ls-files", "--error-unmatch", path,
                index=index).returncode == 0


def test_no_path_is_both_tracked_and_ignored():
    for p in ("data/universe_ranking.json", "data/universe_candidates.json",
              "data/universe_cache/sp500.json", "data/universe_cache/gics_cache.json"):
        ign, trk = _is_ignored(p), _is_tracked(p)
        assert not (ign and trk), f"{p} is BOTH tracked and ignored"
        assert ign, f"{p} should be ignored (it is a local build byproduct)"
    print("  (5a) the data/ build byproducts are ignored and untracked — "
          "one state, not two contradictory ones: OK")

    # The ETF caches are the opposite: committed on purpose, so they must NOT
    # be ignored, or the rotation's write-back would silently do nothing.
    etfs = [f for f in os.listdir(os.path.join(REPO, "data", "universe_cache"))
            if f.startswith("etf_") and f.endswith(".json")]
    assert etfs, "no ETF caches on disk to check"
    for f in etfs:
        p = f"data/universe_cache/{f}"
        assert not _is_ignored(p), \
            f"{p} is ignored — the rotation's write-back would be a no-op"
        assert _is_tracked(p), \
            (f"{p} is not ignored but is not tracked either — 'committed' is "
             "the claim; being merely un-ignored does not make it true")
    print(f"  (5b) all {len(etfs)} ETF holdings caches are tracked AND not "
          "ignored, so the rotation records what it actually sourced: OK")

    # The committed pool inputs must actually BE committed. Pin 1's whole
    # premise is "read from a committed file"; nothing else checks that the
    # file is in the index, and a fresh CI checkout without it hard-fails the
    # rotation on UniverseSourceError.
    for p in ("data/pool/nasdaq100.json", "data/pool/inclusions.json"):
        assert not _is_ignored(p), f"{p} is ignored — CI would never see it"
        assert _is_tracked(p), \
            (f"{p} is untracked. The Nasdaq-100 path has no fetch fallback by "
             "design, so a rotation on a fresh checkout would raise "
             "UniverseSourceError and never run.")
    print("  (5d) the committed pool inputs are genuinely tracked, so a fresh "
          "checkout has the files the loader has no fallback for: OK")

    # THE FAILURE, DEMONSTRATED — in a throwaway index, so the real one is
    # never touched. `git add -f` is the move that creates the contradiction.
    tmpdir = tempfile.mkdtemp()
    victim = os.path.join(REPO, "data", "universe_ranking.json")
    created = not os.path.exists(victim)
    try:
        if created:
            with open(victim, "w") as f:
                json.dump({"_": "throwaway for the pin"}, f)
        idx = os.path.join(tmpdir, "index")
        shutil.copyfile(os.path.join(REPO, ".git", "index"), idx)
        add = _git("add", "-f", "data/universe_ranking.json", index=idx)
        assert add.returncode == 0, f"force-add failed: {add.stderr}"
        assert _is_tracked("data/universe_ranking.json", index=idx)
        assert _is_ignored("data/universe_ranking.json")
        print("  (5c) force-adding the file makes it tracked AND ignored, "
              "and this pin flags that state: OK")
    finally:
        if created and os.path.exists(victim):
            os.remove(victim)
        shutil.rmtree(tmpdir, ignore_errors=True)


# ----------------------------------------------------------------------
# 6. A degraded ETF fetch cannot look like a healthy one.
# ----------------------------------------------------------------------
class _FakeYF:
    """Stand in for yfinance so the failure modes are reachable in a test."""

    def __init__(self, behaviour):
        self.behaviour = behaviour

    def Ticker(self, _sym):
        beh = self.behaviour

        class _T:
            @property
            def funds_data(self):
                if beh == "raise":
                    raise RuntimeError("simulated Yahoo outage")

                class _F:
                    top_holdings = None
                return _F()
        return _T()


def _with_fake_yf(behaviour, fn):
    saved = sys.modules.get("yfinance")
    sys.modules["yfinance"] = _FakeYF(behaviour)
    try:
        return fn()
    finally:
        if saved is None:
            sys.modules.pop("yfinance", None)
        else:
            sys.modules["yfinance"] = saved


def test_degraded_etf_fetch_is_visible():
    import datetime
    cfg = us.load_universe_config()
    tmpdir = tempfile.mkdtemp()
    try:
        path = os.path.join(tmpdir, "etf_TEST.json")
        stale_at = (us._now_utc() - datetime.timedelta(days=30)).isoformat()
        real = ["AAA", "BBB", "CCC", "DDD", "EEE"]
        with open(path, "w") as f:
            json.dump({"tickers": real, "fetched_at": stale_at}, f)

        # (a) The fetch raises. The stale holdings are still served — that is
        #     the intended resilience — but the build must SAY it did that.
        src = us.UniverseSource(cfg, cache_dir=tmpdir)
        got = _with_fake_yf("raise", lambda: src._fetch_etf_holdings("TEST"))
        cov = src.etf_coverage["TEST"]
        assert got == real, "stale fallback stopped serving its holdings"
        assert cov["outcome"] == "stale_fallback", cov
        assert cov["age_days"] and cov["age_days"] > 29, cov
        print("  (6a) a failed fetch serves stale holdings AND records "
              f"stale_fallback at {cov['age_days']} days old: OK")

        # (b) THE FAILURE, DEMONSTRATED. The returned holdings are byte-for-
        #     byte what a healthy run returns. Coverage is the ONLY thing that
        #     separates an outage from a good build — which is why recording
        #     the outcome alone would let the outage impersonate safety.
        healthy = us.UniverseSource(cfg, cache_dir=tmpdir)
        with open(path, "w") as f:
            json.dump({"tickers": real, "fetched_at": us._now_utc().isoformat()}, f)
        got_healthy = healthy._fetch_etf_holdings("TEST")
        assert got_healthy == got, "precondition: the two paths return the same"
        assert healthy.etf_coverage["TEST"]["outcome"] == "cache_hit"
        assert healthy.etf_coverage["TEST"]["outcome"] != cov["outcome"], \
            "an outage and a healthy build are indistinguishable"
        print("  (6b) both paths return identical holdings; only the coverage "
              "record tells them apart: OK")

        # (c) An empty response must not overwrite holdings on record. Writing
        #     [] with a fresh timestamp would put a false statement into a
        #     COMMITTED file.
        with open(path, "w") as f:
            json.dump({"tickers": real, "fetched_at": stale_at}, f)
        src2 = us.UniverseSource(cfg, cache_dir=tmpdir)
        got2 = _with_fake_yf("none", lambda: src2._fetch_etf_holdings("TEST"))
        with open(path) as f:
            on_disk = json.load(f)
        assert got2 == real, "an empty response wiped real holdings"
        assert on_disk["tickers"] == real, "the cache file was overwritten"
        assert on_disk["fetched_at"] == stale_at, \
            "a fresh timestamp was stamped on holdings that were not refetched"
        assert src2.etf_coverage["TEST"]["outcome"] == "stale_fallback"
        print("  (6c) an empty holdings response does not overwrite holdings "
              "on record, and does not re-stamp the timestamp: OK")

        # (d) The summary surfaces a degraded count a reader can act on.
        s = us._coverage_summary({"A": {"outcome": "fresh"},
                                  "B": {"outcome": "cache_hit"},
                                  "C": {"outcome": "stale_fallback"},
                                  "D": {"outcome": "unavailable"}})
        assert s["degraded"] == 2, s
        assert us._coverage_summary({"A": {"outcome": "fresh"}})["degraded"] == 0
        print("  (6d) the artifact summary counts degraded sources; zero is "
              "the healthy state: OK")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_cache_ttl_is_below_the_rotation_interval():
    """The TTL must expire before the next rotation, or the fetch is skipped.

    The TTL check expires only on age STRICTLY greater than the limit, and the
    rotation cron is weekly. At ttl == 7 the two are the same length, so a run
    that starts marginally earlier than the previous one reads its own last
    cache as fresh and never refetches. Before the caches were committed this
    was unreachable — a fresh CI checkout had nothing to hit.
    """
    import datetime
    ttl = us.load_universe_config().get("cache", {}).get("universe_ttl_days")
    assert ttl is not None and ttl < 7, \
        f"universe_ttl_days={ttl} is not strictly below the 7-day rotation"

    cfg = us.load_universe_config()
    tmpdir = tempfile.mkdtemp()
    try:
        p = os.path.join(tmpdir, "c.json")
        src = us.UniverseSource(cfg, cache_dir=tmpdir)
        # A cache written one second short of a full week — the realistic case
        # when CI queueing shaves a minute off the interval.
        almost = (us._now_utc() - datetime.timedelta(days=7, seconds=-1)).isoformat()
        with open(p, "w") as f:
            json.dump({"tickers": ["X"], "fetched_at": almost}, f)
        assert src._read_cache_obj(p, ttl) is None, \
            "a nearly-week-old cache still reads fresh — the fetch is skipped"
        assert src._read_cache_obj(p, 7) is not None, \
            "precondition: at ttl=7 this same cache DOES read fresh"
        print(f"  (7) universe_ttl_days={ttl} expires a 6d23h59m cache that "
              "ttl=7 would have served: OK")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ----------------------------------------------------------------------
# 8. A shrunken pool refuses to rotate.
# ----------------------------------------------------------------------
def test_pool_retention_floor():
    """A degraded feed must stop the rotation, not quietly narrow the book.

    The floor exists because the OUTPUT of a depleted build looks healthy: 15
    groups and ~70 tickers can be assembled from a badly thinned pool, so the
    viability floor (3 groups / 8 tickers) never fires. Only the pool SIZE
    moves, and only against the previous rotation.
    """
    assert 0 < ub.POOL_RETENTION_FLOOR < 1, ub.POOL_RETENTION_FLOOR
    ok = ub.pool_retention_ok          # the predicate the builder itself uses
    prev = 548

    assert ok(548, prev), "an unchanged pool must not trip the floor"
    assert ok(520, prev), "ordinary churn (-5%) must not trip the floor"
    assert ok(494, prev), "exactly at the floor (-9.85%) must pass"
    assert not ok(493, prev), "-10.0% must trip the floor"
    assert not ok(400, prev), "a badly degraded pool must trip the floor"
    assert ok(600, prev), "a GROWING pool must never be refused"
    print(f"  (8a) floor at {ub.POOL_RETENTION_FLOOR:.0%}: 548->520 passes, "
          "548->493 refuses, growth always passes: OK")

    # THE FAILURE, DEMONSTRATED — the case the viability floor cannot see.
    # A pool cut by a third still yields a full-looking universe, so the
    # existing floors stay silent and only this one speaks.
    depleted = 360                      # -34%: an ETF feed served stale/empty
    assert depleted > ub.MIN_VIABLE_TICKERS, \
        "precondition: a depleted pool still clears the viability floor"
    assert not ok(depleted, prev), \
        "a third of the pool vanished and nothing refused to rotate"
    print(f"  (8b) a pool cut to {depleted} still clears the "
          f"{ub.MIN_VIABLE_TICKERS}-ticker viability floor — the retention "
          "floor is the only check that sees it: OK")

    # A missing baseline must not block: on a first build there is nothing to
    # compare against, and refusing would be unfixable without editing state.
    assert ok(10, None) and ok(10, 0), \
        "a missing baseline refused a rotation it cannot judge"
    print("  (8c) a first build with no previous artifact is not blocked: OK")

    # And the guard runs BEFORE the expensive fetch, so a degraded pool fails
    # in seconds rather than after a full ~550-ticker price download.
    import inspect
    body = inspect.getsource(ub.build_active_universe)
    assert body.index("POOL_RETENTION_FLOOR") < body.index("_fetch_price_batch"), \
        "the retention floor must be checked before the price fetch"
    print("  (8d) the floor is evaluated before the price fetch — a degraded "
          "pool fails fast: OK")


if __name__ == "__main__":
    print("\n=== pool-builder pins (pool v2, 2026-08-10) ===")
    test_nasdaq100_is_read_not_fetched()
    test_exclusion_beats_inclusion()
    test_included_name_without_history_is_silent()
    test_pool_version_in_every_artifact()
    test_no_path_is_both_tracked_and_ignored()
    test_degraded_etf_fetch_is_visible()
    test_cache_ttl_is_below_the_rotation_interval()
    test_pool_retention_floor()
    print("\nAll pool-builder pins passed.\n")
