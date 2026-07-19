#!/usr/bin/env python3
"""
Tests for the serve-layer shape sentinel (stale-serve fix): a regime
endpoint must never serve a shape the current code cannot produce.

Pins, through the real Flask serve path (test_client):
  - a pre-1A-shaped artifact (5 voters, no backdrop_gate/macro_inputs)
    -> 503 with Retry-After AND a refresh kick
  - a current-shape artifact -> 200, no refresh kick
  - a valid-but-old artifact -> 200 + stale_hours flag, no refresh kick

Run: python3 test_serve_guard.py
"""

import datetime
import json
import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ticker_api

GUARDED = ["/api/framework/gauges.json", "/api/framework/latest",
           "/api/framework/latest.json", "/api/framework/signals.json"]


def _now_iso(hours_ago=0):
    return (datetime.datetime.now(datetime.timezone.utc)
            - datetime.timedelta(hours=hours_ago)).isoformat()


def _pre_1a_payload():
    """The exact stale class served 2026-07-06 ~9:15 ET: 5 voters incl.
    spy_vs_200dma + yield_curve, no backdrop_gate, no macro_inputs."""
    g = {"value": 1.0, "signal": "risk_on", "detail": "d"}
    return {
        "generated_at": _now_iso(),
        "framework_version": "1.0",
        "regime": {
            "regime": "Risk-on / Trending",
            "risk_on_count": 5, "caution_count": 0, "risk_off_count": 0,
            "gauges": {k: dict(g) for k in
                       ("spy_vs_200dma", "vix_5d_avg", "hy_spread",
                        "breadth", "yield_curve")},
            "action": "Full deployment.",
        },
        "position_signals": {"tickers": {}, "transitions": []},
    }


def _parliament_payload(hours_ago=0):
    """A Build-1A parliament artifact — structurally sound for its era but
    STALE under the chassis engine (D-008 cutover: the guard must refuse to
    serve the other engine's output as current)."""
    g = {"value": 1.0, "signal": "risk_on", "detail": "d"}
    return {
        "generated_at": _now_iso(hours_ago),
        "schema": "regime-1a-3voter",
        "framework_version": "1.0",
        "regime": {
            "regime": "Risk-on / Trending",
            "risk_on_count": 3, "caution_count": 0, "risk_off_count": 0,
            "gauges": {k: dict(g) for k in
                       ("vix_5d_avg", "hy_spread", "breadth")},
            "backdrop_gate": {"gauge": "spy_vs_200dma", "open": True,
                              "capped": False, "reason": None, "value": 8.0},
            "macro_inputs": {"yield_curve": {"value": 0.8,
                                             "signal": "risk_on"}},
            "action": "Full deployment.",
        },
        "position_signals": {"tickers": {"IWM": {"state": "HELD"}},
                             "transitions": []},
    }


def _current_payload(hours_ago=0):
    """A Gauge B chassis artifact — the current schema (regime-b-chassis)."""
    p = _parliament_payload(hours_ago)
    p["schema"] = "regime-b-chassis"
    p["regime"]["engine"] = "chassis"
    p["regime"]["chassis"] = {
        "engine": "chassis",
        "raw_state": "In-Trend-Full",
        "confirmed_state": "In-Trend-Full",
        "regime": "Risk-on / Trending",
        "exposure_ceiling_pct": 90.0,
        "trend_in": True,
        "throttles": {"vix": {"firing": False}, "hy": {"firing": False},
                      "breadth": {"firing": False}},
        "throttles_firing": 0,
        "hysteresis": {"up": 0, "down": 0, "n": 2, "mode": "asymmetric"},
        "degraded": False, "degraded_reason": None,
    }
    p["regime"]["backdrop_gate"]["role"] = "trend_chassis"
    return p


class _Env:
    def __init__(self):
        self.tmp = tempfile.mkdtemp(prefix="serve_guard_")
        self.old_public = ticker_api.PUBLIC_DIR
        self.old_refresh = ticker_api._run_framework_refresh
        ticker_api.PUBLIC_DIR = self.tmp
        self.kicks = []
        ticker_api._run_framework_refresh = lambda: self.kicks.append(1)
        ticker_api._framework_status["running"] = False
        self.client = ticker_api.app.test_client()

    def write(self, payload):
        with open(os.path.join(self.tmp, "framework.json"), "w") as f:
            json.dump(payload, f)

    def wait_for_kick(self, timeout=2.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.kicks:
                return True
            time.sleep(0.05)
        return False

    def close(self):
        ticker_api.PUBLIC_DIR = self.old_public
        ticker_api._run_framework_refresh = self.old_refresh
        shutil.rmtree(self.tmp, ignore_errors=True)


def test_pre_1a_shape_503_and_refresh_kick():
    env = _Env()
    try:
        env.write(_pre_1a_payload())
        for ep in GUARDED:
            env.kicks.clear()
            r = env.client.get(ep)
            assert r.status_code == 503, f"{ep}: expected 503, got {r.status_code}"
            assert r.headers.get("Retry-After") == "180", ep
            body = r.get_json()
            assert body["status"] == "warming_up", ep
            assert body["retry_after"] == 180, ep
            assert "framework_status" in body, ep
            assert env.wait_for_kick(), f"{ep}: refresh was not kicked"
        # unparseable file: same treatment, never a 500
        with open(os.path.join(env.tmp, "framework.json"), "w") as f:
            f.write("{corrupt")
        r = env.client.get(GUARDED[0])
        assert r.status_code == 503
    finally:
        env.close()
    print("  pre-1A shape -> 503 + Retry-After + refresh kick (all 4 endpoints): OK")


def test_current_shape_200_no_kick():
    env = _Env()
    try:
        env.write(_current_payload())
        for ep in GUARDED:
            r = env.client.get(ep)
            assert r.status_code == 200, f"{ep}: {r.status_code}"
            body = r.get_json()
            assert "stale_hours" not in json.dumps(body), ep
        g = env.client.get("/api/framework/gauges.json").get_json()
        assert sorted(g["gauges"].keys()) == ["breadth", "hy_spread", "vix_5d_avg"]
        assert g["backdrop_gate"]["open"] is True
        s = env.client.get("/api/framework/signals.json").get_json()
        assert s["tickers"]["IWM"]["state"] == "HELD"
        assert env.kicks == [], "fresh valid payload must not kick a refresh"
    finally:
        env.close()
    print("  current shape -> 200, correct projections, no refresh kick: OK")


def test_valid_but_old_200_with_stale_flag():
    env = _Env()
    try:
        env.write(_current_payload(hours_ago=60))   # Friday close on a Sunday
        for ep in GUARDED:
            r = env.client.get(ep)
            assert r.status_code == 200, f"{ep}: valid-but-old must SERVE, got {r.status_code}"
        g = env.client.get("/api/framework/gauges.json").get_json()
        assert 59 <= g["stale_hours"] <= 61, g.get("stale_hours")
        lj = env.client.get("/api/framework/latest.json").get_json()
        assert 59 <= lj["stale_hours"] <= 61
        s = env.client.get("/api/framework/signals.json").get_json()
        assert 59 <= s["stale_hours"] <= 61
        assert env.kicks == [], "age alone must never kick a refresh"
    finally:
        env.close()
    print("  valid-but-old -> 200 + stale_hours flag, no refresh kick: OK")


def test_parliament_artifact_stale_under_chassis():
    """D-008 cutover: a deploy-baked PARLIAMENT artifact under the chassis
    engine is schema-stale — 503 + refresh kick, exactly like the pre-1A
    class. This is what regenerates the committed artifact after the flip."""
    env = _Env()
    try:
        env.write(_parliament_payload())
        env.kicks.clear()
        r = env.client.get(GUARDED[0])
        assert r.status_code == 503, f"expected 503, got {r.status_code}"
        assert r.get_json()["status"] == "warming_up"
        assert env.wait_for_kick(), "refresh was not kicked"
    finally:
        env.close()
    print("  parliament artifact under chassis engine -> 503 + refresh kick "
          "(cutover regeneration): OK")


def test_chassis_artifact_stale_under_parliament():
    """Reverse cutover direction (review finding): after a revert to the
    parliament engine, a chassis artifact must equally read stale and
    regenerate — the guard is symmetric."""
    env = _Env()
    cfg = ticker_api._regime_cfg()
    old_engine = cfg["engine"]
    try:
        cfg["engine"] = "parliament"
        env.write(_current_payload())              # chassis-era artifact
        env.kicks.clear()
        r = env.client.get(GUARDED[0])
        assert r.status_code == 503, f"expected 503, got {r.status_code}"
        assert env.wait_for_kick(), "refresh was not kicked"
    finally:
        cfg["engine"] = old_engine
        env.close()
    print("  chassis artifact under parliament engine -> 503 + refresh kick "
          "(reverse direction): OK")


def test_universe_leadership_block():
    """D-007 Phase 2: the Layer-2 GICS leadership block — rank-ordered join
    of selected groups + ranking stats + breaker; defensive on malformed
    artifacts; weeks None on pre-Phase-1 artifacts (honest, no fake history);
    and the artifact serves fine with the new optional block present."""
    from framework.framework_runner import build_universe_leadership
    ua = {"groups": {"Biotechnology": {"tickers": ["C"], "sector": "Health Care"},
                     "Semiconductors": {"tickers": ["A", "B"], "sector": "IT",
                                        "weeks_in_universe": 2}},
          "ranking": [{"name": "Semiconductors", "rank": 1, "composite": 20.5,
                       "median_ytd": 31.2},
                      {"name": "Biotechnology", "rank": 2, "composite": 16.2,
                       "median_ytd": 12.0}]}
    sig = {"groups": [{"name": "Semiconductors", "breaker_status": "clear"}]}
    rows = build_universe_leadership(ua, sig)
    assert [r["name"] for r in rows] == ["Semiconductors", "Biotechnology"]
    assert rows[0]["rank"] == 1 and rows[0]["tickers"] == 2
    assert rows[0]["weeks_in_universe"] == 2
    assert rows[0]["breaker_status"] == "clear"
    assert rows[1]["weeks_in_universe"] is None      # pre-Phase-1 artifact
    assert rows[1]["breaker_status"] is None         # no dashboard row
    # malformed shapes contribute nothing, never raise
    assert build_universe_leadership(None, None) == []
    assert build_universe_leadership({"groups": "bad"}, []) == []
    assert build_universe_leadership({"groups": {"X": None},
                                      "ranking": "bad"}, {"groups": [None]}) == []
    # serving tolerates the optional block
    env = _Env()
    try:
        p = _current_payload()
        p["universe_leadership"] = rows
        env.write(p)
        r = env.client.get(GUARDED[1])
        assert r.status_code == 200
        assert r.get_json()["universe_leadership"][0]["name"] == "Semiconductors"
    finally:
        env.close()
    print("  universe leadership block: rank-ordered join, honest Nones, "
          "malformed-safe, serves: OK")


def test_layer2_ticker_expansion_contract():
    """Layer-2 expandable ticker rows (pure presentation): the page joins
    universe_leadership.name against signals.json groups[].name and renders
    each stock's score/trade_signal/price-vs-ma20/rsi with the dashboard's
    chip vocabulary. Pins the DATA CONTRACT on the real committed artifacts
    (the render is client JS — verified live via DOM + screenshots):
      - the join key works for the current artifacts (most selected groups
        have signals rows; absent groups are the honest-gap path)
      - every stock row carries the fields the sub-row renders
      - every live trade_signal value maps into the dashboard's 6 chip
        classes (the gl2TsClass vocabulary)"""
    import subprocess
    here = os.path.dirname(os.path.abspath(__file__))

    def _head(path):
        out = subprocess.run(["git", "show", f"HEAD:{path}"],
                             capture_output=True, text=True, cwd=here)
        assert out.returncode == 0, path
        return json.loads(out.stdout)

    ua = _head("data/universe_active.json")
    sig = _head("data/signals.json")
    from framework.framework_runner import build_universe_leadership
    lead = build_universe_leadership(ua, sig)
    assert len(lead) >= 10, "leadership block unexpectedly small"
    sig_groups = {g["name"]: g for g in sig.get("groups", [])
                  if isinstance(g, dict) and g.get("name")}
    joined = [g["name"] for g in lead if g["name"] in sig_groups]
    assert len(joined) >= 10, \
        f"join broken: only {len(joined)} of {len(lead)} groups match"

    def ts_class(ts):                    # mirrors the page's gl2TsClass
        if not ts:
            return "ts-hold"
        t = ts.upper()
        if t == "BUY NOW":
            return "ts-buy-now"
        if "WAIT" in t:
            return "ts-wait"
        if "ACCUMULATE" in t:
            return "ts-accumulate"
        if "HOLD" in t:
            return "ts-hold"
        if "REDUCE" in t:
            return "ts-reduce"
        if t == "AVOID":
            return "ts-avoid"
        return "ts-hold"

    known = {"ts-buy-now", "ts-wait", "ts-accumulate", "ts-hold",
             "ts-reduce", "ts-avoid"}
    checked = 0
    for name in joined:
        for s in sig_groups[name].get("stocks", []) or []:
            assert s.get("ticker"), name
            for f in ("score", "price", "ma20", "rsi"):
                assert f in s, (name, s.get("ticker"), f)
            assert ts_class(s.get("trade_signal")) in known, \
                (name, s.get("ticker"), s.get("trade_signal"))
            checked += 1
    assert checked >= 30, f"only {checked} stock rows checked"
    print(f"  Layer-2 expansion contract: {len(joined)}/{len(lead)} groups "
          f"join, {checked} stock rows field-complete, all trade signals "
          f"map to the 6 dashboard chips: OK")


def test_layer2_enrichment_shared_renderers():
    """Layer-2 sub-row enrichment: tracked names inline the watchlist row's
    components via the SAME renderers the Position Signals panel uses.
    Source-structure pin: the enriched row must CALL psBadge/psGrade/psPips/
    psExt, and the page must define each renderer exactly ONCE — a copied
    or forked renderer is a test failure (Lab-law spirit: one implementation).
    Plus the join contract on artifacts: every assessment-tracked ticker
    carries the fields the enrichment renders, and the both-direction join
    stays additive (tracked-not-in-signals resolvable via the row's group)."""
    here = os.path.dirname(os.path.abspath(__file__))
    html = open(os.path.join(here, "public", "framework.html")).read()

    # exactly one definition of each shared renderer
    for fn in ("psBadge", "psGrade", "psPips", "psExt"):
        assert html.count(f"function {fn}(") == 1, \
            f"{fn} defined {html.count(f'function {fn}(')} times — fork!"
    # the enriched sub-row calls all four (no inline re-implementations)
    tick_row = html[html.index("function gl2TickRow"):
                    html.index("async function gl2Fill")]
    for fn in ("psBadge(", "psGrade(", "psPips(", "psExt("):
        assert fn in tick_row, f"gl2TickRow does not call {fn}"
    # no duplicate pip/chip drawing inside the sub-row template
    assert "class=\"pip" not in tick_row, "inline pip markup — use psPips"
    assert "ps-grade-" not in tick_row, "inline grade markup — use psGrade"
    # the join sources: gl2Fill reads ASSESS-tracked rows + signals stocks,
    # additive both directions
    fill = html[html.index("async function gl2Fill"):
                html.index("function gl2Toggle")]
    assert "gl2Tracked()" in fill and "extras" in fill
    assert "x.group===groupName" in fill, "tracked-not-in-signals join leg"

    # artifact contract: tracked watchers carry what the enrichment renders
    fw_path = os.path.join(here, "public", "framework.json")
    fw = json.load(open(fw_path))
    rows = (fw.get("position_signals") or {}).get("tickers") or {}
    watchers = {t: x for t, x in rows.items()
                if isinstance(x, dict) and x.get("kind") == "watching"
                and not x.get("insufficient_data")}
    assert watchers, "no watchers in artifact"
    graded = any(isinstance(x.get("grade"), dict) for x in watchers.values())
    for t, x in watchers.items():
        assert x.get("state"), t
        assert isinstance(x.get("conditions"), dict) and \
            "5_thesis" in x["conditions"], t
        assert "extension_atr" in x, t
        if graded:
            # post-Phase-1 artifact: every watcher must be graded + grouped
            assert isinstance(x.get("grade"), dict) and \
                x["grade"].get("grade"), t
            assert "group" in x, t
    if not graded:
        # pre-Phase-1 committed artifact (before the first post-ship cron):
        # the strict grade/group assertions activate with the next artifact
        print("  (note: artifact predates Phase 1 — grade/group assertions "
              "activate on the next committed run)")
    # the ruled exhibit, asserted when present (self-activates once the
    # artifact includes the 2026-07-18 adds): CRWD tracked with a grade
    if "CRWD" in watchers:
        c = watchers["CRWD"]
        assert c["grade"]["grade"] in ("A+", "B", "C")
        note = f"CRWD tracked: {c['state']} [{c['grade']['grade']}]"
    else:
        note = "CRWD not yet in this artifact (activates after next run)"
    print(f"  Layer-2 enrichment: renderers defined once + all four called, "
          f"join legs present, {len(watchers)} watchers field-complete; "
          f"{note}: OK")


def test_candidates_page_contract():
    """D-017 page law (ruling 11725 Q2/Q3): un-tracked candidate rows get
    the D-011 grade chip + reasons hover through the SAME psGrade renderer
    and a +watch copy-the-prompt affordance — and NOTHING else. Chip-only
    is the visual grammar: no state badge, no pips (a five-pip un-tracked
    row would read as READY). Plus the artifact contract, era-aware."""
    here = os.path.dirname(os.path.abspath(__file__))
    html = open(os.path.join(here, "public", "framework.html")).read()

    # the seam was built — the placeholder comment must be gone
    assert "SEAM (do not build yet)" not in html, "seam comment left behind"

    # candidate branch: chip + watch only, through the shared renderer
    tick_row = html[html.index("function gl2TickRow"):
                    html.index("async function gl2Fill")]
    assert "!tracked&&cand&&cand.grade" in tick_row, \
        "candidate chip must require un-tracked AND a non-null grade"
    cand_branch = tick_row[tick_row.index("!tracked&&cand&&cand.grade"):
                           tick_row.index("return `")]
    assert "psGrade(cand)" in cand_branch, "chip must go through psGrade"
    assert "psPips(" not in cand_branch and "psBadge(" not in cand_branch, \
        "chip-ONLY law: no pips/badge on candidates"
    assert "ps-grade-" not in cand_branch, "inline grade markup — use psGrade"
    assert "gl2-watch" in cand_branch and "cwOpen(" in cand_branch
    # the row TEMPLATE carries the candidate slot exactly once, adjacent
    # to the tracked slot — markup added in the template itself (outside
    # the guarded branch) cannot evade the chip-only law (review finding)
    assert "${cd}${tr}" in tick_row, "candidate slot moved/duplicated"
    assert tick_row.count("${cd}") == 1

    # grades come from the SAME signals artifact as the rows, and only
    # for un-tracked names
    fill = html[html.index("async function gl2Fill"):
                html.index("function gl2Toggle")]
    assert "sig.candidate_grades" in fill
    assert "!tracked[s.ticker])?cands[s.ticker]:null" in fill.replace("\n", "")

    # the lazy fetch caches the PROMISE (review finding: caching the
    # result returned undefined to every concurrent caller after the
    # first — expand-all rendered honest-gap notes for covered rows)
    assert "GL2_SIG_PROMISE" in html and "GL2_FETCHED" not in html

    # the copy-the-prompt appointment (Q3: promotion NEVER automatic; no
    # fake durable write from an ephemeral dyno)
    for needle in ("function cwOpen(", "function cwCopy(",
                   "function cwClose(", 'id="cwModal"', "const GL2CAND"):
        assert needle in html, f"missing {needle}"
    assert "human-curated committed set (D-003)" in html, \
        "the modal must state the honest constraint"
    assert html.count("function cwOpen(") == 1

    # artifact contract (era-aware): when a signals artifact carries the
    # block, every key is un-tracked and every value has the chip's shape
    sig_path = os.path.join(here, "public", "signals.json")
    fw_path = os.path.join(here, "public", "framework.json")
    sig = json.load(open(sig_path))
    fw = json.load(open(fw_path))
    cand = sig.get("candidate_grades")
    if isinstance(cand, dict) and cand:
        rows = (fw.get("position_signals") or {}).get("tickers") or {}
        tracked = set(rows.keys())
        overlap = tracked & set(cand.keys())
        assert not overlap, f"tracked names graded as candidates: {overlap}"
        for t, c in cand.items():
            assert isinstance(c, dict) and "grade" in c and "group" in c, t
            assert c["grade"] in ("A+", "B", "C", None), (t, c["grade"])
            if c["grade"] != "A+":
                assert c.get("reasons"), f"{t}: non-A+ needs reasons (hover)"
        aplus = sorted(t for t, c in cand.items() if c["grade"] == "A+")
        note = (f"{len(cand)} candidates in artifact — {len(aplus)} A+ "
                f"({', '.join(aplus) or 'none'})")
        # the framework artifact carries the same block for notify/API
        assert isinstance(fw.get("candidate_grades"), dict), \
            "signals annotated but framework.json missing the block"
    else:
        note = ("no candidate block yet (pre-emission artifact — "
                "activates on the first post-ship run)")
    print(f"  D-017 candidates page contract: chip-only via psGrade, "
          f"+watch prompt, honest constraint; {note}: OK")


def test_missing_file_still_404():
    env = _Env()
    try:
        r = env.client.get("/api/framework/gauges.json")
        assert r.status_code == 404
        assert env.kicks == []
    finally:
        env.close()
    print("  missing framework.json -> 404 (unchanged), no kick: OK")


if __name__ == "__main__":
    print("\n=== Serve-layer shape sentinel tests ===")
    test_pre_1a_shape_503_and_refresh_kick()
    test_current_shape_200_no_kick()
    test_valid_but_old_200_with_stale_flag()
    test_parliament_artifact_stale_under_chassis()
    test_chassis_artifact_stale_under_parliament()
    test_universe_leadership_block()
    test_layer2_ticker_expansion_contract()
    test_layer2_enrichment_shared_renderers()
    test_candidates_page_contract()
    test_missing_file_still_404()
    print("\nAll serve-guard tests passed.\n")
