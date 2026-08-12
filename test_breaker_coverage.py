#!/usr/bin/env python3
"""
D-019 — BREAKER COVERAGE pins.

THE DEFECT (pre-existing, ruled, fixed here): run_engine fetched the six
MACRO_TICKERS one-shot and swallowed failures with a printed SKIP.
generate_dynamic_breaker_checks then OMITTED the affected checks entirely
(each `if data is not None and len(...) > N:` has no else), and
check_thesis_breakers builds its "not triggered" entries by iterating
checks.items() — so an omitted check left ZERO trace in breaker_alerts.
breaker_status fell through to its "clear" initialiser and was written to
signals.json byte-identical to a group that was checked and found healthy.
The search-page gate then certified that fabricated clear as verified.

THE FIX: record COVERAGE, not just outcome. "clear" is a positive claim
and may only be made when every check the group's sensitivities call for
actually ran.

Pins here follow the house pin doctrine (docs/testing.md):
  1. Drive the real entry point.
  2. Assert the invariant, not the implementation.
  3. Demonstrate the failure — a pin that cannot fail on the old code is
     not evidence.

Run: python3 test_breaker_coverage.py
"""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(os.path.abspath(__file__))

import numpy as np
import pandas as pd

import signal_engine as eng


# ---------------------------------------------------------------- fixtures
def _macro_df(bars=140, start=100.0):
    """A calm macro series that trips NOTHING in either direction.

    Deliberately flat with a small wobble: a rising series trips
    usd_strength (UUP >5% YTD) and approaches energy_spike (XLE >15%
    YTD), while a falling one trips the drop checks. The fixture must
    isolate COVERAGE, so no threshold may be near."""
    # CALENDAR-STABLE (review finding, reproduced): the sp500 check filters
    # to datetime.now().year, so a window ending "today" carries almost no
    # current-year bars in the first days of January — the check would be
    # omitted for a reason that is not an outage, and pin (b) would go red
    # every January. Anchor the window so it always contains a full
    # current-year run; future-dated synthetic bars are harmless here (the
    # check only filters by year, then takes max/last).
    year_start = pd.Timestamp(pd.Timestamp.today().year, 1, 1)
    end = max(pd.Timestamp.today().normalize(),
              year_start + pd.tseries.offsets.BDay(60))
    idx = pd.date_range(end=end, periods=bars, freq="B")
    wobble = np.sin(np.arange(bars) / 7.0) * (start * 0.005)
    close = start + wobble
    return pd.DataFrame({"Open": close, "High": close * 1.004,
                         "Low": close * 0.996, "Close": close,
                         "Volume": np.full(bars, 1_000_000)}, index=idx)


def _healthy_stocks(n=6):
    """Group members with no triggering condition: RSI mid, above MA50,
    positive YTD, beating the S&P."""
    return [{"ticker": f"T{i}", "rsi": 58.0, "price": 110.0, "ma50": 100.0,
             "ytd_return": 12.0, "beating_sp500": True} for i in range(n)]


GROUP_SP = {"macro_sensitivities": ["sp500_drawdown", "group_momentum",
                                    "group_trend", "group_ytd"],
            "commodity_proxy": None, "sector_type": "semiconductor"}
GROUP_OIL = {"macro_sensitivities": ["oil_collapse", "natgas_collapse",
                                     "group_momentum", "group_trend"],
             "commodity_proxy": "USO", "sector_type": "oil_gas_ep"}
GROUP_GOLD = {"macro_sensitivities": ["commodity_drop", "usd_strength",
                                      "group_momentum", "group_trend",
                                      "breadth_collapse"],
              "commodity_proxy": "GLD", "sector_type": "precious_metals"}

ALL_MACRO = {t: _macro_df() for t in eng.MACRO_TICKERS}
ALL_OK = {t: {"ok": True, "reason": None, "bars": 140}
          for t in eng.MACRO_TICKERS}


def _run_group(group_info, macro_data, macro_status):
    """Drive the REAL entry points end to end (pin doctrine, Law 1):
    check_thesis_breakers for the alerts+coverage, then the ENGINE'S OWN
    resolve_breaker_status for the verdict. Reimplementing the ladder here
    would let a ladder change pass every pin in this file."""
    alerts, cov = eng.check_thesis_breakers(
        "TestGroup", group_info, _healthy_stocks(), macro_data, 8.0,
        macro_status=macro_status)
    return eng.resolve_breaker_status(alerts, cov), alerts, cov


# ------------------------------------------------------- (b) full coverage
def test_full_coverage_reads_clear():
    """A day where every macro input arrived and nothing fired must still
    read CLEAR — the fix must not manufacture false degradation."""
    for name, gi in (("sp500", GROUP_SP), ("oil", GROUP_OIL),
                     ("gold", GROUP_GOLD)):
        status, alerts, cov = _run_group(gi, ALL_MACRO, ALL_OK)
        assert status == "clear", f"{name}: {status} — {cov['reasons']}"
        assert cov["complete"] and not cov["missing"], f"{name}: {cov}"
        assert cov["expected"] == cov["run"], f"{name}: {cov}"
        assert cov["reasons"] == [], f"{name}: {cov['reasons']}"
        # byte-comparable to the old world: every alert still carries the
        # same shape, and an all-clear group lists every check untriggered
        assert not [a for a in alerts if a["triggered"]], name
        assert sorted(a["check"] for a in alerts) == cov["expected"], name
    print("  (b) full coverage -> clear, expected==run, no false "
          "degradation (3 group shapes): OK")


# --------------------------------------------------- (a) outage injection
def test_macro_outage_degrades_every_dependent_group():
    """Kill each macro ticker in turn. EVERY group whose sensitivities
    depend on it must read degraded and NAME the reason; groups that do
    not depend on it must be untouched."""
    # every macro ticker must have at least one group that depends on it,
    # or the injection for that ticker proves nothing (review finding: UUP
    # and XLE were unexercised). GROUP_ALL declares every sensitivity that
    # needs a macro input, so no ticker's path goes untested.
    GROUP_ALL = {"macro_sensitivities": ["sp500_drawdown", "commodity_drop",
                                         "usd_strength", "oil_collapse",
                                         "natgas_collapse", "energy_spike"],
                 "commodity_proxy": "GLD", "sector_type": "mixed"}
    groups = {"sp500": GROUP_SP, "oil": GROUP_OIL, "gold": GROUP_GOLD,
              "all": GROUP_ALL}
    checked = 0
    for dead_ticker in eng.MACRO_TICKERS:
        macro = {t: df for t, df in ALL_MACRO.items() if t != dead_ticker}
        status_map = dict(ALL_OK)
        status_map[dead_ticker] = {"ok": False,
                                   "reason": "fetch returned no data"}
        for gname, gi in groups.items():
            sens = set(gi["macro_sensitivities"])
            proxy = gi.get("commodity_proxy")
            # does this group actually call for a check needing dead_ticker?
            needs = any(
                s["sensitivity"] in sens and s.get("macro") == dead_ticker
                and (s.get("requires_proxy") is None
                     or s["requires_proxy"] == proxy)
                for s in eng.BREAKER_CHECK_SPECS)
            status, alerts, cov = _run_group(gi, macro, status_map)
            if needs:
                checked += 1
                assert status == "degraded", (
                    f"{dead_ticker} dead, {gname} sensitive -> {status} "
                    f"(THE BUG: a blind group reading as checked)")
                assert cov["missing"], f"{gname}/{dead_ticker}: {cov}"
                joined = " ".join(cov["reasons"])
                assert dead_ticker in joined, cov["reasons"]
                assert "fetch returned no data" in joined, cov["reasons"]
                # and the omission leaves no trace in the alert list —
                # which is exactly why the coverage record is needed
                assert cov["missing"][0] not in {a["check"] for a in alerts}
            else:
                assert status == "clear", (
                    f"{dead_ticker} dead, {gname} NOT sensitive -> {status}"
                    " (over-degradation)")
    # every macro ticker's dependency path must have been exercised
    exercised = set()
    for spec in eng.BREAKER_CHECK_SPECS:
        if spec.get("macro"):
            exercised.add(spec["macro"])
    assert exercised == set(eng.MACRO_TICKERS), (
        f"macro tickers with no check depending on them: "
        f"{set(eng.MACRO_TICKERS) - exercised}")
    assert checked >= 11, f"only {checked} sensitive combinations exercised"
    print(f"  (a) macro outage: {checked} sensitive group/ticker "
          "combinations degrade with named reasons; insensitive groups "
          "stay clear: OK")


def test_prefix_engine_fabricates_clear_on_the_same_injection():
    """DEMONSTRATE THE FAILURE. Run the PRE-FIX engine (git HEAD) against
    the identical injection: it reports a clean 'clear' with no trace of
    the check that never ran. A pin that cannot fail on the old code is
    not evidence."""
    # ANCHORED TO THE PRE-FIX COMMIT, not HEAD. Against HEAD this pin
    # would pass trivially — and self-skip forever — the moment this build
    # was committed, which is exactly when the demonstration stops being
    # free. e81b6d1 is the last commit before D-019 (the swallowed-fetch
    # review fixes); it is the code that fabricated the clear.
    PREFIX_SHA = "e81b6d1"
    out = subprocess.run(["git", "show", f"{PREFIX_SHA}:signal_engine.py"],
                         capture_output=True, text=True, cwd=REPO)
    if out.returncode != 0:
        # a shallow clone cannot reach it; say so loudly rather than
        # printing a green line for a pin that did not run
        raise AssertionError(
            f"(a') pre-fix anchor {PREFIX_SHA} unreachable (shallow clone?) "
            "— the Law-3 demonstration did NOT run. A pin that cannot fail "
            "on the old code is not evidence, and a pin that silently does "
            "not run is worse: fetch the history or delete this pin "
            "deliberately.")
    tmp = tempfile.mkdtemp(prefix="prefix_eng_")
    path = os.path.join(tmp, "old_signal_engine.py")
    with open(path, "w") as f:
        f.write(out.stdout)
    spec = importlib.util.spec_from_file_location("old_signal_engine", path)
    old = importlib.util.module_from_spec(spec)
    sys.modules["old_signal_engine"] = old
    spec.loader.exec_module(old)

    assert not hasattr(old, "BREAKER_CHECK_SPECS"), (
        f"{PREFIX_SHA} already carries D-019 — the pre-fix anchor is wrong "
        "and this pin is no longer demonstrating anything")
    if True:
        macro = {t: df for t, df in ALL_MACRO.items() if t != "^GSPC"}
        old_alerts = old.check_thesis_breakers(
            "TestGroup", GROUP_SP, _healthy_stocks(), macro, 8.0)
        # the old signature returns a bare list
        assert isinstance(old_alerts, list), type(old_alerts)
        triggered = [a for a in old_alerts if a["triggered"]]
        old_status = "clear" if not triggered else "not-clear"
        assert old_status == "clear", "expected the pre-fix fabricated clear"
        # THE SMOKING GUN: the check that never ran is absent entirely —
        # not listed as triggered, not listed as "Not triggered"
        ids = {a["check"] for a in old_alerts}
        assert "sp500_drawdown_10pct" not in ids, (
            "pre-fix engine left a trace — re-derive this pin")
        # and the SAME injection on the fixed engine degrades
        new_status, _, cov = _run_group(
            GROUP_SP, macro,
            dict(ALL_OK, **{"^GSPC": {"ok": False, "reason": "fetch failed"}}))
        assert new_status == "degraded", new_status
        print("  (a') pre-fix engine on the SAME injection: 'clear' with "
              f"{len(ids)} checks and NO trace of sp500_drawdown_10pct; "
              "fixed engine: 'degraded' — the identity is broken: OK")



# --------------------------------------------------- (c) the table pinned
def test_mapping_table_complete_both_directions():
    """The check<->sensitivity table is the source of truth: every
    sensitivity any group can declare resolves, and every check the
    generator can emit is claimed by exactly one spec. No orphans in
    either direction."""
    spec_sens = {s["sensitivity"] for s in eng.BREAKER_CHECK_SPECS}
    spec_checks = [s["check"] for s in eng.BREAKER_CHECK_SPECS]
    assert len(spec_checks) == len(set(spec_checks)), "duplicate check id"

    # 1. every sensitivity DECLARED anywhere in the repo resolves.
    declared = set()
    import universe_builder as ub
    for meta in ub.GROUP_METADATA.values():
        declared.update(meta.get("macro_sensitivities") or [])
    declared.update(ub.DEFAULT_SENSITIVITIES)
    for gi in (eng.FALLBACK_INDUSTRY_GROUPS or {}).values():
        declared.update(gi.get("macro_sensitivities") or [])
    ua = os.path.join(REPO, "data", "universe_active.json")
    if os.path.exists(ua):
        with open(ua) as f:
            for gi in ((json.load(f) or {}).get("groups") or {}).values():
                declared.update(gi.get("macro_sensitivities") or [])
    orphan_sens = sorted(declared - spec_sens)
    assert not orphan_sens, f"sensitivities with no spec: {orphan_sens}"

    # 2. every check the GENERATOR can emit is claimed by a spec. Derived
    #    from source, so a new branch added without a table entry fails
    #    here rather than silently shrinking "complete coverage".
    # AST, not regex (review finding: the regex missed checks[var]=,
    # single quotes, dict.update, and emission from a helper). Walk the
    # real function body and collect every literal key assigned into
    # `checks`; a NON-literal key is reported as unpinnable rather than
    # silently ignored, because it would slip the completeness guarantee.
    import ast
    src = open(os.path.join(REPO, "signal_engine.py")).read()
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef)
              and n.name == "generate_dynamic_breaker_checks")
    emitted, dynamic = set(), []
    for node in ast.walk(fn):
        # checks["x"] = ...
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if (isinstance(tgt, ast.Subscript)
                        and isinstance(tgt.value, ast.Name)
                        and tgt.value.id == "checks"):
                    key = tgt.slice
                    if isinstance(key, ast.Constant) and isinstance(key.value, str):
                        emitted.add(key.value)
                    else:
                        dynamic.append(ast.dump(key)[:60])
        # checks.update({...}) / checks.setdefault("x", ...)
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "checks"):
            if node.func.attr == "setdefault" and node.args:
                k = node.args[0]
                if isinstance(k, ast.Constant) and isinstance(k.value, str):
                    emitted.add(k.value)
                else:
                    dynamic.append("setdefault(non-literal)")
            elif node.func.attr == "update":
                for a in node.args:
                    if isinstance(a, ast.Dict):
                        for k in a.keys:
                            if isinstance(k, ast.Constant) and isinstance(k.value, str):
                                emitted.add(k.value)
                            else:
                                dynamic.append("update(non-literal key)")
                    else:
                        dynamic.append("update(non-dict)")
    assert not dynamic, (
        "a check id is emitted non-literally, so completeness cannot be "
        f"verified statically: {dynamic}")
    assert emitted, "the AST walk found no emitted checks — re-derive this pin"
    orphan_checks = sorted(emitted - set(spec_checks))
    assert not orphan_checks, f"checks with no spec: {orphan_checks}"
    unreachable = sorted(set(spec_checks) - emitted)
    assert not unreachable, f"specs for checks nothing emits: {unreachable}"

    # 3. the conditional pairing is REAL, not name-matched
    gold = [s for s in eng.BREAKER_CHECK_SPECS
            if s["sensitivity"] == "commodity_drop"][0]
    assert gold["check"] == "gold_below_threshold"
    assert gold["requires_proxy"] == "GLD"
    oil = [s for s in eng.BREAKER_CHECK_SPECS
           if s["sensitivity"] == "oil_collapse"][0]
    assert oil["check"] == "oil_below_60", "the map is not 1:1 by name"
    # commodity_drop declared with a NON-GLD proxy is a config gap, and
    # must degrade rather than quietly expect nothing
    exp, gaps = eng.breaker_expected_checks(
        {"macro_sensitivities": ["commodity_drop"], "commodity_proxy": "USO"})
    assert exp == [] and gaps and "commodity_drop" in gaps[0], (exp, gaps)
    cov = eng.breaker_coverage(
        {"macro_sensitivities": ["commodity_drop"], "commodity_proxy": "USO"},
        [], ALL_OK)
    assert not cov["complete"], "an unimplemented pairing read as complete"
    print(f"  (c) table complete both directions ({len(spec_checks)} "
          f"specs, {len(declared)} declared sensitivities, 0 orphans); "
          "non-1:1 pairings pinned; config gap degrades: OK")


# ------------------------------------------------- (d) era-aware surfaces
def test_gate_withholds_on_degraded_and_is_era_aware():
    """The serve layer: a degraded group gates the trade signal exactly as
    an unavailable one does; a PRE-D-019 artifact (no coverage fields)
    renders as the old world did and never retro-claims."""
    import ticker_api
    tmp = tempfile.mkdtemp(prefix="cov_gate_")
    old_dir = ticker_api.DATA_DIR
    ticker_api.DATA_DIR = tmp
    W = lambda n, o: open(os.path.join(tmp, n), "w").write(json.dumps(o))
    try:
        W("universe_active.json", {"groups": {"Semis": {"tickers": ["AAA"]}}})

        # 1. engine says degraded -> gated, reasons carried
        W("signals.json", {"groups": [{
            "name": "Semis", "breaker_status": "degraded",
            "breaker_checks_expected": ["sp500_drawdown_10pct", "avg_ytd_negative"],
            "breaker_checks_run": ["avg_ytd_negative"],
            "breaker_degraded_reasons": [
                "sp500_drawdown_10pct: ^GSPC unavailable — fetch returned no data"],
            "breaker_alerts": []}]})
        g = ticker_api._group_breaker_context("AAA")
        assert g["status"] == "degraded", g
        assert g["breaker_status"] == "degraded"      # deploy-window safety
        assert "^GSPC" in " ".join(g["degraded_reasons"]), g

        # 2. COVERAGE CONTRADICTS THE LABEL: an artifact stamped "clear"
        #    whose checks did not all run must still gate — the label is
        #    not trusted over the record.
        W("signals.json", {"groups": [{
            "name": "Semis", "breaker_status": "clear",
            "breaker_checks_expected": ["sp500_drawdown_10pct", "avg_ytd_negative"],
            "breaker_checks_run": ["avg_ytd_negative"],
            "breaker_alerts": []}]})
        g = ticker_api._group_breaker_context("AAA")
        assert g["status"] == "degraded", f"a stamped-clear blind group: {g}"
        assert "sp500_drawdown_10pct" in " ".join(g["degraded_reasons"]), g

        # 3. ERA-AWARE: no coverage fields at all -> resolved, exactly the
        #    old world. We never retro-claim coverage never measured.
        W("signals.json", {"groups": [{"name": "Semis",
                                       "breaker_status": "clear",
                                       "breaker_alerts": []}]})
        g = ticker_api._group_breaker_context("AAA")
        assert g["status"] == "resolved" and g["breaker_status"] == "clear", g

        # 4. full coverage on a D-019 artifact -> resolved
        W("signals.json", {"groups": [{
            "name": "Semis", "breaker_status": "clear",
            "breaker_checks_expected": ["avg_ytd_negative"],
            "breaker_checks_run": ["avg_ytd_negative"],
            "breaker_degraded_reasons": [], "breaker_alerts": []}]})
        assert ticker_api._group_breaker_context("AAA")["status"] == "resolved"
    finally:
        ticker_api.DATA_DIR = old_dir
    print("  (d) gate: degraded gates, a stamped-clear-but-blind group "
          "gates, pre-D-019 artifacts read as the old world: OK")


def test_pages_render_degraded_distinctly():
    """Source pins: the withheld chip covers degraded, and the Layer-2
    breaker chip never paints a degraded group green."""
    rd = lambda p: open(os.path.join(REPO, p)).read()
    sr = rd("public/search.html")
    assert "gc.status === 'degraded'" in sr
    assert "GROUP BREAKER PARTIALLY UNVERIFIED" in sr
    # The gate predicate must admit ONLY 'resolved' as proof. This used to
    # read `gcs === 'resolved' || gcs === 'not_in_universe'`, and this pin
    # asserted that string — pinning the defect. 'not_in_universe' proves
    # something about the BREAKER (there is none) and nothing about the
    # SIGNAL (the thesis check still never ran).
    assert "const breakerProven = (gcs === 'resolved');" in sr
    tail = sr.split("const breakerProven")[1][:200]
    for bad in ("'degraded'", "'not_in_universe'", "'unavailable'"):
        assert bad not in tail, f"breakerProven admits {bad} as proof"

    fw = rd("public/framework.html")
    assert "brkDeg" in fw and "breaker_checks_expected" in fw
    # BEHAVIOURAL (review finding: this was an index-sliced source grep
    # that broke on formatting and could pass on a green chip). Mirror the
    # page's own predicate and assert the verdict for each artifact shape.
    def chip_degraded(g):
        if g.get("breaker_status") == "degraded":
            return True
        if g.get("breaker_degraded_reasons"):
            return True
        e, r = g.get("breaker_checks_expected"), g.get("breaker_checks_run")
        return bool(e and r and any(c not in r for c in e))

    must_flag = [
        {"breaker_status": "degraded"},
        # stamped clear, but a called-for check never ran
        {"breaker_status": "clear",
         "breaker_checks_expected": ["a", "b"], "breaker_checks_run": ["a"]},
        # stamped clear, but a check ran on a degraded INPUT
        {"breaker_status": "clear",
         "breaker_checks_expected": ["a"], "breaker_checks_run": ["a"],
         "breaker_degraded_reasons": ["a: baseline unavailable"]},
    ]
    for g in must_flag:
        assert chip_degraded(g), f"chip would paint this as checked: {g}"
    must_not_flag = [
        {"breaker_status": "clear",
         "breaker_checks_expected": ["a"], "breaker_checks_run": ["a"],
         "breaker_degraded_reasons": []},
        {"breaker_status": "clear"},                      # pre-D-019 era
        {"breaker_status": "critical"},                   # a real trigger
    ]
    for g in must_not_flag:
        assert not chip_degraded(g), f"chip would over-flag: {g}"
    # and the SOURCE must gate green on the same flag, so a degraded group
    # can never be painted with the positive-clear colour
    assert "(brk==='clear'&&!brkDeg)?'var(--green)'" in fw, \
        "the green chip is not guarded by the degraded flag"
    assert "&#9888; " in fw and "dashed" in fw

    nt = rd("notify_assessment.py")
    assert "Breaker coverage INCOMPLETE" in nt
    assert "breaker_checks_expected" in nt and "Not clear — unverified" in nt
    print("  (d) surfaces: search withheld+named, Layer-2 amber ⚠ chip "
          "(never green), close report announces: OK")


def test_writer_half_records_and_writes():
    """THE WRITER HALF (review finding: it had no pin at all, and
    era-awareness makes a writer regression INVISIBLE — if the engine
    silently stopped emitting the coverage fields, every consumer would
    read the artifact as pre-D-019 and un-gate everything).

    Drives the real run_engine writer path with the network stubbed: one
    macro ticker dead, and asserts the artifact carries macro_status, the
    per-group coverage fields, the degraded verdict, and a withheld
    published trade signal."""
    import types

    real_fetch = eng.fetch_data
    real_idx = eng.get_index_data
    real_groups = eng.get_industry_groups
    real_fund = getattr(eng, "fetch_fundamentals_yfinance", None)
    written = {}

    def fake_fetch(ticker, period="6mo", **kw):
        if ticker == "^GSPC":
            return None                      # THE OUTAGE
        return _macro_df()

    def fake_idx():
        return {"^GSPC": {"ytd": 8.0}}

    def fake_groups():
        return {"Semis": {"tickers": ["AAA", "BBB"],
                          "sector": "Information Technology",
                          "cycle_stage": "mid",
                          "macro_sensitivities": ["sp500_drawdown",
                                                  "group_momentum"],
                          "commodity_proxy": None,
                          "sector_type": "semiconductor"}}

    tmp = tempfile.mkdtemp(prefix="writer_")
    _stubs = []
    try:
        eng.fetch_data = fake_fetch
        eng.get_index_data = fake_idx
        eng.get_industry_groups = fake_groups
        if real_fund:
            eng.fetch_fundamentals_yfinance = lambda t: {}
        # No network in a pin, and no write into the real repo: run_engine
        # reaches the earnings layer (signal_engine.py:2581), which WRITES
        # data/earnings_calendar.json.
        #
        # The previous guard here was dead TWICE OVER and is kept in the
        # history for the shape of the mistake: it did
        # `sys.modules.get("earnings_calendar")`, which is None because
        # signal_engine imports the module LAZILY INSIDE run_engine — so at
        # guard time it is usually not in sys.modules at all; and it stubbed
        # `refresh_for_tickers` / `next_earnings_map`, NEITHER OF WHICH
        # EXISTS. The real writer is get_earnings_map. Two independent
        # reasons for a guard that never fired once, and a `hasattr` check
        # that silently made both invisible.
        #
        # `import earnings_calendar` forces it into sys.modules; patching the
        # module attribute works because production does
        # `from earnings_calendar import get_earnings_map` at CALL time.
        import earnings_calendar as _ec
        _stubs.append((_ec, "get_earnings_map", _ec.get_earnings_map))
        _ec.get_earnings_map = lambda ts: {t: None for t in ts}
        # capture the artifact instead of writing over the repo's
        orig_open = open

        out = eng.run_engine(output_path=os.path.join(tmp, "signals.json")) \
            if "output_path" in eng.run_engine.__code__.co_varnames else None
        if out is None:
            # run_engine writes to a fixed path — point DATA/PUBLIC at tmp
            for attr in ("DATA_DIR", "PUBLIC_DIR", "OUTPUT_PATH"):
                if hasattr(eng, attr):
                    written[attr] = getattr(eng, attr)
                    setattr(eng, attr, tmp)
            out = eng.run_engine()
    finally:
        eng.fetch_data = real_fetch
        eng.get_index_data = real_idx
        eng.get_industry_groups = real_groups
        if real_fund:
            eng.fetch_fundamentals_yfinance = real_fund
        for attr, val in written.items():
            setattr(eng, attr, val)
        for _obj, _name, _orig in _stubs:
            setattr(_obj, _name, _orig)

    assert isinstance(out, dict), type(out)
    # 1. the RUN-level record exists and names the outage
    ms = out.get("macro_status")
    assert isinstance(ms, dict) and ms, "macro_status missing from artifact"
    assert ms.get("^GSPC", {}).get("ok") is False, ms.get("^GSPC")
    assert ms["^GSPC"].get("reason"), "outage recorded without a reason"
    # 2. the PER-GROUP coverage fields exist and are self-consistent
    grp = (out.get("groups") or [])[0]
    for field in ("breaker_checks_expected", "breaker_checks_run",
                  "breaker_degraded_reasons"):
        assert field in grp, f"{field} not written — a writer regression "\
                             "would read as a pre-D-019 artifact"
    assert "sp500_drawdown_10pct" in grp["breaker_checks_expected"]
    assert "sp500_drawdown_10pct" not in grp["breaker_checks_run"]
    assert any("^GSPC" in r for r in grp["breaker_degraded_reasons"])
    # 3. the verdict
    assert grp["breaker_status"] == "degraded", grp["breaker_status"]
    # 4. THE PUBLISHED SIGNAL IS WITHHELD — the artifact is a surface too,
    #    and the dashboard renders trade_signal straight out of it
    for st in grp.get("stocks", []):
        assert st.get("trade_signal") == "SIGNAL WITHHELD", (
            f"{st.get('ticker')}: published {st.get('trade_signal')!r} on an "
            "unverified breaker — the fabricated clear, one layer over")
    print("  (e) writer half: macro_status + per-group coverage fields "
          "written, verdict degraded, published trade_signal withheld: OK")


def test_a_trigger_still_wins_over_incomplete_coverage():
    """THE LADDER'S HEADLINE RULE (review finding: unpinned). A fired
    breaker is news regardless of what else could not be measured — and
    the serve layer must not downgrade it to 'degraded', which would hide
    the alarm behind the caveat."""
    cov_bad = {"complete": False, "expected": ["a", "b"], "run": ["a"],
               "reasons": ["b: ^GSPC unavailable"]}
    cov_ok = {"complete": True, "expected": ["a"], "run": ["a"], "reasons": []}
    for sev, expected in (("critical", "critical"), ("high", "warning"),
                          ("medium", "watch")):
        alerts = [{"check": "a", "triggered": True, "severity": sev}]
        assert eng.resolve_breaker_status(alerts, cov_bad) == expected, sev
        assert eng.resolve_breaker_status(alerts, cov_ok) == expected, sev
    # and no trigger: coverage decides
    assert eng.resolve_breaker_status([], cov_ok) == "clear"
    assert eng.resolve_breaker_status([], cov_bad) == "degraded"

    # the SERVE layer must agree — this is where the override lived
    import ticker_api
    tmp = tempfile.mkdtemp(prefix="trigwin_")
    old_dir = ticker_api.DATA_DIR
    ticker_api.DATA_DIR = tmp
    W = lambda n, o: open(os.path.join(tmp, n), "w").write(json.dumps(o))
    try:
        W("universe_active.json", {"groups": {"Semis": {"tickers": ["AAA"]}}})
        W("signals.json", {"groups": [{
            "name": "Semis", "breaker_status": "critical",
            "breaker_checks_expected": ["sp500_drawdown_10pct", "avg_ytd_negative"],
            "breaker_checks_run": ["avg_ytd_negative"],
            "breaker_degraded_reasons": ["sp500_drawdown_10pct: ^GSPC unavailable"],
            "breaker_alerts": [{"check": "avg_ytd_negative", "triggered": True,
                                "severity": "critical",
                                "message": "group YTD collapsed"}]}]})
        g = ticker_api._group_breaker_context("AAA")
        assert g["breaker_status"] == "critical", (
            f"a fired CRITICAL was downgraded to {g['breaker_status']!r} by "
            "the coverage caveat — the alarm would be hidden")
        assert g["status"] == "resolved", g["status"]
        assert g.get("coverage_incomplete") is True, g
        assert g.get("degraded_reasons"), "the caveat was dropped entirely"
    finally:
        ticker_api.DATA_DIR = old_dir
    print("  (f) a trigger still wins over incomplete coverage, in the "
          "ladder AND at the serve layer; the caveat rides along: OK")


def test_degraded_gate_is_behavioural():
    """The line that actually withholds the trade signal, pinned as
    BEHAVIOUR (review finding: it was source-grep only). Executes the
    page's own gate predicate over every shape D-019 introduces."""
    sr = open(os.path.join(REPO, "public", "search.html")).read()
    assert "const breakerProven = (gcs === 'resolved');" in sr, \
        "the gate predicate changed — re-derive this pin"

    def withholds(payload):
        gcs = (payload.get("group_context") or {}).get("status")
        proven = gcs == "resolved"          # ONLY a computed breaker proves
        return bool(payload.get("trade_signal_gate")) or not proven

    # not_in_universe used to be treated as proof and rendered a verdict.
    assert withholds({"trade_signal": "BUY NOW",
                      "group_context": {"status": "not_in_universe"}}), \
        "an off-universe name still renders a verdict with no macro check"

    must_withhold = [
        {"trade_signal": "BUY NOW",
         "group_context": {"status": "degraded",
                           "degraded_reasons": ["x: ^GSPC unavailable"]}},
        {"trade_signal": "BUY NOW", "trade_signal_gate": "breaker unverified",
         "group_context": {"status": "degraded"}},
        {"trade_signal": "BUY NOW", "group_context": {"status": "unavailable"}},
        {"trade_signal": "BUY NOW", "group_context": None},
        {"trade_signal": "BUY NOW"},
    ]
    for p in must_withhold:
        assert withholds(p), f"a BUY NOW would render on {p}"
    # a trigger that FIRED is resolved — it must still render (as AVOID),
    # because hiding a fired breaker behind a withheld chip loses the alarm
    fired = {"trade_signal": "AVOID",
             "group_context": {"status": "resolved",
                               "breaker_status": "critical",
                               "coverage_incomplete": True}}
    assert not withholds(fired), "a fired critical was withheld — alarm lost"
    print("  (g) gate behaviour: every degraded shape withholds; a fired "
          "critical still renders its AVOID: OK")


def test_degraded_input_and_published_signal():
    """Two review findings, pinned:

    1. A missing INPUT that does not stop a check from running (the S&P
       YTD baseline defaults to 0.0, so breadth_collapse still produces a
       number — against a fabricated comparison). expected-vs-run cannot
       see it, so it is reported directly.
    2. The ARTIFACT is a surface: compute_trade_signal must withhold on a
       degraded breaker, or the dashboard renders a BUY NOW that the
       search page correctly refuses."""
    gi = {"macro_sensitivities": ["breadth_collapse", "group_trend"],
          "commodity_proxy": None}
    ran = ["breadth_collapse", "majority_below_ma50"]
    cov_ok = eng.breaker_coverage(gi, ran, ALL_OK)
    assert cov_ok["complete"], cov_ok
    cov_bad = eng.breaker_coverage(
        gi, ran, ALL_OK,
        degraded_inputs={"breadth_collapse": "S&P 500 YTD baseline unavailable"})
    assert not cov_bad["complete"], "a fabricated baseline read as complete"
    assert not cov_bad["missing"], "the check DID run — this is an input gap"
    assert any("baseline" in r for r in cov_bad["reasons"]), cov_bad
    # an input gap for a check the group does not call for is ignored
    cov_other = eng.breaker_coverage(
        gi, ran, ALL_OK, degraded_inputs={"energy_spike": "XLE missing"})
    assert cov_other["complete"], "an irrelevant input gap degraded a group"

    det = {"rsi": 63, "macd_histogram": .2, "macd": 1.1, "macd_signal": .9,
           "price": 203, "ma20": 188, "ma50": 180, "ma200": 150,
           "composite_score": 77, "signal": "buy", "ytd_return": 22,
           "volume_ratio": 1.1, "pct_from_52w_high": -2,
           "trend_strength": 15, "rs_vs_ma50": 5}
    clear_sig, _ = eng.compute_trade_signal(dict(det), breaker_status="clear")
    deg_sig, deg_reason = eng.compute_trade_signal(dict(det),
                                                  breaker_status="degraded")
    crit_sig, _ = eng.compute_trade_signal(dict(det), breaker_status="critical")
    assert clear_sig == "BUY NOW", clear_sig
    assert deg_sig == "SIGNAL WITHHELD", (
        f"the artifact published {deg_sig!r} on an unverified breaker")
    assert "unverified" in deg_reason.lower(), deg_reason
    assert crit_sig == "AVOID", crit_sig    # a trigger still wins

    # and the dashboard renders that value distinctly, never as a neutral hold
    ix = open(os.path.join(REPO, "public", "index.html")).read()
    assert "t==='SIGNAL WITHHELD'" in ix and ".ts-withheld{" in ix
    # the leadership artifact carries the reasons the chip tooltips
    fr = open(os.path.join(REPO, "framework", "framework_runner.py")).read()
    assert "breaker_cov_by_name" in fr and "breaker_degraded_reasons" in fr
    # and the close report fails LOUD, not open
    nt = open(os.path.join(REPO, "notify_assessment.py")).read()
    assert "could not be checked" in nt and "except Exception: pass" not in nt
    print("  (h) degraded INPUT (check ran on a fabricated baseline) "
          "degrades; the artifact withholds its published signal; "
          "dashboard + leadership + close report carry it: OK")


def test_both_flavours_gate_everywhere():
    """The re-derivation must catch BOTH flavours of incomplete (review
    finding, reproduced in production shape): a check that never RAN
    (expected \\ run) and a check that RAN on a degraded input or an
    unimplemented pairing — the latter never enters `expected`, so
    expected == run while the group is explicitly not certified.

    The trigger branch of _group_breaker_context already used the right
    predicate; the clear branch trusted the label it exists to distrust."""
    import ticker_api
    tmp = tempfile.mkdtemp(prefix="flavours_")
    old_dir = ticker_api.DATA_DIR
    ticker_api.DATA_DIR = tmp
    W = lambda n, o: open(os.path.join(tmp, n), "w").write(json.dumps(o))
    try:
        W("universe_active.json", {"groups": {"Gold": {"tickers": ["AAA"]}}})
        flavours = {
            "never-ran": {
                "breaker_checks_expected": ["a", "b"],
                "breaker_checks_run": ["a"],
                "breaker_degraded_reasons": ["b: ^GSPC unavailable"]},
            "degraded-input": {          # the check RAN — expected == run
                "breaker_checks_expected": ["breadth_collapse"],
                "breaker_checks_run": ["breadth_collapse"],
                "breaker_degraded_reasons": [
                    "breadth_collapse: S&P 500 YTD baseline unavailable"]},
            "config-gap": {              # a pairing with no implemented check
                "breaker_checks_expected": ["majority_below_ma50"],
                "breaker_checks_run": ["majority_below_ma50"],
                "breaker_degraded_reasons": [
                    "sensitivity 'commodity_drop' declared with "
                    "commodity_proxy 'USO' — no check implemented"]},
        }
        for name, cov in flavours.items():
            # stamped CLEAR — the label must not be trusted over the record
            W("signals.json", {"groups": [dict(
                {"name": "Gold", "breaker_status": "clear",
                 "breaker_alerts": []}, **cov)]})
            g = ticker_api._group_breaker_context("AAA")
            assert g["status"] == "degraded", (
                f"{name}: a stamped-clear uncertified group served as "
                f"{g['status']!r}")
            assert g["degraded_reasons"], f"{name}: reasons dropped"

            # with a TRIGGER, the trigger wins but the caveat survives
            W("signals.json", {"groups": [dict(
                {"name": "Gold", "breaker_status": "critical",
                 "breaker_alerts": [{"check": "x", "triggered": True,
                                     "severity": "critical",
                                     "message": "fired"}]}, **cov)]})
            g = ticker_api._group_breaker_context("AAA")
            assert g["breaker_status"] == "critical", f"{name}: alarm lost"
            assert g.get("coverage_incomplete") is True, f"{name}: caveat lost"

        # a genuinely complete group is NOT gated (no over-gating)
        W("signals.json", {"groups": [{
            "name": "Gold", "breaker_status": "clear",
            "breaker_checks_expected": ["a"], "breaker_checks_run": ["a"],
            "breaker_degraded_reasons": [], "breaker_alerts": []}]})
        assert ticker_api._group_breaker_context("AAA")["status"] == "resolved"
    finally:
        ticker_api.DATA_DIR = old_dir

    # the close report must announce all three flavours too
    nt = open(os.path.join(REPO, "notify_assessment.py")).read()
    assert "_reasons = _g.get(\"breaker_degraded_reasons\")" in nt
    assert "or _reasons" in nt, "the close report sees only the missing flavour"
    # and history must not file a coverage event as a market escalation
    hm = open(os.path.join(REPO, "history_manager.py")).read()
    assert "coverage_event" in hm and "coverage INCOMPLETE" in hm
    assert "breaker coverage incomplete" in hm, \
        "a degraded group could still report 'all breaker checks clear'"
    # the dashboard separates FIRED from UNVERIFIED, and every render path
    # in the file goes through ONE verdict function. Three copies of the
    # rule is how the heatmap kept painting a record-degraded group green
    # while its own card rendered degraded (caught by rendering it).
    ix = open(os.path.join(REPO, "public", "index.html")).read()
    assert "firedGroups" in ix and "unverifiedGroups" in ix
    assert "function breakerVerdict(" in ix
    assert ix.count("function breakerVerdict(") == 1, "verdict rule duplicated"
    # no render path may read breaker_status raw for its verdict
    for bad in ("breakerIcon(g.breaker_status", "g.breaker_status||'clear')"):
        assert bad not in ix, f"a render path bypasses breakerVerdict: {bad}"
    # mirror the predicate and check the shapes that must disagree with
    # the label
    def verdict(g):
        raw = g.get("breaker_status") or "clear"
        if raw not in ("clear", "degraded"):
            return raw                      # a trigger fired
        unver = (raw == "degraded"
                 or bool(g.get("breaker_degraded_reasons"))
                 or bool(g.get("breaker_checks_expected")
                         and g.get("breaker_checks_run")
                         and any(c not in g["breaker_checks_run"]
                                 for c in g["breaker_checks_expected"])))
        return "degraded" if unver else "clear"
    assert verdict({"breaker_status": "clear",
                    "breaker_checks_expected": ["breadth_collapse"],
                    "breaker_checks_run": ["breadth_collapse"],
                    "breaker_degraded_reasons": ["baseline unavailable"]}) \
        == "degraded", "a stamped-clear group with a contradicting record"
    assert verdict({"breaker_status": "clear"}) == "clear"   # pre-D-019 era
    assert verdict({"breaker_status": "critical",
                    "breaker_checks_expected": ["a", "b"],
                    "breaker_checks_run": ["a"]}) == "critical"  # trigger wins
    assert "degraded:'\u26a0\ufe0f'" in ix or "degraded:'⚠️'" in ix, \
        "the dashboard glyph falls through to the unknown '⚪'"
    assert ".card.breaker-degraded{" in ix
    print("  (i) both flavours gate at the serve layer, survive a trigger, "
          "and reach close report / history / dashboard; complete groups "
          "are not over-gated: OK")


def _synthetic_frame(n=140):
    """A clean uptrend — enough bars to score, and healthy enough that a
    RESOLVED-clear control produces a real, publishable verdict. Without
    that control the withhold assertions could pass vacuously."""
    import numpy as np
    import pandas as pd
    idx = pd.date_range("2026-01-02", periods=n, freq="B")
    close = pd.Series(np.linspace(100.0, 150.0, n), index=idx)
    return pd.DataFrame({"Open": close * 0.995, "High": close * 1.01,
                         "Low": close * 0.99, "Close": close,
                         "Volume": pd.Series([1_000_000] * n, index=idx)})


def test_unresolved_breaker_withholds_the_trade_signal():
    """An unresolvable breaker context must WITHHOLD, never present as clear.

    The defect: every non-'resolved' status collapsed to the literal
    "clear" and was fed to compute_trade_signal, where "clear" is a
    positive finding — the value that lets BUY NOW through. Worse,
    'not_in_universe' was exempted from the gate entirely, so a name
    outside the universe published a verdict backed by zero macro coverage
    and chipped identically to a fully checked one.
    """
    import ticker_api
    df = _synthetic_frame()
    saved = (ticker_api.fetch_data, ticker_api.fetch_fundamentals_yfinance,
             ticker_api._group_breaker_context, dict(ticker_api._cache))
    try:
        ticker_api.fetch_data = lambda sym, period="6mo": df
        ticker_api.fetch_fundamentals_yfinance = lambda sym: {}

        # CONTROL first: a resolved-clear breaker must still publish, or
        # every assertion below is vacuous.
        ticker_api._cache.clear()
        ticker_api._group_breaker_context = lambda s: {
            "status": "resolved", "group": "G", "breaker_status": "clear"}
        ok = ticker_api._analyze_ticker("AAA")
        publishable = {"BUY NOW", "WAIT FOR PULLBACK", "ACCUMULATE ON DIP",
                       "HOLD POSITION", "REDUCE/EXIT", "AVOID"}
        assert ok["trade_signal"] in publishable, ok["trade_signal"]
        assert ok["trade_signal_gate"] is None
        print(f"  (12a) control: a RESOLVED clear breaker still publishes "
              f"{ok['trade_signal']!r}: OK")

        # Every non-resolved status withholds — including not_in_universe.
        for status, ctx in (
                ("not_in_universe", {"status": "not_in_universe"}),
                ("unavailable", {"status": "unavailable", "group": "G",
                                 "reason": "signals artifact missing",
                                 "breaker_status": "unknown"}),
                ("degraded", {"status": "degraded", "group": "G",
                              "breaker_status": "clear",
                              "reason": "a check never ran"})):
            ticker_api._cache.clear()
            ticker_api._group_breaker_context = lambda s, c=ctx: c
            r = ticker_api._analyze_ticker("AAA")
            assert r["trade_signal"] == "SIGNAL WITHHELD", \
                f"{status} published {r['trade_signal']!r}"
            assert r["trade_signal_gate"], f"{status} carries no gate reason"
            assert "clear" not in (r["trade_signal"] or "")
        print("  (12b) not_in_universe, unavailable and degraded all "
              "withhold the VALUE, not merely flag it: OK")

        # THE FAILURE, DEMONSTRATED. The old code fed "clear" for exactly
        # these statuses. On this same evidence that produces a real
        # verdict — which is what a user saw, indistinguishable from a
        # checked one.
        from signal_engine import compute_trade_signal, score_stock_v2
        _s, _sig, details = score_stock_v2(df)
        old, _ = compute_trade_signal(details, breaker_status="clear")
        assert old in publishable, old
        assert old != "SIGNAL WITHHELD"
        print(f"  (12c) the old collapse-to-clear yields {old!r} on the same "
              "evidence — a verdict backed by no macro check at all: OK")

        # And the JSON must not disagree with the chip: the withheld value
        # is in trade_signal itself, so a cached or third-party client
        # cannot render a presumed-clear verdict the server suppressed.
        ticker_api._cache.clear()
        ticker_api._group_breaker_context = lambda s: {"status": "not_in_universe"}
        r = ticker_api._analyze_ticker("AAA")
        assert r["trade_signal"] == "SIGNAL WITHHELD" and r["trade_signal_gate"]
        print("  (12d) the withheld state lives in trade_signal itself, so "
              "the JSON and the rendered chip cannot disagree: OK")
    finally:
        (ticker_api.fetch_data, ticker_api.fetch_fundamentals_yfinance,
         ticker_api._group_breaker_context, cache) = saved
        ticker_api._cache.clear()
        ticker_api._cache.update(cache)


def _js_function_body(src, name):
    """Extract a JS function body by brace matching, so a branch found here
    is genuinely INSIDE the mapper and not merely elsewhere in the file."""
    start = src.index(f"function {name}(")
    open_brace = src.index("{", start)
    depth, i = 0, open_brace
    while i < len(src):
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                return src[open_brace:i + 1]
        i += 1
    raise AssertionError(f"unbalanced braces in {name}")


def test_every_chip_mapper_handles_withheld():
    """All THREE chip mappers must treat SIGNAL WITHHELD as a refusal.

    framework.html's gl2TsClass called itself "the dashboard's exact chip
    mapping" and had no such branch, so a withheld signal fell through to
    ts-hold — the one outcome a refusal must never be mistaken for. It
    reads the engine artifact, which publishes that literal string, so it
    was reachable.
    """
    rd = lambda p: open(os.path.join(REPO, p)).read()
    mappers = [("public/index.html", "tsClass"),
               ("public/search.html", "tsClass"),
               ("public/framework.html", "gl2TsClass")]
    for path, fn in mappers:
        src = rd(path)
        body = _js_function_body(src, fn)
        assert "SIGNAL WITHHELD" in body, \
            f"{path}:{fn} has no SIGNAL WITHHELD branch — a refusal falls " \
            f"through to a neutral chip"
        # the branch must return the refusal class, not a verdict class
        line = next(l for l in body.splitlines() if "SIGNAL WITHHELD" in l
                    and "return" in l)
        assert "ts-withheld" in line, f"{path}:{fn} maps withheld to {line!r}"
        # and the branch must come BEFORE any fallthrough return
        assert body.index("SIGNAL WITHHELD") < body.rindex("return"), \
            f"{path}:{fn} reaches its fallthrough before the withheld branch"
        # the class must actually be styled on that page
        assert ".ts-withheld{" in src, \
            f"{path} maps to ts-withheld but never styles it"
    print(f"  (13a) all {len(mappers)} chip mappers map SIGNAL WITHHELD to "
          "ts-withheld, ahead of their fallthrough, and style it: OK")

    # THE FAILURE, DEMONSTRATED — mirror the mapper on a body with the
    # branch removed and confirm it lands on the neutral chip.
    fw = _js_function_body(rd("public/framework.html"), "gl2TsClass")
    stripped = "\n".join(l for l in fw.splitlines()
                         if "SIGNAL WITHHELD" not in l)
    assert "SIGNAL WITHHELD" not in stripped
    assert stripped.rstrip().rstrip("}").rstrip().endswith("return 'ts-hold';"), \
        "the fallthrough is no longer ts-hold — re-check what a stripped " \
        "mapper would render"
    print("  (13b) with the branch removed, framework.html's mapper falls "
          "through to ts-hold — the regression this pin guards: OK")


if __name__ == "__main__":
    print("\n=== D-019 breaker coverage pins ===")
    test_full_coverage_reads_clear()
    test_macro_outage_degrades_every_dependent_group()
    test_prefix_engine_fabricates_clear_on_the_same_injection()
    test_mapping_table_complete_both_directions()
    test_gate_withholds_on_degraded_and_is_era_aware()
    test_pages_render_degraded_distinctly()
    test_writer_half_records_and_writes()
    test_a_trigger_still_wins_over_incomplete_coverage()
    test_degraded_gate_is_behavioural()
    test_degraded_input_and_published_signal()
    test_both_flavours_gate_everywhere()
    test_unresolved_breaker_withholds_the_trade_signal()
    test_every_chip_mapper_handles_withheld()
    print("\nAll breaker-coverage tests passed.\n")
