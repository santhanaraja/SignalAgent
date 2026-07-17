#!/usr/bin/env python3
"""
Gauge B production chassis pins (D-008 build).

THE critical pin (extraction discipline, Build 4 step-0 pattern): the
production transcription in framework/regime_calculator.py must reproduce the
VALIDATED analysis function in scripts/backtest_gauge_b.py — identical state
sequences on identical inputs, step-by-step and over the full cached-data
replay. If production and backtest disagree on any date, the transcription is
wrong.

Plus: locked-calibration config pin, throttle boundary pins, hysteresis
semantics, percentile/lookahead sanity, replay seed-independence, the R28
ceiling integration, calculator chassis/parliament paths with a stub fetcher,
state-record persistence + outage fallback, and the simulate endpoint parity.
"""
import datetime
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

from framework.regime_calculator import (  # noqa: E402
    CHASSIS_STATES, CHASSIS_RANK, CHASSIS_TO_REGIME, CHASSIS_LADDER,
    chassis_raw_state, chassis_step, new_chassis_hysteresis, pctile_of_last,
    replay_chassis, artifact_schema, RegimeCalculator,
)
import backtest_gauge_b as bt  # noqa: E402  — the validated analysis source

LOCKED = {"n": 2, "mode": "asymmetric", "vix_thr": 22.0, "hy_cut": 90.0,
          "hy_window": 60, "breadth_thr": -0.5, "require_k": 1}


def test_locked_config():
    """config.yaml must carry EXACTLY the locked calibration (comment 11724)."""
    import yaml
    with open(os.path.join(REPO, "framework", "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    r = cfg["regime"]
    assert r.get("engine") == "chassis", r.get("engine")
    ch = r["chassis"]
    assert ch["hysteresis_n"] == 2 and ch["hysteresis_mode"] == "asymmetric"
    assert ch["vix_throttle"] == 22.0 and ch["hy_pctile_cut"] == 90.0
    assert ch["hy_pctile_window"] == 60 and ch["breadth_throttle"] == -0.5
    assert ch["require_k"] == 1
    print("  locked config: engine=chassis, k1/vix22/hy90/br-0.5, asym N=2, pctile_60d: OK")


def test_transcription_step_equivalence():
    """Step-by-step: production chassis_step == backtest compute_regime_chassis
    on 2000 randomized inputs with the locked throttle, threading each side's
    own carry. Exposure and confirmed state must match at every step."""
    rng = np.random.default_rng(42)
    h_prod = new_chassis_hysteresis()
    h_bt = bt.new_hysteresis_state()
    throttle = {"vix_thr": 22.0, "breadth_thr": -0.5, "require_k": 1}
    for i in range(2000):
        trend = float(rng.uniform(-15, 15))
        vix = float(rng.uniform(10, 40))
        hy = bool(rng.random() < 0.25)
        breadth = float(rng.uniform(-3, 3))
        rp = chassis_step(trend, vix, hy, breadth, h_prod, n=2,
                          mode="asymmetric", vix_thr=22.0, breadth_thr=-0.5,
                          require_k=1)
        rb = bt.compute_regime_chassis(trend, vix, None, breadth, h_bt, n=2,
                                       mode="asymmetric", hy_stress=hy,
                                       throttle=throttle)
        assert rp["state"] == rb["state"], (i, rp["state"], rb["state"])
        assert rp["exposure"] == rb["exposure"], (i, rp, rb)
        h_prod = rp["hysteresis_state"]
        h_bt = rb["hysteresis_state"]
        assert h_prod == h_bt, (i, h_prod, h_bt)
    print("  transcription step pin: 2000 randomized steps, production == backtest exactly: OK")


def test_transcription_full_replay_pin():
    """THE pin: replay the full cached-data window (2015->present) through the
    production replay and the backtest replay at the LOCKED config — the state
    sequences must be identical on every date. Skips loudly without the cache."""
    cache = os.path.join(REPO, "data", "backtest_cache")
    if not os.path.isdir(cache):
        print("  FULL-REPLAY PIN SKIPPED — data/backtest_cache absent "
              "(run on the analysis machine; the synthetic step pin still holds)")
        return
    raw = bt.load_series()
    end = str(raw["SPY"].index[-1].date())
    inputs = bt.build_inputs(raw, "2015-01-01", end)
    pct = bt.warmed_pctile_value(raw, inputs.index)
    hy_stress = (pct >= 90.0)

    bt_states = bt.replay_states(
        inputs, hy_stress, n=2, mode="asymmetric",
        throttle={"vix_thr": 22.0, "breadth_thr": -0.5, "require_k": 1})

    prod_states, _, _ = replay_chassis(
        [float(x) for x in inputs["spy_vs_200dma"].values],
        [float(x) for x in inputs["vix_5d"].values],
        [bool(x) for x in hy_stress.reindex(inputs.index).fillna(False).values],
        [float(x) for x in inputs["breadth_20d"].values],
        n=2, mode="asymmetric", vix_thr=22.0, breadth_thr=-0.5, require_k=1)

    mismatches = [(str(d.date()), a, b) for d, a, b in
                  zip(inputs.index, bt_states.values, prod_states) if a != b]
    assert not mismatches, f"{len(mismatches)} mismatching dates, first: {mismatches[:3]}"
    print(f"  FULL-REPLAY PIN: {len(prod_states)} trading days 2015->{end}, "
          f"production == backtest on every date: OK")


def test_throttle_boundaries():
    h = "In-Trend"
    # vix >= 22 fires; 21.99 does not
    assert chassis_raw_state(True, 21.99, False, 0.0) == "In-Trend-Full"
    assert chassis_raw_state(True, 22.0, False, 0.0) == "In-Trend-Throttled"
    # breadth < -0.5 fires; exactly -0.5 does NOT (strict <)
    assert chassis_raw_state(True, 15.0, False, -0.5) == "In-Trend-Full"
    assert chassis_raw_state(True, 15.0, False, -0.51) == "In-Trend-Throttled"
    # hy stress fires alone
    assert chassis_raw_state(True, 15.0, True, 0.0) == "In-Trend-Throttled"
    # out-of-trend: defensive by default; Risk-off on vix OR hy — NEVER breadth
    assert chassis_raw_state(False, 15.0, False, -3.0) == "Out-Defensive"
    assert chassis_raw_state(False, 25.0, False, 0.0) == "Out-Risk-off"
    assert chassis_raw_state(False, 15.0, True, 0.0) == "Out-Risk-off"
    # require_k=2: single throttle stays Full
    assert chassis_raw_state(True, 25.0, False, 0.0, require_k=2) == "In-Trend-Full"
    assert chassis_raw_state(True, 25.0, True, 0.0, require_k=2) == "In-Trend-Throttled"
    print("  throttle boundaries: vix>=22, breadth<-0.5 strict, hy bool, "
          "out-of-trend ignores breadth, require_k: OK")


def test_hysteresis_semantics():
    # upgrade needs N=2 consecutive better closes and jumps to the CURRENT raw
    h = new_chassis_hysteresis("Out-Defensive")
    r = chassis_step(5.0, 25.0, False, 0.0, h)          # raw Throttled (vix)
    assert r["state"] == "Out-Defensive" and r["hysteresis_state"]["up"] == 1
    r = chassis_step(5.0, 15.0, False, 0.0, r["hysteresis_state"])  # raw Full
    assert r["state"] == "In-Trend-Full", r              # jumped to CURRENT raw
    # downgrade is instant (asymmetric crash brake)
    r2 = chassis_step(-1.0, 15.0, False, 0.0, r["hysteresis_state"])
    assert r2["state"] == "Out-Defensive"
    # equal raw resets both counters
    h3 = {"confirmed": "In-Trend-Full", "up": 1, "down": 0}
    r3 = chassis_step(5.0, 15.0, False, 0.0, h3)
    assert r3["hysteresis_state"]["up"] == 0 and r3["state"] == "In-Trend-Full"
    # symmetric control: downgrade also waits N
    h4 = new_chassis_hysteresis("In-Trend-Full")
    r4 = chassis_step(-1.0, 15.0, False, 0.0, h4, mode="symmetric")
    assert r4["state"] == "In-Trend-Full" and r4["hysteresis_state"]["down"] == 1
    print("  hysteresis: N=2 upgrade jumps to current raw, instant downgrade, "
          "equal resets, symmetric control: OK")


def test_pctile_and_lookahead():
    # semantics: percentile of the LAST value within the trailing window
    # inclusive, (a <= a[-1]).mean() * 100 — pinned against the backtest's
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert pctile_of_last(vals, 5) == 100.0
    assert pctile_of_last([5, 4, 3, 2, 1], 5) == 20.0
    assert pctile_of_last([2, 2, 2, 2, 2], 5) == 100.0   # ties are <=
    assert pctile_of_last(vals, 6) is None               # un-warmed -> None
    s = pd.Series(np.random.default_rng(7).uniform(2, 6, 300))
    bt_p = bt._rolling_pctile_of_last(s, 60)
    for i in (60, 150, 299):
        mine = pctile_of_last(list(s.iloc[:i + 1]), 60)
        assert abs(mine - float(bt_p.iloc[i])) < 1e-9, i
    # lookahead: truncated history reproduces the full series' value at T
    full = pctile_of_last(list(s), 60)
    assert full == pctile_of_last(list(s.iloc[:300]), 60)
    print("  percentile: backtest-identical on 300-pt series, ties<=, "
          "un-warmed None, trailing-only: OK")


def test_replay_seed_independence():
    # any realistic 60-day window converges regardless of the hysteresis seed
    rng = np.random.default_rng(3)
    trend = list(rng.uniform(-5, 10, 60))
    vix = list(rng.uniform(12, 30, 60))
    hy = [bool(x < 0.2) for x in rng.random(60)]
    br = list(rng.uniform(-2, 2, 60))
    finals = set()
    for seed in CHASSIS_STATES:
        states, carry, _ = replay_chassis(trend, vix, hy, br, seed_state=seed)
        finals.add((states[-1], carry["up"], carry["down"]))
    assert len(finals) == 1, finals
    print("  replay seed-independence: 4 seeds -> identical final state+carry: OK")


def test_r28_ceiling_integration():
    """Each chassis state -> regime label -> the REAL assess_portfolio ceiling
    must be the D-008 Q4 ladder (90/50/25/5) — one integration point."""
    from framework.portfolio_rules import assess_portfolio, REGIME_CEILINGS
    expected = {"In-Trend-Full": 90.0, "In-Trend-Throttled": 50.0,
                "Out-Defensive": 25.0, "Out-Risk-off": 5.0}
    for chassis_state, pct in expected.items():
        label = CHASSIS_TO_REGIME[chassis_state]
        assert REGIME_CEILINGS[label] == pct, (chassis_state, label)
        assert CHASSIS_LADDER[chassis_state] * 100.0 == pct
        r = assess_portfolio({"holdings": []}, {}, 97500, label)
        assert r["status"] == "ok" and r["ceiling_pct"] == pct, (label, r)
        assert r["ceiling_state"] == label
    print("  R28 integration: 4 chassis states -> labels -> assess_portfolio "
          "ceilings 90/50/25/5: OK")


# ---- calculator-path tests with a stub fetcher -----------------------------

def _bdays(n, end=None):
    # dynamic end (today) — a hardcoded date here is a time bomb: the
    # stale-record decay compares against the real clock
    end = end or pd.Timestamp.now().normalize()
    return pd.bdate_range(end=end, periods=n)


_STUB_LAST_DAY = str(_bdays(1)[-1].date())


def _ohlcv(closes, end=None):
    idx = _bdays(len(closes), end)
    c = pd.Series(closes, index=idx, dtype=float)
    return pd.DataFrame({"Open": c, "High": c, "Low": c, "Close": c,
                         "Volume": 1_000_000.0}, index=idx)


def _stub_fetcher(spy_up=True, vix_level=15.0):
    n = 520
    spy = list(np.linspace(400, 500, n)) if spy_up \
        else list(np.linspace(500, 380, n))
    frames = {
        "SPY": _ohlcv(spy),
        "^VIX": _ohlcv([vix_level] * 130),
        "RSP": _ohlcv([x * 0.4 for x in spy[-130:]]),
        "HYG": _ohlcv(list(np.linspace(78, 80, 130))),
        "IEF": _ohlcv([95.0] * 130),
        "^TYX": _ohlcv([4.8] * 25),
        "^IRX": _ohlcv([4.2] * 25),
    }

    def fetch(ticker, period="1y"):
        return frames.get(ticker)
    return fetch


def _calc(tmpdir, fetch, oas_series="declining"):
    import yaml
    with open(os.path.join(REPO, "framework", "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    calc = RegimeCalculator(cfg, fetch)
    calc.STATE_DIR = tmpdir
    if oas_series == "declining":
        idx = _bdays(200)
        calc._fetch_oas_series = lambda c, limit=250: pd.Series(
            np.linspace(4.0, 2.5, 200), index=idx)     # falling OAS -> low pctile
    elif oas_series == "spiking":
        idx = _bdays(200)
        vals = [3.0] * 195 + [3.5, 3.8, 4.2, 4.6, 5.0]  # fresh spike -> pctile 100
        calc._fetch_oas_series = lambda c, limit=250: pd.Series(
            vals, index=idx, dtype=float)
    else:
        calc._fetch_oas_series = lambda c, limit=250: None
    # keep tests offline: yield curve + parliament HY scalar fetch stubbed
    calc._compute_yield_curve = lambda c: {"value": None, "signal": "unavailable",
                                           "detail": "stubbed"}
    calc._try_fred_hy_spread = lambda c: None
    return calc


def test_calculator_chassis_path():
    tmp = tempfile.mkdtemp(prefix="chassis_state_")
    try:
        calc = _calc(tmp, _stub_fetcher(spy_up=True, vix_level=15.0))
        r = calc.compute()
        assert r["engine"] == "chassis"
        ch = r["chassis"]
        assert ch is not None and ch["confirmed_state"] == "In-Trend-Full", ch
        assert r["regime"] == "Risk-on / Trending"
        assert ch["throttles_firing"] == 0 and ch["trend_in"] is True
        assert ch["exposure_ceiling_pct"] == 90.0
        assert ch["replay"]["days_used"] == 60
        # state record written with the correct shape
        rec = json.load(open(os.path.join(tmp, "regime_chassis_state.json")))
        assert rec["confirmed"] == "In-Trend-Full" and rec["engine"] == "chassis"
        assert rec["as_of"] == _STUB_LAST_DAY
        # gauges/counts still emitted for display + back-compat
        assert set(r["gauges"].keys()) == {"vix_5d_avg", "hy_spread", "breadth"}
        assert isinstance(r["risk_on_count"], int)
        # elevated VIX -> throttled -> Choppy label (downgrade instant)
        calc2 = _calc(tmp, _stub_fetcher(spy_up=True, vix_level=25.0))
        r2 = calc2.compute()
        assert r2["chassis"]["confirmed_state"] == "In-Trend-Throttled", r2["chassis"]
        assert r2["regime"] == "Risk-on / Choppy"
        # downtrend + calm vol -> Out-Defensive -> Caution
        calc3 = _calc(tmp, _stub_fetcher(spy_up=False, vix_level=15.0))
        r3 = calc3.compute()
        assert r3["chassis"]["confirmed_state"] == "Out-Defensive", r3["chassis"]
        assert r3["regime"] == "Caution"
        assert r3["backdrop_gate"]["open"] is False
        # downtrend + hy spike -> Out-Risk-off -> Risk-off
        calc4 = _calc(tmp, _stub_fetcher(spy_up=False, vix_level=15.0),
                      oas_series="spiking")
        r4 = calc4.compute()
        assert r4["chassis"]["confirmed_state"] == "Out-Risk-off", r4["chassis"]
        assert r4["regime"] == "Risk-off"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  calculator chassis path: Full/Trending, vix->Throttled/Choppy, "
          "downtrend->Defensive/Caution, +hy spike->Risk-off; record written: OK")


def test_calculator_tz_mixed_fetcher():
    """Runner-fetcher regression: yfinance indexes daily bars at midnight of
    the EXCHANGE tz — ^VIX is America/Chicago while SPY/RSP are New York, so
    instant-based reindex aligns NOTHING and the chassis silently degraded
    (caught live in the runner integration test). Alignment must key on the
    trading DATE."""
    tmp = tempfile.mkdtemp(prefix="chassis_tz_")
    try:
        base = _stub_fetcher(spy_up=True, vix_level=15.0)

        def tz_fetch(ticker, period="1y"):
            df = base(ticker, period)
            if df is None:
                return None
            df = df.copy()
            tz = "America/Chicago" if ticker == "^VIX" else "America/New_York"
            df.index = df.index.tz_localize(tz)
            return df
        calc = _calc(tmp, tz_fetch)
        r = calc.compute()
        ch = r["chassis"]
        assert ch["degraded"] is False, ch
        assert ch["confirmed_state"] == "In-Trend-Full", ch
        assert ch["replay"]["days_used"] == 60
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  tz-mixed fetcher (VIX Chicago vs SPY New York): date-keyed "
          "alignment, no degrade: OK")


def test_phantom_bar_guard():
    """Review finding: on market holidays / weekend runs the fetcher appends
    a synthetic 'today' row (fast_info quote, Volume 0) — it must NOT count
    as a hysteresis close. The trailing zero-volume SPY row is dropped; the
    replay (and the record's as_of) end at the last REAL close."""
    tmp = tempfile.mkdtemp(prefix="chassis_phantom_")
    try:
        base = _stub_fetcher(spy_up=True, vix_level=15.0)

        def phantom_fetch(ticker, period="1y"):
            df = base(ticker, period)
            if df is None or ticker != "SPY":
                return df
            df = df.copy()
            # synthetic Saturday row: duplicated close, Volume 0
            import pandas as _pd
            phantom = df.iloc[[-1]].copy()
            phantom.index = [df.index[-1] + _pd.Timedelta(days=1, hours=14)]
            phantom["Volume"] = 0.0
            return _pd.concat([df, phantom])
        calc = _calc(tmp, phantom_fetch)
        r = calc.compute()
        ch = r["chassis"]
        assert ch["replay"]["end"] == _STUB_LAST_DAY, ch["replay"]
        rec = json.load(open(os.path.join(tmp, "regime_chassis_state.json")))
        assert rec["as_of"] == _STUB_LAST_DAY, rec
        # identical outcome to the phantom-free fetch — the phantom step is gone
        calc2 = _calc(tmp, base)
        r2 = calc2.compute()
        assert r2["chassis"]["confirmed_state"] == ch["confirmed_state"]
        assert r2["chassis"]["hysteresis"] == ch["hysteresis"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  phantom-bar guard: trailing Volume-0 SPY row dropped, replay ends "
          "at the last real close: OK")


def test_fallback_basis_flagged_degraded():
    """Review finding: the HYG/IEF credit fallback keeps the gauge alive but
    is NOT the D-008-validated basis — it must surface degraded, never pass
    as healthy."""
    tmp = tempfile.mkdtemp(prefix="chassis_fb_")
    try:
        calc = _calc(tmp, _stub_fetcher(), oas_series="none")  # FRED dead
        r = calc.compute()
        ch = r["chassis"]
        assert ch["hy_basis"] == "hyg_ief_inverted_pctile", ch["hy_basis"]
        assert ch["degraded"] is True
        assert ch["degraded_reason"] == "hy_fallback_basis"
        assert ch["confirmed_state"] in CHASSIS_RANK      # still computes
        # the trend outage-transparency must NOT trip on a credit fallback
        assert r["backdrop_gate"]["capped"] is False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  fallback credit basis: computes but flagged degraded "
          "(hy_fallback_basis), gate untouched: OK")


def test_unwarmed_hy_window_degrades():
    """Review finding: an un-warmed percentile window must never silently
    read hy_stress=False. Short HYG history (< window) -> no valid pctile ->
    the honest hy_data_unavailable degrade, not a healthy-looking block."""
    tmp = tempfile.mkdtemp(prefix="chassis_warm_")
    try:
        base = _stub_fetcher()

        def short_hyg(ticker, period="1y"):
            df = base(ticker, period)
            if ticker in ("HYG", "IEF") and df is not None:
                return df.iloc[-40:]                       # < hy_window rows
            return df
        calc = _calc(tmp, short_hyg, oas_series="none")
        r = calc.compute()
        ch = r["chassis"]
        assert ch["hy_basis"] is None, ch["hy_basis"]
        assert ch["degraded"] is True
        assert ch["degraded_reason"] == "hy_data_unavailable"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  un-warmed HY window: no faked stress, degrades to "
          "hy_data_unavailable: OK")


def test_stale_record_decay():
    """Review finding: the outage fallback must not serve a weeks-old
    In-Trend-Full — records older than 7 days decay to Out-Defensive
    (fail closed, the system's outage convention)."""
    tmp = tempfile.mkdtemp(prefix="chassis_stale_")
    try:
        with open(os.path.join(tmp, "regime_chassis_state.json"), "w") as f:
            json.dump({"as_of": "2026-06-01", "confirmed": "In-Trend-Full",
                       "raw_state": "In-Trend-Full", "up": 0, "down": 0,
                       "engine": "chassis"}, f)
        calc = _calc(tmp, lambda t, period="1y": None, oas_series="none")
        r = calc.compute()
        ch = r["chassis"]
        assert ch["degraded"] is True
        assert ch["degraded_reason"] == "data_unavailable_stale"
        assert ch["confirmed_state"] == "Out-Defensive"
        assert r["regime"] == "Caution"
        assert r["backdrop_gate"]["capped"] is True        # outage-transparent
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  stale-record decay: >7d-old record -> Out-Defensive/Caution, "
          "outage-transparent: OK")


def test_lab_carry_reproduces_today():
    """Review finding: the lab must step from the carry production held
    BEFORE today's step (carry_pre_final), so an untouched seeded lab
    reproduces today's confirmed state instead of double-stepping the bar."""
    tmp = tempfile.mkdtemp(prefix="chassis_labcarry_")
    try:
        calc = _calc(tmp, _stub_fetcher(spy_up=True, vix_level=15.0))
        r = calc.compute()
        ch = r["chassis"]
        rec = json.load(open(os.path.join(tmp, "regime_chassis_state.json")))
        pre = rec["carry_pre_final"]
        step = chassis_step(ch["trend_pct"],
                            ch["throttles"]["vix"]["value"],
                            ch["throttles"]["hy"]["firing"],
                            ch["throttles"]["breadth"]["value"],
                            pre, n=2, mode="asymmetric", vix_thr=22.0,
                            breadth_thr=-0.5, require_k=1)
        assert step["state"] == ch["confirmed_state"], (step, ch)
        assert step["raw_state"] == ch["raw_state"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  lab carry: carry_pre_final + today's inputs reproduce today's "
          "confirmed state (no double-step): OK")


def test_calculator_outage_fallback_and_parliament():
    tmp = tempfile.mkdtemp(prefix="chassis_state_")
    try:
        # seed a recorded state, then kill all fetches -> served degraded
        calc = _calc(tmp, _stub_fetcher())
        calc.compute()
        calc_dead = _calc(tmp, lambda t, period="1y": None, oas_series="none")
        r = calc_dead.compute()
        ch = r["chassis"]
        assert ch["degraded"] is True and ch["degraded_reason"] == "data_unavailable"
        assert ch["confirmed_state"] == "In-Trend-Full"      # from the record
        assert r["regime"] == "Risk-on / Trending"
        assert r["backdrop_gate"]["capped"] is True
        assert r["backdrop_gate"]["reason"] == "data_unavailable"
        # parliament path intact behind the flag (reversibility)
        import yaml
        with open(os.path.join(REPO, "framework", "config.yaml")) as f:
            cfg = yaml.safe_load(f)
        cfg["regime"]["engine"] = "parliament"
        calc_p = RegimeCalculator(cfg, _stub_fetcher())
        calc_p.STATE_DIR = tmp
        calc_p._compute_yield_curve = lambda c: {"value": None,
                                                 "signal": "unavailable",
                                                 "detail": "stubbed"}
        calc_p._try_fred_hy_spread = lambda c: None
        rp = calc_p.compute()
        assert rp["engine"] == "parliament" and rp["chassis"] is None
        assert rp["regime"] in ("Risk-on / Trending", "Risk-on / Choppy",
                                "Caution", "Risk-off")
        assert rp["backdrop_gate"]["role"] == "backdrop_gate"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  outage fallback (recorded state, degraded, gate outage-transparent) "
          "+ parliament reversibility: OK")


def test_artifact_schema_tags():
    assert artifact_schema({"engine": "chassis"}) == "regime-b-chassis"
    assert artifact_schema({"engine": "parliament"}) == "regime-1a-3voter"
    assert artifact_schema({}) == "regime-b-chassis"          # default = chassis
    assert artifact_schema(None) == "regime-b-chassis"
    print("  artifact schema tags: chassis/parliament/default: OK")


def test_simulate_endpoint_chassis():
    import ticker_api
    client = ticker_api.app.test_client()
    # fixed OAS tail + a fresh carry so the parity check is deterministic
    ticker_api._OAS_TAIL["vals"] = [3.0] * 200
    ticker_api._OAS_TAIL["ts"] = 9e12
    inputs = {"vix_5d": 16.0, "hy_oas": 2.7, "breadth_20d": 0.2,
              "spy_vs_200dma_pct": 8.0}
    resp = client.post("/api/regime/simulate", json=inputs)
    assert resp.status_code == 200, resp.get_json()
    d = resp.get_json()
    assert d["engine"] == "chassis"
    ch = d["chassis"]
    # parity with the pure functions on the same inputs + persisted carry
    from framework.regime_calculator import chassis_step, pctile_of_last
    ccfg = ticker_api._regime_cfg()["chassis"]
    window = [3.0] * (ccfg["hy_window"] - 1) + [2.7]
    p = pctile_of_last(window, ccfg["hy_window"])
    hy_stress = p is not None and p >= ccfg["hy_cut"]
    pure = chassis_step(8.0, 16.0, hy_stress, 0.2, ticker_api._chassis_carry(),
                        n=ccfg["n"], mode=ccfg["mode"], vix_thr=ccfg["vix_thr"],
                        breadth_thr=ccfg["breadth_thr"],
                        require_k=ccfg["require_k"])
    assert ch["confirmed_state"] == pure["state"], (ch, pure)
    assert ch["raw_state"] == pure["raw_state"]
    assert d["state"] == CHASSIS_TO_REGIME[pure["state"]]
    # throttle flip distances present for available inputs
    assert set(d["flip_distances"].keys()) == {"vix_5d", "hy_oas",
                                               "breadth_20d",
                                               "spy_vs_200dma_pct"}
    vf = d["flip_distances"]["vix_5d"]
    assert vf and vf["to"] in ("throttle_firing", "clear")
    # hostile input still 400s
    assert client.post("/api/regime/simulate", json={"vix_5d": "x"}).status_code == 400
    assert client.post("/api/regime/simulate",
                       data="nope", content_type="application/json").status_code == 400
    print("  simulate endpoint (chassis): parity with pure chassis_step, "
          "throttle flips, 400s intact: OK")


if __name__ == "__main__":
    print("\n=== Gauge B production chassis pins (D-008 build) ===")
    test_locked_config()
    test_transcription_step_equivalence()
    test_transcription_full_replay_pin()
    test_throttle_boundaries()
    test_hysteresis_semantics()
    test_pctile_and_lookahead()
    test_replay_seed_independence()
    test_r28_ceiling_integration()
    test_calculator_chassis_path()
    test_calculator_tz_mixed_fetcher()
    test_phantom_bar_guard()
    test_fallback_basis_flagged_degraded()
    test_unwarmed_hy_window_degrades()
    test_stale_record_decay()
    test_lab_carry_reproduces_today()
    test_calculator_outage_fallback_and_parliament()
    test_artifact_schema_tags()
    test_simulate_endpoint_chassis()
    print("\nAll Gauge B chassis pins green.")
