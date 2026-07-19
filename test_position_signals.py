#!/usr/bin/env python3
"""
Tests for the position signal engine (Build 1B): the 5-condition re-entry
rule, the HELD -> EXIT_FIRED -> WATCHING -> RE_ENTRY_ARMING -> RE_ENTRY_READY
state machine, history-event emission, and the MRVL June replay pinned
against real recorded bars.

Run: python3 test_position_signals.py
"""

import copy
import datetime
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from framework.position_signals import (
    PositionSignalEngine, HELD, EXIT_FIRED, WATCHING,
    RE_ENTRY_ARMING, RE_ENTRY_READY,
)

TRENDING = "Risk-on / Trending"
CHOPPY = "Risk-on / Choppy"
CAUTION = "Caution"

CONFIG = {
    "positions": {"sma_period": 20, "confirmation_closes": 2,
                  "atr_period": 14, "atr_mult": 0.5, "slope_lookback_days": 5},
    "themes": {
        "watchlist": [{"name": "Semis", "proxy": "SMH"},
                      {"name": "Biotech", "proxy": "XBI"}],
        "entry_rule": {"requires_top_n_rank": 2},
    },
}

UNMAPPED = {"met": True, "no_group_mapping": True, "detail": "no mapping"}


def _bars(closes, spread=1.0):
    closes = list(closes)
    idx = pd.date_range(end="2026-07-02", periods=len(closes), freq="B")
    c = np.array(closes, dtype=float)
    return pd.DataFrame({"Open": c, "High": c + spread, "Low": c - spread,
                         "Close": c, "Volume": [1000] * len(c)}, index=idx)


def _engine():
    return PositionSignalEngine(copy.deepcopy(CONFIG), lambda t, period="6mo": None)


# ------------------------------------------------------------------
# Conditions
# ------------------------------------------------------------------

def test_confirmation_consecutive_closes():
    eng = _engine()
    # wide bars (spread 3 -> ATR ~6): closes above SMA but well inside the
    # ATR-break threshold, so ONLY the consecutive-closes route can confirm
    df = _bars([100.0] * 28 + [101.2, 101.4], spread=3.0)
    r = eng.evaluate({}, "watching", df, TRENDING, UNMAPPED, WATCHING)
    c = r["conditions"]
    assert c["1_trigger"]["met"]
    assert c["2_confirmation"]["met"]
    assert c["2_confirmation"]["consecutive_closes_above"] == 2
    assert "consecutive closes" in c["2_confirmation"]["detail"]
    assert "ATR" not in c["2_confirmation"]["detail"]
    # one close above, inside the ATR threshold: unconfirmed -> ARMING
    df1 = _bars([100.0] * 29 + [100.6], spread=3.0)
    r1 = eng.evaluate({}, "watching", df1, TRENDING, UNMAPPED, WATCHING)
    assert r1["conditions"]["1_trigger"]["met"]
    assert not r1["conditions"]["2_confirmation"]["met"]
    assert r1["state"] == RE_ENTRY_ARMING
    print("  confirmation via 2 consecutive closes (ATR route excluded): OK")


def test_confirmation_via_atr_break():
    eng = _engine()
    # single close far above SMA: > SMA20 + 0.5*ATR(14) confirms same-day.
    # spread=3 keeps ATR high enough that the jump stays inside the
    # extension guard (~1.2xATR) — the guard has its own tests below.
    df = _bars([100.0] * 29 + [108.0], spread=3.0)
    r = eng.evaluate({}, "watching", df, TRENDING, UNMAPPED, WATCHING)
    c2 = r["conditions"]["2_confirmation"]
    assert c2["met"] and "ATR14" in c2["detail"]
    assert r["state"] == RE_ENTRY_READY   # all five hold (slope flat = ok)
    # informational extension fields (display only, never gate):
    # cross-check both against the returned close/sma20/ATR values
    atr = c2["atr14"]
    assert abs(r["extension_pct"]
               - round((108.0 - r["sma20"]) / r["sma20"] * 100, 2)) < 0.02
    assert abs(r["extension_atr"]
               - round((108.0 - r["sma20"]) / atr, 2)) < 0.02
    assert r["extension_pct"] > 0 and r["extension_atr"] > 0
    print("  confirmation via ATR break + extension fields: OK")


def test_regime_gate_modes():
    eng = _engine()
    # rising: trigger/confirm/slope met; spread=4 keeps extension ~0.6xATR
    # (inside the guard) so READY is reachable
    df = _bars(np.linspace(90, 110, 40), spread=4.0)
    r_t = eng.evaluate({}, "watching", df, TRENDING, UNMAPPED, WATCHING)
    assert r_t["state"] == RE_ENTRY_READY
    assert r_t["conditions"]["3_regime_gate"]["mode"] == "full"
    assert "a_plus_only" not in r_t
    r_c = eng.evaluate({}, "watching", df, CHOPPY, UNMAPPED, WATCHING)
    assert r_c["state"] == RE_ENTRY_READY
    assert r_c["a_plus_only"] is True
    r_x = eng.evaluate({}, "watching", df, CAUTION, UNMAPPED, WATCHING)
    assert r_x["state"] == RE_ENTRY_ARMING   # blocked gate, still armed
    assert not r_x["conditions"]["3_regime_gate"]["met"]
    r_o = eng.evaluate({}, "watching", df, "Risk-off", UNMAPPED, WATCHING)
    assert not r_o["conditions"]["3_regime_gate"]["met"]
    print("  regime gate: full / conditional (A+ only) / blocked: OK")


def test_slope_condition():
    eng = _engine()
    # falling tape that pops above its falling SMA: slope must block READY
    closes = list(np.linspace(130, 100, 35)) + [110.0]
    r = eng.evaluate({}, "watching", _bars(closes), TRENDING, UNMAPPED, WATCHING)
    assert r["conditions"]["1_trigger"]["met"]
    assert not r["conditions"]["4_slope"]["met"]
    assert r["state"] == RE_ENTRY_ARMING
    print("  slope condition blocks READY under a falling SMA20: OK")


def test_group_status_variants():
    eng = _engine()
    gm = {"AAA": "Semiconductors", "CCC": "Biotechnology"}
    selected = {"Semiconductors"}
    weeks = {"Semiconductors": 3}
    # in-universe group: met, exact detail wording (D-007 spec), weeks
    s = eng._group_status({"ticker": "AAA"}, gm, selected, weeks, 15)
    assert s["met"] and s["detail"] == "group 'Semiconductors' in universe (top-15)"
    assert s["weeks_in_universe"] == 3
    # resolvable group NOT selected: honest fail
    s = eng._group_status({"ticker": "CCC"}, gm, selected, weeks, 15)
    assert not s["met"] and "not in current universe" in s["detail"]
    # manual row override beats the map (R28 precedence)
    s = eng._group_status({"ticker": "CCC", "group": "Semiconductors"},
                          gm, selected, weeks, 15)
    assert s["met"]
    # universe unavailable: fail CLOSED with honest wording (outage never
    # opens the entry gate, and never claims the group ranked out)
    s = eng._group_status({"ticker": "AAA"}, gm, set(), {}, 15)
    assert not s["met"] and "universe unavailable" in s["detail"]
    # unmapped: passes but FLAGGED (the old no_theme_mapping convention)
    s = eng._group_status({"ticker": "ZZZ", "theme": "SmallCap/Broad"},
                          gm, selected, weeks, 15)
    assert s["met"] and s.get("no_group_mapping") is True
    assert "SmallCap/Broad" in s["detail"]
    print("  condition 5 (group gate): in-universe / not-selected / manual "
          "override / unmapped-flagged: OK")


def test_stop_display():
    eng = _engine()
    df = _bars(np.linspace(90, 110, 40))
    r = eng.evaluate({"stop_on_entry": "sma20_close"}, "holding", df,
                     TRENDING, UNMAPPED, None)
    assert r["state"] == HELD
    assert r["stop"]["type"] == "sma20_close"
    assert abs(r["stop"]["level"] - r["sma20"]) < 1e-9
    # short MAs (producer amendment): on a rising series sma5 > sma10 > sma20
    assert r["sma5"] > r["sma10"] > r["sma20"], (r["sma5"], r["sma10"], r["sma20"])
    r2 = eng.evaluate({}, "holding", df, TRENDING, UNMAPPED, None)
    assert r2["stop"]["type"] is None
    print("  effective stop for HELD (sma20_close + none defined): OK")


# ------------------------------------------------------------------
# State machine full cycle
# ------------------------------------------------------------------

def _cycle(kind):
    eng = _engine()
    closes = list(np.linspace(90, 110, 45))          # rising: above SMA20
    closes += [95.0]                                  # break below SMA20
    closes += [94.0]                                  # still below
    closes += [118.0]                                 # violent reclaim (ATR break)
    closes += [96.0]                                  # back below
    seq, prev = [], None
    # spread=5 keeps the reclaim inside the extension guard (~1.2xATR);
    # the guard's own cycle is tested separately
    for i in range(40, len(closes) + 1):
        r = eng.evaluate({}, kind, _bars(closes[:i], spread=5.0), TRENDING,
                         UNMAPPED, prev)
        prev = r["state"]
        seq.append(r["state"])
    return seq


def test_full_cycle_watcher():
    seq = _cycle("watching")
    assert seq[0] == RE_ENTRY_READY          # rising watcher: all five
    assert seq[-4] == WATCHING, seq[-6:]     # break: watcher has no EXIT state
    assert seq[-2] == RE_ENTRY_READY, seq[-6:]
    assert seq[-1] == WATCHING, seq[-6:]
    print("  watcher cycle WATCHING -> READY -> WATCHING: OK")


def test_full_cycle_holding_returns_to_held():
    seq = _cycle("holding")
    assert seq[0] == HELD
    assert seq[-4] == EXIT_FIRED, seq[-6:]
    assert seq[-3] == WATCHING, seq[-6:]
    # a HOLDING whose re-entry conditions complete returns to HELD (stop
    # resumes) — HELD must not be unreachable after one exit
    assert seq[-2] == HELD, seq[-6:]
    # and from HELD, the next break fires the exit signal again
    assert seq[-1] == EXIT_FIRED, seq[-6:]
    print("  holding cycle HELD -> EXIT_FIRED -> WATCHING -> HELD -> EXIT_FIRED: OK")


def test_ready_blocked_by_thesis():
    # everything met EXCEPT condition 5 (mapped theme, not qualified):
    # READY must be unreachable — pins cond5's membership in the all-five gate
    eng = _engine()
    df = _bars(np.linspace(90, 110, 40))
    dead_theme = {"met": False, "detail": "theme 'Biotech' not qualified"}
    r = eng.evaluate({}, "watching", df, TRENDING, dead_theme, WATCHING)
    assert r["conditions_met"] == 4
    assert not r["conditions"]["5_thesis"]["met"]
    assert r["state"] == RE_ENTRY_ARMING, r["state"]
    # holding variant must not sneak back to HELD via the READY mapping
    r2 = eng.evaluate({}, "holding", df, TRENDING, dead_theme, WATCHING)
    assert r2["state"] == RE_ENTRY_ARMING, r2["state"]
    print("  thesis condition gates READY (watcher + holding): OK")


def test_synthetic_last_bar_stripped():
    eng = _engine()
    df = _bars(np.linspace(90, 110, 40))
    # append fetch_data's live-quote signature: V=0, O==H==L==C, crash price
    synth = pd.DataFrame({"Open": [80.0], "High": [80.0], "Low": [80.0],
                          "Close": [80.0], "Volume": [0]},
                         index=[df.index[-1] + pd.Timedelta(days=1)])
    poisoned = pd.concat([df, synth])
    stripped = eng._strip_synthetic_last_bar(poisoned)
    assert len(stripped) == len(df)
    assert float(stripped["Close"].iloc[-1]) == float(df["Close"].iloc[-1])
    # a real bar (volume > 0) is never stripped
    assert len(eng._strip_synthetic_last_bar(df)) == len(df)
    # end-to-end: the synthetic crash bar must not fire a false EXIT
    r = eng.evaluate({}, "holding", stripped, TRENDING, UNMAPPED, HELD)
    assert r["state"] == HELD
    print("  synthetic live-quote bar stripped (no false intraday EXIT): OK")


# ------------------------------------------------------------------
# Extension guard (PER-508 item 20)
# ------------------------------------------------------------------

EXTENDED_DF_ARGS = (list(np.linspace(90, 110, 40)),)  # spread=1 -> ext ~2.4xATR


def test_extension_guard_suppresses_extended_ready():
    """The MRNA case: all 5 conditions met, but far above the mean —
    a watcher must print EXTENDED_HOLD, never READY."""
    eng = _engine()
    df = _bars(*EXTENDED_DF_ARGS)                   # default spread=1
    r = eng.evaluate({}, "watching", df, TRENDING, UNMAPPED, WATCHING)
    assert r["extension_atr"] > 1.5, r["extension_atr"]
    assert r["all_conditions_met"] is True          # pips stay honest
    assert r["conditions_met"] == 5
    assert r["state"] == "EXTENDED_HOLD", r["state"]
    assert "re-entry suppressed" in r["extension_guard"]
    assert "> 1.8×" in r["extension_guard"]
    assert "a_plus_only" not in r                   # not READY -> no A+ flag
    print("  extension guard: all-5 + extended watcher -> EXTENDED_HOLD: OK")


def test_extension_guard_permits_healthy_trending_distance():
    """Calibration pin (2026-07-09): Monday's actual fills came in at
    1.06-1.64xATR extension — ARWR at 1.64x was the week's best entry.
    A watcher in that class (1.5 < ext <= 1.8) must stay READY; the old
    1.5 default would have suppressed exactly these entries."""
    eng = _engine()
    # flat 100s + one strong close: ext lands ~1.6xATR (computed, asserted)
    df = _bars([100.0] * 29 + [107.0], spread=1.9)
    r = eng.evaluate({}, "watching", df, TRENDING, UNMAPPED, WATCHING)
    assert 1.5 < r["extension_atr"] <= 1.8, r["extension_atr"]
    assert r["state"] == RE_ENTRY_READY, r["state"]
    assert "extension_guard" not in r
    print("  extension guard: 1.5-1.8x trending distance stays READY: OK")


def test_extension_guard_healthy_reclaim_unaffected():
    """Genuine near-SMA20 reclaim (well under 0.5xATR extension) stays
    READY — the guard must not false-trip healthy entries."""
    eng = _engine()
    df = _bars([100.0] * 28 + [101.2, 101.4], spread=3.0)
    r = eng.evaluate({}, "watching", df, TRENDING, UNMAPPED, WATCHING)
    assert r["extension_atr"] < 0.5, r["extension_atr"]
    assert r["state"] == RE_ENTRY_READY
    assert "extension_guard" not in r
    print("  extension guard: near-SMA20 reclaim stays READY: OK")


def test_extension_guard_holdings_exempt():
    """Holdings are NEVER touched by the guard — an extended held winner
    is trailing-stop territory, not a forced exit."""
    eng = _engine()
    df = _bars(*EXTENDED_DF_ARGS)                   # same ~2.4xATR extension
    for prev in (None, HELD, WATCHING):
        r = eng.evaluate({}, "holding", df, TRENDING, UNMAPPED, prev)
        assert r["state"] == HELD, (prev, r["state"])
        assert "extension_guard" not in r
    print("  extension guard: holdings exempt at any prev state: OK")


def test_extension_guard_boundary_and_config():
    """Threshold is configurable; comparison is strictly greater-than —
    test both sides of the boundary."""
    df = _bars(*EXTENDED_DF_ARGS)
    probe = _engine().evaluate({}, "watching", df, TRENDING, UNMAPPED, WATCHING)
    ext = probe["extension_atr"]

    def eng_with(threshold):
        cfg = copy.deepcopy(CONFIG)
        cfg["positions"]["extension_guard_max"] = threshold
        return PositionSignalEngine(cfg, lambda t, period="6mo": None)

    # threshold just above the observed extension: no trip
    r_hi = eng_with(ext + 0.05).evaluate({}, "watching", df, TRENDING,
                                         UNMAPPED, WATCHING)
    assert r_hi["state"] == RE_ENTRY_READY, r_hi["state"]
    # threshold just below: trips
    r_lo = eng_with(ext - 0.05).evaluate({}, "watching", df, TRENDING,
                                         UNMAPPED, WATCHING)
    assert r_lo["state"] == "EXTENDED_HOLD", r_lo["state"]
    assert f"> {ext - 0.05}×" in r_lo["extension_guard"]
    print("  extension guard: configurable threshold, strict > boundary: OK")


def test_extension_guard_no_flapping_events():
    """READY->EXTENDED_HOLD and EXTENDED_HOLD->READY emit ONE event each;
    repeated runs at constant extension emit nothing."""
    tmp = tempfile.mkdtemp(prefix="extguard_")
    state_dir, data_dir, public_dir = (os.path.join(tmp, d)
                                       for d in ("state", "data", "public"))
    for d in (state_dir, data_dir, public_dir):
        os.makedirs(d)
    olds = (PositionSignalEngine.STATE_DIR, PositionSignalEngine.DATA_DIR,
            PositionSignalEngine.PUBLIC_DIR)
    PositionSignalEngine.STATE_DIR = state_dir
    PositionSignalEngine.DATA_DIR = data_dir
    PositionSignalEngine.PUBLIC_DIR = public_dir
    try:
        with open(os.path.join(state_dir, "positions.json"), "w") as f:
            json.dump({"holdings": [],
                       "watching": [{"ticker": "EXT", "theme": "X"}]}, f)
        # condition-5 basis (post-D-007): EXT's group in the universe — an
        # EMPTY data dir now fails the gate closed (total-outage guard), so
        # the fixture must exist for READY/EXTENDED_HOLD to be reachable
        with open(os.path.join(data_dir, "universe_active.json"), "w") as f:
            json.dump({"groups": {"TestGroup": {"tickers": ["EXT"]}},
                       "ranking": []}, f)
        import earnings_calendar
        old_gem = earnings_calendar.get_earnings_map
        earnings_calendar.get_earnings_map = lambda ts: {t: None for t in ts}

        extended = lambda t, period="6mo": _bars(*EXTENDED_DF_ARGS)
        calm = lambda t, period="6mo": _bars([100.0] * 28 + [101.2, 101.4],
                                             spread=3.0)
        regime = {"regime": TRENDING}

        def events():
            with open(os.path.join(data_dir, "position_events.json")) as f:
                return json.load(f)["changes"]

        eng = PositionSignalEngine(copy.deepcopy(CONFIG), extended)
        r1 = eng.compute(regime, None)
        assert r1["tickers"]["EXT"]["state"] == "EXTENDED_HOLD"
        assert len(r1["transitions"]) == 1          # untracked -> EXTENDED_HOLD
        assert r1["transitions"][0]["detail"]["extension_guard"]
        assert len(events()) == 1

        r2 = eng.compute(regime, None)              # same extension: no re-emit
        assert r2["transitions"] == []
        assert len(events()) == 1, "flapping: event re-emitted at constant extension"

        eng_calm = PositionSignalEngine(copy.deepcopy(CONFIG), calm)
        r3 = eng_calm.compute(regime, None)         # extension fell back
        assert r3["tickers"]["EXT"]["state"] == RE_ENTRY_READY
        assert len(r3["transitions"]) == 1
        assert r3["transitions"][0]["detail"]["from_state"] == "EXTENDED_HOLD"
        assert r3["transitions"][0]["detail"]["to_state"] == RE_ENTRY_READY
        assert len(events()) == 2
        earnings_calendar.get_earnings_map = old_gem
    finally:
        (PositionSignalEngine.STATE_DIR, PositionSignalEngine.DATA_DIR,
         PositionSignalEngine.PUBLIC_DIR) = olds
        shutil.rmtree(tmp, ignore_errors=True)
    print("  extension guard: one event per crossing, no flapping: OK")


# ------------------------------------------------------------------
# MRVL June replay — pinned against real recorded bars
# ------------------------------------------------------------------

FIXTURE_BARS = [
    ("2026-04-28", 147.91, 156.00, 146.85, 153.23),
    ("2026-04-29", 153.77, 157.21, 151.30, 156.57),
    ("2026-04-30", 160.34, 165.61, 156.36, 165.15),
    ("2026-05-01", 162.35, 166.39, 159.26, 164.95),
    ("2026-05-04", 165.48, 166.82, 162.26, 163.66),
    ("2026-05-05", 168.28, 172.98, 164.58, 168.75),
    ("2026-05-06", 172.60, 175.80, 165.00, 172.15),
    ("2026-05-07", 171.20, 171.52, 158.55, 160.01),
    ("2026-05-08", 164.69, 170.59, 162.90, 170.13),
    ("2026-05-11", 163.67, 174.16, 162.49, 170.84),
    ("2026-05-12", 165.46, 168.73, 157.96, 164.50),
    ("2026-05-13", 169.09, 182.31, 168.91, 177.95),
    ("2026-05-14", 180.88, 192.15, 177.33, 182.58),
    ("2026-05-15", 173.90, 182.14, 173.34, 176.89),
    ("2026-05-18", 181.77, 182.71, 165.10, 168.93),
    ("2026-05-19", 164.61, 181.64, 162.85, 176.27),
    ("2026-05-20", 183.45, 193.32, 182.28, 186.80),
    ("2026-05-21", 192.51, 194.58, 188.20, 190.69),
    ("2026-05-22", 194.72, 198.40, 192.22, 196.33),
    ("2026-05-26", 211.24, 217.45, 200.04, 208.26),
    ("2026-05-27", 217.98, 218.26, 196.25, 198.70),
    ("2026-05-28", 198.75, 207.40, 194.70, 204.83),
    ("2026-05-29", 204.44, 208.76, 199.20, 205.00),
    ("2026-06-01", 198.91, 225.14, 195.12, 219.43),
    ("2026-06-02", 253.46, 291.30, 252.43, 290.79),
    ("2026-06-03", 317.63, 324.20, 294.01, 301.65),
    ("2026-06-04", 282.95, 321.50, 277.56, 316.43),
    ("2026-06-05", 299.50, 300.72, 261.39, 263.47),
    ("2026-06-08", 288.69, 304.96, 281.36, 288.85),
    ("2026-06-09", 299.76, 302.40, 244.00, 266.88),
    ("2026-06-10", 263.50, 272.47, 252.26, 252.59),
    ("2026-06-11", 260.54, 282.32, 258.43, 280.71),
    ("2026-06-12", 270.07, 287.98, 267.31, 279.70),
    ("2026-06-15", 296.71, 312.98, 288.09, 308.88),
    ("2026-06-16", 299.13, 317.00, 278.13, 278.67),
    ("2026-06-17", 292.89, 307.37, 283.32, 289.54),
    ("2026-06-18", 305.47, 329.88, 302.36, 310.58),
    ("2026-06-22", 313.39, 314.17, 298.18, 307.86),
    ("2026-06-23", 278.82, 290.95, 276.25, 279.04),
    ("2026-06-24", 281.95, 281.95, 263.66, 276.70),
    ("2026-06-25", 291.18, 292.51, 263.72, 281.26),
    ("2026-06-26", 268.37, 274.20, 262.00, 266.77),
    ("2026-06-29", 272.41, 278.28, 250.52, 277.75),
    ("2026-06-30", 278.75, 300.00, 275.50, 297.89),
    ("2026-07-01", 282.47, 292.50, 271.35, 272.05),
    ("2026-07-02", 269.50, 274.95, 237.20, 245.29),
]

# New-gauge regime per day (approved ladder replayed on recorded voters)
REPLAY_REGIME = {
    "2026-06-18": TRENDING, "2026-06-22": CHOPPY, "2026-06-23": TRENDING,
    "2026-06-24": CAUTION, "2026-06-25": CAUTION, "2026-06-26": CAUTION,
    "2026-06-29": CAUTION, "2026-06-30": CAUTION, "2026-07-01": CHOPPY,
    "2026-07-02": TRENDING,
}

EXPECTED_TRACE = {
    "2026-06-18": HELD, "2026-06-22": HELD, "2026-06-23": HELD,
    "2026-06-24": HELD, "2026-06-25": HELD,
    "2026-06-26": EXIT_FIRED,          # close 266.77 < SMA20 278.24
    "2026-06-29": WATCHING,
    "2026-06-30": RE_ENTRY_ARMING,     # reclaim 297.89; 1/2 confirmations;
                                       # regime Caution (approved ladder) = blocked
    "2026-07-01": WATCHING,
    "2026-07-02": WATCHING,
}


def _fixture_df():
    idx = pd.to_datetime([b[0] for b in FIXTURE_BARS])
    return pd.DataFrame({
        "Open": [b[1] for b in FIXTURE_BARS],
        "High": [b[2] for b in FIXTURE_BARS],
        "Low": [b[3] for b in FIXTURE_BARS],
        "Close": [b[4] for b in FIXTURE_BARS],
        "Volume": [0] * len(FIXTURE_BARS),
    }, index=idx)


def test_mrvl_june_replay():
    eng = _engine()
    df = _fixture_df()
    entry = {"ticker": "MRVL", "theme": "Semis", "stop_on_entry": "sma20_close"}
    semis = {"met": True, "detail": "theme 'Semis' ranked #1 (top 2)"}
    prev = None
    for day, regime in REPLAY_REGIME.items():
        window = df[df.index <= pd.Timestamp(day)]
        r = eng.evaluate(entry, "holding", window, regime, semis, prev)
        assert r["state"] == EXPECTED_TRACE[day], \
            f"{day}: expected {EXPECTED_TRACE[day]}, got {r['state']}"
        prev = r["state"]
        if day == "2026-06-26":
            assert r["close"] == 266.77 and r["sma20"] == 278.24
            assert r["stop"]["level"] == 278.24   # the stop that fired
        if day == "2026-06-30":
            c = r["conditions"]
            assert c["1_trigger"]["met"]            # reclaim
            assert not c["2_confirmation"]["met"]   # 1/2, no ATR break
            assert not c["3_regime_gate"]["met"]    # Caution -> blocked
            assert c["4_slope"]["met"]
            assert c["5_thesis"]["met"]
            assert r["conditions_met"] == 3
    print("  MRVL June replay pinned (real bars, approved-ladder regimes): OK")


# ------------------------------------------------------------------
# compute(): persistence + history events
# ------------------------------------------------------------------

def test_compute_persists_and_emits():
    tmp = tempfile.mkdtemp(prefix="pos_test_")
    state_dir = os.path.join(tmp, "state")
    data_dir = os.path.join(tmp, "data")
    public_dir = os.path.join(tmp, "public")
    for d in (state_dir, data_dir, public_dir):
        os.makedirs(d)
    olds = (PositionSignalEngine.STATE_DIR, PositionSignalEngine.DATA_DIR,
            PositionSignalEngine.PUBLIC_DIR)
    PositionSignalEngine.STATE_DIR = state_dir
    PositionSignalEngine.DATA_DIR = data_dir
    PositionSignalEngine.PUBLIC_DIR = public_dir
    try:
        with open(os.path.join(state_dir, "positions.json"), "w") as f:
            json.dump({"schema_version": "1.0", "account": "TOS",
                       "holdings": [{"ticker": "AAA", "theme": "Semis",
                                     "stop_on_entry": "sma20_close"}],
                       "watching": [{"ticker": "BBB", "theme": "SmallCap/Broad"},
                                    {"ticker": "CCC", "theme": "Biotech"}]},
                      f)
        # qualified_themes still written — proving the gate NO LONGER reads
        # it (D-007 Phase 1: condition 5 is the universe, themes display-only)
        with open(os.path.join(state_dir, "qualified_themes.json"), "w") as f:
            json.dump({"active": [{"name": "Semis", "proxy": "SMH"}]}, f)
        # condition-5 basis: the current universe + dashboard artifacts
        with open(os.path.join(data_dir, "universe_active.json"), "w") as f:
            json.dump({"groups": {"Semiconductors": {
                           "tickers": ["AAA"], "weeks_in_universe": 3}},
                       "ranking": [{"name": "Biotechnology",
                                    "tickers": [{"ticker": "CCC"}]}]}, f)
        with open(os.path.join(data_dir, "signals.json"), "w") as f:
            json.dump({"groups": [{"name": "Semiconductors",
                                   "breaker_status": "clear",
                                   "stocks": [{"ticker": "AAA"}]}]}, f)

        def fetcher(ticker, period="6mo"):
            if ticker == "AAA":
                return _bars(np.linspace(90, 110, 40))    # above SMA -> HELD
            if ticker == "BBB":
                return _bars(list(np.linspace(110, 100, 35)) + [95.0])  # below
            return None                                    # CCC: fetch failure

        eng = PositionSignalEngine(copy.deepcopy(CONFIG), fetcher)
        regime = {"regime": TRENDING}
        themes = {"ranked_themes": [{"name": "Semis", "rank": 1},
                                    {"name": "Biotech", "rank": 2}]}

        # Keep tests offline: stub the earnings map (PER-510) — AAA reports
        # earnings in 3 days, others unknown
        import earnings_calendar
        old_gem = earnings_calendar.get_earnings_map
        aaa_er = (datetime.date.today() + datetime.timedelta(days=3)).isoformat()
        earnings_calendar.get_earnings_map = \
            lambda ts: {t: (aaa_er if t == "AAA" else None) for t in ts}

        r1 = eng.compute(regime, themes)
        # Earnings layer: fields present; note only on HELD/READY with ER ≤7d
        assert r1["tickers"]["AAA"]["days_to_earnings"] == 3
        assert r1["tickers"]["AAA"]["earnings_note"] == \
            "earnings in 3d — R8: binary catalyst window"
        assert r1["tickers"]["BBB"]["days_to_earnings"] is None
        assert "earnings_note" not in r1["tickers"]["BBB"]   # WATCHING + no date
        assert r1["tickers"]["AAA"]["state"] == HELD
        assert r1["tickers"]["AAA"]["stop"]["type"] == "sma20_close"
        # thesis via the UNIVERSE gate (D-007): group in current universe
        assert r1["tickers"]["AAA"]["conditions"]["5_thesis"]["met"]
        assert "in universe" in r1["tickers"]["AAA"]["conditions"]["5_thesis"]["detail"]
        assert r1["tickers"]["AAA"]["group"] == "Semiconductors"
        assert r1["tickers"]["AAA"]["weeks_in_universe"] == 3
        assert r1["tickers"]["BBB"]["state"] == WATCHING
        assert "distance_to_sma20_pct" in r1["tickers"]["BBB"]
        assert r1["tickers"]["BBB"]["extension_pct"] < 0   # below SMA20
        assert r1["tickers"]["AAA"]["extension_pct"] > 0   # above SMA20
        assert r1["tickers"]["BBB"]["conditions"]["5_thesis"].get("no_group_mapping")
        # D-011: watchers carry the grade (BBB below SMA20 -> conditions fail -> C)
        assert r1["tickers"]["BBB"]["grade"]["grade"] == "C"
        assert r1["tickers"]["BBB"]["grade_inputs"] is not None
        assert "grade" not in r1["tickers"]["AAA"]          # holdings ungraded
        # fetch failure: reported, but never seeds/poisons persisted state
        assert r1["tickers"]["CCC"].get("insufficient_data") is True
        assert len(r1["transitions"]) == 2   # AAA + BBB initialized, not CCC

        # events go to position_events.json (SEPARATE from history.json —
        # history_manager rewrites that file and would clobber a co-writer)
        assert not os.path.exists(os.path.join(data_dir, "history.json"))
        with open(os.path.join(data_dir, "position_events.json")) as f:
            hist = json.load(f)
        assert len(hist["changes"]) == 2
        ev = [e for e in hist["changes"] if e["ticker"] == "AAA"][0]
        assert ev["type"] == "position_state_change"
        assert ev["detail"]["to_state"] == HELD
        assert ev["severity"] == "medium"
        assert ev["detail"]["extension_pct"] is not None   # surfaced in events
        assert ev["detail"]["extension_atr"] is not None
        with open(os.path.join(public_dir, "position_events.json")) as f:
            assert len(json.load(f)["changes"]) == 2   # mirrored to public

        # second run, nothing changed: no new transitions, no new events
        r2 = eng.compute(regime, themes)
        assert r2["transitions"] == []
        with open(os.path.join(data_dir, "position_events.json")) as f:
            assert len(json.load(f)["changes"]) == 2

        with open(os.path.join(state_dir, "position_state.json")) as f:
            st = json.load(f)
        assert st["AAA"]["state"] == HELD and st["BBB"]["state"] == WATCHING
        assert "CCC" not in st

        # unreadable positions.json: engine skips, state preserved
        with open(os.path.join(state_dir, "positions.json"), "w") as f:
            f.write("{not json")
        r3 = eng.compute(regime, themes)
        assert "error" in r3 and r3["tickers"] == {}
        with open(os.path.join(state_dir, "position_state.json")) as f:
            assert "AAA" in json.load(f)   # state NOT wiped
    finally:
        (PositionSignalEngine.STATE_DIR, PositionSignalEngine.DATA_DIR,
         PositionSignalEngine.PUBLIC_DIR) = olds
        earnings_calendar.get_earnings_map = old_gem
        shutil.rmtree(tmp, ignore_errors=True)
    print("  compute(): state persisted, events emitted once, earnings layer: OK")


# ------------------------------------------------------------------
# D-011 — the A+ grade (grade_setup fixtures from the decision record)
# ------------------------------------------------------------------

def test_runway_sessions():
    from framework.position_signals import runway_sessions_before as rsb
    # THE record's arithmetic (D-011 amendment 2): MRNA evaluated
    # 2026-07-12 (Sun) / 07-13 (Mon) vs the 2026-07-31 print -> 14 sessions
    # STRICTLY before the print day
    assert rsb("2026-07-31", datetime.date(2026, 7, 13)) == 14
    assert rsb("2026-07-31", datetime.date(2026, 7, 12)) == 14   # Sunday eval
    # ARWR Jul-6 anatomy: ~31 days out -> comfortably >= 15
    assert rsb("2026-08-18", datetime.date(2026, 7, 6)) == 31
    # THE PRINT DAY: 0 sessions — must FAIL the bar, never read as
    # "no known print" (review finding)
    assert rsb("2026-07-13", datetime.date(2026, 7, 13)) == 0
    # past print -> None (re-qualified); None/garbage -> None
    assert rsb("2026-07-01", datetime.date(2026, 7, 13)) is None
    assert rsb(None) is None
    assert rsb("not-a-date") is None
    print("  runway (amendment 2): MRNA=14 strictly-before-print, Sunday "
          "eval, edges: OK")


def test_grade_fixtures():
    from framework.position_signals import grade_setup
    ARWR = dict(all_conditions_met=True, extension_atr=1.64, close=76.0,
                sma5=74.0, up_close_since_swing_low=True, rsi14=62.0,
                quality_score=86, breaker_status="clear", runway_sessions=31)
    MRNA = dict(all_conditions_met=True, extension_atr=0.43, close=68.27,
                sma5=76.0, up_close_since_swing_low=True, rsi14=55.0,
                quality_score=55, breaker_status="critical",
                runway_sessions=14)

    # ARWR Jul-6 anatomy (the A+ taken): all seven rows -> A+
    g = grade_setup(**ARWR)
    assert g["grade"] == "A+" and not g["failing"], g

    # MRNA Jul-12 anatomy (the knife): approach fail ESCALATES to C
    # (amendment 1) with score/breaker/runway as further named failures
    g = grade_setup(**MRNA)
    assert g["grade"] == "C", g
    assert "3_approach" in g["failing"]
    assert "5_score" in g["failing"] and "6_breaker" in g["failing"]
    assert "7_runway" in g["failing"]
    assert g["rows"]["4_rsi"]["met"]     # RSI 55 IS in the 45-70 band (record †)

    # B case: all pass except score 72 -> B with the named reason
    g = grade_setup(**dict(ARWR, quality_score=72))
    assert g["grade"] == "B" and g["failing"] == ["5_score"], g
    assert "72" in g["reasons"]

    # runway boundary (strictly-before-print): 14 -> B named; 15 -> pass
    g = grade_setup(**dict(ARWR, runway_sessions=14))
    assert g["grade"] == "B" and g["failing"] == ["7_runway"]
    g = grade_setup(**dict(ARWR, runway_sessions=15))
    assert g["grade"] == "A+"

    # C-escalation is regime-INDEPENDENT: the pure function has no regime
    # input at all — a knife grades C everywhere (Trending included); the
    # endpoint test proves it through a Trending simulate call
    g = grade_setup(**dict(ARWR, close=70.0, sma5=74.0))
    assert g["grade"] == "C" and "3_approach" in g["failing"]

    # conditions fail -> C; guard fail -> C
    g = grade_setup(**dict(ARWR, all_conditions_met=False))
    assert g["grade"] == "C"
    g = grade_setup(**dict(ARWR, extension_atr=2.1))
    assert g["grade"] == "C"

    # rsi boundaries inclusive: 45 and 70 pass, 44.9 / 70.1 fail (B)
    assert grade_setup(**dict(ARWR, rsi14=45.0))["grade"] == "A+"
    assert grade_setup(**dict(ARWR, rsi14=70.0))["grade"] == "A+"
    assert grade_setup(**dict(ARWR, rsi14=44.9))["grade"] == "B"

    # unavailable data can never be A+ (proven, not defaulted) — but the
    # index-vehicle waiver passes the score row
    g = grade_setup(**dict(ARWR, rsi14=None))
    assert g["grade"] == "B" and "4_rsi" in g["failing"]
    g = grade_setup(**dict(ARWR, quality_score=None, score_waived=True))
    assert g["grade"] == "A+" and g["rows"]["5_score"].get("waived")
    # no known print -> unbounded runway, passes
    g = grade_setup(**dict(ARWR, runway_sessions=None))
    assert g["grade"] == "A+"
    # print DAY: runway 0 -> fails the bar (review finding)
    g = grade_setup(**dict(ARWR, runway_sessions=0))
    assert g["grade"] == "B" and "7_runway" in g["failing"]
    # approach data unavailable -> B (named), NOT the C-escalation — the
    # amendment escalates a PROVEN knife, not an unknown (review finding)
    g = grade_setup(**dict(ARWR, sma5=None))
    assert g["grade"] == "B" and "3_approach" in g["failing"]
    print("  grade fixtures: ARWR A+ / MRNA C-escalation / B named / runway "
          "14-15 boundary / RSI bounds / waiver / unknowns: OK")


def test_grade_gate_choppy():
    """Q4 enforcement through evaluate(): Choppy READY without A+ carries
    grade_gate (renders blocked); Trending READY does not (advisory)."""
    eng = _engine()
    df = _bars(np.linspace(90, 110, 45), spread=4.0)   # READY-able series
    ctx = {"breaker_status": "clear", "next_earnings_date": None,
           "score_waived": True}                        # waive score (synthetic df)
    r = eng.evaluate({}, "watching", df, CHOPPY, UNMAPPED, WATCHING,
                     grade_ctx=ctx)
    assert r["state"] == RE_ENTRY_READY and r["a_plus_only"] is True
    assert "grade" in r
    if r["grade"]["grade"] != "A+":
        assert "READY blocked" in r["grade_gate"]
    else:
        assert "grade_gate" not in r
    # Trending: advisory — same grade, never gated
    r2 = eng.evaluate({}, "watching", df, TRENDING, UNMAPPED, WATCHING,
                      grade_ctx=ctx)
    assert r2["state"] == RE_ENTRY_READY
    assert "grade_gate" not in r2
    # holdings never carry a grade
    r3 = eng.evaluate({}, "holding", df, CHOPPY, UNMAPPED, None,
                      grade_ctx=ctx)
    assert "grade" not in r3
    print("  Q4 enforcement: Choppy hard gate (READY blocked without A+), "
          "Trending advisory, holdings ungraded: OK")


def test_review_hardening():
    """Review-finding pins: weeks counter week-keyed; extension-boundary
    consistency; malformed-aplus sanitize; resolver defensiveness."""
    from universe_builder import _weeks_in_universe
    prev = {"Semis": {"weeks_in_universe": 4}, "Old": {}}
    # new week: present advances, field-less seeds 1->2, new group starts 1
    assert _weeks_in_universe("Semis", prev, "2026-07-10", "2026-07-17") == 5
    assert _weeks_in_universe("Old", prev, "2026-07-10", "2026-07-17") == 2
    assert _weeks_in_universe("New", prev, "2026-07-10", "2026-07-17") == 1
    # SAME-week rebuild (--force): carries unchanged — never a phantom week
    assert _weeks_in_universe("Semis", prev, "2026-07-17", "2026-07-17") == 4
    assert _weeks_in_universe("New", prev, "2026-07-17", "2026-07-17") == 1
    # no previous artifact
    assert _weeks_in_universe("Semis", {}, None, "2026-07-17") == 1

    # extension boundary: guard (unrounded, strict >) and grade row 2 agree —
    # 1.803xATR fires the guard AND fails row 2 (never EXTENDED_HOLD + A+)
    from framework.position_signals import grade_setup
    g = grade_setup(all_conditions_met=True, extension_atr=1.803, close=101.8,
                    sma5=100.5, up_close_since_swing_low=True, rsi14=60.0,
                    quality_score=86, breaker_status="clear",
                    runway_sessions=31)
    assert g["grade"] == "C" and "2_extension" in g["failing"]

    # malformed aplus config sanitizes to ruled defaults (never crashes)
    cfg = copy.deepcopy(CONFIG)
    cfg["positions"]["aplus"] = {"rsi_min": None, "index_vehicles": None}
    eng = PositionSignalEngine(cfg, lambda t, period="6mo": None)
    assert eng.aplus_cfg["rsi_min"] == 45.0
    assert "SPY" in eng.aplus_cfg["index_vehicles"]
    df = _bars(np.linspace(90, 110, 45), spread=4.0)
    r = eng.evaluate({}, "watching", df, CHOPPY, UNMAPPED, WATCHING,
                     grade_ctx={"breaker_status": "clear",
                                "next_earnings_date": None,
                                "score_waived": True})
    assert "grade" in r or "grade_error" in r      # never raises

    # resolver: parseable-but-wrong shapes contribute nothing, never raise
    from framework.portfolio_rules import resolve_group_map
    assert resolve_group_map([], "junk") == {}
    assert resolve_group_map({"ranking": [None, {"tickers": "x"}],
                              "groups": "bad"},
                             {"groups": [None, {"stocks": "y"}]}) == {}
    print("  review hardening: week-keyed counter, guard/grade boundary "
          "agreement, aplus sanitize, resolver shapes: OK")


def test_watchers_replay_real_artifacts():
    """THE Phase-1 license pin, on the REAL COMMITTED artifacts.

    Reads framework.json + universe_active.json + signals.json from git HEAD
    (not the working tree) so the pin is immune to sweep-order artifact
    mutation (review finding: test_pipeline regenerates signals.json
    mid-sweep) and always asserts against the genuinely recorded state.

    The invariants pinned (rotation-proof, no dated time bomb):
      - STATES replay identically through the rewired engine
      - conditions 1-4 met-status identical
      - condition 5 equals the shared resolver's verdict on the same
        committed artifacts (data-driven, not hardcoded)
    PLUS the dated exhibit, asserted only while it holds: the 2026-07-18
    rotation dropped Biotechnology, so MRNA's c5 diverges (old True ->
    honest False) — the first live D-007 divergence. If a later rotation
    readmits Biotechnology, the sub-assertion self-retires with a note.
    """
    import subprocess
    from framework.position_signals import assess_position
    from framework.portfolio_rules import resolve_group_map

    def _head(path):
        out = subprocess.run(["git", "show", f"HEAD:{path}"],
                             capture_output=True, text=True,
                             cwd=os.path.dirname(os.path.abspath(__file__)))
        assert out.returncode == 0, f"git show failed for {path}"
        return json.loads(out.stdout)

    fw = _head("public/framework.json")
    universe = _head("data/universe_active.json")
    signals = _head("data/signals.json")
    rows = (fw.get("position_signals") or {}).get("tickers") or {}
    assert "ARWR" in rows and "MRNA" in rows, "watchers missing from artifact"

    gmap = resolve_group_map(universe, signals)
    selected = set((universe.get("groups") or {}).keys())
    eng = _engine()

    for t in ("ARWR", "MRNA"):
        row = rows[t]
        si = dict(row["assess_inputs"])
        old_c5 = si.pop("theme_qualified", None)
        if old_c5 is None:
            old_c5 = si.pop("group_in_universe")   # post-rewire artifact
        thesis = eng._group_status({"ticker": t, "theme": row.get("theme")},
                                   gmap, selected, {}, len(selected) or 15)
        got = assess_position(
            si["close"], si["sma20"], si["sma20_5d_ago"], si["atr14"],
            si["consecutive_closes_above"], si["regime_state"],
            thesis["met"], si["kind"], thesis_detail=thesis["detail"])
        # states identical — the license
        assert got["state"] == row["state"], \
            f"{t}: {got['state']} != recorded {row['state']}"
        # conditions 1-4 met-status identical
        for k in ("1_trigger", "2_confirmation", "3_regime_gate", "4_slope"):
            assert got["conditions"][k]["met"] == \
                row["conditions"][k]["met"], f"{t}:{k}"
        # condition 5 == the resolver's verdict (data-driven invariant)
        assert got["conditions"]["5_thesis"]["met"] == thesis["met"], t

    # the dated exhibit — self-retiring when Biotechnology rotates back in
    if "Biotechnology" not in selected and gmap.get("MRNA") == "Biotechnology":
        m = eng._group_status({"ticker": "MRNA", "theme": "Biotech"},
                              gmap, selected, {}, len(selected) or 15)
        assert m["met"] is False, "MRNA divergence exhibit regressed"
        note = "MRNA divergence exhibit ACTIVE (Biotechnology out since 2026-07-18)"
    else:
        note = "divergence exhibit retired (Biotechnology back in universe)"
    print(f"  REPLAY PIN (committed artifacts): ARWR+MRNA states + c1-c4 "
          f"identical, c5 == resolver verdict; {note}: OK")


# ------------------------------------------------------------------
# D-017 Candidates tier: shared input helpers, emission parity, the
# stateless grade pass, and THE parity pin (stateless == watcher path
# on the same inputs)
# ------------------------------------------------------------------

def test_grade_inputs_helpers():
    """The extracted helpers + grade_inputs_from_df: same values the
    watcher path derives, None on unusable history, and the emission
    (signal_engine.compute_grade_inputs) strips the synthetic bar."""
    from framework.position_signals import (
        grade_inputs_from_df, atr_mean, consec_closes_above,
        up_close_off_swing_low, strip_synthetic_last_bar)

    closes = [100 + 0.4 * i for i in range(40)]      # steady uptrend
    closes[-3] = closes[-4] - 2.0                     # dip -> swing low
    df = _bars(closes)
    gi = grade_inputs_from_df(df)
    assert gi is not None
    close = df["Close"]
    sma = close.rolling(20).mean()
    assert gi["close"] == float(close.iloc[-1])
    assert gi["sma20"] == float(sma.iloc[-1])
    assert gi["sma20_5d_ago"] == float(sma.iloc[-6])
    assert gi["sma5"] == float(close.rolling(5).mean().iloc[-1])
    assert gi["atr14"] == atr_mean(df, 14) == PositionSignalEngine._atr(df, 14)
    assert gi["consecutive_closes_above"] == consec_closes_above(close, sma)
    assert gi["up_close_since_swing_low"] == \
        up_close_off_swing_low(close, 20) is True
    assert gi["params"] == {"sma_period": 20, "slope_lookback": 5,
                            "atr_period": 14, "swing_lookback": 20}
    # unusable history -> None (the unavailable-data convention starts here)
    assert grade_inputs_from_df(None) is None
    assert grade_inputs_from_df(_bars(closes[:10])) is None

    # emission: rsi/score added, synthetic live-quote bar stripped (the
    # close-basis law — grades are on confirmed closes)
    from signal_engine import compute_grade_inputs
    emitted = compute_grade_inputs(df)
    for k, v in gi.items():
        assert emitted[k] == v, f"emission drift on {k}"
    assert emitted["rsi14"] is not None
    assert emitted["quality_score"] is not None
    syn = df.iloc[-1].copy()
    px = float(df["Close"].iloc[-1]) + 3.0
    syn_row = pd.DataFrame({"Open": [px], "High": [px], "Low": [px],
                            "Close": [px], "Volume": [0]},
                           index=[df.index[-1] + pd.Timedelta(days=1)])
    df_syn = pd.concat([df, syn_row])
    assert strip_synthetic_last_bar(df_syn).equals(df)
    assert compute_grade_inputs(df_syn) == emitted
    print("  D-017 grade-input helpers + emission (synthetic bar stripped): OK")


def _cand_gi(eng, **over):
    """A+-shaped grade_inputs under Trending + in-universe (each test
    breaks one thing)."""
    gi = {"close": 25.0, "sma20": 24.0, "sma20_5d_ago": 23.5, "sma5": 24.8,
          "atr14": 1.0, "consecutive_closes_above": 2,
          "up_close_since_swing_low": True, "rsi14": 55.0,
          "quality_score": 84.0,
          "params": {"sma_period": eng.sma_period,
                     "slope_lookback": eng.slope_lookback,
                     "atr_period": eng.atr_period,
                     "swing_lookback": eng.aplus_cfg["approach_swing_lookback"]}}
    gi.update(over)
    return gi


def test_candidates_tier():
    """grade_candidates: the stateless grade pass over un-tracked signals
    rows — tracked excluded, era-aware None on pre-emission artifacts,
    honest nulls, c5 from the current universe, outage fails closed."""
    eng = _engine()
    today = datetime.date(2026, 7, 18)
    ned = "2026-09-10"                       # runway >> 15 sessions
    signals = {"groups": [
        {"name": "Tech Hardware", "stocks": [
            {"ticker": "HPQ", "next_earnings_date": ned,
             "grade_inputs": _cand_gi(eng)},
            {"ticker": "NOGI", "next_earnings_date": ned},   # no emission
            {"ticker": "TRK", "next_earnings_date": ned,
             "grade_inputs": _cand_gi(eng)},                  # tracked
            {"ticker": "KNIFE", "next_earnings_date": ned,
             "grade_inputs": _cand_gi(eng, sma5=25.5)},       # close < SMA5
            {"ticker": "BADP", "next_earnings_date": ned,
             "grade_inputs": _cand_gi(eng, params={"sma_period": 50})},
        ]},
        {"name": "Soft", "stocks": [
            {"ticker": "SFT", "next_earnings_date": ned,
             "grade_inputs": _cand_gi(eng)},                  # breaker unknown
            {"ticker": "RSIU", "next_earnings_date": ned,
             "grade_inputs": _cand_gi(eng, rsi14=None)},      # rsi unavailable
            {"ticker": "SC0", "next_earnings_date": ned,
             "grade_inputs": _cand_gi(eng, quality_score=None)},
            {"ticker": "AP0", "next_earnings_date": ned,
             "grade_inputs": _cand_gi(eng, sma5=None)},       # approach unavail
        ]},
        {"name": "Outside", "stocks": [
            {"ticker": "OUT1", "next_earnings_date": ned,
             "grade_inputs": _cand_gi(eng)},                  # c5 fails
        ]},
    ]}
    selected = {"Tech Hardware", "Soft"}
    breakers = {"Tech Hardware": "clear", "Outside": "clear"}
    out = eng.grade_candidates(signals, selected, breakers, TRENDING,
                               tracked={"TRK"}, today=today)
    assert "TRK" not in out, "tracked names must be EXCLUDED"
    assert out["HPQ"]["grade"] == "A+" and out["HPQ"]["group"] == "Tech Hardware"
    assert out["NOGI"]["grade"] is None and "not emitted" in out["NOGI"]["reasons"]
    # amendment 1 via the candidate path: a proven knife is C in EVERY regime
    assert out["KNIFE"]["grade"] == "C" and "3_approach" in out["KNIFE"]["failing"]
    assert out["BADP"]["grade"] is None and "parameters" in out["BADP"]["reasons"]
    assert out["SFT"]["grade"] == "B" and "6_breaker" in out["SFT"]["failing"]
    # unavailable-data-never-A+ per FIELD (review finding): unknowns grade
    # honest-B with the row named — never A+ (defaulted) and never C
    # (the amendment escalates the proven knife, not the unknown)
    assert out["RSIU"]["grade"] == "B" and "4_rsi" in out["RSIU"]["failing"] \
        and "RSI unavailable" in out["RSIU"]["reasons"]
    assert out["SC0"]["grade"] == "B" and "5_score" in out["SC0"]["failing"]
    assert out["AP0"]["grade"] == "B" and "3_approach" in out["AP0"]["failing"] \
        and "approach unavailable" in out["AP0"]["reasons"]
    # c5 failure names its REAL basis in the reasons (review finding)
    assert out["OUT1"]["grade"] == "C" and \
        "1_conditions" in out["OUT1"]["failing"] and \
        "not in current universe" in out["OUT1"]["reasons"]
    # era-aware: NO row emitted -> None (block omitted, never fabricated)
    bare = {"groups": [{"name": "Tech Hardware",
                        "stocks": [{"ticker": "HPQ"}]}]}
    assert eng.grade_candidates(bare, selected, breakers, TRENDING,
                                tracked=set(), today=today) is None
    # universe outage: c5 fails CLOSED -> C, and the reasons say OUTAGE,
    # never a generic "conditions not met" masquerading as a weak tape
    out2 = eng.grade_candidates(signals, set(), breakers, TRENDING,
                                tracked=set(), today=today)
    assert out2["HPQ"]["grade"] == "C" and "1_conditions" in out2["HPQ"]["failing"]
    assert "universe unavailable" in out2["HPQ"]["reasons"]
    # regime threads through the candidate path (review finding): blocked
    # regime -> conditions fail -> C; Choppy (conditional) -> grade stands
    outC = eng.grade_candidates(signals, selected, breakers, CAUTION,
                                tracked=set(), today=today)
    assert outC["HPQ"]["grade"] == "C" and "1_conditions" in outC["HPQ"]["failing"]
    outCh = eng.grade_candidates(signals, selected, breakers, CHOPPY,
                                 tracked=set(), today=today)
    assert outCh["HPQ"]["grade"] == "A+"
    print("  D-017 candidates tier fixtures (exclusion, honest nulls + "
          "per-field unknowns, c5 reasons, outage, regime threading): OK")


def test_candidate_watcher_parity():
    """THE D-017 pin: the stateless candidate path and the stateful
    watcher path produce the SAME grade from the SAME inputs. Era-aware:
    replays every watcher in the live artifact that carries grade +
    grade_inputs + assess_inputs + group; skips honestly otherwise."""
    import yaml
    from framework.position_signals import runway_sessions_before
    art_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "framework", "output", "latest.json")
    if not os.path.exists(art_path):
        print("  D-017 PARITY PIN: no framework artifact — skipped")
        return
    with open(art_path) as f:
        art = json.load(f)
    ps = art.get("position_signals") or {}
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "framework", "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    eng = PositionSignalEngine(cfg, lambda t, period="6mo": None)
    base_day = None
    try:
        base_day = datetime.date.fromisoformat(
            str(art.get("generated_at", ""))[:10])
    except ValueError:
        pass
    if base_day is None:
        print("  D-017 PARITY PIN: artifact has no parsable generated_at "
              "— skipped")
        return
    checked = []
    eligible = []
    for t, row in (ps.get("tickers") or {}).items():
        ai, gi, g = (row.get("assess_inputs"), row.get("grade_inputs"),
                     row.get("grade"))
        grp = row.get("group")
        if not (isinstance(ai, dict) and isinstance(gi, dict)
                and isinstance(g, dict) and grp
                and row.get("kind") == "watching"):
            continue
        eligible.append(t)
        if bool(gi.get("score_waived")) != (t in eng.aplus_cfg["index_vehicles"]):
            continue          # per-row vehicle override — not reconstructable
        # recover the evaluation day so the stateless runway matches the
        # recorded one (data-driven, no dated assertion)
        today = None
        ned = row.get("next_earnings_date")
        rec_runway = gi.get("runway_sessions")
        if ned is None or rec_runway is None:
            today = base_day
            if runway_sessions_before(ned, today) != rec_runway:
                continue
        else:
            for delta in (0, -1, 1, -2, 2):
                d = base_day + datetime.timedelta(days=delta)
                if runway_sessions_before(ned, d) == rec_runway:
                    today = d
                    break
            if today is None:
                continue
        cand_row = {"ticker": t, "next_earnings_date": ned,
                    "grade_inputs": {
                        "close": ai["close"], "sma20": ai["sma20"],
                        "sma20_5d_ago": ai["sma20_5d_ago"],
                        "atr14": ai["atr14"],
                        "consecutive_closes_above":
                            ai["consecutive_closes_above"],
                        "sma5": gi["sma5"],
                        "up_close_since_swing_low":
                            gi["up_close_since_swing_low"],
                        "rsi14": gi["rsi14"],
                        "quality_score": gi["quality_score"],
                        "params": {
                            "sma_period": eng.sma_period,
                            "slope_lookback": eng.slope_lookback,
                            "atr_period": eng.atr_period,
                            "swing_lookback":
                                eng.aplus_cfg["approach_swing_lookback"]}}}
        signals_fx = {"groups": [{"name": grp, "stocks": [cand_row]}]}
        selected = {grp} if ai.get("group_in_universe") else {"__other__"}
        out = eng.grade_candidates(
            signals_fx, selected, {grp: gi.get("breaker_status")},
            ai.get("regime_state"), tracked=set(), today=today)
        got = out[t]["grade"]
        want = g.get("grade")
        assert got == want, (
            f"D-017 PARITY BROKEN for {t}: candidate path graded {got}, "
            f"watcher path graded {want} on identical inputs")
        checked.append(f"{t}={want}")
    # RATCHET (review finding): era-skip is honest only while the artifact
    # is genuinely ungraded. Once graded watcher rows EXIST, replaying
    # zero of them means the pin quietly disengaged (renamed field,
    # broken today-recovery) — that must FAIL, never skip-green.
    if eligible:
        assert checked, (
            f"D-017 parity pin DISENGAGED: {len(eligible)} graded watcher "
            f"rows in the artifact ({', '.join(eligible)}) but 0 replayed")
        print(f"  D-017 PARITY PIN (stateless == watcher on same inputs): "
              f"{', '.join(checked)}"
              + (f"; {len(eligible) - len(checked)} skipped honestly"
                 if len(checked) < len(eligible) else "")
              + ": OK")
    else:
        print("  D-017 PARITY PIN: no graded watcher rows in artifact "
              "(pre-grade era) — skipped")


if __name__ == "__main__":
    print("\n=== Position signal engine tests (Build 1B) ===")
    test_confirmation_consecutive_closes()
    test_confirmation_via_atr_break()
    test_regime_gate_modes()
    test_slope_condition()
    test_group_status_variants()
    test_stop_display()
    test_full_cycle_watcher()
    test_full_cycle_holding_returns_to_held()
    test_ready_blocked_by_thesis()
    test_synthetic_last_bar_stripped()
    test_extension_guard_suppresses_extended_ready()
    test_extension_guard_permits_healthy_trending_distance()
    test_extension_guard_healthy_reclaim_unaffected()
    test_extension_guard_holdings_exempt()
    test_extension_guard_boundary_and_config()
    test_extension_guard_no_flapping_events()
    test_mrvl_june_replay()
    test_compute_persists_and_emits()
    test_runway_sessions()
    test_grade_fixtures()
    test_grade_gate_choppy()
    test_review_hardening()
    test_watchers_replay_real_artifacts()
    test_grade_inputs_helpers()
    test_candidates_tier()
    test_candidate_watcher_parity()
    print("\nAll position-signal tests passed.\n")
