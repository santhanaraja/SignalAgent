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

UNMAPPED = {"met": True, "no_theme_mapping": True, "detail": "no mapping"}


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
    # single close far above SMA: > SMA20 + 0.5*ATR(14) confirms same-day
    df = _bars([100.0] * 29 + [108.0])
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
    df = _bars(np.linspace(90, 110, 40))   # rising: trigger/confirm/slope met
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


def test_theme_status_variants():
    eng = _engine()
    wl = {"Semis", "Biotech"}
    s = eng._theme_status("Biotech", wl, ["Biotech"], {}, 2)
    assert s["met"] and "active" in s["detail"]
    s = eng._theme_status("Biotech", wl, [], {"Biotech": 2}, 2)
    assert s["met"] and "#2" in s["detail"]
    s = eng._theme_status("Biotech", wl, [], {"Biotech": 4}, 2)
    assert not s["met"]
    s = eng._theme_status("SmallCap/Broad", wl, [], {}, 2)
    assert s["met"] and s.get("no_theme_mapping") is True
    s = eng._theme_status("external:Uranium", wl, [], {}, 2)
    assert s["met"] and s.get("no_theme_mapping") is True
    print("  thesis condition: active / ranked / unqualified / unmapped: OK")


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
    for i in range(40, len(closes) + 1):
        r = eng.evaluate({}, kind, _bars(closes[:i]), TRENDING, UNMAPPED, prev)
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
        # qualified_themes read from STATE_DIR — the framework_runner call
        # path (no qualified_active argument)
        with open(os.path.join(state_dir, "qualified_themes.json"), "w") as f:
            json.dump({"active": [{"name": "Semis", "proxy": "SMH"}]}, f)

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
        # thesis via the qualified_themes.json load path (Semis active)
        assert r1["tickers"]["AAA"]["conditions"]["5_thesis"]["met"]
        assert "active" in r1["tickers"]["AAA"]["conditions"]["5_thesis"]["detail"]
        assert r1["tickers"]["BBB"]["state"] == WATCHING
        assert "distance_to_sma20_pct" in r1["tickers"]["BBB"]
        assert r1["tickers"]["BBB"]["extension_pct"] < 0   # below SMA20
        assert r1["tickers"]["AAA"]["extension_pct"] > 0   # above SMA20
        assert r1["tickers"]["BBB"]["conditions"]["5_thesis"].get("no_theme_mapping")
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


if __name__ == "__main__":
    print("\n=== Position signal engine tests (Build 1B) ===")
    test_confirmation_consecutive_closes()
    test_confirmation_via_atr_break()
    test_regime_gate_modes()
    test_slope_condition()
    test_theme_status_variants()
    test_stop_display()
    test_full_cycle_watcher()
    test_full_cycle_holding_returns_to_held()
    test_ready_blocked_by_thesis()
    test_synthetic_last_bar_stripped()
    test_mrvl_june_replay()
    test_compute_persists_and_emits()
    print("\nAll position-signal tests passed.\n")
