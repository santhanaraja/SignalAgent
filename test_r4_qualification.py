#!/usr/bin/env python3
"""
Tests for the R4 qualification fix: signals fire fast, state moves slow.

- Regime-degradation exit SIGNALS stay regime-triggered and advisory.
- Theme QUALIFICATION (qualified_themes.json) mutates only on the weekly
  (Sunday) review, and regime-degradation removals additionally require the
  confirmation protocol (consecutive degraded weekly closes).
- Consecutive-Sunday counters run over weekly closes, not daily snapshots.

Run: python3 test_r4_qualification.py
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

from framework import regime_calculator as rc_mod
from framework import theme_ranker as tr_mod
from framework.regime_calculator import RegimeCalculator
from framework.theme_ranker import ThemeRanker

TRENDING = "Risk-on / Trending"
CHOPPY = "Risk-on / Choppy"
CAUTION = "Caution"
RISK_OFF = "Risk-off"

CONFIG = {
    "regime": {
        "gauges": {},   # not exercised in these tests
        "states": [],
        "change_protocol": {"consecutive_confirmations_required": 2},
    },
    "themes": {
        "watchlist": [
            {"name": "Semis", "proxy": "SMH", "constituents": []},
            {"name": "Biotech", "proxy": "XBI", "constituents": []},
        ],
        "ranking": {"lookback_4w": 20, "lookback_12w": 60,
                    "composite_method": "rank_average", "max_active_themes": 2},
        "entry_rule": {"requires_top_n_rank": 2, "consecutive_sundays_required": 2,
                       "regime_required": [TRENDING, CHOPPY],
                       "discretionary_review_required": True},
        "exit_rule": {"drops_below_rank": 3, "consecutive_sundays_required": 3,
                      "regime_degradation_triggers_exit": True},
        "concentration_limits": {},
    },
}


def _df(closes):
    closes = list(closes)
    idx = pd.date_range(end="2026-07-02", periods=len(closes), freq="B")
    return pd.DataFrame({"Open": closes, "High": closes, "Low": closes,
                         "Close": closes, "Volume": [1000] * len(closes)}, index=idx)


def theme_fetcher(ticker, period="6mo"):
    """SMH outruns XBI; both have 130 days so both rank."""
    if ticker == "SMH":
        return _df(np.linspace(500, 650, 130))
    if ticker == "XBI":
        return _df(np.linspace(140, 160, 130))
    return None


class _Env:
    """Temp STATE_DIR + injectable today for ThemeRanker replays."""

    def __init__(self, active=None, last_review=None):
        self.tmp = tempfile.mkdtemp(prefix="r4_test_")
        self.old_state_dir = ThemeRanker.STATE_DIR
        ThemeRanker.STATE_DIR = self.tmp
        self.old_tr_today = tr_mod._today
        self.old_rc_today = rc_mod._today
        qualified = {"active": active or [], "pending": []}
        if last_review:
            qualified["last_weekly_review"] = last_review
        with open(os.path.join(self.tmp, "qualified_themes.json"), "w") as f:
            json.dump(qualified, f)

    def set_today(self, iso):
        d = datetime.date.fromisoformat(iso)
        tr_mod._today = lambda: d
        rc_mod._today = lambda: d

    def qualified(self):
        with open(os.path.join(self.tmp, "qualified_themes.json")) as f:
            return json.load(f)

    def active_names(self):
        return [a["name"] if isinstance(a, dict) else a
                for a in self.qualified().get("active", [])]

    def close(self):
        ThemeRanker.STATE_DIR = self.old_state_dir
        tr_mod._today = self.old_tr_today
        rc_mod._today = self.old_rc_today
        shutil.rmtree(self.tmp, ignore_errors=True)


ACTIVES = [{"name": "Semis", "proxy": "SMH", "entry_date": "2026-06-14",
            "entry_price": 600.0},
           {"name": "Biotech", "proxy": "XBI", "entry_date": "2026-06-14",
            "entry_price": 150.0}]


def _meta(incl, completed, needed=2):
    return {"confirmations_needed": needed,
            "consecutive_degraded_weeks": incl,
            "consecutive_degraded_weeks_completed": completed}


# ------------------------------------------------------------------
# Regime side: degraded-week streaks
# ------------------------------------------------------------------

# regime_history mimicking June under the new gauge (no weekend runs —
# matches production local state; weekly closes are the Friday entries)
JUNE_HISTORY = [
    {"date": "2026-06-18", "regime": TRENDING},
    {"date": "2026-06-19", "regime": CHOPPY},     # week 25 close (Juneteenth stale)
    {"date": "2026-06-22", "regime": CHOPPY},
    {"date": "2026-06-23", "regime": TRENDING},
    {"date": "2026-06-24", "regime": CAUTION},
    {"date": "2026-06-25", "regime": CAUTION},
    {"date": "2026-06-26", "regime": CAUTION},    # week 26 close: Caution
    {"date": "2026-06-29", "regime": CAUTION},
    {"date": "2026-06-30", "regime": CAUTION},
    {"date": "2026-07-01", "regime": CHOPPY},
    {"date": "2026-07-02", "regime": TRENDING},   # week 27 close: Trending
]


def _calc():
    return RegimeCalculator({"regime": CONFIG["regime"]}, lambda t, period="1y": None)


def _streaks_on(day_iso, state, history, gate=None):
    d = datetime.date.fromisoformat(day_iso)
    old = rc_mod._today
    rc_mod._today = lambda: d
    try:
        hist = [h for h in history if h["date"] < day_iso]
        return _calc()._degraded_week_streaks(hist, state, gate)
    finally:
        rc_mod._today = old


def test_degraded_streaks_june_replay():
    # Wed Jun 24, first Caution print: prior completed week closed Choppy
    incl, comp = _streaks_on("2026-06-24", CAUTION, JUNE_HISTORY)
    assert (incl, comp) == (1, 0), (incl, comp)
    # Mon Jun 29: completed week (Jun 22-28) closed Caution (Jun 26 entry)
    incl, comp = _streaks_on("2026-06-29", CAUTION, JUNE_HISTORY)
    assert (incl, comp) == (2, 1), (incl, comp)
    # Sunday Jul 5, state Trending (recovered): incl streak dead. Jul 1/2
    # entries belong to Jul 5's OWN ISO week (Jun 29 - Jul 5) and are
    # excluded, so 'completed' still sees the Caution close of Jun 22-28.
    incl, comp = _streaks_on("2026-07-05", TRENDING, JUNE_HISTORY)
    assert (incl, comp) == (0, 1), (incl, comp)
    print("  degraded-week streaks (June replay): OK")


def test_degraded_streaks_details():
    # Hypothetical: the Jun 29 week ALSO degrades -> Sunday Jul 12 Caution
    # confirms (completed: wk Jun29-Jul5 close Jul 2 Caution + wk Jun22-28
    # Caution = 2; current run adds 1)
    hist = [h for h in JUNE_HISTORY if h["date"] < "2026-07-01"]
    hist += [{"date": "2026-07-01", "regime": CAUTION},
             {"date": "2026-07-02", "regime": CAUTION}]
    incl, comp = _streaks_on("2026-07-12", CAUTION, hist)
    assert (incl, comp) == (3, 2), (incl, comp)
    # Outage-capped Caution week is transparent: neither counts nor breaks
    hist2 = [h for h in JUNE_HISTORY if h["date"] < "2026-07-01"]
    hist2 += [{"date": "2026-07-02", "regime": CAUTION,
               "backdrop_gate": {"capped": True, "reason": "data_unavailable"}}]
    incl, comp = _streaks_on("2026-07-12", CAUTION, hist2)
    # wk Jun29-Jul5 transparent, wk Jun22-28 Caution counts
    assert (incl, comp) == (2, 1), (incl, comp)
    # Current run itself outage-capped: doesn't count itself
    gate = {"capped": True, "reason": "data_unavailable"}
    incl, comp = _streaks_on("2026-06-29", CAUTION, JUNE_HISTORY, gate=gate)
    assert (incl, comp) == (1, 1), (incl, comp)
    print("  degraded-week streaks (hypotheticals + outage transparency): OK")


# ------------------------------------------------------------------
# Theme side: the June sequence — signals fire, state survives
# ------------------------------------------------------------------

def test_june_replay_actives_survive():
    """Grounding narrative as a test: Caution prints Jun 24-30 warn but never
    liquidate; the delayed Monday reviews reach the same verdict the missed
    Sundays would have; the streak dies when week 27 closes Trending."""
    env = _Env(active=copy.deepcopy(ACTIVES), last_review="2026-06-14")
    try:
        ranker = ThemeRanker(copy.deepcopy(CONFIG), theme_fetcher)
        # (date, state, incl_current, completed) — streak values as the
        # regime calculator computes them for each day (validated above)
        days = [
            ("2026-06-18", TRENDING, 0, 0),
            ("2026-06-19", CHOPPY,   0, 0),
            ("2026-06-22", CHOPPY,   0, 0),   # Mon: delayed review for wk25
            ("2026-06-23", TRENDING, 0, 0),
            ("2026-06-24", CAUTION,  1, 0),
            ("2026-06-25", CAUTION,  1, 0),
            ("2026-06-26", CAUTION,  1, 0),
            ("2026-06-29", CAUTION,  2, 1),   # Mon: delayed review for wk26
            ("2026-06-30", CAUTION,  2, 1),
            ("2026-07-01", CHOPPY,   0, 1),
            ("2026-07-02", TRENDING, 0, 1),
        ]
        for date_iso, state, incl, comp in days:
            env.set_today(date_iso)
            result = ranker.compute(state, _meta(incl, comp))
            ranker.save_weekly_snapshot(result)
            regime_exits = [s for s in result["exit_signals"]
                            if s.get("reason") == "regime_degradation"]
            if state in (CAUTION, RISK_OFF):
                assert len(regime_exits) == 2, (date_iso, regime_exits)
                for s in regime_exits:
                    assert s["confirmed"] is False, (date_iso, s)
                    assert "Warning — regime degraded" in s["action"], s["action"]
                    assert "unconfirmed" in s["action"]
                    assert "EXIT SIGNAL" not in s["action"]
            else:
                assert regime_exits == [], (date_iso, regime_exits)
            assert sorted(env.active_names()) == ["Biotech", "Semis"], \
                f"{date_iso}: actives must survive the whole June sequence"
        # Sunday Jul 5, recovered to Trending: review runs, still no exits
        env.set_today("2026-07-05")
        result = ranker.compute(TRENDING, _meta(0, 1))
        assert [s for s in result["exit_signals"]
                if s.get("reason") == "regime_degradation"] == []
        assert sorted(env.active_names()) == ["Biotech", "Semis"]
        assert env.qualified()["last_weekly_review"] == "2026-07-05"
    finally:
        env.close()
    print("  June replay: warnings fire, qualification survives: OK")


def test_confirmed_degradation_removes_only_on_review_day():
    """Confirmed EXIT SIGNAL fires midweek but mutation waits for Sunday."""
    env = _Env(active=copy.deepcopy(ACTIVES), last_review="2026-07-05")
    try:
        ranker = ThemeRanker(copy.deepcopy(CONFIG), theme_fetcher)
        # Wed Jul 8: confirmed on the incl-current basis is IRRELEVANT —
        # midweek uses completed-basis (1) -> signal shows unconfirmed 1/2
        env.set_today("2026-07-08")
        result = ranker.compute(CAUTION, _meta(2, 1))
        sig = [s for s in result["exit_signals"]
               if s.get("reason") == "regime_degradation"][0]
        assert sig["confirmed"] is False and "1/2" in sig["action"]
        assert sorted(env.active_names()) == ["Biotech", "Semis"]
        # Sunday Jul 12, second degraded weekly close: confirmed -> removed
        env.set_today("2026-07-12")
        result = ranker.compute(CAUTION, _meta(2, 1))
        sigs = [s for s in result["exit_signals"]
                if s.get("reason") == "regime_degradation"]
        assert all(s["confirmed"] for s in sigs)
        assert all("EXIT SIGNAL" in s["action"] for s in sigs)
        assert env.active_names() == [], "confirmed Sunday review must remove actives"
    finally:
        env.close()
    print("  confirmed degradation: signal midweek, removal on Sunday: OK")


def test_delayed_review_uses_completed_basis():
    """Monday catch-up review must reach the missed Sunday's verdict:
    completed-weeks basis, not Monday's provisional week."""
    env = _Env(active=copy.deepcopy(ACTIVES), last_review="2026-06-21")
    try:
        ranker = ThemeRanker(copy.deepcopy(CONFIG), theme_fetcher)
        # Mon Jun 29: review due (Sun Jun 28 missed). incl-current = 2 but
        # completed = 1 -> unconfirmed -> actives survive.
        env.set_today("2026-06-29")
        result = ranker.compute(CAUTION, _meta(2, 1))
        assert sorted(env.active_names()) == ["Biotech", "Semis"]
        assert env.qualified()["last_weekly_review"] == "2026-06-29"
        # Tue Jun 30: review already done this week -> even a confirmed
        # signal cannot mutate until next Sunday.
        env.set_today("2026-06-30")
        ranker.compute(CAUTION, _meta(2, 2))
        assert sorted(env.active_names()) == ["Biotech", "Semis"]
    finally:
        env.close()
    print("  delayed Monday review uses completed-weeks basis: OK")


def test_entry_needs_weekly_closes_not_daily_runs():
    """Two consecutive DAILY top-2 runs in the same ISO week must not
    satisfy the 2-consecutive-Sunday entry rule (the Jun 30 bug)."""
    env = _Env(active=[], last_review="2026-06-21")
    try:
        ranker = ThemeRanker(copy.deepcopy(CONFIG), theme_fetcher)
        # Tue Jun 23 + Wed Jun 24: first-ever snapshots, same ISO week
        env.set_today("2026-06-23")
        r1 = ranker.compute(TRENDING, _meta(0, 0))
        ranker.save_weekly_snapshot(r1)
        env.set_today("2026-06-24")
        r2 = ranker.compute(TRENDING, _meta(0, 0))
        ranker.save_weekly_snapshot(r2)
        entries = [s for s in r2["entry_signals"] if "ENTRY SIGNAL" in s["action"]]
        assert entries == [], f"same-week dailies must not qualify: {entries}"
        building = [s for s in r2["entry_signals"] if "Building" in s["action"]]
        assert building and "1/2" in building[0]["action"]
        assert env.active_names() == []
        # Next week: Fri Jun 26 close exists (week 26 completed for wk27
        # runs); Sunday Jul 5 run -> 2 weekly closes -> entry + mutation
        env.set_today("2026-06-26")
        ranker.save_weekly_snapshot(ranker.compute(TRENDING, _meta(0, 0)))
        env.set_today("2026-07-05")
        r3 = ranker.compute(TRENDING, _meta(0, 0))
        entries = [s for s in r3["entry_signals"] if "ENTRY SIGNAL" in s["action"]]
        assert len(entries) == 2, r3["entry_signals"]
        assert sorted(env.active_names()) == ["Biotech", "Semis"]
    finally:
        env.close()
    print("  entry requires weekly closes, not daily runs: OK")


def test_bootstrap_missing_marker_reviews_immediately():
    """No last_weekly_review marker (bootstrap / corrupted state): the
    first run must be review-eligible even on a weekday — the marker's
    only writer is the review itself, so 'Sundays only' would deadlock
    under a weekday-only run cadence."""
    env = _Env(active=copy.deepcopy(ACTIVES))   # no marker seeded
    try:
        ranker = ThemeRanker(copy.deepcopy(CONFIG), theme_fetcher)
        env.set_today("2026-07-08")             # Wednesday
        # fully confirmed degradation on the completed-weeks basis
        ranker.compute(CAUTION, _meta(3, 2))
        assert env.active_names() == [], "bootstrap review must apply confirmed exits"
        assert env.qualified()["last_weekly_review"] == "2026-07-08"
        # and the very next weekday run is no longer review-eligible
        env.set_today("2026-07-09")
        ranker.compute(TRENDING, _meta(0, 0))
        assert env.qualified()["last_weekly_review"] == "2026-07-08"
    finally:
        env.close()
    print("  bootstrap (no marker) reviews on first run, then re-arms: OK")


RANK_CONFIG = copy.deepcopy(CONFIG)
RANK_CONFIG["themes"]["watchlist"] = [
    {"name": "Semis", "proxy": "SMH", "constituents": []},
    {"name": "Biotech", "proxy": "XBI", "constituents": []},
    {"name": "Gold", "proxy": "GLD", "constituents": []},
    {"name": "Weak", "proxy": "WK", "constituents": []},
]


def rank_fetcher(ticker, period="6mo"):
    ramps = {"SMH": (500, 650), "XBI": (140, 160), "GLD": (180, 190),
             "WK": (100, 101)}
    if ticker in ramps:
        a, b = ramps[ticker]
        return _df(np.linspace(a, b, 130))
    return None


def _rank_snapshot(date_iso):
    return {"date": date_iso,
            "rankings": {"Semis": {"rank": 1}, "Biotech": {"rank": 2},
                         "Gold": {"rank": 3}, "Weak": {"rank": 4}}}


def test_rank_exit_needs_three_weekly_closes():
    """Rank-degradation exit: 3 consecutive WEEKLY closes below #3 — three
    daily snapshots in one week must not satisfy it."""
    active = [{"name": "Weak", "proxy": "WK", "entry_date": "2026-06-01",
               "entry_price": 100.0}]
    env = _Env(active=copy.deepcopy(active), last_review="2026-06-28")
    try:
        ranker = ThemeRanker(copy.deepcopy(RANK_CONFIG), rank_fetcher)
        hist_path = os.path.join(env.tmp, "theme_history.json")
        # (a) three sub-rank dailies inside ONE ISO week -> 1 weekly close
        with open(hist_path, "w") as fh:
            json.dump([_rank_snapshot(d) for d in
                       ("2026-06-30", "2026-07-01", "2026-07-02")], fh)
        env.set_today("2026-07-05")   # Sunday of that SAME ISO week: the three
        # dailies are the in-progress week -> excluded; live rank is this
        # week's provisional close -> only 1/3
        r = ranker.compute(TRENDING, _meta(0, 0))
        weak = [s for s in r["exit_signals"] if s["theme"] == "Weak"
                and s.get("reason", "").startswith("rank")]
        assert weak and "Warning" in weak[0]["action"], weak
        assert "1/3" in weak[0]["action"], weak[0]["action"]
        assert env.active_names() == ["Weak"], "three dailies in one week = 1/3 only"
        # (b) two completed weekly closes + this Sunday = 3/3 -> exit + removal
        with open(hist_path, "w") as fh:
            json.dump([_rank_snapshot("2026-06-26"),   # week Jun22-28 close
                       _rank_snapshot("2026-07-02")],  # week Jun29-Jul5 close
                      fh)
        env.set_today("2026-07-12")   # next Sunday
        r = ranker.compute(TRENDING, _meta(0, 0))
        weak = [s for s in r["exit_signals"] if s["theme"] == "Weak"
                and s.get("reason") == "rank_degradation"]
        assert weak and "EXIT SIGNAL" in weak[0]["action"], r["exit_signals"]
        # Weak removed by the rank exit; top-2 themes enter on the same
        # Sunday review (2 completed weekly closes + Trending regime)
        assert "Weak" not in env.active_names(), env.active_names()
        assert sorted(env.active_names()) == ["Biotech", "Semis"]
    finally:
        env.close()
    print("  rank exit: 3 weekly closes required, dailies don't count: OK")


def test_weekly_dedupe_latest_wins():
    hist = [{"date": "2026-06-22", "rankings": {"A": {"rank": 4}}},
            {"date": "2026-06-24", "rankings": {"A": {"rank": 1}}},
            {"date": "2026-06-30", "rankings": {"A": {"rank": 2}}}]
    weekly = ThemeRanker._weekly_dedupe(hist)
    assert len(weekly) == 2
    assert weekly[0]["date"] == "2026-06-24", "latest snapshot per week must win"
    assert weekly[1]["date"] == "2026-06-30"
    # exclude_week drops the in-progress week
    wk = datetime.date(2026, 6, 30).isocalendar()[:2]
    weekly = ThemeRanker._weekly_dedupe(hist, exclude_week=wk)
    assert [w["date"] for w in weekly] == ["2026-06-24"]
    print("  weekly dedupe: latest-per-week wins, exclude_week honored: OK")


def test_retention_keeps_53_true_weeks():
    env = _Env()
    try:
        ranker = ThemeRanker(copy.deepcopy(CONFIG), theme_fetcher)
        env.set_today("2026-07-05")
        # seed 400 daily snapshots going back >53 weeks
        hist = [{"date": (datetime.date(2026, 7, 5) - datetime.timedelta(days=i)).isoformat(),
                 "rankings": {}} for i in range(400, 0, -1)]
        with open(os.path.join(env.tmp, "theme_history.json"), "w") as f:
            json.dump(hist, f)
        result = ranker.compute(TRENDING, _meta(0, 0))
        ranker.save_weekly_snapshot(result)
        with open(os.path.join(env.tmp, "theme_history.json")) as f:
            kept = json.load(f)
        cutoff = (datetime.date(2026, 7, 5) - datetime.timedelta(weeks=53)).isoformat()
        assert all(h["date"] >= cutoff for h in kept)
        assert len(kept) > 300, "a year of dailies must survive (old cap kept 52)"
        # pin the boundary exactly: the entry AT the cutoff date survives,
        # the one just before it is dropped (distinguishes 53w from 52w)
        dates = {h["date"] for h in kept}
        assert cutoff in dates, f"entry at cutoff {cutoff} must be retained"
        day_before = (datetime.date.fromisoformat(cutoff)
                      - datetime.timedelta(days=1)).isoformat()
        assert day_before not in dates, "entry before cutoff must be pruned"
    finally:
        env.close()
    print("  theme_history retention: 53 true weeks of dailies: OK")


if __name__ == "__main__":
    print("\n=== R4 qualification tests (signals fast, state slow) ===")
    test_degraded_streaks_june_replay()
    test_degraded_streaks_details()
    test_june_replay_actives_survive()
    test_confirmed_degradation_removes_only_on_review_day()
    test_delayed_review_uses_completed_basis()
    test_entry_needs_weekly_closes_not_daily_runs()
    test_bootstrap_missing_marker_reviews_immediately()
    test_rank_exit_needs_three_weekly_closes()
    test_weekly_dedupe_latest_wins()
    test_retention_keeps_53_true_weeks()
    print("\nAll R4 tests passed.\n")
