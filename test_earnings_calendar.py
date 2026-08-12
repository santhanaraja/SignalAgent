#!/usr/bin/env python3
"""
Tests for the earnings-calendar cache (PER-510 Enhancement A).
Fully offline: the yfinance fetcher, today-clock, and cache path are all
injected. Run: python3 test_earnings_calendar.py
"""

import datetime
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import earnings_calendar as ec


class _Env:
    def __init__(self, today="2026-07-05"):
        self.tmp = tempfile.mkdtemp(prefix="er_test_")
        self.olds = (ec.CACHE_PATH, ec.DATA_DIR, ec._fetch_next_earnings, ec._today_et)
        ec.DATA_DIR = self.tmp
        ec.CACHE_PATH = os.path.join(self.tmp, "earnings_calendar.json")
        self.fetch_log = []
        self.dates = {}
        ec._fetch_next_earnings = self._fetch
        self.set_today(today)

    def _fetch(self, ticker):
        self.fetch_log.append(ticker)
        v = self.dates.get(ticker)
        return datetime.date.fromisoformat(v) if v else None

    def set_today(self, iso):
        d = datetime.date.fromisoformat(iso)
        ec._today_et = lambda: d

    def cache(self):
        with open(ec.CACHE_PATH) as f:
            return json.load(f)

    def close(self):
        (ec.CACHE_PATH, ec.DATA_DIR, ec._fetch_next_earnings, ec._today_et) = self.olds
        shutil.rmtree(self.tmp, ignore_errors=True)


def test_days_to_earnings():
    today = datetime.date(2026, 7, 5)
    assert ec.days_to_earnings("2026-07-08", today) == 3
    assert ec.days_to_earnings("2026-07-05", today) == 0
    assert ec.days_to_earnings("2026-07-01", today) is None   # past: never a chip
    assert ec.days_to_earnings(None, today) is None
    assert ec.days_to_earnings("garbage", today) is None
    print("  days_to_earnings (incl. past-date guard): OK")


def test_daily_refresh_then_cache_reads():
    env = _Env(today="2026-07-05")
    try:
        env.dates = {"AAA": "2026-07-09"}
        m = ec.get_earnings_map(["AAA", "IWM"])          # first run of the day
        assert m == {"AAA": "2026-07-09", "IWM": None}   # ETF: silently None
        assert env.fetch_log == ["AAA", "IWM"]
        env.fetch_log.clear()
        m = ec.get_earnings_map(["AAA", "IWM"])          # 15-min cycle: no refetch
        assert env.fetch_log == []
        assert m["AAA"] == "2026-07-09"
        # next ET day: full refresh
        env.set_today("2026-07-06")
        ec.get_earnings_map(["AAA", "IWM"])
        assert env.fetch_log == ["AAA", "IWM"]
    finally:
        env.close()
    print("  daily refresh once, cache reads within the day: OK")


def test_topup_only_missing_names_midday():
    env = _Env()
    try:
        env.dates = {"AAA": "2026-07-09", "NEW": "2026-07-15"}
        ec.get_earnings_map(["AAA"])
        env.fetch_log.clear()
        m = ec.get_earnings_map(["AAA", "NEW"])          # NEW added midday
        assert env.fetch_log == ["NEW"], env.fetch_log   # only the missing name
        assert m == {"AAA": "2026-07-09", "NEW": "2026-07-15"}
    finally:
        env.close()
    print("  midday top-up fetches only missing names: OK")


def test_failed_fetch_retains_previous_date():
    env = _Env(today="2026-07-05")
    try:
        env.dates = {"AAA": "2026-07-09"}
        ec.get_earnings_map(["AAA"])
        env.set_today("2026-07-06")                      # next-day refresh...
        env.dates = {}                                   # ...endpoint flakes out
        m = ec.get_earnings_map(["AAA"])
        assert m["AAA"] == "2026-07-09", "stale date must beat a flaky miss"
        # a ticker that never resolved stays an explicit null, no error
        m = ec.get_earnings_map(["GHOST"])
        assert m["GHOST"] is None
    finally:
        env.close()
    print("  flaky fetch retains previous date; unknown stays null: OK")


def test_never_raises_even_with_broken_cache():
    env = _Env()
    try:
        with open(ec.CACHE_PATH, "w") as f:
            f.write("{corrupt")
        m = ec.get_earnings_map(["AAA"])
        assert m == {"AAA": None}
    finally:
        env.close()
    print("  corrupt cache degrades to empty, never raises: OK")


def test_needs_resolution_truth_table():
    T = datetime.date(2026, 8, 11)
    assert ec.needs_resolution(None, T) is True
    assert ec.needs_resolution("", T) is True
    assert ec.needs_resolution("2026-08-06", T) is True     # past (ARWR's case)
    assert ec.needs_resolution("2026-08-10", T) is True     # past by one day
    # A print date of TODAY is a REAL measurement — runway 0, row 7 fails on
    # it correctly. Re-resolving it away would destroy the one case the row
    # most needs to see.
    assert ec.needs_resolution("2026-08-11", T) is False
    assert ec.needs_resolution("2026-11-24", T) is False    # future
    # Structurally unreachable through the writer (see the module note), but
    # a hand-edited cache must be handled rather than trusted.
    assert ec.needs_resolution("August 6, 2026", T) is True
    assert ec.needs_resolution("2026-08-06T00:00:00", T) is True
    print("  needs_resolution: null/past/unparseable resolve, today and "
          "future do not: OK")


def test_stale_and_null_dates_are_re_resolved():
    """The fail-open fix: an absence is re-resolved, never reused.

    Row 7 reads a null/past/unparseable date as "runway unbounded" and
    PASSES. The old refresh policy only fetched names ABSENT from the
    cache, so a null or a consumed date sat there being silently
    satisfying for the rest of the day.
    """
    env = _Env(today="2026-08-11")
    try:
        # Seed a same-day cache so `stale` is False — this is the midday
        # path, the one that used to skip these entirely.
        env.dates = {"STALE": "2026-11-24", "NULLED": "2026-09-02",
                     "FUTURE": "2026-12-01"}
        ec.get_earnings_map(["FUTURE"])
        with open(ec.CACHE_PATH) as f:
            c = json.load(f)
        c["dates"]["STALE"] = "2026-08-06"      # consumed print, 5 days past
        c["dates"]["NULLED"] = None             # never knew (CRDO's shape)
        with open(ec.CACHE_PATH, "w") as f:
            json.dump(c, f)

        env.fetch_log.clear()
        m, cov = ec.get_earnings_map(["STALE", "NULLED", "FUTURE"],
                                     with_coverage=True)
        assert sorted(env.fetch_log) == ["NULLED", "STALE"], env.fetch_log
        assert m["STALE"] == "2026-11-24", m
        assert m["NULLED"] == "2026-09-02", m
        assert m["FUTURE"] == "2026-12-01", "a usable date was refetched"
        assert cov["degraded"] == 0 and cov["unresolved"] == []
        print("  (a) midday: a PAST date and a NULL are both re-resolved; a "
              "future date is left alone: OK")

        # THE FAILURE, DEMONSTRATED — the old predicate was absence-only, so
        # neither of those two would have been fetched at all.
        old_pred = lambda t, cache: t not in cache
        assert not old_pred("STALE", c["dates"]), \
            "the old rule would have refetched STALE — re-derive this pin"
        assert not old_pred("NULLED", c["dates"])
        print("  (b) under the old absence-only rule neither would have been "
              "fetched — they stayed fail-open all day: OK")
    finally:
        env.close()


def test_failed_resolution_does_not_preserve_a_past_date():
    """A future date survives a flaky fetch; a PAST one must not.

    The old rule kept any previously known date over a failed fetch. For a
    future date that is right. For a past one it preserved precisely the
    value that reads as unbounded runway.
    """
    env = _Env(today="2026-08-11")
    try:
        env.dates = {"KEEP": "2026-12-01"}
        ec.get_earnings_map(["KEEP"])
        with open(ec.CACHE_PATH) as f:
            c = json.load(f)
        c["dates"]["GONE"] = "2026-08-06"       # consumed
        with open(ec.CACHE_PATH, "w") as f:
            json.dump(c, f)

        env.dates = {}                          # endpoint flakes for everyone
        env.set_today("2026-08-12")             # next ET day: full refresh
        m, cov = ec.get_earnings_map(["KEEP", "GONE"], with_coverage=True)

        assert m["KEEP"] == "2026-12-01", "a usable future date was dropped"
        assert cov["by_ticker"]["KEEP"] == "kept_cached"
        assert m["GONE"] is None, \
            "a consumed date survived a failed re-resolution and keeps " \
            "reading as unbounded runway"
        assert cov["by_ticker"]["GONE"] == "unresolved"
        assert cov["degraded"] == 1 and cov["unresolved"] == ["GONE"]
        print("  (c) a future date survives a flaky fetch; a consumed one "
              "becomes an explicit null and is counted degraded: OK")
    finally:
        env.close()


if __name__ == "__main__":
    print("\n=== Earnings calendar tests (PER-510) ===")
    test_days_to_earnings()
    test_daily_refresh_then_cache_reads()
    test_topup_only_missing_names_midday()
    test_failed_fetch_retains_previous_date()
    test_never_raises_even_with_broken_cache()
    test_needs_resolution_truth_table()
    test_stale_and_null_dates_are_re_resolved()
    test_failed_resolution_does_not_preserve_a_past_date()
    print("\nAll earnings-calendar tests passed.\n")
