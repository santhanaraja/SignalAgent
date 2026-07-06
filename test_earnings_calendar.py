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


if __name__ == "__main__":
    print("\n=== Earnings calendar tests (PER-510) ===")
    test_days_to_earnings()
    test_daily_refresh_then_cache_reads()
    test_topup_only_missing_names_midday()
    test_failed_fetch_retains_previous_date()
    test_never_raises_even_with_broken_cache()
    print("\nAll earnings-calendar tests passed.\n")
