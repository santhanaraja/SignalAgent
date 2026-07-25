#!/usr/bin/env python3
"""
Deliberate activation of a previously-dead production path (Build 5,
ruling 5). NOT a silent change.

THE FINDING (Build 5 Step-0 recon): earnings_calendar._fetch_next_earnings
has two sources — `.calendar` (primary) and `get_earnings_dates()`
(fallback). The fallback parses Yahoo's earnings table with
pandas.read_html, which needs lxml or html5lib. requirements.txt pinned
NEITHER, so on every production host the call raised

    ImportError: Missing optional dependency 'lxml'

and the bare `except Exception` swallowed it. The fallback has never once
executed. Every ticker whose `.calendar` came back empty silently degraded
to "no known earnings date" — which the D-011 runway row treats as
UNBOUNDED RUNWAY, i.e. an automatic pass.

Adding lxml to requirements ACTIVATES that path. This file is the proof
that activating it is safe and that it actually works, per the pin
doctrine's Law 3 (demonstrate the failure): the first test FAILS on a
host without lxml, which is exactly the production state this fixes.

Run: python3 test_earnings_fallback.py
"""

import datetime
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import earnings_calendar as ec


def test_lxml_present_so_read_html_works():
    """The dependency the fallback needs. On a host without lxml this
    fails — and that WAS production."""
    try:
        import lxml                                    # noqa: F401
    except ImportError:
        raise AssertionError(
            "lxml is missing — get_earnings_dates() will raise ImportError "
            "and earnings_calendar's fallback will silently degrade every "
            "ticker to 'no known earnings' (an automatic row-7 pass). "
            "This is the exact production state Build 5 fixes; install "
            "requirements.txt.")
    import pandas as pd
    # the actual mechanism: read_html must parse a table without raising
    html = ("<table><tr><th>Earnings Date</th><th>EPS Estimate</th></tr>"
            "<tr><td>Jul 30, 2026, 4 PM EDT</td><td>1.23</td></tr></table>")
    frames = pd.read_html(html)
    assert frames and "Earnings Date" in frames[0].columns, frames
    print("  lxml present; pandas.read_html parses the earnings table: OK")


def test_fallback_path_is_reachable_and_shaped():
    """Drive _fetch_next_earnings with `.calendar` FORCED EMPTY, so the
    fallback is the only source. Proves the previously-dead branch now
    runs and returns the documented shape (a future date or None) rather
    than raising into the swallow."""
    calls = {"calendar": 0, "fallback": 0}

    class _FakeTicker:
        def __init__(self, sym):
            self.symbol = sym

        @property
        def calendar(self):
            calls["calendar"] += 1
            return {}                       # the empty-primary case

        def get_earnings_dates(self, limit=12):
            calls["fallback"] += 1
            import pandas as pd
            today = datetime.date.today()
            idx = pd.to_datetime([
                today + datetime.timedelta(days=21),
                today - datetime.timedelta(days=70),
            ])
            return pd.DataFrame({"EPS Estimate": [1.2, 1.1]}, index=idx)

    import yfinance as yf
    real = yf.Ticker
    yf.Ticker = _FakeTicker
    try:
        got = ec._fetch_next_earnings("TEST")
    finally:
        yf.Ticker = real

    assert calls["fallback"] == 1, (
        "the fallback was never reached — the primary did not yield to it")
    assert isinstance(got, datetime.date), f"expected a date, got {got!r}"
    assert got > datetime.date.today(), (
        f"the fallback returned a PAST print ({got}) — it must filter to "
        "upcoming only, or the runway row would compute against a print "
        "that already happened")
    print(f"  fallback reached and returned the next upcoming print "
          f"({got}): OK")


def test_activation_does_not_change_the_happy_path():
    """A ticker whose `.calendar` DOES answer must not touch the fallback.
    Activating a dead path must not re-route the working one."""
    calls = {"calendar": 0, "fallback": 0}
    want = datetime.date.today() + datetime.timedelta(days=14)

    class _FakeTicker:
        def __init__(self, sym):
            self.symbol = sym

        @property
        def calendar(self):
            calls["calendar"] += 1
            return {"Earnings Date": [want]}

        def get_earnings_dates(self, limit=12):
            calls["fallback"] += 1
            raise AssertionError("the fallback ran on the happy path")

    import yfinance as yf
    real = yf.Ticker
    yf.Ticker = _FakeTicker
    try:
        got = ec._fetch_next_earnings("TEST")
    finally:
        yf.Ticker = real

    assert calls["calendar"] == 1 and calls["fallback"] == 0, calls
    assert got == want, (got, want)
    print("  primary still answers alone; the fallback is not re-routed: OK")


def test_failure_still_degrades_to_none_not_an_exception():
    """Both sources dead -> None, not a raise. The swallow was hiding a
    permanent ImportError; it must still absorb a genuine data outage."""
    class _FakeTicker:
        def __init__(self, sym):
            pass

        @property
        def calendar(self):
            raise RuntimeError("network down")

        def get_earnings_dates(self, limit=12):
            raise RuntimeError("network down")

    import yfinance as yf
    real = yf.Ticker
    yf.Ticker = _FakeTicker
    try:
        assert ec._fetch_next_earnings("TEST") is None
    finally:
        yf.Ticker = real
    print("  both sources dead -> None (a real outage still degrades): OK")


if __name__ == "__main__":
    print("\n=== earnings fallback activation (Build 5, ruling 5) ===")
    test_lxml_present_so_read_html_works()
    test_fallback_path_is_reachable_and_shaped()
    test_activation_does_not_change_the_happy_path()
    test_failure_still_degrades_to_none_not_an_exception()
    print("\nAll earnings-fallback tests passed.\n")
