#!/usr/bin/env python3
"""Pins for the fundamentals surface (2026-08-11).

Three invariants, each demonstrating the failure it guards:

  1. Market cap survives a `.info` that omits it, and the fallback can only
     run after a SUCCESSFUL call — a failed fetch never gets a cap.
  2. A fetch that never landed is distinguishable from a field Yahoo did not
     carry (D-019: an outage must not wear the same dash as a real gap).
  3. Dividend yield renders AS DELIVERED, and the delivered scale is pinned
     to a plausible band so an upstream scale change breaks loudly.

On (3), the scale cannot be inferred at runtime and this pin is the whole
protection. 0.5% is 0.005 as a fraction and 0.5 as a percent — both below 1,
so no magnitude branch can tell them apart. A heuristic would silently pick
wrong for exactly the low-yield names where it matters. A test that goes red
is the only honest guard.
"""

import json
import os
import re
import statistics
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

import signal_engine as se

SEARCH_HTML = os.path.join(REPO, "public", "search.html")

# Long-standing dividend payers. Several, because one company cutting its
# dividend must not turn this pin red — an upstream SCALE change moves all
# of them together, which is what the median below tests for.
PAYERS = ["KO", "JNJ", "PG", "XOM", "VZ", "PEP", "CVX", "MO"]

# The plausible band for a known payer's RENDERED yield, in percent.
#
# The floor is 0.5, NOT 0. A zero floor cannot detect the regression this
# pin exists for: if Yahoo reverted to fractions, a 4%-yielding name would
# deliver 0.04 and render "0.04%", which sits comfortably inside 0-15 and
# would pass. 0.5% is below every name on the PAYERS list and above every
# value a fraction-scale revert could produce for them, so the band bites in
# both directions.
PLAUSIBLE_MIN, PLAUSIBLE_MAX = 0.5, 15.0


# ----------------------------------------------------------------------
# The page's own scaling factor, read out of the page.
# ----------------------------------------------------------------------
def page_dividend_multiplier():
    """The multiplier public/search.html applies to `dividend_yield`.

    Parsed from the page rather than restated here, so this pin tests what
    the page actually does. Re-adding `*100` moves this to 100 and every
    band assertion below fails.
    """
    with open(SEARCH_HTML) as f:
        src = f.read()
    line = next((l for l in src.splitlines()
                 if "'Dividend Yield'" in l and "dividend_yield" in l), None)
    assert line, "the Dividend Yield stat line is gone from search.html"
    scaled = re.search(r"dividend_yield\s*\*\s*([0-9.]+)", line)
    return float(scaled.group(1)) if scaled else 1.0


def render_yield(delivered, multiplier):
    """Reproduce the page's rendered number for a delivered value."""
    return round(delivered * multiplier, 2)


# ----------------------------------------------------------------------
# 1 + 2. Market cap fallback and fetch coverage.
# ----------------------------------------------------------------------
class _FakeTicker:
    """yfinance stand-in: choose what .info and .fast_info do."""

    def __init__(self, info=None, fast=None, raise_on_info=False,
                 raise_on_fast=False):
        self._info, self._fast = info, fast
        self._raise_info, self._raise_fast = raise_on_info, raise_on_fast

    @property
    def info(self):
        if self._raise_info:
            raise RuntimeError("simulated Yahoo outage")
        return self._info

    @property
    def fast_info(self):
        if self._raise_fast:
            raise RuntimeError("fast_info unavailable")
        return self._fast


def _with_ticker(factory, fn):
    import yfinance as yf
    orig = yf.Ticker
    yf.Ticker = lambda t, *a, **k: factory(t)
    try:
        return fn()
    finally:
        yf.Ticker = orig


def test_market_cap_survives_info_omitting_it():
    full = {"marketCap": None, "forwardPE": 9.9, "trailingPE": 11.0,
            "sector": "Technology", "industry": "Computer Hardware"}

    # (a) .info answers WITHOUT marketCap -> fast_info supplies it.
    f = _with_ticker(
        lambda t: _FakeTicker(info=dict(full), fast={"marketCap": 27_252_775_464}),
        lambda: se.fetch_fundamentals_yfinance("HPQ"))
    assert f["fetch_status"] == "ok", f["fetch_status"]
    assert f["market_cap"] == 27_252_775_464, f["market_cap"]
    assert f["market_cap_source"] == "fast_info", f["market_cap_source"]
    assert f["forward_pe"] == 9.9, "the rest of the payload was lost"
    print("  (1a) .info omits marketCap while answering everything else: "
          "fast_info supplies it, other fields intact: OK")

    # (b) .info carries it -> fast_info is NOT consulted. A working endpoint
    #     must not be second-guessed by a fallback.
    f = _with_ticker(
        lambda t: _FakeTicker(info={"marketCap": 414_542_102_528},
                              raise_on_fast=True),
        lambda: se.fetch_fundamentals_yfinance("AMAT"))
    assert f["market_cap"] == 414_542_102_528
    assert f["market_cap_source"] == "info", f["market_cap_source"]
    print("  (1b) .info carries the cap: fast_info is never called (it would "
          "have raised): OK")

    # (c) THE FAILURE, DEMONSTRATED. The whole call fails. The fallback must
    #     NOT run — a cap invented after a failed fetch would paper over the
    #     outage the status field exists to report.
    f = _with_ticker(
        lambda t: _FakeTicker(raise_on_info=True, fast={"marketCap": 1}),
        lambda: se.fetch_fundamentals_yfinance("HPQ"))
    assert f["fetch_status"] == "failed", f["fetch_status"]
    assert f["market_cap"] is None, "a failed fetch produced a market cap"
    assert f["market_cap_source"] is None
    assert "simulated Yahoo outage" in (f["fetch_error"] or "")
    print("  (1c) a FAILED fetch gets no fallback cap and reports the "
          "reason: OK")


def test_outage_is_distinguishable_from_a_missing_field():
    missing = _with_ticker(
        lambda t: _FakeTicker(info={"forwardPE": 9.9}, fast={}),
        lambda: se.fetch_fundamentals_yfinance("X"))
    outage = _with_ticker(
        lambda t: _FakeTicker(raise_on_info=True),
        lambda: se.fetch_fundamentals_yfinance("X"))

    # Identical where it counts — which is exactly the problem this solves.
    assert missing["market_cap"] is None and outage["market_cap"] is None
    assert missing["sector"] is None and outage["sector"] is None
    # ...and separable only by the status.
    assert missing["fetch_status"] == "ok"
    assert outage["fetch_status"] == "failed"
    assert missing["fetch_error"] is None and outage["fetch_error"]
    print("  (2) a missing FIELD and a failed FETCH carry identical None "
          "values; only fetch_status separates them: OK")


# ----------------------------------------------------------------------
# 3. Dividend yield: rendered as delivered, on a pinned scale.
# ----------------------------------------------------------------------
def test_dividend_yield_renders_as_delivered():
    mult = page_dividend_multiplier()
    assert mult == 1.0, (
        f"search.html scales dividend_yield by {mult}. Yahoo delivers a "
        "PERCENT; scaling it renders 3.99% as 399.00%. If the upstream "
        "scale genuinely changed, fix it here AND update this pin — do not "
        "add a runtime magnitude branch, which cannot distinguish 0.5% as "
        "0.005 from 0.5% as 0.5.")
    print("  (3a) search.html applies no scaling to dividend_yield: OK")

    # THE FAILURE, DEMONSTRATED. The regression this pin exists for, on the
    # real observed value.
    assert render_yield(3.99, 100.0) == 399.0
    assert not (PLAUSIBLE_MIN <= render_yield(3.99, 100.0) <= PLAUSIBLE_MAX)
    assert PLAUSIBLE_MIN <= render_yield(3.99, mult) <= PLAUSIBLE_MAX
    print("  (3b) HPQ's 3.99 renders 3.99% as delivered and 399.00% under "
          "the old ×100 — outside the plausible band: OK")

    # And the band bites the OTHER way too: a fraction-scale revert.
    assert not (PLAUSIBLE_MIN <= render_yield(0.0399, 1.0) <= PLAUSIBLE_MAX), \
        "the band's floor is too low to detect a fraction-scale revert"
    print(f"  (3c) a fraction-scale revert (0.0399 -> 0.04%) falls below the "
          f"{PLAUSIBLE_MIN}% floor and fails: OK")


def test_delivered_dividend_scale_is_percent():
    """LIVE: the upstream contract. Skipped loudly, never silently passed."""
    mult = page_dividend_multiplier()
    got = {}
    for t in PAYERS:
        try:
            v = se.fetch_fundamentals_yfinance(t)
            if v["fetch_status"] == "ok" and v["dividend_yield"] is not None:
                got[t] = render_yield(v["dividend_yield"], mult)
        except Exception:
            continue

    if len(got) < 3:
        print(f"  (3d) SKIPPED — only {len(got)}/{len(PAYERS)} payers "
              "resolved; upstream unreachable, scale NOT verified this run")
        return

    med = statistics.median(got.values())
    outliers = {k: v for k, v in got.items()
                if not (PLAUSIBLE_MIN <= v <= PLAUSIBLE_MAX)}
    assert PLAUSIBLE_MIN <= med <= PLAUSIBLE_MAX, (
        f"median rendered yield across {len(got)} known dividend payers is "
        f"{med}%, outside [{PLAUSIBLE_MIN}, {PLAUSIBLE_MAX}] — the upstream "
        f"dividendYield SCALE has changed. Values: {got}. Fix the render in "
        "public/search.html and this band together; do not add a runtime "
        "heuristic.")
    print(f"  (3d) {len(got)} known payers render a median {med:.2f}% "
          f"(band {PLAUSIBLE_MIN}-{PLAUSIBLE_MAX}%): OK"
          + (f"  [individual outliers, not fatal: {outliers}]" if outliers else ""))


if __name__ == "__main__":
    print("\n=== fundamentals pins ===")
    test_market_cap_survives_info_omitting_it()
    test_outage_is_distinguishable_from_a_missing_field()
    test_dividend_yield_renders_as_delivered()
    test_delivered_dividend_scale_is_percent()
    print("\nAll fundamentals pins passed.\n")
