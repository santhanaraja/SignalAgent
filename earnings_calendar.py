#!/usr/bin/env python3
"""
Earnings-calendar cache (PER-510 Enhancement A). Display-only v1 —
no scoring or tally changes anywhere.

Next-earnings dates per ticker via yfinance, cached in
data/earnings_calendar.json and refreshed at most once per ET day
(the daily first run / rotation refreshes; the 15-minute cycles read
the cache). yfinance's calendar endpoint is flaky and ETFs have no
earnings, so a missing date is normal: it means "no chip", never an
error, and a fetch failure retains a previously cached FUTURE date
(dates rarely move; stale beats blank).

THIS FEEDS GRADE ROW 7, AND ITS ABSENCES FAIL OPEN (2026-08-11).
------------------------------------------------------------------
`runway_sessions_before()` returns None for a null, a past, or an
unparseable date, and grade row 7 reads None as "no known earnings
print — runway unbounded" and PASSES. So every gap here is not a
missing chip; it is a silently satisfied grade row.

The census that prompted this: 3 of today's 73 board names passed row
7 on the fail-open, all three because they had just reported and the
cache had not caught up. All three happened to have true runways of
62-75 sessions, so the board's blast radius was zero — but off-board,
CRDO sat at null with a TRUE runway of 16 sessions, two sessions from
crossing under the 15-session bar while its entry said "unbounded".

THE FIX IS TO RESOLVE, NOT TO FAIL CLOSED. Failing closed on "unknown"
would block CRDO, whose real runway passes — the same false negative
as a breaker gate firing on missing data, run in the other direction.
The cache is stale, not empty. See get_earnings_map.

THE UNPARSEABLE CASE IS STRUCTURALLY UNREACHABLE, AND THAT IS ON THE
RECORD SO NOBODY GOES LOOKING FOR IT. The only writer of a date into
this cache is `cache["dates"][t] = d.isoformat()` on a `datetime.date`
returned by `_fetch_next_earnings`, which coerces through `_to_date`.
It can emit nothing but strict YYYY-MM-DD or None. On Python 3.9,
`date.fromisoformat` would reject 'August 6, 2026', '08/06/2026' or
'2026-08-06T00:00:00' — but no such value can reach the field through
this writer, and there is no second producer. Verified 2026-08-11: all
133 non-null cache values and all 67 next_earnings_date values in
public/signals.json parse. `needs_resolution()` still treats an
unparseable value as needing resolution, so a hand-edited cache is
handled rather than trusted. THE DEFECT IS NULLS AND STALE DATES ONLY.

Standalone module: imports nothing from the project, so signal_engine
and framework/* can both use it without import cycles. The best-effort
fetch mirrors framework_runner.fetch_next_earnings (kept there for the
constituent ranker, untouched).
"""

import datetime
import json
import os

try:
    from zoneinfo import ZoneInfo
    _ET = ZoneInfo("America/New_York")
except Exception:  # pragma: no cover
    _ET = None

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
CACHE_PATH = os.path.join(DATA_DIR, "earnings_calendar.json")


def _today_et():
    """Trading-day date — module-level so tests can inject dates."""
    if _ET is not None:
        return datetime.datetime.now(_ET).date()
    return datetime.date.today()


def _to_date(val):
    """Best-effort coercion of a yfinance/pandas value to datetime.date."""
    try:
        if isinstance(val, datetime.datetime):
            return val.date()
        if isinstance(val, datetime.date):
            return val
        if hasattr(val, "to_pydatetime"):
            return val.to_pydatetime().date()
        if hasattr(val, "date"):
            return val.date()
    except Exception:
        pass
    return None


def _fetch_next_earnings(ticker):
    """
    Next upcoming earnings date (datetime.date) or None. Best-effort
    across yfinance versions; every failure degrades to None. ETFs and
    symbols without a calendar simply return None (handled silently).
    """
    today = _today_et()
    try:
        import yfinance as yf
        tk = yf.Ticker(ticker)

        # Preferred: .calendar (dict in recent yfinance, DataFrame in older)
        try:
            cal = tk.calendar
            dates = None
            if isinstance(cal, dict):
                dates = cal.get("Earnings Date")
            elif cal is not None and hasattr(cal, "loc"):
                try:
                    dates = list(cal.loc["Earnings Date"])
                except Exception:
                    dates = None
            if dates is not None:
                if not isinstance(dates, (list, tuple)):
                    dates = [dates]
                upcoming = sorted(d for d in (_to_date(x) for x in dates)
                                  if d is not None and d >= today)
                if upcoming:
                    return upcoming[0]
        except Exception:
            pass

        # Fallback: .get_earnings_dates() DataFrame indexed by date
        try:
            df = tk.get_earnings_dates(limit=12)
            if df is not None and len(df) > 0:
                upcoming = sorted(d for d in (_to_date(x) for x in df.index)
                                  if d is not None and d >= today)
                if upcoming:
                    return upcoming[0]
        except Exception:
            pass
    except Exception:
        pass
    return None


def _load_cache():
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, "r") as f:
                cache = json.load(f)
            if isinstance(cache, dict) and isinstance(cache.get("dates"), dict):
                return cache
        except (json.JSONDecodeError, IOError):
            pass
    return {"fetched_on": None, "dates": {}}


def _save_cache(cache):
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(CACHE_PATH, "w") as f:
            json.dump(cache, f, indent=2)
    except IOError:
        pass  # cache is an optimization, never a blocker


def needs_resolution(date_iso, today=None):
    """True when a cached value cannot support a runway measurement.

    Three cases, and they are the whole defect surface:
      · None      — nothing was ever known, or a fetch failed and stored null
      · PAST date — the name reported and the cache never caught up
      · unparseable — structurally unreachable (see the module note), kept
                      here so a hand-edited cache cannot slip through

    All three make runway_sessions_before() return None, which grade row 7
    reads as "no known earnings print — runway unbounded" and PASSES. That
    fail-open is why these must be re-resolved rather than reused.
    """
    if not date_iso:
        return True
    try:
        d = datetime.date.fromisoformat(str(date_iso)[:10])
    except ValueError:
        return True
    return d < (today or _today_et())


def get_earnings_map(tickers, with_coverage=False):
    """
    {ticker: "YYYY-MM-DD" or None} for the requested tickers.

    Refresh policy: a full refresh at most once per ET day (fetched_on
    stamp); within the day, tickers absent from the cache are topped up
    AND any cached value that cannot support a runway — null, or a date
    already past — is RE-RESOLVED on demand. Never raises.

    WHY RE-RESOLVE RATHER THAN FAIL CLOSED. An absent date is not evidence
    of an imminent print; it is usually evidence of a stale cache. CRDO is
    the case that settles it: its cached value is null while its true
    runway is 16 sessions, which PASSES. Failing closed on "unknown" would
    block a name the doctrine admits — the breaker-gate error run in the
    opposite direction. The cache is stale, not empty, so the fix is to go
    and get the date.

    THE STALE-PAST-DATE TRAP THIS REMOVES. The old rule kept a previously
    known date whenever a fetch returned None, to ride out a flaky
    endpoint. That is right for a FUTURE date and wrong for a past one: it
    preserved exactly the value that reads as unbounded runway, so a name
    that had just reported stayed fail-open indefinitely. A past date is
    now only kept when re-resolution also fails, and the failure is
    recorded rather than dressed as a date.
    """
    tickers = [t for t in dict.fromkeys(tickers) if t]
    cache = _load_cache()
    today = _today_et()
    today_iso = today.isoformat()
    stale = cache.get("fetched_on") != today_iso

    # ONE resolution attempt per ET day per ticker. Without this, anything
    # that can never resolve — an ETF has no earnings at all — is unusable
    # by definition and would be refetched on every 15-minute cycle, for
    # ever. `attempted` records the day we last tried, so a genuine
    # unknown costs one call a day rather than one call a cycle.
    attempted = cache.setdefault("attempted", {})
    if not isinstance(attempted, dict):
        attempted = cache["attempted"] = {}

    def _unusable(t):
        return (t in cache["dates"]
                and needs_resolution(cache["dates"].get(t), today)
                and attempted.get(t) != today_iso)

    unresolvable = [t for t in tickers if not stale and _unusable(t)]
    to_fetch = [t for t in tickers
                if stale or t not in cache["dates"] or _unusable(t)]

    coverage = {}
    if to_fetch:
        why = "daily refresh" if stale else "cache top-up"
        if unresolvable:
            why += f" + {len(unresolvable)} unresolvable (null/past)"
        print(f"[earnings] refreshing {len(to_fetch)} tickers ({why})")
        for t in to_fetch:
            prev = cache["dates"].get(t)
            try:
                d = _fetch_next_earnings(t)
            except Exception:
                d = None
            if d is not None:
                cache["dates"][t] = d.isoformat()
                attempted.pop(t, None)
                coverage[t] = "resolved"
            elif needs_resolution(prev, today):
                # Nothing usable before and nothing usable now. Store the
                # null explicitly so the artifact says "unknown" rather
                # than showing a consumed date as if it were upcoming, and
                # stamp the attempt so this costs one call a day, not one
                # a cycle.
                cache["dates"][t] = None
                attempted[t] = today_iso
                coverage[t] = "unresolved"
            else:
                # A usable FUTURE date survives a flaky fetch, as before.
                coverage[t] = "kept_cached"
        cache["fetched_on"] = today_iso
        _save_cache(cache)

    out = {t: cache["dates"].get(t) for t in tickers}
    if not with_coverage:
        return out
    for t in tickers:
        coverage.setdefault(t, "cache_hit")
    unresolved = sorted(t for t in tickers if coverage.get(t) == "unresolved")
    return out, {
        "by_ticker": coverage,
        "unresolved": unresolved,
        # Zero in the healthy state. Non-zero means these names reach row 7
        # with no measurable runway and will PASS it on the fail-open.
        "degraded": len(unresolved),
    }


def days_to_earnings(date_iso, today=None):
    """Whole days until date_iso (0 = today), or None. Past dates -> None
    (a stale pre-earnings date must not render as an upcoming event)."""
    if not date_iso:
        return None
    try:
        d = datetime.date.fromisoformat(str(date_iso)[:10])
    except ValueError:
        return None
    delta = (d - (today or _today_et())).days
    return delta if delta >= 0 else None
