#!/usr/bin/env python3
"""
Earnings-calendar cache (PER-510 Enhancement A). Display-only v1 —
no scoring or tally changes anywhere.

Next-earnings dates per ticker via yfinance, cached in
data/earnings_calendar.json and refreshed at most once per ET day
(the daily first run / rotation refreshes; the 15-minute cycles read
the cache). yfinance's calendar endpoint is flaky and ETFs have no
earnings, so a missing date is normal: it means "no chip", never an
error, and a fetch failure retains the previously cached date (dates
rarely move; stale beats blank).

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


def get_earnings_map(tickers):
    """
    {ticker: "YYYY-MM-DD" or None} for the requested tickers.

    Refresh policy: a full refresh at most once per ET day (fetched_on
    stamp); within the day, only tickers absent from the cache are
    topped up (e.g. a name just added to positions.json). A failed
    fetch keeps the previously cached date. Never raises.
    """
    tickers = [t for t in dict.fromkeys(tickers) if t]
    cache = _load_cache()
    today_iso = _today_et().isoformat()
    stale = cache.get("fetched_on") != today_iso

    to_fetch = [t for t in tickers if stale or t not in cache["dates"]]
    if to_fetch:
        print(f"[earnings] refreshing {len(to_fetch)} tickers "
              f"({'daily refresh' if stale else 'cache top-up'})")
        for t in to_fetch:
            try:
                d = _fetch_next_earnings(t)
            except Exception:
                d = None
            if d is not None:
                cache["dates"][t] = d.isoformat()
            else:
                # keep a previously known date over a flaky-endpoint miss;
                # store an explicit null only when nothing was known
                cache["dates"].setdefault(t, None)
        cache["fetched_on"] = today_iso
        _save_cache(cache)

    return {t: cache["dates"].get(t) for t in tickers}


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
