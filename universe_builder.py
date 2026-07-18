#!/usr/bin/env python3
"""
Universe Builder (Phase 2)
==========================
Builds the *active* dashboard universe from the candidate universe
(universe_source + gics_classifier), replacing the hardcoded
INDUSTRY_GROUPS constant with a weekly-rotated dynamic top-N:

  1. Classify all candidates by canonical GICS sub-industry (aliases applied).
  2. Fetch 1y price history for every candidate; compute YTD/3M/1M returns
     and the EXISTING dashboard composite score (signal_engine.score_stock).
  3. Rank every sub-industry by group composite:
         0.50 * median(YTD) + 0.30 * median(3M) + 0.20 * median(1M)
     (medians over the group's tickers with valid price data).
  4. Keep the top `rotation.top_n` ELIGIBLE groups: at least
     `rotation.min_candidates` candidate tickers (no solo/duo groups) and
     at least one QUALIFYING ticker: score >= min_composite_score, market
     cap >= min_market_cap USD (FX-converted), 63-day avg volume >=
     min_avg_volume, and >= min_history_days calendar days of history.
  5. Select up to `rotation.max_tickers_per_group` qualifiers per group,
     ordered by dashboard composite score (no force-fill; fewer is fine).
     Per-ticker YTD is kept in the selection detail so a YTD ordering
     stays derivable for a later UI toggle.
  6. Emit the result in the exact INDUSTRY_GROUPS schema signal_engine
     consumes, cached at data/universe_active.json (a committed artifact —
     data/universe_cache/ is gitignored and absent on fresh CI checkouts).
     Also emit the full audit table (every group, every candidate ticker,
     selection/exclusion status) to data|public/universe_ranking.json,
     served read-only at /api/universe/ranking.json.

Rotation: the universe rebuilds only when the cached week_key no longer
matches the current rotation week (boundary: Friday 20:00 ET, after the
weekly close — the build serves the FOLLOWING trading week), or with
--force. Within that week the cached membership is stable Mon-Fri.

Safety: a build below the MIN_VIABLE_* floor (data outage, rate-limit wave,
or the first January rotation before the year has 2 trading days of data)
raises UniverseBuildError instead of writing, so the previous week's
universe keeps serving until a viable build succeeds.

Standalone usage:
    python universe_builder.py --dry-run   # build, print report + diff, write nothing
    python universe_builder.py --force     # rebuild + write data/universe_active.json
"""

import argparse
import datetime
import json
import math
import os
import statistics
import time
from zoneinfo import ZoneInfo

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ACTIVE_PATH = os.path.join(BASE_DIR, "data", "universe_active.json")

ET = ZoneInfo("America/New_York")
# Rotation boundary: Friday 20:00 ET (after the weekly close). The build made
# at/after this boundary is the active universe FOR THE FOLLOWING TRADING WEEK
# (Mon-Fri), computed on the completed week's data and reviewable over the
# weekend before Monday deployment decisions.
ROTATION_WEEKDAY = 4   # Friday (Monday=0)
ROTATION_HOUR_ET = 20  # 8 PM ET

# Minimum-viability floor: a build below any of these is treated as a data
# failure (raises UniverseBuildError) rather than written/served — otherwise a
# Yahoo outage or rate-limit wave at rotation time would cache an empty
# universe as the week's truth and the stale-cache fallback would never engage.
MIN_VIABLE_GROUPS = 3
MIN_VIABLE_TICKERS = 8
MIN_PRICE_COVERAGE = 0.60   # share of candidates with usable price history
MIN_MCAP_COVERAGE = 0.70    # share of metric-valid tickers with a market cap

# Static FX fallbacks (approximate; only used when the live FX fetch fails)
# for converting listing-currency market caps to USD before the 500M floor.
FX_USD_FALLBACK = {
    "USD": 1.0, "KRW": 0.0007, "TWD": 0.031, "EUR": 1.10, "JPY": 0.0067,
    "GBP": 1.27, "GBp": 0.0127, "HKD": 0.13, "CAD": 0.73, "CHF": 1.12,
    "AUD": 0.66, "CNY": 0.14, "INR": 0.012, "DKK": 0.15, "SEK": 0.095,
    "NOK": 0.092,
}


class UniverseBuildError(RuntimeError):
    """Raised when a build is too degraded to trust (see MIN_VIABLE_*)."""


def _atomic_write_json(path, obj):
    """Write JSON via tmp-file + rename so API readers never see a partial
    document. Non-finite floats become null (bare NaN is invalid JSON for
    strict parsers). Raises on failure — artifact writes must not fail
    silently."""
    from signal_engine import sanitize_for_json
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(sanitize_for_json(obj), f, indent=2)
    os.replace(tmp, path)

DEFAULT_ROTATION = {
    "top_n": 15,
    "min_candidates": 3,
    "max_tickers_per_group": 7,
    "min_composite_score": 50,
    "min_market_cap": 500_000_000,
    "min_avg_volume": 500_000,
    "min_history_days": 90,
    "composite_weights": {"ytd": 0.50, "r3m": 0.30, "r1m": 0.20},
}

RANKING_FILENAME = "universe_ranking.json"

DEFAULT_SENSITIVITIES = ["sp500_drawdown", "group_momentum", "group_trend", "group_ytd"]

# Group metadata for the INDUSTRY_GROUPS schema, carried over verbatim from
# the legacy hardcoded groups (keyed by post-alias canonical names; the
# "Semiconductors" entry covers the retired "— Memory & HBM" custom group).
# Groups outside this map get neutral defaults. sector_type/commodity_proxy
# drive the commodity-specific thesis/breaker branches in signal_engine, so
# only groups listed here get those behaviors.
GROUP_METADATA = {
    "Technology Hardware, Storage & Peripherals": {
        "gics_code": "45202030", "sector": "Information Technology",
        "industry_group": "Technology Hardware & Equipment",
        "cycle_stage": "mid", "sector_type": "tech_hardware",
        "commodity_proxy": None,
        "macro_sensitivities": DEFAULT_SENSITIVITIES,
    },
    "Semiconductors": {
        "gics_code": "45301020", "sector": "Information Technology",
        "industry_group": "Semiconductors & Semiconductor Equipment",
        "cycle_stage": "early-mid", "sector_type": "semiconductor",
        "commodity_proxy": None,
        "macro_sensitivities": DEFAULT_SENSITIVITIES,
    },
    "Semiconductor Materials & Equipment": {
        "gics_code": "45301010", "sector": "Information Technology",
        "industry_group": "Semiconductors & Semiconductor Equipment",
        "cycle_stage": "mid", "sector_type": "semiconductor",
        "commodity_proxy": None,
        "macro_sensitivities": DEFAULT_SENSITIVITIES,
    },
    "Gold": {
        "gics_code": "15104030", "sector": "Materials",
        "industry_group": "Metals & Mining",
        "cycle_stage": "mid", "sector_type": "precious_metals",
        "commodity_proxy": "GLD",
        "macro_sensitivities": ["commodity_drop", "usd_strength", "group_momentum",
                                "group_trend", "breadth_collapse"],
    },
    "Copper": {
        "gics_code": "15104025", "sector": "Materials",
        "industry_group": "Metals & Mining",
        "cycle_stage": "mid", "sector_type": "industrial_metals",
        "commodity_proxy": None,
        "macro_sensitivities": ["group_momentum", "group_trend", "group_ytd", "sp500_drawdown"],
    },
    "Specialty Chemicals": {
        "gics_code": "15101050", "sector": "Materials",
        "industry_group": "Chemicals",
        "cycle_stage": "early", "sector_type": "chemicals",
        "commodity_proxy": None,
        "macro_sensitivities": ["group_momentum", "group_trend", "group_ytd", "energy_spike"],
    },
    "Oil & Gas Equipment & Services": {
        "gics_code": "10101020", "sector": "Energy",
        "industry_group": "Energy Equipment & Services",
        "cycle_stage": "mid", "sector_type": "oil_services",
        "commodity_proxy": "USO",
        "macro_sensitivities": ["oil_collapse", "group_momentum", "group_trend"],
    },
    "Oil & Gas Exploration & Production": {
        "gics_code": "10102020", "sector": "Energy",
        "industry_group": "Oil, Gas & Consumable Fuels",
        "cycle_stage": "mid", "sector_type": "oil_gas_ep",
        "commodity_proxy": "USO",
        "macro_sensitivities": ["oil_collapse", "natgas_collapse", "group_momentum", "group_trend"],
    },
    "Aerospace & Defense": {
        "gics_code": "20101010", "sector": "Industrials",
        "industry_group": "Capital Goods",
        "cycle_stage": "early-mid", "sector_type": "defense",
        "commodity_proxy": None,
        "macro_sensitivities": ["group_momentum", "group_trend", "group_ytd"],
    },
    "Independent Power Producers & Energy Traders": {
        "gics_code": "55105020", "sector": "Utilities",
        "industry_group": "Independent Power and Renewable Electricity Producers",
        "cycle_stage": "early-mid", "sector_type": "power_nuclear",
        "commodity_proxy": None,
        "macro_sensitivities": ["group_momentum", "group_trend", "group_ytd", "sp500_drawdown"],
    },
    "Coal & Consumable Fuels": {
        "gics_code": "10102050", "sector": "Energy",
        "industry_group": "Oil, Gas & Consumable Fuels",
        "cycle_stage": "early", "sector_type": "uranium",
        "commodity_proxy": None,
        "macro_sensitivities": ["group_momentum", "group_trend", "group_ytd"],
    },
    "Oil & Gas Refining & Marketing": {
        "gics_code": "10102030", "sector": "Energy",
        "industry_group": "Oil, Gas & Consumable Fuels",
        "cycle_stage": "mid-late", "sector_type": "refining",
        "commodity_proxy": "USO",
        "macro_sensitivities": ["group_momentum", "group_trend", "group_ytd"],
    },
}


def _now_utc():
    return datetime.datetime.now(datetime.timezone.utc)


def rotation_week_key(now=None):
    """Date (ISO) of the most recent Friday 20:00 ET rotation boundary.

    The key names the FRIDAY the universe was (or would have been) rotated
    on; that universe is active until the next Friday 20:00 ET, i.e. for the
    following trading week. Example: key '2026-07-03' covers Fri 2026-07-03
    20:00 ET through Fri 2026-07-10 19:59 ET — so a Monday signal run and the
    Friday-evening build that preceded it agree, and the self-heal path in
    get_industry_groups() does not rebuild over a fresh Friday rotation.
    """
    now_et = (now or _now_utc()).astimezone(ET)
    days_since = (now_et.weekday() - ROTATION_WEEKDAY) % 7
    friday = (now_et - datetime.timedelta(days=days_since)).date()
    boundary = datetime.datetime.combine(
        friday, datetime.time(ROTATION_HOUR_ET, 0), tzinfo=ET)
    if now_et < boundary:
        friday -= datetime.timedelta(days=7)
    return friday.isoformat()


def effective_week_of(week_key):
    """Monday (ISO) of the trading week a rotation serves (key + 3 days)."""
    friday = datetime.date.fromisoformat(week_key)
    return (friday + datetime.timedelta(days=3)).isoformat()


# ----------------------------------------------------------------------
# Data fetching
# ----------------------------------------------------------------------
def _clean_frame(sub):
    if sub is None or sub.empty:
        return None
    sub = sub.dropna(subset=["Close"])
    if sub.empty:
        return None
    sub = sub.copy()
    try:
        sub.index = sub.index.tz_localize(None)
    except TypeError:
        pass
    return sub


def _fetch_price_batch(tickers, period="1y", chunk_size=100, max_single_retries=200):
    """Batch-download OHLCV history; returns {ticker: DataFrame}.

    Uses yf.download in chunks (threaded) — vastly faster than per-ticker
    calls for 500+ names — then retries whatever the batch missed (failed
    chunk, ticker absent from the frame) one-by-one via signal_engine's
    fetch_data. Frames match signal_engine.fetch_data format (naive index,
    Open/High/Low/Close/Volume, auto-adjusted closes).
    """
    import pandas as pd
    import yfinance as yf
    import signal_engine as se

    out = {}
    tickers = sorted(set(tickers))
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i + chunk_size]
        print(f"[builder] downloading prices {i + 1}-{i + len(chunk)} of {len(tickers)}...")
        try:
            df = yf.download(chunk, period=period, interval="1d", group_by="ticker",
                             auto_adjust=True, threads=True, progress=False)
        except Exception as e:
            print(f"[builder] batch download failed for chunk ({e}); "
                  f"will retry these tickers individually")
            df = None
        if df is None or df.empty:
            continue
        # single-ticker chunks can come back with flat (non-MultiIndex) columns
        if not isinstance(df.columns, pd.MultiIndex) and len(chunk) == 1:
            sub = _clean_frame(df)
            if sub is not None:
                out[chunk[0]] = sub
            continue
        for t in chunk:
            try:
                if t in df.columns.get_level_values(0):
                    sub = _clean_frame(df[t])
                    if sub is not None:
                        out[t] = sub
            except Exception:
                continue

    missing = [t for t in tickers if t not in out]
    if missing:
        if len(missing) > max_single_retries:
            print(f"[builder] {len(missing)} tickers missing after batch download — "
                  f"too many for single retries; build viability check will decide")
        else:
            print(f"[builder] retrying {len(missing)} missing tickers individually...")
            for t in missing:
                sub = _clean_frame(se.fetch_data(t, period=period))
                if sub is not None:
                    out[t] = sub
                time.sleep(0.2)
            still = len([t for t in missing if t not in out])
            print(f"[builder] single retries recovered {len(missing) - still}, "
                  f"still missing {still}")
    return out


def _fetch_market_caps(tickers, workers=6):
    """{ticker: {market_cap, currency, market_cap_usd}} via yfinance fast_info.

    Threaded, with one retry pass for failures (a transient rate-limit wave
    must not become a mass 'mcap unavailable' disqualification). Caps are
    converted to USD so the min_market_cap floor is meaningful for
    foreign-listed candidates (KRW/TWD/EUR).
    """
    import yfinance as yf
    from concurrent.futures import ThreadPoolExecutor

    def one(t):
        try:
            fi = yf.Ticker(t).fast_info
            cap = fi["marketCap"]
            return t, {"market_cap": cap, "currency": fi.get("currency")}
        except Exception:
            return t, {"market_cap": None, "currency": None}

    tickers = sorted(set(tickers))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        caps = dict(ex.map(one, tickers))

    failed = [t for t, v in caps.items() if v["market_cap"] is None]
    if failed:
        print(f"[builder] retrying market cap for {len(failed)} tickers...")
        time.sleep(2)
        with ThreadPoolExecutor(max_workers=2) as ex:
            caps.update(dict(ex.map(one, failed)))

    fx = _fx_to_usd({v["currency"] for v in caps.values() if v["currency"]})
    for v in caps.values():
        cap, cur = v["market_cap"], v["currency"] or "USD"
        rate = fx.get(cur)
        v["market_cap_usd"] = cap * rate if (cap is not None and rate) else (
            cap if cur == "USD" else None)
    return caps


def _fx_to_usd(currencies):
    """{currency: rate_to_usd} via yfinance FX pairs, static fallback on failure."""
    import yfinance as yf

    rates = {"USD": 1.0}
    for cur in sorted(c for c in currencies if c and c != "USD"):
        pair = ("GBP" if cur == "GBp" else cur) + "USD=X"
        rate = None
        try:
            rate = float(yf.Ticker(pair).fast_info["lastPrice"])
            if cur == "GBp":
                rate /= 100.0
        except Exception:
            rate = FX_USD_FALLBACK.get(cur)
            print(f"[builder] live FX failed for {cur}; using fallback {rate}")
        if rate:
            rates[cur] = rate
    return rates


def _load_subindustry_sector_map(cache_dir, aliases):
    """{canonical sub-industry: GICS sector} from the S&P 500 datahub CSV
    (majority vote per sub-industry). Cached; stale cache accepted on failure."""
    import csv
    from io import StringIO
    from collections import Counter, defaultdict

    from universe_source import SP500_CSV_URL, _HEADERS

    cache_path = os.path.join(cache_dir, "subindustry_sectors.json")
    try:
        with open(cache_path) as f:
            cached = json.load(f)
        fetched = datetime.datetime.fromisoformat(cached["fetched_at"])
        if (_now_utc() - fetched).total_seconds() < 7 * 86400:
            return cached["map"]
    except Exception:
        cached = None

    votes = defaultdict(Counter)
    try:
        import requests
        r = requests.get(SP500_CSV_URL, headers=_HEADERS, timeout=30)
        r.raise_for_status()
        for row in csv.DictReader(StringIO(r.text)):
            sub = (row.get("GICS Sub-Industry") or "").strip()
            sector = (row.get("GICS Sector") or "").strip()
            if sub and sector:
                votes[aliases.get(sub, sub)][sector] += 1
        result = {sub: c.most_common(1)[0][0] for sub, c in votes.items()}
        try:
            with open(cache_path, "w") as f:
                json.dump({"fetched_at": _now_utc().isoformat(), "map": result}, f, indent=2)
        except Exception:
            pass
        return result
    except Exception as e:
        print(f"[builder] sector map fetch failed: {e}")
        return cached["map"] if cached else {}


# ----------------------------------------------------------------------
# Metrics + qualification
# ----------------------------------------------------------------------
def _weeks_in_universe(name, prev_groups, prev_week, this_week):
    """Consecutive weekly rotations `name` has been selected (D-007 Phase 1,
    observe-only). Advances ONLY when the rotation week changed; a same-week
    rebuild (--force, retry, dispatch) carries the previous value unchanged.
    A group present last week without the field (pre-Phase-1 artifact) seeds
    at 1 so the next new week reads 2; a newly selected group starts at 1."""
    if name not in (prev_groups or {}):
        return 1
    prev = int((prev_groups[name] or {}).get("weeks_in_universe") or 1)
    return prev if prev_week == this_week else prev + 1


def _ticker_metrics(ticker, df, sp500_ytd=None):
    """Per-ticker returns + dashboard composite score, via signal_engine."""
    import signal_engine as se

    if df is None or len(df) < 63:
        return None
    current_year = datetime.datetime.now().year
    if len(df[df.index.year == current_year]) < 2:
        return None

    ytd = se.compute_ytd_return(df)
    momentum = se.compute_momentum_metrics(df)
    score, signal, _details = se.score_stock(df)
    span_days = (df.index[-1] - df.index[0]).days
    avg_volume = float(df["Volume"].iloc[-63:].mean())
    if math.isnan(avg_volume):  # Yahoo returns null volumes for some listings
        avg_volume = 0.0
    return {
        "ticker": ticker,
        "ytd": ytd,
        "r3m": momentum["return_3m"],
        "r1m": momentum["return_1m"],
        "score": score,
        "score_components": _details.get("score_components"),
        # Raw price>MA50 boolean (the +6 term behind the score `ma` component),
        # surfaced explicitly so the Fear & Greed "Market Internals" breadth row
        # (D-012) reads it directly instead of reverse-engineering it from the
        # component points. Uses score_inputs (pre-round) to match the component.
        "above_50dma": (_details.get("score_inputs") or {}).get("above_ma50"),
        "signal": signal,
        "history_days": int(span_days),
        "rows": int(len(df)),
        "avg_volume": round(avg_volume),
    }


def _qualify(m, caps, rot):
    """Return (qualified: bool, list of failed filter names)."""
    fails = []
    if m["score"] < rot["min_composite_score"]:
        fails.append(f"score {m['score']}<{rot['min_composite_score']}")
    if m["history_days"] < rot["min_history_days"]:
        fails.append(f"history {m['history_days']}d<{rot['min_history_days']}d")
    if m["avg_volume"] < rot["min_avg_volume"]:
        fails.append(f"avg_vol {m['avg_volume']:,}<{rot['min_avg_volume']:,}")
    cap = (caps.get(m["ticker"]) or {}).get("market_cap_usd")
    if cap is None:
        fails.append("mcap unavailable")
    elif cap < rot["min_market_cap"]:
        fails.append(f"mcap ${cap / 1e6:,.0f}M<${rot['min_market_cap'] / 1e6:,.0f}M")
    return (not fails), fails


def _gate_status(fails):
    """Ticker status for the first failed qualifier gate."""
    first = fails[0]
    if first.startswith("score"):
        return "failed_score_gate"
    if first.startswith("history"):
        return "failed_history_gate"
    if first.startswith("avg_vol"):
        return "failed_volume_gate"
    return "failed_mcap_gate"


def _group_composite(members, weights):
    """Weighted median composite over the group's valid tickers."""
    med_ytd = statistics.median(m["ytd"] for m in members)
    med_3m = statistics.median(m["r3m"] for m in members)
    med_1m = statistics.median(m["r1m"] for m in members)
    composite = (weights["ytd"] * med_ytd + weights["r3m"] * med_3m
                 + weights["r1m"] * med_1m)
    return round(composite, 2), round(med_ytd, 2), round(med_3m, 2), round(med_1m, 2)


def _sector_via_yahoo(ticker):
    """Last-resort sector lookup (one .info call) for groups the S&P CSV
    doesn't cover — e.g. on a fresh CI checkout where the CSV fetch failed."""
    try:
        import yfinance as yf
        return (yf.Ticker(ticker).info or {}).get("sector") or ""
    except Exception:
        return ""


def _group_schema_entry(name, tickers, sector_map):
    """One INDUSTRY_GROUPS-schema entry for a selected group."""
    meta = GROUP_METADATA.get(name, {})
    sector = meta.get("sector") or sector_map.get(name, "")
    if not sector and tickers:
        sector = _sector_via_yahoo(tickers[0])
    return {
        "gics_code": meta.get("gics_code", ""),
        "gics_level": "Sub-Industry",
        "sector": sector,
        "industry_group": meta.get("industry_group", ""),
        "tickers": tickers,
        "cycle_stage": meta.get("cycle_stage", "mid"),
        "sector_type": meta.get("sector_type", ""),
        "commodity_proxy": meta.get("commodity_proxy"),
        "macro_sensitivities": list(meta.get("macro_sensitivities", DEFAULT_SENSITIVITIES)),
    }


# ----------------------------------------------------------------------
# Build
# ----------------------------------------------------------------------
def load_rotation_config():
    from universe_source import load_universe_config
    uni_cfg = load_universe_config()
    rot = dict(DEFAULT_ROTATION)
    rot.update(uni_cfg.get("rotation", {}) or {})
    weights = dict(DEFAULT_ROTATION["composite_weights"])
    weights.update((uni_cfg.get("rotation", {}) or {}).get("composite_weights", {}) or {})
    rot["composite_weights"] = weights
    return uni_cfg, rot


def build_active_universe(write=True, verbose=True):
    """Build the active universe from live data. Returns the full artifact:
    {built_at, week_key, groups: INDUSTRY_GROUPS-schema, ranking, detail}."""
    from universe_source import build_universe_candidates, DEFAULT_CACHE_DIR

    import signal_engine as se

    uni_cfg, rot = load_rotation_config()
    weights = rot["composite_weights"]

    # Cheap pre-flight before the ~530-ticker fetch: one index call proves
    # Yahoo is reachable AND the current year has >=2 trading days (the first
    # January build cannot produce YTD metrics — fail in seconds, not minutes,
    # so a self-healing weekday run falls back to the stale universe cheaply).
    spx = se.fetch_data("^GSPC", period="1mo")
    if spx is None or len(spx[spx.index.year == datetime.datetime.now().year]) < 2:
        raise UniverseBuildError(
            "pre-flight failed: index data unavailable or <2 trading days in the "
            "current year (YTD undefined) — keeping previous universe")

    # 1. candidates + canonical GICS classification (aliases applied).
    # Candidates artifact is NOT written here — only after the viability
    # floor passes (a degraded candidate list must never be published).
    out, tickers, by_gics = build_universe_candidates(write=False)
    by_gics = {k: v for k, v in by_gics.items() if k != "_unclassified"}
    if verbose:
        print(f"[builder] {len(tickers)} candidates in {len(by_gics)} sub-industries "
              f"({out['unclassified']} unclassified excluded)")

    if not tickers:
        raise UniverseBuildError("candidate universe is empty (all feeds failed?)")

    # 2. price history + per-ticker metrics
    prices = _fetch_price_batch(tickers)
    metrics = {}
    for t in tickers:
        try:
            m = _ticker_metrics(t, prices.get(t))
        except Exception as e:
            print(f"[builder] metrics failed for {t}: {e} — skipping")
            m = None
        if m:
            metrics[t] = m
    if verbose:
        print(f"[builder] price data OK for {len(prices)}, valid metrics for {len(metrics)}")
    if len(prices) < MIN_PRICE_COVERAGE * len(tickers):
        raise UniverseBuildError(
            f"price data for only {len(prices)}/{len(tickers)} candidates "
            f"(<{MIN_PRICE_COVERAGE:.0%}) — refusing to build from degraded data")

    # 3. market caps (only tickers that still matter: valid metrics)
    caps = _fetch_market_caps(list(metrics.keys()))
    caps_ok = sum(1 for v in caps.values() if v.get("market_cap_usd") is not None)
    if metrics and caps_ok < MIN_MCAP_COVERAGE * len(metrics):
        raise UniverseBuildError(
            f"market cap for only {caps_ok}/{len(metrics)} tickers "
            f"(<{MIN_MCAP_COVERAGE:.0%}) — refusing to build from degraded data")

    # 4. rank all groups (every sub-industry appears; no-data groups unranked)
    ranking = []
    for name, members in by_gics.items():
        valid = [metrics[t] for t in members if t in metrics]
        qualified, disqualified = [], {}
        for m in valid:
            ok, fails = _qualify(m, caps, rot)
            if ok:
                qualified.append(m)
            else:
                disqualified[m["ticker"]] = fails
        # score desc, then YTD desc, then ticker asc — deterministic
        qualified.sort(key=lambda m: (-m["score"], -m["ytd"], m["ticker"]))
        if valid:
            composite, med_ytd, med_3m, med_1m = _group_composite(valid, weights)
        else:
            composite = med_ytd = med_3m = med_1m = None
        ranking.append({
            "name": name,
            "composite": composite,
            "median_ytd": med_ytd,
            "median_3m": med_3m,
            "median_1m": med_1m,
            "candidates": len(members),
            "members": sorted(members),
            "valid": len(valid),
            "qualifier_count": len(qualified),
            "qualifiers": [
                {"ticker": m["ticker"], "score": m["score"], "ytd": m["ytd"],
                 "r3m": m["r3m"], "r1m": m["r1m"], "avg_volume": m["avg_volume"],
                 "market_cap": (caps.get(m["ticker"]) or {}).get("market_cap"),
                 "market_cap_usd": (caps.get(m["ticker"]) or {}).get("market_cap_usd"),
                 "currency": (caps.get(m["ticker"]) or {}).get("currency")}
                for m in qualified
            ],
            "disqualified": disqualified,
        })
    ranked = [g for g in ranking if g["composite"] is not None]
    ranked.sort(key=lambda g: (-g["composite"], g["name"]))
    for i, g in enumerate(ranked):
        g["rank"] = i + 1
    unranked = sorted((g for g in ranking if g["composite"] is None),
                      key=lambda g: g["name"])
    for g in unranked:
        g["rank"] = None
    ranking = ranked + unranked

    # 5. eligibility + selection: top-N eligible groups, up to max tickers each
    for g in ranking:
        g["eligible"] = (g["candidates"] >= rot["min_candidates"]
                         and g["qualifier_count"] >= 1)
    eligible = [g for g in ranked if g["eligible"]]
    selected = eligible[:rot["top_n"]]
    selected_names = {g["name"] for g in selected}
    for g in ranking:
        if g["name"] in selected_names:
            g["exclusion_reason"] = None
        elif g["composite"] is None:
            g["exclusion_reason"] = "no_valid_data"
        elif g["candidates"] < rot["min_candidates"]:
            g["exclusion_reason"] = "below_min_candidates"
        elif g["qualifier_count"] == 0:
            g["exclusion_reason"] = "no_qualifiers"
        else:
            g["exclusion_reason"] = "outranked"

    # 6. per-ticker audit status (for the read-only ranking endpoint)
    for g in ranking:
        chosen = {q["ticker"] for q in g["qualifiers"][:rot["max_tickers_per_group"]]}
        qual_by_ticker = {q["ticker"]: q for q in g["qualifiers"]}
        tickers_out = []
        for t in g["members"]:
            m = metrics.get(t)
            if m is None:
                tickers_out.append({"ticker": t, "score": None, "ytd": None,
                                    "status": "no_valid_data", "fails": []})
                continue
            entry = {"ticker": t, "score": m["score"], "ytd": m["ytd"],
                     "components": m.get("score_components"),
                     "above_50dma": m.get("above_50dma"), "fails": []}
            if t in qual_by_ticker:
                if g["exclusion_reason"] == "below_min_candidates":
                    entry["status"] = "group_below_min_candidates"
                elif g["exclusion_reason"] == "outranked":
                    entry["status"] = "group_outranked"
                elif t in chosen:
                    entry["status"] = "selected"
                else:
                    entry["status"] = "outranked_within_group"
            else:
                entry["fails"] = g["disqualified"][t]
                entry["status"] = _gate_status(entry["fails"])
            tickers_out.append(entry)
        tickers_out.sort(key=lambda e: (e["score"] is None, -(e["score"] or 0),
                                        e["ticker"]))
        g["tickers"] = tickers_out
        del g["members"]

    sector_map = _load_subindustry_sector_map(DEFAULT_CACHE_DIR,
                                              uni_cfg.get("gics_aliases") or {})
    # weeks_in_universe (D-007 Phase 1, OBSERVE ONLY — no persistence gate):
    # consecutive weekly ROTATIONS a group has been selected. The counter
    # advances only when the rotation week actually changed — a same-week
    # rebuild (--force, flaky-fetch retry, manual dispatch) CARRIES the
    # previous value unchanged (review finding: build-counting would let
    # every --force re-run add a phantom week to the exact dataset the
    # future persistence-gate decision observes). A group present last week
    # without the field (pre-Phase-1 artifact) seeds at 1 so a new week
    # reads 2.
    prev_groups = {}
    prev_week = None
    prev_active = load_cached_active()
    if prev_active:
        prev_groups = prev_active.get("groups") or {}
        prev_week = prev_active.get("week_key")
    this_week = rotation_week_key()
    groups = {}
    for g in selected:
        chosen = [q["ticker"] for q in g["qualifiers"][:rot["max_tickers_per_group"]]]
        entry = _group_schema_entry(g["name"], chosen, sector_map)
        # Compact per-group audit for the dashboard's "near misses" strip —
        # embedded here so signals.json (hourly) can never drift from the
        # rotation-week selection it was built with.
        entry["near_misses"] = _near_misses_from_ranking(g)
        entry["weeks_in_universe"] = _weeks_in_universe(
            g["name"], prev_groups, prev_week, this_week)
        groups[g["name"]] = entry

    total_selected = sum(len(g["tickers"]) for g in groups.values())
    if len(groups) < MIN_VIABLE_GROUPS or total_selected < MIN_VIABLE_TICKERS:
        raise UniverseBuildError(
            f"build produced only {len(groups)} groups / {total_selected} tickers "
            f"(floor: {MIN_VIABLE_GROUPS} groups / {MIN_VIABLE_TICKERS} tickers) — "
            f"not writing; stale cache stays authoritative")

    week_key = rotation_week_key()
    active = {
        "built_at": _now_utc().isoformat(),
        "week_key": week_key,
        "effective_week_of": effective_week_of(week_key),
        "rotation_config": rot,
        "groups": groups,
        "ranking": ranking,
        "candidates_total": len(tickers),
    }

    if write:
        # Viability floor passed — publish the full artifact set atomically.
        # Any write failure raises: the rotation must fail visibly rather
        # than commit a mutually inconsistent set.
        from universe_source import write_candidates_artifact
        _atomic_write_json(ACTIVE_PATH, active)
        _write_ranking_json(active)
        write_candidates_artifact(out)
        if verbose:
            print(f"[builder] active universe written to {ACTIVE_PATH}")
    return active


def build_ranking_payload(active):
    """Read-only observability payload for /api/universe/ranking.json:
    the COMPLETE group ranking with per-ticker audit statuses."""
    return {
        "generated_at": active["built_at"],
        "week_key": active["week_key"],
        "rotation_config": active["rotation_config"],
        "candidates_total": active["candidates_total"],
        "total_groups": len(active["ranking"]),
        "selected_groups": len(active["groups"]),
        "status_legend": {
            "selected": "in the active universe",
            "outranked_within_group": "qualified, but group already has max_tickers_per_group better scores",
            "group_outranked": "qualified, but group ranked below top_n",
            "group_below_min_candidates": "qualified, but group has fewer than min_candidates tickers",
            "failed_score_gate": "dashboard composite score below min_composite_score",
            "failed_history_gate": "less than min_history_days of price history",
            "failed_volume_gate": "63-day average volume below min_avg_volume",
            "failed_mcap_gate": "market cap below min_market_cap USD (or unavailable)",
            "no_valid_data": "no usable price data (needs >=63 trading days incl. current year)",
        },
        "groups": [
            {"rank": g["rank"], "name": g["name"],
             "selected": g["exclusion_reason"] is None,
             "eligible": g["eligible"],
             "exclusion_reason": g["exclusion_reason"],
             "composite": g["composite"], "median_ytd": g["median_ytd"],
             "median_3m": g["median_3m"], "median_1m": g["median_1m"],
             "candidates": g["candidates"], "valid": g["valid"],
             "qualifier_count": g["qualifier_count"],
             "tickers": g["tickers"]}
            for g in active["ranking"]
        ],
    }


def _write_ranking_json(active):
    payload = build_ranking_payload(active)
    for d in (os.path.join(BASE_DIR, "data"), os.path.join(BASE_DIR, "public")):
        _atomic_write_json(os.path.join(d, RANKING_FILENAME), payload)


# ----------------------------------------------------------------------
# Loader for signal_engine (cache-aware)
# ----------------------------------------------------------------------
def _near_misses_from_ranking(rank_entry):
    """Compact non-selected candidates of a group (score desc, matching the
    audit ordering): outranked_within_group + gate failures. no_valid_data
    tickers are omitted (nothing meaningful to show)."""
    out = []
    for t in (rank_entry or {}).get("tickers", []):
        if t.get("status") in ("selected", "no_valid_data"):
            continue
        nm = {"ticker": t["ticker"], "score": t["score"], "ytd": t["ytd"],
              "status": t["status"]}
        if t.get("components"):
            nm["components"] = t["components"]
        if t.get("fails"):
            nm["fails"] = t["fails"]
        out.append(nm)
    return out


def _with_near_misses(cached):
    """Groups view with near_misses guaranteed. Artifacts built before the
    near-miss embed lack the key on group entries — retrofit it from the
    same artifact's ranking section (same build, so still consistent)."""
    groups = cached["groups"]
    by_name = {r.get("name"): r for r in cached.get("ranking", [])}
    for name, entry in groups.items():
        if "near_misses" not in entry:
            entry["near_misses"] = _near_misses_from_ranking(by_name.get(name))
    return groups


def load_cached_active():
    """Read data/universe_active.json; None if missing, malformed, or not
    viable (malformed == not viable == rebuild; never crash the caller)."""
    try:
        with open(ACTIVE_PATH) as f:
            cached = json.load(f)
        groups = cached.get("groups")
        if (isinstance(groups, dict) and len(groups) >= MIN_VIABLE_GROUPS
                and all(isinstance(g, dict) and g.get("tickers")
                        for g in groups.values())):
            return cached
    except Exception:
        return None
    print(f"[builder] cached universe at {ACTIVE_PATH} is not viable — ignoring")
    return None


def get_active_industry_groups(force=False):
    """INDUSTRY_GROUPS-schema dict for the current rotation week.

    Returns the cached universe if it belongs to the current week; otherwise
    rebuilds (self-healing if the Friday rotation was missed). A rebuild that
    fails or is below the viability floor raises inside build_active_universe
    and never overwrites the cache, so the stale universe keeps serving.
    Returns None only when there is no viable universe at all — the caller
    decides its own fallback.
    """
    cached = load_cached_active()
    if (not force) and cached and cached.get("week_key") == rotation_week_key():
        return _with_near_misses(cached)
    try:
        return build_active_universe(write=True)["groups"]
    except Exception as e:
        print(f"[builder] rebuild failed: {e}")
        if cached:
            print(f"[builder] falling back to stale universe from {cached.get('built_at')}")
            return _with_near_misses(cached)
        return None


# ----------------------------------------------------------------------
# Dry-run reporting
# ----------------------------------------------------------------------
def _mark(g):
    return {"below_min_candidates": "m", "no_qualifiers": "x",
            "no_valid_data": "-", None: "*"}.get(g["exclusion_reason"], " ")


def print_report_and_diff(active, baseline_groups, top_context=30,
                          baseline_label="legacy hardcoded universe"):
    rot = active["rotation_config"]
    ranking = active["ranking"]
    selected_names = list(active["groups"].keys())

    print(f"\n{'=' * 100}")
    print(f"DRY RUN — would-be active universe (week {active['week_key']}, "
          f"built {active['built_at'][:16]}, {active['candidates_total']} candidates)")
    print(f"{'=' * 100}")
    print(f"FULL RANKING (top {top_context} of {len(ranking)} groups; "
          f"full table -> universe_ranking.json / /api/universe/ranking.json)")
    print(f"{'rk':>3} {'group':48} {'comp':>8} {'medYTD':>8} {'med3M':>7} "
          f"{'med1M':>7} {'n':>3} {'ok':>3} {'q':>3}  exclusion / selection")
    for g in ranking[:top_context]:
        chosen = [q["ticker"] for q in g["qualifiers"][:rot["max_tickers_per_group"]]]
        tail = ",".join(chosen) if g["exclusion_reason"] is None else g["exclusion_reason"]
        comp = f"{g['composite']:>8.2f}" if g["composite"] is not None else f"{'—':>8}"
        meds = (f"{g['median_ytd']:>8.2f} {g['median_3m']:>7.2f} {g['median_1m']:>7.2f}"
                if g["composite"] is not None else f"{'—':>8} {'—':>7} {'—':>7}")
        print(f"{g['rank'] or '—':>3} {_mark(g)}{g['name']:47} {comp} {meds} "
              f"{g['candidates']:>3} {g['valid']:>3} {g['qualifier_count']:>3}  {tail}")
    print("   (* = selected, m = below min_candidates, x = no qualifying ticker, "
          "- = no valid data)")

    print(f"\nSELECTED GROUPS — tickers (dashboard score, YTD%):")
    for name in selected_names:
        g = next(r for r in ranking if r["name"] == name)
        chosen = g["qualifiers"][:rot["max_tickers_per_group"]]
        picks = ", ".join(f"{q['ticker']}({q['score']}, {q['ytd']:+.1f}%)" for q in chosen)
        foreign = [q["ticker"] for q in chosen if q.get("currency") not in (None, "USD")]
        extra = f"  [non-USD mcap: {','.join(foreign)}]" if foreign else ""
        print(f"  #{g['rank']:>2} {name} — {len(chosen)}/{g['qualifier_count']} "
              f"qualifiers used: {picks}{extra}")
        if len(chosen) < rot["max_tickers_per_group"] and g["disqualified"]:
            reasons = "; ".join(f"{t}: {', '.join(f)}" for t, f in
                                sorted(g["disqualified"].items())[:6])
            print(f"       under {rot['max_tickers_per_group']} — disqualified: {reasons}")

    # ---- diff vs baseline ----
    base_canon = {}   # canonical name -> tickers (baseline)
    from universe_source import load_universe_config
    aliases = (load_universe_config().get("gics_aliases") or {})
    for name, info in baseline_groups.items():
        base_canon.setdefault(aliases.get(name, name), []).extend(info["tickers"])

    new_groups = set(selected_names)
    old_groups = set(base_canon.keys())
    new_tickers = {t for g in active["groups"].values() for t in g["tickers"]}
    old_tickers = {t for ts in base_canon.values() for t in ts}

    print(f"\n{'=' * 100}")
    print(f"DIFF vs {baseline_label} "
          f"({len(old_groups)} groups / {len(old_tickers)} tickers "
          f"→ {len(new_groups)} groups / {len(new_tickers)} tickers)")
    print(f"{'=' * 100}")

    by_name = {g["name"]: g for g in ranking}
    print("\nGROUPS KEPT:")
    for n in sorted(new_groups & old_groups):
        print(f"  = {n}")
    print("GROUPS ADDED:")
    for n in sorted(new_groups - old_groups):
        g = by_name.get(n)
        print(f"  + {n}  (rank {g['rank']}, composite {g['composite']})" if g
              else f"  + {n}")
    print("GROUPS DROPPED:")
    for n in sorted(old_groups - new_groups):
        g = by_name.get(n)
        if g is None:
            print(f"  - {n}  (no candidate tickers in universe)")
        elif g["composite"] is None:
            print(f"  - {n}  (no candidates with valid data)")
        else:
            print(f"  - {n}  (ranked #{g['rank']}, composite {g['composite']}, "
                  f"qualifiers {g['qualifier_count']}, "
                  f"reason: {g['exclusion_reason']})")

    print(f"\nTICKERS KEPT ({len(new_tickers & old_tickers)}): "
          f"{', '.join(sorted(new_tickers & old_tickers))}")
    print(f"TICKERS ADDED ({len(new_tickers - old_tickers)}): "
          f"{', '.join(sorted(new_tickers - old_tickers))}")
    print(f"TICKERS DROPPED ({len(old_tickers - new_tickers)}): "
          f"{', '.join(sorted(old_tickers - new_tickers))}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build the weekly active dashboard universe")
    ap.add_argument("--dry-run", action="store_true",
                    help="build from live data, print report+diff, write nothing")
    ap.add_argument("--force", action="store_true",
                    help="rebuild even if the cached universe is current")
    ap.add_argument("--out", help="also dump the full artifact JSON to this path")
    args = ap.parse_args()

    # Capture the diff baseline BEFORE building — a write path overwrites
    # data/universe_active.json, which is exactly the previous universe we
    # want to diff against.
    prev = load_cached_active()

    if args.dry_run:
        active = build_active_universe(write=False)
    elif (not args.force) and prev and prev.get("week_key") == rotation_week_key():
        print(f"[builder] cached universe is current (week {prev['week_key']}); "
              f"use --force to rebuild")
        active = prev
    else:
        active = build_active_universe(write=True)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(active, f, indent=2)
        print(f"[builder] artifact dumped to {args.out}")

    # Diff baseline: the previous active universe when one exists (normal
    # weekly rotation audit), else the legacy hardcoded groups (first cutover).
    if prev is not None and prev.get("built_at") != active.get("built_at"):
        print_report_and_diff(active, prev["groups"],
                              baseline_label=f"previous active universe ({prev['week_key']})")
    else:
        from signal_engine import FALLBACK_INDUSTRY_GROUPS
        print_report_and_diff(active, FALLBACK_INDUSTRY_GROUPS,
                              baseline_label="legacy hardcoded universe")
