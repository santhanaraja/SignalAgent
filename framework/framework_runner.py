#!/usr/bin/env python3
"""
Framework Runner — Daily orchestrator that runs Regime → Theme → Rules
pipeline and writes output to framework/output/latest.json.
"""

import datetime
import json
import os
import traceback

import yaml
import yfinance as yf
import pandas as pd

from .regime_calculator import RegimeCalculator, artifact_schema
from .theme_ranker import ThemeRanker
from .rule_engine import RuleEngine
from .constituent_ranker import ConstituentRanker
from .position_signals import PositionSignalEngine


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.yaml")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
STATE_DIR = os.path.join(BASE_DIR, "state")


def load_config() -> dict:
    """Load framework config from YAML."""
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def fetch_data(ticker: str, period: str = "1y") -> pd.DataFrame:
    """
    Fetch OHLCV data via yfinance.
    Appends a real-time intraday row if market is open and history
    only contains yesterday's close.
    Returns DataFrame or None on failure.
    """
    try:
        tk = yf.Ticker(ticker)
        df = tk.history(period=period, auto_adjust=True)
        if df is None or df.empty:
            return None
        # Flatten multi-index columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Check if the latest bar is stale (yesterday or older)
        # If so, fetch a real-time quote and append it
        try:
            last_date = df.index[-1]
            today = pd.Timestamp.now(tz=last_date.tzinfo if last_date.tzinfo else "America/New_York").normalize()
            if last_date.normalize() < today:
                # History doesn't include today — try fast_info for current price
                info = tk.fast_info
                current_price = getattr(info, "last_price", None)
                if current_price and current_price > 0:
                    now_ts = pd.Timestamp.now(tz=last_date.tzinfo if last_date.tzinfo else "America/New_York")
                    new_row = pd.DataFrame({
                        "Open": [current_price],
                        "High": [current_price],
                        "Low": [current_price],
                        "Close": [current_price],
                        "Volume": [0],
                    }, index=[now_ts])
                    df = pd.concat([df, new_row])
                    print(f"[framework] {ticker}: appended live price ${current_price:.2f} (history was through {last_date.date()})")
        except Exception as e:
            pass  # Silently continue with historical data only

        return df
    except Exception as e:
        print(f"[framework] fetch_data({ticker}, {period}) failed: {e}")
        return None


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


def fetch_next_earnings(ticker: str):
    """
    Return the next upcoming earnings date (datetime.date) or None.
    Best-effort across yfinance versions; failures degrade to None so the
    "earnings_within_7d" warning is simply skipped.
    """
    today = datetime.date.today()
    try:
        tk = yf.Ticker(ticker)

        # Preferred: .calendar (dict in recent yfinance, DataFrame in older).
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

        # Fallback: .get_earnings_dates() returns a DataFrame indexed by date.
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


def build_universe_leadership(universe_active, signals_data):
    """The Layer-2 GICS leadership block (D-007 Phase 2): the current
    universe's selected groups joined with their ranking stats and the
    dashboard's breaker state. Pure + defensive — malformed artifacts
    contribute nothing. weeks_in_universe is None on pre-Phase-1 artifacts
    (the counter starts accruing at the 2026-07-25 rotation; the page says
    so honestly rather than faking history)."""
    if not isinstance(universe_active, dict):
        return []
    selected = universe_active.get("groups")
    if not isinstance(selected, dict) or not selected:
        return []
    rank_by_name = {}
    for g in universe_active.get("ranking") or []:
        if isinstance(g, dict) and g.get("name"):
            rank_by_name[g["name"]] = g
    breaker_by_name = {}
    if isinstance(signals_data, dict):
        for g in signals_data.get("groups") or []:
            if isinstance(g, dict) and g.get("name"):
                breaker_by_name[g["name"]] = g.get("breaker_status")
    rows = []
    for name, g in selected.items():
        if not isinstance(g, dict):
            continue
        r = rank_by_name.get(name) or {}
        rows.append({
            "name": name,
            "rank": r.get("rank"),
            "composite": r.get("composite"),
            "median_ytd": r.get("median_ytd"),
            "sector": g.get("sector"),
            "tickers": len(g.get("tickers") or []),
            "weeks_in_universe": g.get("weeks_in_universe"),
            "breaker_status": breaker_by_name.get(name),
        })
    rows.sort(key=lambda x: (x["rank"] is None, x["rank"] or 0, x["name"]))
    return rows


def load_regime_history() -> list:
    """Load regime_history.json."""
    path = os.path.join(STATE_DIR, "regime_history.json")
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return []


def save_regime_history(history: list, new_entry: dict):
    """Append new regime result to history and save."""
    os.makedirs(STATE_DIR, exist_ok=True)
    # Avoid duplicate dates
    history = [h for h in history if h.get("date") != new_entry.get("date")]
    # Store gauge values for comparison between runs
    gauge_snapshot = {}
    for gname, gdata in new_entry.get("gauges", {}).items():
        gauge_snapshot[gname] = {
            "value": gdata.get("value"),
            "signal": gdata.get("signal"),
            "detail": gdata.get("detail"),
        }

    entry = {
        "date": new_entry.get("date"),
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "regime": new_entry.get("regime"),
        "risk_on_count": new_entry.get("risk_on_count"),
        "caution_count": new_entry.get("caution_count"),
        "risk_off_count": new_entry.get("risk_off_count"),
        "gauges": gauge_snapshot,
    }
    # 3-voter era: keep the 200DMA gate and macro series in history so the
    # SPY/yield-curve time series survives their removal from the voter dict.
    # (Counts range 0-3 from these entries onward, 0-5 before.)
    bg = new_entry.get("backdrop_gate")
    if bg is not None:
        entry["backdrop_gate"] = {
            "open": bg.get("open"),
            "capped": bg.get("capped"),
            "reason": bg.get("reason"),
            "value": bg.get("value"),
            "detail": bg.get("detail"),
        }
    if new_entry.get("macro_inputs"):
        entry["macro_inputs"] = {
            name: {"value": m.get("value"), "signal": m.get("signal"),
                   "detail": m.get("detail")}
            for name, m in new_entry["macro_inputs"].items()
        }
    # NaN/Inf -> null before the write: this file is re-served verbatim by
    # /api/framework/history*, where a bare NaN token breaks JSON.parse in
    # the browser. Deferred import, same pattern as universe_builder.
    from signal_engine import sanitize_for_json
    history.append(sanitize_for_json(entry))
    # Keep last 52 weeks
    history = history[-52:]
    path = os.path.join(STATE_DIR, "regime_history.json")
    with open(path, "w") as f:
        json.dump(history, f, indent=2)


def run_framework(force_fetch: bool = False) -> dict:
    """
    Run the full framework pipeline.

    Args:
        force_fetch: If True, fetch fresh data even if recently run.

    Returns:
        Complete framework result dict.
    """
    print(f"[framework] Starting framework run at {datetime.datetime.now(datetime.timezone.utc).isoformat()}")

    config = load_config()

    # --- Layer 1: Regime ---
    print("[framework] Computing regime...")
    regime_history = load_regime_history()
    regime_calc = RegimeCalculator(config, fetch_data)
    regime_result = regime_calc.compute(regime_history)
    _gate = regime_result.get("backdrop_gate", {})
    _gate_note = "gate open" if _gate.get("open") else f"gate SHUT: {_gate.get('reason')}"
    print(f"[framework] Regime: {regime_result['regime']} ({regime_result['risk_on_count']} risk-on, "
          f"{regime_result['caution_count']} caution, {regime_result['risk_off_count']} risk-off "
          f"of 3 voters; {_gate_note})")

    # Save regime history
    save_regime_history(regime_history, regime_result)

    # --- Layer 2: Theme Rotation ---
    print("[framework] Computing theme rankings...")
    theme_ranker = ThemeRanker(config, fetch_data)
    theme_result = theme_ranker.compute(regime_result["regime"], regime_result)

    # Save theme history snapshot
    theme_ranker.save_weekly_snapshot(theme_result)

    top_themes = [t for t in theme_result["ranked_themes"] if t.get("rank") is not None]
    for t in top_themes[:3]:
        print(f"[framework]   #{t['rank']} {t['name']} ({t['proxy']}): "
              f"4w {t.get('return_4w', 'N/A')}%, 12w {t.get('return_12w', 'N/A')}%")

    # --- Layer 2.5: Constituent leaders for qualified themes ---
    print("[framework] Ranking constituents for qualified themes...")
    max_active = config.get("themes", {}).get("ranking", {}).get("max_active_themes", 2)
    qualified_names = [t["name"] for t in theme_result["ranked_themes"]
                       if t.get("rank") is not None and t["rank"] <= max_active]
    # Always include currently-held (active) themes so their leaders show too.
    for name in theme_result.get("active_themes", []):
        if name not in qualified_names:
            qualified_names.append(name)

    # Rank constituents for ALL themes (informational visibility); only the
    # qualified subset gets the full warning pipeline.
    all_theme_names = [t["name"] for t in config.get("themes", {}).get("watchlist", [])]
    constituent_ranker = ConstituentRanker(config, fetch_data, fetch_next_earnings)
    theme_leaders = constituent_ranker.compute(all_theme_names, qualified_names)
    print(f"[framework] Constituent leaders ranked for {len(theme_leaders)} themes "
          f"(qualified: {', '.join(qualified_names) or 'none'})")

    # --- Shared ticker->group resolution (ONE resolver — D-007 Phase 1):
    # condition 5 and R28 both consume this map. Loaded once here; the whole
    # block is best-effort (an unreadable/misshapen artifact degrades the
    # gate and R28, never the run — the old block lived inside R28's try).
    universe_active = {}
    signals_data = {}
    group_map = {}
    try:
        from .portfolio_rules import resolve_group_map
        try:
            with open(os.path.join(BASE_DIR, "..", "data",
                                   "universe_active.json")) as f:
                universe_active = json.load(f)
        except Exception:
            universe_active = {}
        try:
            with open(os.path.join(BASE_DIR, "..", "data",
                                   "signals.json")) as f:
                signals_data = json.load(f)
        except Exception:
            signals_data = {}
        if not isinstance(universe_active, dict):
            universe_active = {}
        group_map = resolve_group_map(universe_active, signals_data)
    except Exception as e:
        print(f"[framework] group resolution unavailable: {e}")
        universe_active, signals_data, group_map = {}, {}, {}

    # Active groups = the REAL holdings' GICS groups (D-007 Phase 2: the
    # rule engine's active-state trigger). Manual row 'group' overrides the
    # map; unmapped holdings bucket as "TICKER (ungrouped)" — R28's own
    # convention. Flat book -> [] (reminders idle).
    active_groups = None      # None = unknown (outage); [] = proven flat
    try:
        with open(os.path.join(BASE_DIR, "state", "positions.json")) as f:
            _pos = json.load(f)
        holdings = _pos.get("holdings")
        if not isinstance(holdings, list):
            if holdings not in (None, []):
                raise ValueError("positions.json holdings is not a list")
            holdings = []
        seen_groups = set()
        for h in holdings:
            if not isinstance(h, dict) or not isinstance(h.get("ticker"), str) \
                    or not h.get("ticker"):
                continue
            t = h["ticker"]
            # str() coercion: a hand-typo'd non-string manual group must
            # never crash the join downstream (review finding). Multi-lot
            # rows with CONFLICTING groups union deliberately — any group
            # with active money elevates the reminders (R28 aggregates
            # lots first-wins for the CAPS; different question).
            seen_groups.add(str(h.get("group") or group_map.get(t)
                                or f"{t} (ungrouped)"))
        active_groups = sorted(seen_groups)
        print(f"[framework] Active groups (real holdings): "
              f"{', '.join(active_groups) or 'none — flat book'}")
    except Exception as e:
        active_groups = None
        print(f"[framework] active-group resolution unavailable: {e} — "
              f"rules will report active state UNKNOWN, not flat")

    # --- Layer 2.75: Position signal engine (exit / re-entry state machine) ---
    print("[framework] Evaluating position signals...")
    try:
        pos_engine = PositionSignalEngine(config, fetch_data)
        position_signals = pos_engine.compute(
            regime_result, theme_result,
            universe_active=universe_active, group_map=group_map)
        n_pos = len(position_signals.get("tickers", {}))
        n_trans = len(position_signals.get("transitions", []))
        print(f"[framework] Position signals: {n_pos} tickers evaluated, "
              f"{n_trans} state transition(s)")
    except Exception as e:
        # A malformed positions.json or engine fault must not take down the
        # regime/theme/rules layers.
        print(f"[framework] Position signal engine failed: {e}")
        position_signals = {"error": f"position engine failed: {e}",
                            "tickers": {}, "transitions": []}

    # --- Layer 3: Rules ---
    print("[framework] Evaluating rules...")
    rule_engine = RuleEngine(config)
    rules_result = rule_engine.evaluate(regime_result, theme_result,
                                        active_groups=active_groups)
    summary = rules_result["summary"]
    print(f"[framework] Rules: {summary['compliant']} compliant, "
          f"{summary['action_needed']} action needed, {summary['violations']} violations")

    # --- R28: real-dollar portfolio enforcement (PER-508 Phase 0) ---
    # Pure assessment on data already computed this run: positions.json
    # holdings, EOD closes from the position engine's rows, the regime
    # state, capital from config. Never fails the run.
    print("[framework] R28 portfolio enforcement...")
    try:
        from .portfolio_rules import assess_portfolio
        positions_path = os.path.join(BASE_DIR, "state", "positions.json")
        with open(positions_path) as f:
            positions = json.load(f)
        # Group resolution: the SHARED resolver built above (universe
        # selected + ranking audit rows + signals.json fallback) — a holding
        # whose group rotated OUT of the universe still aggregates for the
        # per-group caps. A manual "group" key on the row overrides it.
        ps_rows = (position_signals or {}).get("tickers") or {}
        prices = {t: r.get("close") for t, r in ps_rows.items()
                  if isinstance(r, dict)}
        for h in (positions.get("holdings") or []):
            if isinstance(h, dict) and not h.get("group") \
                    and h.get("ticker") in group_map:
                h["group"] = group_map[h["ticker"]]
        r28_cfg = config.get("r28", {}) or {}
        r28_result = assess_portfolio(
            positions, prices,
            config.get("account_capital_usd"), regime_result["regime"],
            group_max_pct=r28_cfg.get("group_max_pct", 20),
            group_max_positions=r28_cfg.get("group_max_positions", 3))
        if r28_result.get("status") == "ok":
            s = r28_result["summary"]
            print(f"[framework] R28: deployed {r28_result['deployed_pct']}% "
                  f"vs {r28_result['ceiling_pct']:.0f}% ceiling — "
                  f"{s['violations']} violations, {s['action_needed']} "
                  f"action needed, {s['warnings']} warnings")
        else:
            print(f"[framework] R28: {r28_result.get('status')}")
    except Exception as e:
        r28_result = {"status": "error", "message": f"R28 failed: {e}"}
        print(f"[framework] R28 failed (non-fatal): {e}")

    # --- Assemble output ---
    output = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        # Serve-layer shape sentinel (mirrored in ticker_api): derived from
        # the configured regime engine so a flag flip (chassis <-> parliament)
        # forces the committed artifact to regenerate rather than serving the
        # other engine's output as current.
        "schema": artifact_schema(config.get("regime")),
        "framework_version": config.get("framework", {}).get("version", "1.0"),
        "regime": regime_result,
        "themes": theme_result,
        # Layer 2 primary content (D-007 Phase 2): the GICS leadership view;
        # the themes block above is the legacy display-only strip until
        # Phase 3 deletes it
        "universe_leadership": build_universe_leadership(universe_active,
                                                         signals_data),
        "theme_leaders": theme_leaders,
        "position_signals": position_signals,
        "r28": r28_result,
        "rules": rules_result,
        "standing_rules_text": config.get("standing_rules", {}),
    }

    # --- Write output ---
    # NaN/Inf -> null guard on the full payload (same policy as every
    # signals.json write). Deferred import mirrors universe_builder.py and
    # avoids a module-level cycle.
    from signal_engine import sanitize_for_json
    output = sanitize_for_json(output)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Latest
    latest_path = os.path.join(OUTPUT_DIR, "latest.json")
    with open(latest_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"[framework] Written to {latest_path}")

    # Date-stamped archive
    date_str = datetime.date.today().isoformat()
    archive_path = os.path.join(OUTPUT_DIR, f"framework_{date_str}.json")
    with open(archive_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"[framework] Archived to {archive_path}")

    # Also write to public dir for frontend access
    public_dir = os.path.join(BASE_DIR, "..", "public")
    if os.path.isdir(public_dir):
        public_path = os.path.join(public_dir, "framework.json")
        with open(public_path, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"[framework] Written to {public_path}")

    print(f"[framework] Framework run complete.")
    return output


if __name__ == "__main__":
    try:
        result = run_framework(force_fetch=True)
        print(json.dumps(result, indent=2, default=str)[:2000])
    except Exception as e:
        print(f"[framework] ERROR: {e}")
        traceback.print_exc()
