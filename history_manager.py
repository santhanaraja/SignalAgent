#!/usr/bin/env python3
"""
History Manager — Tracks signal changes between runs, saves snapshots,
and generates a change log for the history dashboard.

Detects:
- Signal changes (buy→sell, buy→hold, etc.)
- New tickers added to tracking
- Tickers removed from tracking
- New industry groups added
- Industry groups removed
- Score changes > threshold
- Breaker status changes per group (clear/watch/warning/critical)
- Swing regime transitions (read from public/framework.json)
- Trade signal transitions in/out of the actionable poles (BUY NOW, AVOID)
- Score crossings of the >=50 qualifier line

Single-writer constraint: history.json (and its public mirror) is written
ONLY here. Other event producers get their own file (see
position_signals.emit_history_events / position_events.json).
"""

import json
import math
import os
import shutil
import datetime

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SNAPSHOTS_DIR = os.path.join(DATA_DIR, "snapshots")
PUBLIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "public")
FRAMEWORK_JSON = os.path.join(PUBLIC_DIR, "framework.json")

SCORE_CHANGE_THRESHOLD = 10  # Log score changes > this amount
QUALIFIER_SCORE = 50         # gate line a universe ticker must clear to qualify

# The actionable poles of the trade-signal ladder. Transitions in or out of
# these are logged; churn among the middle states (HOLD POSITION,
# ACCUMULATE ON DIP, WAIT FOR PULLBACK, REDUCE/EXIT) is noise.
TRADE_SIGNAL_POLES = {"BUY NOW", "AVOID"}

# Events are effectively permanent (the backfill reaches to 2026-02); the cap
# is a runaway-growth backstop, not a retention policy. ~50 events/day at the
# current cadence -> this holds roughly a year beyond the backfill.
HISTORY_MAX_CHANGES = 25000

# PER-508 item 6: snapshot retention — keep this many days of full intraday
# snapshots; older dates thin to the weekly archive (last snapshot of each
# Friday). Raw snapshots are re-derivation material only; the EVENTS distilled
# from them live permanently in history.json.
RETENTION_FULL_DAYS = 30


def load_json(path):
    """Load JSON file, return None if not found."""
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


def save_json(path, data):
    """Save data as JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _num(v):
    """A numeric value, or None if missing/NaN. Snapshots that predate the
    pipeline sanitizer carry bare NaN — treat it as absent, never emit it."""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
        return v
    return None


def _breaker_reason(group):
    """Human-readable reason for a group's current breaker status, from the
    triggered checks. Falls back gracefully on old snapshot schemas."""
    triggered = [a for a in (group.get("breaker_alerts") or [])
                 if isinstance(a, dict) and a.get("triggered")]
    parts = []
    for a in triggered:
        msg = a.get("message")
        if not msg or msg == "Not triggered":
            msg = a.get("description") or a.get("check")
        if msg:
            parts.append(str(msg))
    if parts:
        return "; ".join(parts)
    if (group.get("breaker_status") or "").lower() == "clear":
        return "all breaker checks clear"
    return "no triggered checks recorded"


def detect_coverage_events(previous, current):
    """PER-508 item 18: breaker_change, trade_signal_change, gate_crossing.

    Kept separate from detect_changes' hard-indexed walk so the backfill can
    replay the full snapshot archive through the EXACT same logic — every
    field is .get()ed and NaN-tolerant because old snapshots predate several
    schema additions (breaker/trade_signal exist from 2026-02-17).
    """
    events = []
    if not previous or not current:
        return events
    timestamp = current.get("timestamp", datetime.datetime.now().isoformat())

    prev_groups = {g.get("name"): g for g in previous.get("groups", [])
                   if isinstance(g, dict) and g.get("name")}
    curr_groups = {g.get("name"): g for g in current.get("groups", [])
                   if isinstance(g, dict) and g.get("name")}

    # --- breaker_change: any change of a group's breaker status ---
    for name in curr_groups.keys() & prev_groups.keys():
        prev_bs = (prev_groups[name].get("breaker_status") or "").lower() or None
        curr_bs = (curr_groups[name].get("breaker_status") or "").lower() or None
        if not prev_bs or not curr_bs or prev_bs == curr_bs:
            continue
        severity = {"critical": "critical", "warning": "high"}.get(curr_bs, "medium")
        events.append({
            "timestamp": timestamp,
            "type": "breaker_change",
            "severity": severity,
            "group": name,
            "ticker": None,
            "description": f"{name}: Breaker {prev_bs.upper()} → {curr_bs.upper()}",
            "detail": {
                "from_status": prev_bs,
                "to_status": curr_bs,
                "reason": _breaker_reason(curr_groups[name])
            }
        })

    # Keyed by TICKER, not group|ticker: the spec defines trade-signal and
    # gate events per ticker, and group renames/re-clustering (Feb 16-17
    # re-clustering; Phase 2 dynamic-universe cutover pending) must not
    # swallow a transition. First occurrence wins if a ticker somehow
    # appears in two groups.
    def _stocks(groups):
        out = {}
        for name, g in groups.items():
            for s in g.get("stocks", []):
                if isinstance(s, dict) and s.get("ticker") and s["ticker"] not in out:
                    out[s["ticker"]] = (s, g, name)
        return out

    prev_stocks = _stocks(prev_groups)
    curr_stocks = _stocks(curr_groups)

    for ticker in curr_stocks.keys() & prev_stocks.keys():
        prev_s, _, _ = prev_stocks[ticker]
        curr_s, curr_g, group_name = curr_stocks[ticker]

        # --- trade_signal_change: transitions in/out of {BUY NOW, AVOID} ---
        pts, cts = prev_s.get("trade_signal"), curr_s.get("trade_signal")
        if pts and cts and pts != cts and (
                pts in TRADE_SIGNAL_POLES or cts in TRADE_SIGNAL_POLES):
            if pts in TRADE_SIGNAL_POLES and cts in TRADE_SIGNAL_POLES:
                severity = "critical"          # pole-to-pole flip
            elif cts in TRADE_SIGNAL_POLES:
                severity = "high"              # entering a pole
            else:
                severity = "medium"            # leaving a pole to neutral
            events.append({
                "timestamp": timestamp,
                "type": "trade_signal_change",
                "severity": severity,
                "group": group_name,
                "ticker": ticker,
                "description": f"{ticker}: Trade signal {pts} → {cts}",
                "detail": {
                    "from_trade_signal": pts,
                    "to_trade_signal": cts,
                    "score": _num(curr_s.get("score")),
                    "breaker_status": (curr_g.get("breaker_status") or None)
                }
            })

        # --- gate_crossing: score crossing the >=50 qualifier line ---
        ps, cs = _num(prev_s.get("score")), _num(curr_s.get("score"))
        if ps is not None and cs is not None:
            was_in, now_in = ps >= QUALIFIER_SCORE, cs >= QUALIFIER_SCORE
            if was_in != now_in:
                events.append({
                    "timestamp": timestamp,
                    "type": "gate_crossing",
                    "severity": "medium",
                    "group": group_name,
                    "ticker": ticker,
                    "description": (f"{ticker}: Score crossed "
                                    f"{'above' if now_in else 'below'} "
                                    f"{QUALIFIER_SCORE} ({ps} → {cs})"),
                    "detail": {
                        "from_score": ps,
                        "to_score": cs,
                        "direction": "up" if now_in else "down",
                        "signal": curr_s.get("signal"),
                        "trade_signal": curr_s.get("trade_signal")
                    }
                })

    return events


def detect_regime_change(prev_regime, curr_regime, timestamp):
    """PER-508 item 18: swing regime transition, with voter counts.

    prev/curr are {regime, risk_on_count, caution_count, risk_off_count}
    dicts (prev is the last_states entry tracked in history.json; curr comes
    from framework.json). Bootstrap (no previous) records silently — the
    first observation of a regime is not a transition.
    """
    if not isinstance(curr_regime, dict) or not curr_regime.get("regime"):
        return []
    if not isinstance(prev_regime, dict) or not prev_regime.get("regime"):
        return []
    if prev_regime["regime"] == curr_regime["regime"]:
        return []
    to_r = curr_regime["regime"]
    ro = _num(curr_regime.get("risk_on_count"))
    ca = _num(curr_regime.get("caution_count"))
    rf = _num(curr_regime.get("risk_off_count"))
    counts = (f" ({ro}/{ca}/{rf})"
              if ro is not None and ca is not None and rf is not None else "")
    return [{
        "timestamp": timestamp,
        "type": "regime_change",
        "severity": "critical" if to_r == "Risk-off" else "high",
        "group": None,
        "ticker": None,
        "description": f"Swing regime: {prev_regime['regime']} → {to_r}{counts}",
        "detail": {
            "from_regime": prev_regime["regime"],
            "to_regime": to_r,
            "risk_on_count": ro,
            "caution_count": ca,
            "risk_off_count": rf
        }
    }]


def _current_regime_state():
    """The regime block of public/framework.json as a last_states entry, or
    None if the artifact is missing/invalid/mid-write. Never raises — a bad
    framework artifact must not take the signal pipeline down."""
    try:
        with open(FRAMEWORK_JSON, "r") as f:
            fw = json.load(f)
        regime = fw.get("regime")
        if not isinstance(regime, dict) or not regime.get("regime"):
            return None, None
        state = {
            "regime": regime.get("regime"),
            "risk_on_count": regime.get("risk_on_count"),
            "caution_count": regime.get("caution_count"),
            "risk_off_count": regime.get("risk_off_count"),
            "as_of": fw.get("generated_at") or regime.get("date")
        }
        return state, fw.get("generated_at")
    except Exception:
        return None, None


def detect_changes(previous, current):
    """
    Compare previous and current signals.json to detect all changes.
    Returns a list of change events.
    """
    changes = []
    timestamp = current.get("timestamp", datetime.datetime.now().isoformat())

    # Build lookup maps
    prev_groups = {}
    prev_stocks = {}
    if previous:
        for group in previous.get("groups", []):
            prev_groups[group["name"]] = group
            for stock in group.get("stocks", []):
                key = f"{group['name']}|{stock['ticker']}"
                prev_stocks[key] = {**stock, "group_name": group["name"], "group_signal": group["group_signal"]}

    curr_groups = {}
    curr_stocks = {}
    for group in current.get("groups", []):
        curr_groups[group["name"]] = group
        for stock in group.get("stocks", []):
            key = f"{group['name']}|{stock['ticker']}"
            curr_stocks[key] = {**stock, "group_name": group["name"], "group_signal": group["group_signal"]}

    # --- Detect group-level changes ---
    prev_group_names = set(prev_groups.keys())
    curr_group_names = set(curr_groups.keys())

    # New groups added
    for name in curr_group_names - prev_group_names:
        g = curr_groups[name]
        changes.append({
            "timestamp": timestamp,
            "type": "group_added",
            "severity": "high",
            "group": name,
            "ticker": None,
            "description": f"New industry group added: {name}",
            "detail": {
                "group_signal": g["group_signal"],
                "avg_ytd": g["avg_ytd"],
                "stock_count": g["stock_count"]
            }
        })

    # Groups removed
    for name in prev_group_names - curr_group_names:
        g = prev_groups[name]
        changes.append({
            "timestamp": timestamp,
            "type": "group_removed",
            "severity": "high",
            "group": name,
            "ticker": None,
            "description": f"Industry group removed: {name}",
            "detail": {
                "last_signal": g["group_signal"],
                "last_avg_ytd": g["avg_ytd"]
            }
        })

    # Group signal changes
    for name in curr_group_names & prev_group_names:
        curr_g = curr_groups[name]
        prev_g = prev_groups[name]
        if curr_g["group_signal"] != prev_g["group_signal"]:
            changes.append({
                "timestamp": timestamp,
                "type": "group_signal_change",
                "severity": "high",
                "group": name,
                "ticker": None,
                "description": f"{name}: Group signal {prev_g['group_signal'].upper()} → {curr_g['group_signal'].upper()}",
                "detail": {
                    "from_signal": prev_g["group_signal"],
                    "to_signal": curr_g["group_signal"],
                    "from_score": prev_g["avg_score"],
                    "to_score": curr_g["avg_score"]
                }
            })

        # Group rank changes (significant moves)
        if abs(curr_g["rank"] - prev_g["rank"]) >= 2:
            changes.append({
                "timestamp": timestamp,
                "type": "group_rank_change",
                "severity": "medium",
                "group": name,
                "ticker": None,
                "description": f"{name}: Rank #{prev_g['rank']} → #{curr_g['rank']}",
                "detail": {
                    "from_rank": prev_g["rank"],
                    "to_rank": curr_g["rank"]
                }
            })

    # --- Detect stock-level changes ---
    prev_stock_keys = set(prev_stocks.keys())
    curr_stock_keys = set(curr_stocks.keys())

    # New stocks added
    for key in curr_stock_keys - prev_stock_keys:
        s = curr_stocks[key]
        changes.append({
            "timestamp": timestamp,
            "type": "ticker_added",
            "severity": "medium",
            "group": s["group_name"],
            "ticker": s["ticker"],
            "description": f"{s['ticker']} added to {s['group_name']}",
            "detail": {
                "signal": s["signal"],
                "score": s["score"],
                "ytd_return": s["ytd_return"]
            }
        })

    # Stocks removed
    for key in prev_stock_keys - curr_stock_keys:
        s = prev_stocks[key]
        changes.append({
            "timestamp": timestamp,
            "type": "ticker_removed",
            "severity": "medium",
            "group": s["group_name"],
            "ticker": s["ticker"],
            "description": f"{s['ticker']} removed from {s['group_name']}",
            "detail": {
                "last_signal": s["signal"],
                "last_score": s["score"],
                "last_ytd_return": s["ytd_return"]
            }
        })

    # Stock signal changes
    for key in curr_stock_keys & prev_stock_keys:
        curr_s = curr_stocks[key]
        prev_s = prev_stocks[key]

        # Signal change
        if curr_s["signal"] != prev_s["signal"]:
            severity = "high"
            # Downgrade from buy to sell is critical
            buy_signals = {"strong-buy", "buy"}
            sell_signals = {"sell", "strong-sell"}
            if prev_s["signal"] in buy_signals and curr_s["signal"] in sell_signals:
                severity = "critical"
            elif prev_s["signal"] in sell_signals and curr_s["signal"] in buy_signals:
                severity = "critical"

            changes.append({
                "timestamp": timestamp,
                "type": "signal_change",
                "severity": severity,
                "group": curr_s["group_name"],
                "ticker": curr_s["ticker"],
                "description": f"{curr_s['ticker']}: {prev_s['signal'].upper()} → {curr_s['signal'].upper()}",
                "detail": {
                    "from_signal": prev_s["signal"],
                    "to_signal": curr_s["signal"],
                    "from_score": prev_s["score"],
                    "to_score": curr_s["score"],
                    "ytd_return": curr_s["ytd_return"],
                    "rsi": curr_s.get("rsi", None)
                }
            })

        # Significant score change (without signal change)
        elif abs(curr_s["score"] - prev_s["score"]) >= SCORE_CHANGE_THRESHOLD:
            direction = "improved" if curr_s["score"] > prev_s["score"] else "deteriorated"
            changes.append({
                "timestamp": timestamp,
                "type": "score_change",
                "severity": "low",
                "group": curr_s["group_name"],
                "ticker": curr_s["ticker"],
                "description": f"{curr_s['ticker']}: Score {direction} {prev_s['score']} → {curr_s['score']}",
                "detail": {
                    "from_score": prev_s["score"],
                    "to_score": curr_s["score"],
                    "signal": curr_s["signal"]
                }
            })

    # PER-508 item 18: breaker / trade-signal-pole / gate-crossing events
    changes.extend(detect_coverage_events(previous, current))

    # Sort by severity then timestamp
    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    changes.sort(key=lambda x: (severity_order.get(x["severity"], 99), x["timestamp"]))

    return changes


def save_snapshot(current_signals):
    """Save a timestamped snapshot of the full signals state."""
    os.makedirs(SNAPSHOTS_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    snapshot_path = os.path.join(SNAPSHOTS_DIR, f"snapshot_{ts}.json")
    save_json(snapshot_path, current_signals)
    print(f"Snapshot saved: {snapshot_path}")
    return snapshot_path


def prune_snapshots(now=None):
    """PER-508 item 6: enforce snapshot retention on both mirrors.

    Keep every snapshot from the last RETENTION_FULL_DAYS days, plus a
    weekly archive older than that: the LAST snapshot of each Friday (the
    weekly close state — R4/theme review operate on weekly closes). If a
    holiday week has no Friday snapshot, the week's last snapshot stands in
    so no week goes dark. Applied identically to data/snapshots and
    public/snapshots; public/ remains the canonical write-once copy —
    retention only deletes, it never rewrites a surviving file.

    Deletes RAW SNAPSHOT FILES ONLY. history.json events are permanent and
    are never touched here; pruning removes re-derivation material, not the
    derived record.

    Returns (kept, deleted) counts summed across both dirs.
    """
    now = now or datetime.datetime.now()
    cutoff = (now - datetime.timedelta(days=RETENTION_FULL_DAYS)).date()
    public_snapshots = os.path.join(PUBLIC_DIR, "snapshots")
    dirs = [d for d in (SNAPSHOTS_DIR, public_snapshots) if os.path.isdir(d)]

    names = set()
    for d in dirs:
        names.update(f for f in os.listdir(d)
                     if f.startswith("snapshot_") and f.endswith(".json"))

    keep, dated, by_week = set(), {}, {}
    for f in sorted(names):
        try:
            ts = datetime.datetime.strptime(f[9:24], "%Y%m%d_%H%M%S")
        except ValueError:
            keep.add(f)                    # unrecognized name: never delete
            continue
        day = ts.date()
        if day >= cutoff:
            keep.add(f)
        else:
            dated[f] = day
            by_week.setdefault(day.isocalendar()[:2], []).append(f)

    for files in by_week.values():
        files.sort()                       # filename sort == timestamp sort
        fridays = [f for f in files if dated[f].weekday() == 4]
        keep.add(fridays[-1] if fridays else files[-1])

    kept = deleted = 0
    for d in dirs:
        for f in os.listdir(d):
            if not (f.startswith("snapshot_") and f.endswith(".json")):
                continue
            if f in keep:
                kept += 1
            else:
                os.remove(os.path.join(d, f))
                deleted += 1
    return kept, deleted


def get_snapshot_list():
    """Get list of all snapshots sorted by date."""
    os.makedirs(SNAPSHOTS_DIR, exist_ok=True)
    snapshots = []
    for f in sorted(os.listdir(SNAPSHOTS_DIR)):
        if f.startswith("snapshot_") and f.endswith(".json"):
            path = os.path.join(SNAPSHOTS_DIR, f)
            ts_str = f.replace("snapshot_", "").replace(".json", "")
            try:
                # Support both old and new timestamp formats
                try:
                    ts = datetime.datetime.strptime(ts_str, "%Y%m%d_%H%M%S_%f")
                except ValueError:
                    ts = datetime.datetime.strptime(ts_str, "%Y%m%d_%H%M%S")
                data = load_json(path)
                snapshots.append({
                    "filename": f,
                    "timestamp": ts.isoformat(),
                    "path": path,
                    "total_groups": data.get("total_groups", 0) if data else 0,
                    "total_tickers": data.get("total_tickers", 0) if data else 0,
                    "sp500_ytd": data.get("sp500_ytd", 0) if data else 0
                })
            except:
                pass
    return snapshots


def run_history_manager():
    """
    Main entry point.
    1. Load current signals.json
    2. Load previous signals (from last snapshot)
    3. Detect changes
    4. Append to history.json
    5. Save new snapshot
    6. Write history.json to public dir
    """
    print(f"\n{'='*60}")
    print(f"History Manager — {datetime.datetime.now().isoformat()}")
    print(f"{'='*60}\n")

    # Load current signals
    current_path = os.path.join(DATA_DIR, "signals.json")
    current = load_json(current_path)
    if not current:
        print("[ERROR] No signals.json found. Run signal_engine.py first.")
        return

    # Load previous signals (most recent snapshot)
    snapshots = get_snapshot_list()
    previous = None
    if snapshots:
        last_snapshot_path = snapshots[-1]["path"]
        previous = load_json(last_snapshot_path)
        print(f"Previous snapshot: {snapshots[-1]['filename']}")
    else:
        print("No previous snapshot — this is the first run.")

    # Load existing history early — last_states (regime tracking) lives here
    history_path = os.path.join(DATA_DIR, "history.json")
    history = load_json(history_path) or {"changes": [], "snapshots": []}

    # Detect changes
    changes = detect_changes(previous, current)

    # Swing regime transition (PER-508 item 18). The framework artifact is
    # regenerated earlier in the same CI run; compare against the last state
    # this manager recorded. If the artifact is unreadable, keep the old
    # last_states so the transition is caught on the next healthy run.
    curr_regime, fw_generated_at = _current_regime_state()
    if curr_regime:
        prev_regime = (history.get("last_states") or {}).get("regime")
        regime_ts = fw_generated_at or current.get(
            "timestamp", datetime.datetime.now().isoformat())
        changes.extend(detect_regime_change(prev_regime, curr_regime, regime_ts))
        history.setdefault("last_states", {})["regime"] = curr_regime

    print(f"\nDetected {len(changes)} changes:")
    for c in changes:
        icon = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "⚪"}.get(c["severity"], "⚪")
        print(f"  {icon} [{c['type']}] {c['description']}")

    if changes:
        history["changes"].extend(changes)

    # Backstop cap only — see HISTORY_MAX_CHANGES. Events are permanent;
    # snapshot pruning must never touch this list.
    history["changes"] = history["changes"][-HISTORY_MAX_CHANGES:]

    # Save snapshot
    snapshot_path = save_snapshot(current)

    # Enforce retention on every write (PER-508 item 6). Runs before the
    # index rebuild so history["snapshots"] and the public copy loop only
    # ever see survivors.
    kept, deleted = prune_snapshots()
    if deleted:
        print(f"Snapshot retention: kept {kept}, pruned {deleted} "
              f"(>{RETENTION_FULL_DAYS}d non-Friday)")

    # Update snapshot list in history
    history["snapshots"] = get_snapshot_list()

    # Save history
    save_json(history_path, history)
    print(f"History updated: {history_path}")

    # Also copy to public dir
    public_history_path = os.path.join(PUBLIC_DIR, "history.json")
    save_json(public_history_path, history)

    # Copy snapshots to public dir too
    public_snapshots_dir = os.path.join(PUBLIC_DIR, "snapshots")
    os.makedirs(public_snapshots_dir, exist_ok=True)
    for snap in history["snapshots"]:
        src = snap["path"]
        dst = os.path.join(public_snapshots_dir, snap["filename"])
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)

    return changes


if __name__ == "__main__":
    run_history_manager()
