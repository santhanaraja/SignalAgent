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
"""

import json
import os
import shutil
import datetime

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SNAPSHOTS_DIR = os.path.join(DATA_DIR, "snapshots")
PUBLIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "public")

SCORE_CHANGE_THRESHOLD = 10  # Log score changes > this amount


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

    # Detect changes
    changes = detect_changes(previous, current)
    print(f"\nDetected {len(changes)} changes:")
    for c in changes:
        icon = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "⚪"}.get(c["severity"], "⚪")
        print(f"  {icon} [{c['type']}] {c['description']}")

    # Load existing history and append
    history_path = os.path.join(DATA_DIR, "history.json")
    history = load_json(history_path) or {"changes": [], "snapshots": []}

    if changes:
        history["changes"].extend(changes)

    # Keep last 1000 changes max
    history["changes"] = history["changes"][-1000:]

    # Save snapshot
    snapshot_path = save_snapshot(current)

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
