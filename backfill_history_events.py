#!/usr/bin/env python3
"""
One-shot backfill for PER-508 item 18 — reconstructs the four new history
event types from the existing archives:

- breaker_change / trade_signal_change / gate_crossing: replayed over
  consecutive snapshot pairs (data/snapshots + public/snapshots union)
  through history_manager.detect_coverage_events — the EXACT function the
  live pipeline uses, so backfilled and live events are semantically
  identical.
- regime_change: snapshots never carried regime state, so this type is
  reconstructed from framework/state/regime_history.json instead (EOD
  granularity, one entry per date, earliest 2026-06-12). Intraday regime
  flips before the backfill date are unrecoverable.

Every backfilled event carries detail.backfilled=true. The run is
idempotent: events are deduped on (type, timestamp, group, ticker) against
whatever is already in history.json.

Usage:
    python3 backfill_history_events.py            # dry run (report only)
    python3 backfill_history_events.py --apply    # write history.json + mirror
"""

import argparse
import datetime
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from history_manager import (DATA_DIR, PUBLIC_DIR, SNAPSHOTS_DIR,
                             HISTORY_MAX_CHANGES, detect_coverage_events,
                             detect_regime_change, load_json, save_json)

PUBLIC_SNAPSHOTS_DIR = os.path.join(PUBLIC_DIR, "snapshots")
REGIME_HISTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "framework", "state", "regime_history.json")


def _normalize_ts(ts):
    """Naive ISO timestamps in old snapshots were written by a UTC clock
    (CI) — stamp them +00:00 so the browser doesn't shift them to local."""
    if not isinstance(ts, str) or "T" not in ts:
        return ts
    time_part = ts.split("T", 1)[1]
    if "+" in time_part or "-" in time_part or time_part.endswith("Z"):
        return ts
    return ts + "+00:00"


def _snapshot_files():
    """Union of both snapshot dirs, preferring public/ on filename collision.
    The public copy is write-once (history_manager only copies when the
    destination is missing), so it cannot have been clobbered afterwards —
    unlike data/, where pre-sandbox test runs overwrote at least one real
    snapshot (snapshot_20260216_204905.json gained a 'TEST — New Group')."""
    files = {}
    for d in (SNAPSHOTS_DIR, PUBLIC_SNAPSHOTS_DIR):
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if f.startswith("snapshot_") and f.endswith(".json"):
                files[f] = os.path.join(d, f)
    return [files[f] for f in sorted(files)]


def _load_snapshots(paths):
    """[(sort_key, snapshot_dict)] sorted chronologically; unreadable files
    are skipped and counted."""
    loaded, failed = [], []
    for p in paths:
        try:
            data = json.load(open(p))
        except Exception:
            failed.append(os.path.basename(p))
            continue
        ts = data.get("timestamp") or ""
        # filename fallback: snapshot_YYYYMMDD_HHMMSS[_us].json
        if not ts:
            stem = os.path.basename(p)[9:-5]
            try:
                ts = datetime.datetime.strptime(
                    stem[:15], "%Y%m%d_%H%M%S").isoformat()
            except ValueError:
                failed.append(os.path.basename(p))
                continue
        loaded.append((ts, data))
    loaded.sort(key=lambda x: x[0])
    return loaded, failed


def backfill_snapshot_events():
    paths = _snapshot_files()
    snaps, failed = _load_snapshots(paths)
    events = []
    for (_, prev), (_, curr) in zip(snaps, snaps[1:]):
        events.extend(detect_coverage_events(prev, curr))
    return events, len(snaps), failed


def backfill_regime_events():
    entries = load_json(REGIME_HISTORY) or []
    if not isinstance(entries, list):
        return [], None
    entries = [e for e in entries if isinstance(e, dict) and e.get("regime")]
    entries.sort(key=lambda e: e.get("date") or "")
    events = []
    for prev, curr in zip(entries, entries[1:]):
        ts = curr.get("generated_at") or f"{curr.get('date')}T20:00:00+00:00"
        events.extend(detect_regime_change(prev, curr, ts))
    latest = entries[-1] if entries else None
    return events, latest


def _dedupe_key(e):
    return (e.get("type"), e.get("timestamp"),
            e.get("group") or "", e.get("ticker") or "")


def _regime_day_key(e):
    """Date-level identity for regime_change events. Live events are stamped
    with framework.json's generated_at while the backfill uses
    regime_history.json's — two different clock reads in the same run
    (~seconds apart), so exact-timestamp dedupe can NEVER match a
    live-recorded transition and would double-record it. Same transition on
    the same date == same event."""
    detail = e.get("detail") or {}
    return (detail.get("from_regime"), detail.get("to_regime"),
            (e.get("timestamp") or "")[:10])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="write history.json (+ public mirror); default is dry run")
    args = ap.parse_args()

    snap_events, n_snaps, failed = backfill_snapshot_events()
    regime_events, latest_regime = backfill_regime_events()
    new_events = snap_events + regime_events
    for e in new_events:
        e["timestamp"] = _normalize_ts(e["timestamp"])
        e.setdefault("detail", {})["backfilled"] = True

    history_path = os.path.join(DATA_DIR, "history.json")
    history = load_json(history_path) or {"changes": [], "snapshots": []}
    existing_keys = {_dedupe_key(e) for e in history["changes"]}
    existing_regime_days = {_regime_day_key(e) for e in history["changes"]
                            if e.get("type") == "regime_change"}

    fresh, seen = [], set(existing_keys)
    for e in new_events:
        k = _dedupe_key(e)
        if k in seen:
            continue
        if (e["type"] == "regime_change"
                and _regime_day_key(e) in existing_regime_days):
            continue
        seen.add(k)
        fresh.append(e)

    print(f"snapshots scanned: {n_snaps} (pairs: {max(n_snaps - 1, 0)}, "
          f"unreadable: {len(failed)})")
    for f in failed:
        print(f"  unreadable: {f}")
    print(f"events reconstructed: {len(new_events)} "
          f"({len(new_events) - len(fresh)} already present, "
          f"{len(fresh)} new)\n")

    by_type = {}
    for e in fresh:
        by_type.setdefault(e["type"], []).append(e["timestamp"])
    for t in ("breaker_change", "regime_change", "trade_signal_change",
              "gate_crossing"):
        stamps = sorted(by_type.get(t, []))
        if stamps:
            print(f"  {t:22s} {len(stamps):5d} events   "
                  f"{stamps[0][:16]} → {stamps[-1][:16]}")
        else:
            print(f"  {t:22s}     0 events")

    if not args.apply:
        print("\nDRY RUN — nothing written. Re-run with --apply.")
        return 0

    merged = history["changes"] + fresh
    merged.sort(key=lambda e: e.get("timestamp") or "")
    if len(merged) > HISTORY_MAX_CHANGES:
        print(f"\nWARNING: {len(merged)} events exceeds cap "
              f"{HISTORY_MAX_CHANGES}; oldest will be trimmed")
    history["changes"] = merged[-HISTORY_MAX_CHANGES:]

    # Seed regime tracking so the next live run doesn't re-emit the last
    # backfilled transition. Never overwrite a state the live manager has
    # already recorded (it would be newer than regime_history's EOD entry).
    if latest_regime and "regime" not in (history.get("last_states") or {}):
        history.setdefault("last_states", {})["regime"] = {
            "regime": latest_regime.get("regime"),
            "risk_on_count": latest_regime.get("risk_on_count"),
            "caution_count": latest_regime.get("caution_count"),
            "risk_off_count": latest_regime.get("risk_off_count"),
            "as_of": (latest_regime.get("generated_at")
                      or latest_regime.get("date")),
        }

    save_json(history_path, history)
    save_json(os.path.join(PUBLIC_DIR, "history.json"), history)
    print(f"\nAPPLIED: history.json now holds {len(history['changes'])} events "
          f"(data/ + public/ mirrors written)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
