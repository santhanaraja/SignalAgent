#!/usr/bin/env python3
"""
Tests for PER-508 item 6 — snapshot retention in history_manager.

Pins: the 30-day full window (boundary inclusive), last-of-Friday weekly
thinning, holiday-week fallback (a week with no Friday keeps its last
snapshot), identical treatment of both mirrors including public-only
files, refusal to delete unrecognized filenames, and that history.json
is never touched.

Run: python3 test_snapshot_retention.py
"""

import datetime
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import history_manager as hm

NOW = datetime.datetime(2026, 7, 10, 21, 0, 0)   # Friday
CUTOFF = (NOW - datetime.timedelta(days=hm.RETENTION_FULL_DAYS)).date()


def _env():
    tmp = tempfile.mkdtemp(prefix="retention_test_")
    old = (hm.DATA_DIR, hm.SNAPSHOTS_DIR, hm.PUBLIC_DIR)
    hm.DATA_DIR = os.path.join(tmp, "data")
    hm.SNAPSHOTS_DIR = os.path.join(hm.DATA_DIR, "snapshots")
    hm.PUBLIC_DIR = os.path.join(tmp, "public")
    os.makedirs(hm.SNAPSHOTS_DIR)
    os.makedirs(os.path.join(hm.PUBLIC_DIR, "snapshots"))
    return tmp, old


def _mk(name, public_too=True, public_only=False):
    for d, want in ((hm.SNAPSHOTS_DIR, not public_only),
                    (os.path.join(hm.PUBLIC_DIR, "snapshots"),
                     public_too or public_only)):
        if want:
            with open(os.path.join(d, name), "w") as f:
                f.write("{}")


def _listing():
    return (sorted(os.listdir(hm.SNAPSHOTS_DIR)),
            sorted(os.listdir(os.path.join(hm.PUBLIC_DIR, "snapshots"))))


def test_retention_rule():
    tmp, old = _env()
    try:
        # Recent (inside 30d): all kept, including the boundary date itself
        _mk("snapshot_20260709_140000_111111.json")
        _mk(f"snapshot_{CUTOFF.strftime('%Y%m%d')}_150000_222222.json")

        # Old Friday (2026-05-15) with three intraday runs: keep LAST only
        _mk("snapshot_20260515_133000_000001.json")
        _mk("snapshot_20260515_170000_000002.json")
        _mk("snapshot_20260515_202600_000003.json")

        # Old mid-week days die
        _mk("snapshot_20260511_140000_000004.json")   # Monday
        _mk("snapshot_20260513_140000_000005.json")   # Wednesday

        # Holiday week: Good Friday 2026-04-03 has no snapshot — the week's
        # last (Thursday) stands in
        _mk("snapshot_20260330_140000_000006.json")   # Monday
        _mk("snapshot_20260402_140000_000007.json")   # Thursday
        _mk("snapshot_20260402_203000_000008.json")   # Thursday, later

        # Unrecognized name: never deleted
        _mk("snapshot_manual-notes.json")

        # public-only stray old file (mirror drift): pruned there too
        _mk("snapshot_20260512_140000_000009.json", public_only=True)

        kept, deleted = hm.prune_snapshots(now=NOW)
        data_files, public_files = _listing()

        expect = sorted([
            "snapshot_20260709_140000_111111.json",
            f"snapshot_{CUTOFF.strftime('%Y%m%d')}_150000_222222.json",
            "snapshot_20260515_202600_000003.json",   # Friday's last
            "snapshot_20260402_203000_000008.json",   # holiday-week fallback
            "snapshot_manual-notes.json",
        ])
        assert data_files == expect, data_files
        assert public_files == expect, public_files   # 000009 pruned
        assert kept == 10                              # 5 keeper names x 2 dirs
        assert deleted == 13                           # 6 names x 2 dirs + 1 public-only
        print("  retention: 30d window, Friday-last, holiday fallback, "
              "mirrors identical: OK")
    finally:
        (hm.DATA_DIR, hm.SNAPSHOTS_DIR, hm.PUBLIC_DIR) = old
        shutil.rmtree(tmp, ignore_errors=True)


def test_history_json_untouched_and_missing_dirs():
    tmp, old = _env()
    try:
        history_path = os.path.join(hm.DATA_DIR, "history.json")
        payload = {"changes": [{"type": "regime_change", "timestamp": "x"}],
                   "snapshots": []}
        with open(history_path, "w") as f:
            json.dump(payload, f)
        # lone old Monday: its week has no Friday, so the fallback keeps it
        _mk("snapshot_20260511_140000_000001.json")
        kept, deleted = hm.prune_snapshots(now=NOW)
        assert (kept, deleted) == (2, 0)               # kept in both mirrors
        assert json.load(open(history_path)) == payload
        # a missing public dir is tolerated (fresh checkout mid-bootstrap)
        shutil.rmtree(os.path.join(hm.PUBLIC_DIR, "snapshots"))
        kept, deleted = hm.prune_snapshots(now=NOW)
        assert (kept, deleted) == (1, 0)
        print("  history.json untouched; missing mirror dir tolerated: OK")
    finally:
        (hm.DATA_DIR, hm.SNAPSHOTS_DIR, hm.PUBLIC_DIR) = old
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    print("\n=== Snapshot retention tests (PER-508 #6) ===")
    test_retention_rule()
    test_history_json_untouched_and_missing_dirs()
    print("\nAll snapshot retention tests passed.\n")
