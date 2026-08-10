#!/usr/bin/env python3
"""
Grade-history pins.

Per the pin doctrine (docs/testing.md). The load-bearing pin is
BACKFILL REPRODUCIBILITY: every committed row that claims git
provenance must re-derive bit-identically from its source commit —
the series is only trustworthy if it is exactly what the artifacts
said.

Run: python3 test_grade_history.py
"""

import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(os.path.abspath(__file__))

from framework.grade_history import (GRADE_HISTORY_PATH, GRADES, SCHEMA,
                                     append_daily_grade_history,
                                     make_day_entry, upsert_day)
from signal_engine import scorer_era


def test_backfill_reproducibility():
    """Every committed row with a source_commit re-derives exactly from
    `git show <sha>:public/framework.json` — same grades, same
    generated_at, same era. Drives the same extraction the backfill
    ran; a hand-edited or drifted series fails here."""
    doc = json.load(open(GRADE_HISTORY_PATH))
    assert doc["schema"] == SCHEMA
    checked = 0
    for row in doc["days"]:
        sha = row.get("source_commit")
        if not sha:
            continue                      # live-appended rows have no SHA
        r = subprocess.run(["git", "show", f"{sha}:public/framework.json"],
                           capture_output=True, text=True, cwd=REPO)
        assert r.returncode == 0, (row["date"], sha)
        fw = json.loads(r.stdout.replace(": NaN", ": null"))
        cand = fw.get("candidate_grades") or {}
        grades = {t: (rec or {}).get("grade") for t, rec in cand.items()
                  if (rec or {}).get("grade")}
        assert row["grades"] == dict(sorted(grades.items())), row["date"]
        assert row["generated_at"] == fw.get("generated_at"), row["date"]
        assert row["scorer_version"] == scorer_era(fw), row["date"]
        assert row["graded"] == len(grades)
        checked += 1
    assert checked >= 17, f"only {checked} provenanced rows"
    # the era boundary must be visible: both eras present
    eras = {r["scorer_version"] for r in doc["days"]}
    assert "score_stock_v1-era" in eras and "score_stock_v2" in eras, eras
    print(f"  (1) backfill reproducibility: {checked} rows re-derived "
          f"bit-identically from their source commits; both scorer eras "
          f"visible: OK")


def test_upsert_semantics():
    """D-012 pattern: same-day write replaces, other days untouched,
    days stay sorted; an unreadable file rebuilds rather than crashes."""
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "gh.json")
        e1 = make_day_entry("2026-08-01", {"AAA": "A+"}, "t1", "v", "abc")
        e2 = make_day_entry("2026-08-03", {"BBB": "C"}, "t2", "v", "def")
        assert upsert_day(e1, path=p) == "appended"
        assert upsert_day(e2, path=p) == "appended"
        e2b = make_day_entry("2026-08-03", {"BBB": "B"}, "t3", "v", "ghi")
        assert upsert_day(e2b, path=p) == "replaced"
        # out-of-order insert lands sorted, neighbours untouched
        e0 = make_day_entry("2026-08-02", {"CCC": "C"}, "t4", "v", None)
        upsert_day(e0, path=p)
        doc = json.load(open(p))
        assert [d["date"] for d in doc["days"]] == \
            ["2026-08-01", "2026-08-02", "2026-08-03"]
        assert doc["days"][0]["grades"] == {"AAA": "A+"}
        assert doc["days"][2]["grades"] == {"BBB": "B"}
        with open(p, "w") as f:
            f.write("{corrupt")
        upsert_day(e1, path=p)
        assert json.load(open(p))["days"][0]["grades"] == {"AAA": "A+"}
    print("  (2) upsert: append/replace/sort, neighbours untouched, "
          "corrupt file rebuilds: OK")


def test_forward_writer():
    """The framework hook's contract: a FAILED grading run (None)
    writes nothing — an outage never impersonates an empty board; a
    graded-empty run IS recorded; grades land as letters with null
    source_commit and the artifact's era."""
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "gh.json")
        assert append_daily_grade_history(None, "t", "score_stock_v2",
                                          today="2026-08-10", path=p) is None
        assert not os.path.exists(p)
        cand = {"HPE": {"grade": "A+", "reasons": ""},
                "MU": {"grade": "C", "reasons": "x"},
                "BAD": {"grade": None}}
        act = append_daily_grade_history(cand, "ts1", "score_stock_v2",
                                         today="2026-08-10", path=p)
        assert act == "appended"
        row = json.load(open(p))["days"][0]
        assert row["grades"] == {"HPE": "A+", "MU": "C"}     # BAD dropped
        assert row["source_commit"] is None
        assert row["scorer_version"] == "score_stock_v2"
        assert all(g in GRADES for g in row["grades"].values())
        act2 = append_daily_grade_history({}, "ts2", "score_stock_v2",
                                          today="2026-08-11", path=p)
        assert act2 == "appended"
        doc = json.load(open(p))
        assert doc["days"][1]["graded"] == 0                 # real zero-day
    print("  (3) forward writer: failed run writes nothing, zero-grade "
          "day recorded, letters only, null SHA, era carried: OK")


def test_one_implementation():
    """Backfill and forward rows share make_day_entry — identical
    inputs, identical row (modulo provenance), so the series never
    forks into two dialects."""
    g = {"HPE": "A+", "MU": "C"}
    a = make_day_entry("2026-08-10", g, "ts", "score_stock_v2", "abc1234")
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "gh.json")
        append_daily_grade_history(
            {t: {"grade": v} for t, v in g.items()},
            "ts", "score_stock_v2", today="2026-08-10", path=p)
        b = json.load(open(p))["days"][0]
    a["source_commit"] = None
    assert a == b, (a, b)
    print("  (4) one implementation: backfill row == forward row for "
          "identical inputs (modulo provenance): OK")


if __name__ == "__main__":
    test_backfill_reproducibility()
    test_upsert_semantics()
    test_forward_writer()
    test_one_implementation()
    print("All grade-history pins passed.")
