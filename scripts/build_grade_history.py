#!/usr/bin/env python3
"""Grade-history backfill + report.

Walks `git log -- public/framework.json`, takes the LAST artifact of
each day that carries candidate_grades, and upserts each into
data/grade_history.json through the same writer the forward path uses
(framework/grade_history.py — one implementation of the row format).
The extraction is the committed-artifact replay pattern
test_backtest_doctrine.test_production_grade_parity_overlap
demonstrated: `git show <sha>:public/framework.json`, read the
recorded block, never recompute.

Provenance per backfilled row: source commit SHA, the artifact's own
generated_at, and the scorer version by the D-020a era convention
(absent stamp = "score_stock_v1-era" — never retro-relabelled).

Idempotent: re-running re-derives the same rows and upserts them
(live rows for days the backfill also covers are REPLACED by the
git-anchored row, which gains provenance; the content is the same
artifact).

Run: python3 scripts/build_grade_history.py            # backfill + report
     python3 scripts/build_grade_history.py --report   # report only
"""

import argparse
import collections
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from framework.grade_history import (GRADE_HISTORY_PATH, make_day_entry,
                                     upsert_day)
from signal_engine import scorer_era

SINCE = "2026-07-19"          # D-017 ship day — first graded artifact


def backfill(path=None):
    log = subprocess.run(
        ["git", "log", "--format=%H %cI", "--", "public/framework.json"],
        capture_output=True, text=True, cwd=REPO).stdout.split("\n")
    last_of_day = {}                  # date -> sha (log is newest-first)
    for line in log:
        if not line.strip():
            continue
        sha, ciso = line.split()
        day = ciso[:10]
        last_of_day.setdefault(day, sha)
    written = skipped = 0
    for day in sorted(last_of_day):
        if day < SINCE:
            continue
        sha = last_of_day[day]
        r = subprocess.run(["git", "show", f"{sha}:public/framework.json"],
                           capture_output=True, text=True, cwd=REPO)
        if r.returncode != 0:
            skipped += 1
            continue
        try:
            fw = json.loads(r.stdout.replace(": NaN", ": null"))
        except ValueError:
            skipped += 1
            continue
        cand = fw.get("candidate_grades")
        if not isinstance(cand, dict) or not cand:
            skipped += 1
            continue
        grades = {t: (rec or {}).get("grade") for t, rec in cand.items()
                  if (rec or {}).get("grade")}
        entry = make_day_entry(day, grades, fw.get("generated_at"),
                               scorer_era(fw), source_commit=sha[:7])
        upsert_day(entry, path=path)
        written += 1
    print(f"backfill: {written} days written, {skipped} artifact-days "
          f"without grades skipped")
    return written


def report(path=None):
    doc = json.load(open(path or GRADE_HISTORY_PATH))
    days = doc["days"]
    dates = [d["date"] for d in days]
    print(f"\nseries: {len(days)} days, {dates[0]} -> {dates[-1]}; "
          f"eras: {collections.Counter(d['scorer_version'] for d in days)}")

    # --- A+ spells (consecutive SERIES days; last-day spells are
    # right-censored and reported separately) ---
    by_ticker = collections.defaultdict(dict)
    for d in days:
        for t, g in d["grades"].items():
            by_ticker[t][d["date"]] = g
    spells, open_spells = [], []
    for t, series in by_ticker.items():
        run = 0
        for date in dates:
            g = series.get(date)
            if g == "A+":
                run += 1
            else:
                if run:
                    spells.append((t, run))
                run = 0
        if run:
            open_spells.append((t, run))       # still A+ on the last day
    closed = sorted(l for _, l in spells)
    dist = collections.Counter(closed)
    med = closed[len(closed) // 2] if closed else None
    print(f"\nA+ spells: {len(closed)} closed (median {med}, "
          f"distribution {dict(sorted(dist.items()))}), "
          f"{len(open_spells)} right-censored (open on {dates[-1]}: "
          f"{sorted(open_spells)})")

    ever = sorted({t for t, s in by_ticker.items()
                   if any(g == "A+" for g in s.values())})
    print(f"\ndistinct names ever A+ since D-017: {len(ever)}\n  {ever}")

    # --- transition matrix on consecutive-day pairs, split at the
    # scorer-era boundary (a cross-era transition mixes the scorer
    # change with the tape and must not contaminate either era) ---
    def matrix(pairs):
        m = collections.Counter()
        for a, b in pairs:
            m[(a, b)] += 1
        return m
    within, cross = [], []
    for i in range(1, len(days)):
        prev, cur = days[i - 1], days[i]
        bucket = within if prev["scorer_version"] == cur["scorer_version"] \
            else cross
        for t in set(prev["grades"]) | set(cur["grades"]):
            a = prev["grades"].get(t, "—")
            b = cur["grades"].get(t, "—")
            if (a, b) != ("—", "—"):
                bucket.append((a, b))
    for label, pairs in (("within-era", within),
                        ("CROSS-ERA (v1->v2 boundary pair)", cross)):
        m = matrix(pairs)
        keys = ["A+", "B", "C", "—"]
        print(f"\ntransition matrix, {label} "
              f"({sum(m.values())} ticker-day pairs; '—' = not graded "
              f"that day):")
        print("        to: " + "".join(f"{k:>6}" for k in keys))
        for a in keys:
            print(f"  from {a:>3}: " + "".join(
                f"{m.get((a, b), 0):>6}" for b in keys))
    return doc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true",
                    help="report only, no backfill")
    a = ap.parse_args()
    if not a.report:
        backfill()
    report()


if __name__ == "__main__":
    main()
