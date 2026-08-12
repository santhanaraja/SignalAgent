#!/usr/bin/env python3
"""Run the pin suite and REPORT what it dirtied. Never restores anything.

    python3 scripts/run_pins.py              # whole suite
    python3 scripts/run_pins.py test_pool_builder.py test_grade_view.py

WHY REPORT-ONLY, AND WHY THAT IS THE WHOLE POINT.

This replaces a manual restore ritual — a multi-path `git checkout --` run
after every sweep. That ritual was wrong twice over. It reverted eight to
twelve paths when the suite dirties one, and a wide `git checkout` in a tree
where a concurrent session has legitimate uncommitted work DESTROYS that
work. Automating the restore would have industrialised the second failure,
so this tool deliberately does not restore: it tells you the file and the
path, and you decide with your own eyes on your own tree.

WHAT IT WATCHES, AND WHY TWO MECHANISMS.

`git status` cannot see a write into a gitignored directory, and several
caches that feed pinned studies live in exactly those directories
(data/universe_cache/, data/doctrine_cache/). A test that rewrites one of
those leaves git clean and the next study silently non-reproducing — the
Build 7 defect. So mtimes are swept too.

Each file runs in a FRESH interpreter, because these pins monkeypatch
module globals and a shared process would let one file's patch mask another
file's leak.
"""

import json
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TIMEOUT = int(os.environ.get("RUN_PINS_TIMEOUT", "600"))

# A pin file exits 3 to say NOT-RUN: its INPUT was unavailable, so it
# declined to judge. Distinct from pass (0) and fail (non-zero, non-3), and
# this runner is the consumer that gives that signal somewhere to land —
# without it a skip is terminal text nobody aggregates.
NOT_RUN = 3

# Gitignored trees whose contents feed pinned studies — invisible to git.
WATCHED_IGNORED = ("data/universe_cache", "data/doctrine_cache")

# No allow-list, deliberately. Anything already present at the baseline is
# excluded by the before/after diff, so an allow-list would only ever mask a
# NEW leak — which is the one thing this tool exists to see.

# Where to write the machine-readable report. Defaults to nothing: a guard
# that writes into the repo it audits would dirty the tree it is checking.
REPORT_PATH = os.environ.get("RUN_PINS_REPORT")


def _git(*args):
    return subprocess.run(["git", "-C", REPO] + list(args),
                          capture_output=True, text=True).stdout


def snapshot():
    """(tracked-modified set, untracked set, {watched path: mtime}).

    The mtime map covers the gitignored caches AND every already-modified
    tracked path. Without the second part a file that is dirty when a test
    starts and dirtied AGAIN by that test looks clean, because it never
    enters the modified-set difference — so the SECOND writer of a path
    goes unreported. That is not hypothetical: two files write
    data/earnings_calendar.json, and a set-difference-only version of this
    tool attributed it to the first one alone.
    """
    modified, untracked = set(), set()
    for line in _git("status", "--porcelain").splitlines():
        code, path = line[:2], line[3:].strip('"')
        (untracked if code == "??" else modified).add(path)
    mtimes = {}

    def stamp(rel_path):
        try:
            mtimes[rel_path] = os.stat(os.path.join(REPO, rel_path)).st_mtime
        except OSError:
            pass

    for rel in WATCHED_IGNORED:
        for dirpath, _dirs, files in os.walk(os.path.join(REPO, rel)):
            for f in files:
                stamp(os.path.relpath(os.path.join(dirpath, f), REPO))
    for p in modified:
        stamp(p)
    return modified, untracked, mtimes


def delta(before, after):
    bm, bu, bt = before
    am, au, at = after
    rewritten = {p for p, m in at.items()
                 if p in bm and (p not in bt or bt[p] != m)}
    new_mod = sorted((am - bm) | rewritten)
    new_unt = sorted(au - bu)
    # Genuinely-ignored writes only: anything git can already see is
    # reported once, as MODIFIED, above.
    touched = sorted(p for p, m in at.items()
                     if p not in bm and p not in am
                     and (p not in bt or bt[p] != m))
    gone = sorted(set(bt) - set(at))
    return new_mod, new_unt, touched, gone


def main(argv):
    targets = argv[1:] or sorted(f for f in os.listdir(REPO)
                                 if f.startswith("test_") and f.endswith(".py"))
    print(f"run_pins: {len(targets)} file(s), timeout {TIMEOUT}s per file")
    print(f"repo: {REPO}\n")

    base = snapshot()
    if base[0] or base[1]:
        print(f"NOTE: tree is already dirty ({len(base[0])} modified, "
              f"{len(base[1])} untracked). Reporting only what THIS run adds.\n")

    results, failed, dirty, not_run = [], [], [], []
    for i, t in enumerate(targets, 1):
        before = snapshot()
        t0 = time.time()
        try:
            p = subprocess.run([sys.executable, t], cwd=REPO,
                               capture_output=True, text=True, timeout=TIMEOUT)
            # Exit 3 = NOT-RUN: the pin declined to judge because its INPUT
            # was unavailable (a degraded artifact), not because the code is
            # wrong. It must never read as a pass — a skipped check that
            # looks green is an outage impersonating safety — and never as a
            # failure, which would train people to ignore red.
            status = {0: "pass", NOT_RUN: "NOT-RUN"}.get(p.returncode, "FAIL")
            tail = [] if p.returncode in (0, NOT_RUN) else \
                (p.stderr or "").strip().splitlines()[-1:]
        except subprocess.TimeoutExpired:
            status, tail = "TIMEOUT", []
        secs = round(time.time() - t0, 1)
        mod, unt, touched, gone = delta(before, snapshot())

        if status == "NOT-RUN":
            not_run.append(t)
        elif status != "pass":
            failed.append((t, status, " ".join(tail)))
        if mod or unt or touched or gone:
            dirty.append((t, mod, unt, touched, gone))

        flag = "  " if not (mod or unt or touched or gone) else "**"
        print(f"{flag}[{i:>2}/{len(targets)}] {t:<30} {status:<8} {secs:>6}s"
              + (f"  dirtied={mod + unt}" if mod or unt else "")
              + (f"  ignored-cache={touched}" if touched else "")
              + (f"  DELETED={gone}" if gone else ""))
        results.append({"file": t, "status": status, "seconds": secs,
                        "modified": mod, "untracked": unt,
                        "ignored_touched": touched, "ignored_deleted": gone})

    print("\n" + "=" * 68)
    if dirty:
        print(f"DIRTIED THE TREE — {len(dirty)} file(s). NOT restored, by design:")
        for t, mod, unt, touched, gone in dirty:
            for p in mod:
                print(f"  {t}  ->  MODIFIED   {p}")
            for p in unt:
                print(f"  {t}  ->  CREATED    {p}")
            for p in touched:
                print(f"  {t}  ->  ignored cache written  {p}")
            for p in gone:
                print(f"  {t}  ->  DELETED    {p}")
        print("\nInspect with `git diff -- <path>` and decide yourself. Do NOT "
              "run a wide `git checkout --`: another session may hold "
              "uncommitted work in this tree.")
    else:
        print("CLEAN — no test dirtied a tracked path or a watched cache.")
    if failed:
        print(f"\nFAILED/TIMEOUT — {len(failed)}:")
        for t, s, msg in failed:
            print(f"  {t}: {s} {msg}")
    if not_run:
        print(f"\nNOT-RUN — {len(not_run)} pin file(s) declined to judge "
              f"(degraded input, exit {NOT_RUN}). These are NOT passes:")
        for t in not_run:
            print(f"  {t}")

    if REPORT_PATH:
        try:
            with open(REPORT_PATH, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nreport: {REPORT_PATH}")
        except OSError as e:
            print(f"\nreport not written ({e})")
    # 1 = something is wrong (dirt or a real failure). 3 = nothing wrong,
    # but at least one pin could not judge — visible to CI without being
    # conflated with red. 0 = everything ran and everything was clean.
    if dirty or failed:
        return 1
    return NOT_RUN if not_run else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
