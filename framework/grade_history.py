"""Grade history — the committed per-ticker daily candidate-grade series.

Before this module existed, grades were the only first-class doctrine
output with no purpose-built history: everything knowable about grade
evolution was an accident of the snapshot system and git retention.
This series makes it deliberate. One implementation of the row format,
shared by the backfill (scripts/build_grade_history.py, which walks
git history) and the forward writer (framework_runner calls
append_daily_grade_history after each grading run) — the D-012
fear-greed pattern: upsert per day, last write wins, rides the daily
data commit (the workflow's `git add data/` glob).

Row provenance: `source_commit` carries the git SHA for backfilled
rows and null for live-appended rows (the commit that will carry a
live row does not exist at write time — the asymmetry is inherent,
recorded here rather than papered over). `scorer_version` is the
artifact's own era stamp via the D-020a convention: absent stamp =
"score_stock_v1-era", never retro-relabelled.
"""

import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GRADE_HISTORY_PATH = os.path.join(BASE_DIR, "..", "data",
                                  "grade_history.json")

SCHEMA = "grade-history-1"

GRADES = ("A+", "B", "C")          # grade_setup's complete ladder


def make_day_entry(date, grades, generated_at, scorer_version,
                   source_commit=None):
    """One day's row. `grades` is {ticker: grade-letter} exactly as the
    framework artifact's candidate_grades carries them."""
    return {
        "date": date,
        "generated_at": generated_at,
        "source_commit": source_commit,
        "scorer_version": scorer_version,
        "graded": len(grades),
        "grades": dict(sorted(grades.items())),
    }


def upsert_day(entry, path=None):
    """Insert or replace the entry for its date; other days untouched;
    days kept sorted. Returns the action taken."""
    path = path or GRADE_HISTORY_PATH
    doc = {"schema": SCHEMA, "days": []}
    if os.path.exists(path):
        try:
            with open(path) as f:
                loaded = json.load(f)
            if isinstance(loaded, dict) and isinstance(loaded.get("days"),
                                                       list):
                doc = loaded
        except (OSError, ValueError):
            pass                      # unreadable file: rebuilt from scratch
    days = [d for d in doc["days"] if d.get("date") != entry["date"]]
    action = "replaced" if len(days) != len(doc["days"]) else "appended"
    days.append(entry)
    days.sort(key=lambda d: d["date"])
    doc["schema"] = SCHEMA
    doc["days"] = days
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(doc, f, indent=1)
    os.replace(tmp, path)
    return action


def append_daily_grade_history(candidate_grades, generated_at,
                               scorer_version, today=None, path=None):
    """The forward writer (one call from framework_runner per grading
    run). A run whose grading FAILED (candidate_grades is None) writes
    nothing — an outage never impersonates an empty board; a run that
    graded zero candidates (empty dict) is a real observation and is
    recorded."""
    if not isinstance(candidate_grades, dict):
        return None
    import datetime
    today = today or datetime.date.today().isoformat()
    grades = {t: (r or {}).get("grade") for t, r in
              candidate_grades.items() if (r or {}).get("grade")}
    entry = make_day_entry(today, grades, generated_at, scorer_version,
                           source_commit=None)
    return upsert_day(entry, path=path)
