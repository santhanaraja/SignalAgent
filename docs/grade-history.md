# Grade history — the committed candidate-grade series

**Created 2026-08-09.** Series: [data/grade_history.json](../data/grade_history.json)
· Writer: `framework/grade_history.py` (one implementation, shared) ·
Backfill/report: `scripts/build_grade_history.py` · Pins:
`test_grade_history.py` (4).

Before this series, grades were the only first-class doctrine output
with no purpose-built history — everything knowable about grade
evolution was an accident of the snapshot system and git retention
(the read-only investigation of 2026-08-09). This makes it deliberate.

## Contract

- One row per calendar day that produced a graded framework artifact:
  `{date, generated_at, source_commit, scorer_version, graded,
  grades{ticker: A+|B|C}}`.
- **Backfilled rows** (2026-07-19 → 2026-08-09) carry the git SHA of
  the day's LAST `public/framework.json` commit and re-derive
  bit-identically from it (pin 1). **Live rows** are appended by
  `framework_runner` after each grading run (D-012's upsert pattern —
  last write of the day wins, rides the cron's `git add data/`) and
  carry `source_commit: null` — the commit that will carry them does
  not exist at write time. The asymmetry is inherent and recorded, not
  papered over.
- `scorer_version` is the artifact's own D-020a era stamp (absent =
  `score_stock_v1-era`, never retro-relabelled). The v1/v2 boundary
  sits between 2026-08-07 and 2026-08-09.
- A run whose grading FAILED writes nothing (an outage never
  impersonates an empty board); a run that graded zero candidates is
  a real observation and is recorded.
- No history_manager events for grades (ruled 2026-08-09): ~50
  candidates churning daily would drown the feed — the same problem
  the scorer_cutover suppression solved. If events come later, they
  are scoped to A+ transitions only.

## The backfilled record (as of 2026-08-09; regenerate with --report)

- 17 days, 2026-07-19 → 2026-08-09 (16 v1-era, 1 v2; Jul 19 and
  Aug 9 are weekend artifacts — D-017's ship day and the D-020a
  cutover bake).
- **A+ spell length: median 1 day.** 17 closed spells — 9 of length
  1, 5 of length 2, 1 of 3, 2 of 4 — plus 8 right-censored (A+ on the
  final day). The one-day median that CR-2026-08-09-2 flagged as a
  timing defect is now measured from committed artifacts.
- **23 distinct names have touched A+** since D-017 shipped.
- Within-era transitions (771 ticker-day pairs): A+ holds 13, decays
  to C 10 times vs B only 2 — **A+ mostly falls off a cliff, not down
  a step** — and is entered from C (8) more often than from B (4).
  Grade churn B↔C dominates (54 + 59).
- The single cross-era pair (Aug 7 v1 → Aug 9 v2, 91 pairs) is
  reported separately by the tool and must never be pooled with
  within-era transitions: it mixes the scorer change with two days of
  tape and a universe rotation.

## The dead archive (recommendation, nothing changed)

`framework/output/framework_YYYY-MM-DD.json` is a dated archive that
accrues nothing: it is gitignored, and on GitHub Actions it writes to
an ephemeral runner filesystem that vanishes per run — the local
copies stop at 2026-07-25 when local runs stopped. **Recommendation:
DELETE the archive write and the five stale local files.** Its
intended purpose is now served three better ways (git history of
`public/framework.json` per bake, this committed series, and the
snapshot system); "making it work" would mean committing dated copies
via cron, which duplicates what git history already stores per
commit. A directory that looks like history and isn't is worse than
no directory. Awaiting the ruling; no change made.
