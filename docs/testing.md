# Testing — the pin doctrine

This system's tests are called **pins**. The name is the point: a pin
holds a specific behaviour in place so it cannot drift, and every pin
names the failure it exists to prevent. A test suite that only proves
"the code runs" is not a pin set.

The doctrine is three laws. They were not written in the abstract —
each one is a lesson a real defect taught, and the citations are
findings from this repo's own history.

## Law 1 — Drive the real entry point

A pin exercises the function production calls, through the path
production takes. Reimplementing the logic inside the test proves only
that you can write the same bug twice.

> **Where this was learned:** D-018's first close-replay pin asserted an
> identity against a reimplemented engine inside the test file. Both
> sides agreed, the pin was green, and it was *inert* — it could not
> have caught a change to the real `compute()`. It was rewritten to
> drive the actual engine.

Corollaries:

- Prefer a real artifact, a real `test_client()`, a real module import
  over a hand-built stand-in.
- When you must inject (an outage, a clock, a data dir), inject at the
  boundary — patch the source, not the logic under test.
- If the pin needs the production entry point to change shape before it
  can be driven, change the shape. That pressure is the design telling
  you something.

## Law 2 — Assert the invariant, not the implementation

Pin the property that must hold for the system to be correct. Do not
pin the current spelling of the code that happens to deliver it. An
implementation pin passes a refactor that reintroduces the bug, and
fails a refactor that keeps the behaviour — exactly backwards.

> **Where this was learned:** the swallowed-fetch remediation shipped a
> pin asserting that the gate check appeared *earlier in the file* than
> the chip render (`i_gate < i_chip`). It was a textual proxy: any
> refactor that rendered the chip anyway would sail through. It was
> replaced with a pin that **executes the page's own gate predicate**
> over five payload shapes that must withhold and two that must not.

Corollaries:

- Structural over count-based. `count(flag) == 2` cemented an
  incomplete remediation (7 of 9 fallbacks were unflagged and still
  rendered as live readings); "every fallback of this shape carries the
  flag" could not.
- Beware the vacuous assertion. `assert X not in slice` passes
  trivially when `slice` is empty — assert the slice is non-empty
  *first*.
- Name the invariant in the pin's docstring in plain language. If you
  cannot state it in a sentence, you are probably pinning an
  implementation.

## Law 3 — Demonstrate the failure

**A pin that cannot fail on the old code is not evidence.** Before a
pin counts, show it going red against the defect — by running the
pre-fix code, by injecting the outage, by constructing the input that
broke things.

> **Where this was learned:** D-019's breaker-coverage pin loads the
> pre-fix engine, runs it against the *same* macro outage, and asserts it
> reports a clean `clear` with no trace of the check that never ran —
> then asserts the fixed engine reports `degraded`. The pin holds both
> halves of the before/after, so the evidence cannot rot.

**Anchor the "before" to a COMMIT, never to `HEAD`.** The first draft of
that pin read `git show HEAD:signal_engine.py`. It was green, and it was
a trap: the moment the fix committed, `HEAD` *was* the fixed code, the
pin's own guard would see the fix present, print "skipped", and pass
forever. The demonstration would evaporate at exactly the moment it
started costing something to keep. It now names the pre-fix SHA
explicitly, and an unreachable anchor **raises** rather than printing a
green line — a pin that silently does not run is worse than one that
fails, because the suite still reads all-clear.

**The SLIDING-WINDOW pin — same family, slower fuse.** A pin whose
anchor MOVES relative to its data will eventually stop testing what it
was written to test. `test_backtest_doctrine`'s honesty bridge anchored
on the *newest 20 commits* of `public/framework.json` and then filtered
to days the doctrine price cache covers. Green for weeks — until daily
bakes accumulated and the window slid entirely past the cache's
coverage: on 2026-08-10 the pin failed on clean `HEAD` with zero
replayable days, through no defect in anything it guards. The
HEAD-anchored trap dies the moment the fix lands; the sliding window
dies slowly, on a schedule set by how fast the sequence grows — and it
can fail EITHER way: red on healthy code (this instance), or worse,
quietly testing an ever-thinner slice until it tests nothing. The
lesson: **anchor a pin to a fixed commit or a content hash, never to a
position in a growing sequence.** Select the data that satisfies the
pin's intent first (here: cache-covered days), then cap for runtime —
never cap first and hope the intent survives the cap.

Corollaries:

- Prefer injection to mocking: kill the input, don't stub the answer.
- Pin the *negative* too — that the fix does not over-fire. A coverage
  pin must prove a fully-covered day still reads `clear`, or it has
  traded a false-negative for a false-positive.
- When a defect had a specific historical instance (a date, an
  artifact, a reported number), reproduce that instance by name.

## Practical conventions

- Pins are script-style: a `__main__` block that calls each test and
  prints `"  <what was pinned>: OK"`. Run with `python3 test_x.py`.
  There is no pytest in this environment.
- One file per subject area; the full sweep is every `test_*.py`.
- **Read committed artifacts with `git show HEAD:path`, never the
  working tree** — a sweep regenerates artifacts, and a replay pin that
  reads the working tree becomes order-dependent. (Both position pins
  hit this; both were converted.)
- Restore any artifact a test mutates before finishing.
- Era-aware pins: when a record gains fields, pin BOTH the new shape and
  the old one rendering as it always did. A fix must never retro-claim
  about data that was never measured.

## Related

- [D-006](decisions/D-006-build4-protocol.md) — the backtest protocol
  (lookahead pins are tests, not vibes).
- [D-019](decisions/D-019-breaker-coverage.md) — coverage, not outcome;
  the record whose pin set demonstrates Law 3.
