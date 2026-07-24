# D-018 — The close-basis law extends to the position ladder

| | |
|---|---|
| **ID** | D-018 |
| **Date** | 2026-07-23 |
| **Status** | Ruled — implemented 2026-07-23; **amended 2026-07-24** (revision discriminator) |

## The law

**Position-state transitions occur ONLY on confirmed close-basis
evaluations.** Intraday runs render the committed close-basis states with at
most a labeled provisional read of the forming bar. They never transition,
never move a counter, and never emit a transition event.

## Context — the live evidence

The Phase 3 adversarial review's one deliberately-deferred major item: the
regime hysteresis got the close-basis law in commit `4bdf2a5`, but the
position ladder still evaluated forming bars. Two recorded incidents, four
days apart, show both failure modes:

**2026-07-21 — an exit that never happened.** `data/position_events.json`:

| Time (ET) | Bar | Event |
|---|---|---|
| 13:59 | forming 190.96 vs SMA20 190.97 | `HELD → EXIT_FIRED` |
| 15:40 | forming 192.04 | `EXIT_FIRED → HELD` |
| 16:00 | **confirmed close 191.15** | *(above SMA20 — no exit)* |

One cent of intraday wick manufactured a stop-firing, announced it, and took
it back 100 minutes later. R11 says decisions are made on the close; the
ladder was making them on whatever bar happened to be forming.

**2026-07-22 — a real exit mis-recorded (reported only by aggregation).**
The same mechanism running the other way:

| Time (ET) | Bar | Run outcome |
|---|---|---|
| 11:02 | forming 187.51 | `HELD → EXIT_FIRED` logged |
| 12:45 | forming 187.63 | `EXIT_FIRED → WATCHING` logged |
| 16:32 | **confirmed close 188.42** vs SMA20 191.89 | **0 transitions** — the ladder was already in `WATCHING`, so the close run detected nothing |

The confirmed close was genuinely below SMA20 — a **true** stop firing. The
intraday path had already consumed the transition and demoted it, so the
close run itself computed `WATCHING → WATCHING` and produced no event.

**The exit still reached the operator — but by aggregation, not by design.**
`_assessment_changes` collects *every* event in `position_events.json`
stamped with the current run's date, whatever run wrote it, so the 16:32
close report's changes line carried both intraday transitions. What it
carried was wrong in substance: an exit stamped 11:02 ET at 187.51 — a price
that was never a close — immediately followed by its own reversal, so the
day's genuine stop-firing read as intraday noise. The operator saw a flap;
the tape had delivered an exit. (The final *state*, `WATCHING`, happened to
be where a true exit-and-stay-below day belongs — right by accident, wrong by
provenance.)

Under D-018 the same day yields one event: `HELD → EXIT_FIRED` at the close,
at 188.42, correctly stamped.

> Correction, 2026-07-24: this record and the implementing commit
> (`6680efe`) originally read "the close run saw `WATCHING → WATCHING` and
> announced nothing — the one exit that mattered went unreported." The first
> half is confirmed (that run emitted zero transitions); the second half was
> wrong — the changes line did surface the intraday-logged events. The
> failure is provenance and labeling, not silence. Verified against the
> committed Jul-22 artifacts and `_assessment_changes`.

## What changes

- The state machine (`WATCHING` / `RE_ENTRY_ARMING` / `RE_ENTRY_READY` /
  `HELD` / `EXIT_FIRED` and the consecutive-close counters) advances only on
  confirmed closes, using the **same `confirmed_close_frame` and the same
  16:10 ET settle boundary** the regime hysteresis uses (D-008). One law,
  one notion of "settled" — the splitter is imported, never re-implemented.
- Intraday regenerations re-render the committed close from the recorded
  `prev_before` seed (the chassis's `carry_pre_final` precedent — a
  re-render must reproduce, never re-step). **A re-render never moves the
  ladder.** The verdict depends on inputs that are not the bar — the
  regime (c3), the universe (c5), config — so letting a differing verdict
  through would transition intraday off a regime flip, and could walk a
  `HELD` holding out of `HELD` without ever firing `EXIT_FIRED`,
  disarming its stop. Disagreements are reported on the row
  (`render_note`) and settled by the next confirmed close. Rows also
  attach
  `intraday_preview: {as_of, would_state, close, sma20, conditions_met,
  note}` for the forming bar. Display only.
- Grades and the ⛔ gate key off close-basis state, on every path. **The
  D-017 reconciliation:** stripping the *synthetic* bar was never the whole
  law — a live-quote row (Volume 0) went, but the day's forming bar has real
  volume and stayed, so intraday candidate grades were forming-bar-based.
  Both grade paths now apply the same splitter.
- **Data-unavailable carries the committed state** — never transitions,
  never defaults.

## The gap-proofness question (ruled before implementing)

**Are the counters incremental or replay-based? Replay — and it is
mandatory, not stylistic.**

The ladder's counters were *already* stateless: `consec_closes_above`
rescans the window every run. But the **state** is seeded from persistence,
so a missed confirmed close silently swallows its transition. The material
case is the one that carries protective urgency: a holding whose close
pierces its SMA20 on the missed day and recovers the next. Incrementally the
next run sees price back above and reports `HELD` — the stop that fired is
never announced. That is the Jul-22 failure again, reached by a different
road, and the CI framework step is `continue-on-error`, so a missed close
run is not hypothetical.

So `pending_closes` steps **every unevaluated confirmed close, in order**,
from the persisted state. Two deliberate differences from the regime
ladder's fixed-window fresh-seed replay:

1. **The seed is persistence, not a fresh state.** This ladder is *not*
   seed-independent — after `EXIT_FIRED` a holding needs all five conditions
   to return to `HELD`, so a fresh seed would fabricate `HELD`. Persistence
   is authoritative; replay steps forward from it.
2. **Caught-up events keep wall-clock timestamps** and name the bar they came
   from (`detail.bar_date`, `detail.caught_up`, and the description).
   Back-dating them read tidier but hid them from the "what changed" surface
   and the close report — and the entire point of gap-proofness is that a
   missed exit gets *announced*. The normal daily path is unchanged.

**Cutover:** a pre-D-018 record has a state but no `last_close_date`.
Treating that as first sight would re-step a close the old engine already
stepped — and the ladder is not idempotent at `EXIT_FIRED`, so the first run
(an intraday one, most likely) would consume a live exit and write a false
event into the permanent audit log. The migration therefore *adopts* the
record's current close and stamps the bookkeeping, without moving the ladder
or emitting anything.

Bounded at `MAX_CATCHUP_BARS = 10` (two trading weeks); a longer gap resyncs
to the last close and says so on the row (`catchup_truncated`) — an honest
resync beats a flood of stale events.

## Amendment — the revision discriminator (2026-07-24)

As first shipped (`6680efe`), the re-render branch suppressed **every**
verdict disagreement between the recomputed state and the committed state.
That was correct for a non-bar input moving intraday — a regime flip or a
universe drop must never transition a position (critical-#3) — but it also
swallowed a *genuine provider revision of the confirmed close*. A provider
adjusting a settled close **down through the SMA20** would recompute to
`EXIT_FIRED`, the branch would revert it to the committed `HELD`, and the
stop would never fire. The corollary this section originally stated —
"revised data does not move the ladder" — was that bug written down as a
rule. The ruling (PER-508 comment 11726 follow-up) overturns it.

**The two causes are separable by the bar data itself**, and that is the
discriminator: on any re-render whose verdict disagrees with the committed
state, compare the last confirmed bar's **fingerprint** — its OHLC — to the
one persisted with the state.

- **Fingerprint changed → a genuine revision.** Re-step *that* bar (from
  `prev_before`, so the re-step is the seed-entering-the-bar applied to the
  revised data) and emit the transition **loudly**: wall-clock stamped,
  `severity: critical`, both old and new OHLC in `detail.revision`, a
  distinct `detail.revised` flag (not `caught_up`, not a plain close), and
  the old→new close named in the description.
- **Fingerprint identical → a non-bar input moved.** HOLD (the shipped
  behavior, now correctly *scoped* to this case): `render_note` set, no
  event, next confirmed close decides. Critical-#3 is preserved.

**Why full OHLC, not just the close:** the verdict also depends on the high
and low through ATR (the confirmation branch's `atr_break` and the extension
guard), so a revision to any of the four can move the state and all four
must be watched. Rounded to 4 places — a revision that matters is cents;
4dp erases float / JSON round-trip noise (verified stable across re-fetches),
and a NaN/Inf bar fingerprints as `None` (routes to HOLD, never a phantom
from `nan != nan`).

**Corporate actions are not revisions.** The production fetcher runs
`auto_adjust=True`, which multiplicatively re-bases *every* historical bar
on a split/dividend ex-date — so the last-bar OHLC changes without any
provider revising anything, and the change is **scale-invariant to the whole
ladder** (close-vs-SMA20, the consecutive count, the slope, and the
ATR-normalized extension all divide out the common factor). A re-basing is
told from a revision by `uniform_rescale`: if the four new/old OHLC ratios
share one positive factor (tight tolerance, biased toward *emitting* — a
missed exit is worse than a spurious event), it is a re-basing → HOLD +
refresh the fingerprint, never a revision. And a verdict-preserving change
(a split with no coincident move, or a sub-threshold correction) **refreshes
the stored fingerprint on the quiet path**, so it can never leave a stale
fingerprint that a *later* non-bar move would compare against and
phantom-fire.

**The re-step seed is the ENTERING seed, not the pre-revision state.** A
revision re-steps the revised bar from the seed that *entered* it
(`prev_before`), and persists *that* as the new record's `prev_before` — not
the state the revision superseded. Persisting the superseded state instead
drifts the seed across successive revisions of one bar (down→up→down), and
once it lands on a reclaim-cycle state a below-stop re-step yields `WATCHING`
instead of `EXIT_FIRED` — a disarmed stop written to the audit log as a
critical event. Pinned by the oscillation case.

**BOUND (ruled explicitly): only the LAST confirmed bar is fingerprinted
and re-steppable.** A revision arriving for an *older* bar folds into the
trailing window like any other data — it moves the last bar's verdict only
through the SMA20/ATR recompute, leaving the last bar's own fingerprint
identical, so it routes to the HOLD branch and the next close decides. It
never reaches back to re-step history. (Verified: the older-bar case in
pin (a) stays silent.)

**Cutover for the discriminator itself:** records written before this
amendment carry no fingerprint. On the first run one is stamped as
bookkeeping (no state change, no event) — the same shape as the
`last_close_date` migration — so a `null` fingerprint reads as "cannot yet
discriminate → HOLD", never as a spurious revision. A revision that happens
to coincide with that one stamping run is folded silently; a one-time
cutover cost, ruled acceptable, identical in kind to the D-018 cutover.

## Scope fence — what does NOT change

- **The intraday ALERT layer.** Proximity warnings ("approaching stop ·
  close decides, no pre-emption") are the intraday runs' entire purpose.
  Alerts compare live price against the **committed** close-basis stop read
  from the artifact; the alert path holds no price history and cannot
  recompute an SMA. Warning is not transitioning. Pinned.
- Stop levels, exit decision semantics, grading logic, entry doctrine.
- **[D-009](D-009-exit-timing-1230.md) (intraday execution) stays open and
  untouched.**

## Pins (the license)

| Pin | What it proves |
|---|---|
| Forming-bar-cannot-transition | The Jul-21 flap reproduced: the un-split frame **provably** fires `EXIT_FIRED`; the split frame holds `HELD`. Jul-22's real close still fires. |
| Close-replay identity | 3 price paths × holding/watching, 6 distinct states: state sequences identical at every confirmed close, before vs after. |
| Artifact re-render | HPQ + all 9 tracked names re-render to their recorded states from the committed artifact. |
| Gap-proofness | A missed close is caught up in order (the full chain, with correct from-states), the swallowed exit **is** announced and named to its bar; long gaps resync with a flag. |
| Alert-emission parity | The same artifact row + price path emits byte-identical alerts across the whole tier ladder; the stop's provenance is the artifact (source-pinned against recompute). |
| Intraday quiet path | Committed state re-rendered, **zero** events, labeled preview, data-outage carries (and says it is carried). |
| Re-render law — pin (b) | A regime flip and a universe drop on a **byte-identical** last bar leave the ladder still (`render_note`, no event, no revision) — critical-#3 preserved. |
| Revision discriminator — pin (a) | A revised last close **through the SMA20 emits** `EXIT_FIRED` — old→new OHLC, `revised` marker, `critical`; oscillating revisions keep the seed stable (down→up→down → `EXIT_FIRED`, never `WATCHING`). |
| The bound | An in-window older-bar revision that shifts the SMA20 past the unchanged last close **HOLDS** (fingerprint identical) — history is never re-stepped. |
| Corporate-action guard | A uniform split/dividend re-basing is never a revision: alone it refreshes the fingerprint (no later phantom); coincident with a non-bar move it HOLDS. NaN → HOLD; round(4) noise/JSON stable. |
| Aggregation contract — pin (c) | A caught-up **or** revised exit (wall-clock stamped, naming its bar) survives `_assessment_changes`' `startswith(run-date)` filter and reads *as such* in the close report — the once-undesigned load-bearing path, now asserted. |
| Cutover | A pre-D-018 record adopts its close instead of re-stepping it: no consumed exit, no false event, bookkeeping stamped. |
| Grade reconciliation | `compute_grade_inputs` splits the forming bar — intraday grade inputs equal the confirmed close's, and step post-close. |

## Revisit triggers

1. **[D-009](D-009-exit-timing-1230.md) ruling on intraday execution —
   mandatory revisit.** If intraday execution is ever ruled in, this law is
   revisited as part of that ruling.
2. Any live incident where a close-only transition demonstrably delayed a
   protective action the operator needed intraday — that would reopen the
   alert-vs-transition boundary.

## Source

Adversarially verified (5-dimension fan-out, 42 agents: 37 findings, 17
confirmed and fixed — including three critical ones that reshaped the
re-render branch and added the cutover migration; 20 refuted).

The amendment was adversarially verified (4-dimension fan-out, 20 agents:
16 findings, 8 confirmed and fixed — the two serious ones a re-step seed
that drifted across oscillating revisions and an `auto_adjust` corporate-
action re-basing that would phantom-fire; 8 refuted).

PER-508 comment 11726 (2026-07-23, night). Ruled by the user in those terms;
a transcription of an existing law onto a second ladder, reusing the
machinery Phase 3 built. Related: [D-003](D-003-1b-position-engine.md) (the
ladder), [D-008](D-008-gauge-b-architecture.md) (the law's first home),
[D-011](D-011-aplus-doctrine.md) / [D-017](D-017-candidates-tier.md) (the
grade paths reconciled here), `docs/explainers/market-mechanics-primer.md`
(why the close is the price of maximum agreement).
