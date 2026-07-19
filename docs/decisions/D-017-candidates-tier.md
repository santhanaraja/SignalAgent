# D-017 — Candidates tier: auto-grade every scanner name; the human still appoints

| | |
|---|---|
| **ID** | D-017 |
| **Date** | 2026-07-18 |
| **Status** | Ruled — implemented same night (staged for review) |

## Context

Third bump into the tracked/un-tracked seam in one evening: HPQ at 84 BUY NOW —
the highest score on the page — carried no grade because nobody had nominated it
into positions.json. The funnel's mouth was manual: rotation → groups → *silence*
→ hand-picked watchers → state machine → A+ gate. The scanner's own names were
the one tier the machine refused an opinion on.

**Technical basis (what makes this cheap and safe):** the
[D-011](D-011-aplus-doctrine.md) grade is **stateless per run** — all seven rows
(conditions, extension ≤1.8×ATR, approach filter, RSI 45–70, score ≥75, breaker
clear, ≥15-session runway) compute from per-ticker data the engine already holds
each close. Only the state machine (WATCHING→ARMING→READY, consecutive-close
counters) is stateful and remains watchlist-only. Candidates therefore carry a
**GRADE without a STATE**.

## Ruling (PER-508 comment 11725) — all four questions as recommended

- **Q1 — Coverage: ALL signals.json names** (~42–54, bounded by construction to
  the rotation's selected groups). Partial coverage would recreate the "why
  doesn't X have a grade" question one tier down. ~50 pure-function calls on
  already-computed inputs; **zero new fetches**.
- **Q2 — Display: grade chip + failing-reasons hover ONLY** for un-tracked
  candidates. No state badge (no state exists), no pips (pips are the state
  machine's face; a five-pip un-tracked row reads as READY). Visual grammar:
  **chip-only = candidate (the machine's opinion) · chip + badge + pips =
  tracked (the machine's commitment)**. Tracked rows keep the 9b239f5
  enrichment unchanged.
- **Q3 — Promotion: NEVER automatic.** positions.json stays the human-curated
  set ([D-003](D-003-1b-position-engine.md); North Star guardrail). Friction
  drops to one click: a `+watch` affordance with add-time context auto-recorded
  (the fe1de42 note pattern). **Honest constraint:** positions.json is a
  committed artifact and the serving dyno's disk is ephemeral with no push
  credentials, so the one click **copies the appointment prompt** — no durable
  one-click write exists and none is faked.
- **Q4 — Close report: one candidates line** — A+ names spelled out, B/C
  counted (`Candidates: 1 A+ (HPQ) · 4 B · 12 C`). The system now ANSWERS "is
  there anything to buy today?" every close. A+ is rare by construction; when
  the line names one, it earns eyes.

## Implementation (this build)

Engine-side per Lab law 1 ([D-010](D-010-lab-pattern-laws.md)) — one grade
implementation, no drift possible:

- **signal_engine** emits per-row `grade_inputs` (unrounded close/SMA20/
  SMA20-5d/SMA5/ATR14/consecutive-closes/up-close-off-low + rsi14 +
  quality_score), computed on the **synthetic-bar-stripped** df by the same
  shared helpers the watcher path uses (`grade_inputs_from_df` etc. in
  `position_signals`) — the exact watcher recipe, close-basis law.
- **PositionSignalEngine.grade_candidates** feeds those scalars through the
  SAME `assess_position` + `grade_setup` pure functions with the same knobs;
  tracked names excluded; un-gradeable rows carry `grade: null` + the reason
  (unavailable-data-never-A+); inputs computed under different parameters are
  refused, not graded.
- The runner hoists `candidate_grades` to a top-level framework.json key and
  annotates both signals.json copies (grades travel with the rows that
  produced their inputs). All consumers are **era-aware**: absent block →
  nothing renders, no line, no fabricated empty tier.
- Page: chip + hover + `+watch` in the Layer-2 sub-rows (the seam left in
  9b239f5); `/api/assessment` gains a `candidates` section; the Slack close
  report gains the Q4 line.

## Retest recipe

Build 5's historical replay grades CANDIDATE histories too — do A+ candidates
outperform B candidates forward? (D-011's recipe, wider population.)

## Revisit triggers

1. Candidate chips churning day-to-day = threshold miscalibration signal.
2. The `+watch` affordance used for bulk adds = curation eroding → opens a
   watchlist-size-cap discussion.

## Source

PER-508 comment 11725 (2026-07-18, late). Deliberation brief:
`docs/briefs/candidates-tier-brief.md` (dropped in with the build).

The funnel is now designed end-to-end: rotation → groups → auto-graded
candidates → one-click appointment → state machine → A+ gate → human trigger.
