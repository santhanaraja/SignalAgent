# D-010 — The Lab pattern: three laws

| | |
|---|---|
| **ID** | D-010 |
| **Date** | 2026-07-11 |
| **Status** | Ruled |

## Context

The user's hand-built scoring calculator prototype drifted from
production BEFORE it shipped (81 vs 76 at YTD 209% — a stale copy of the
overextension branch). A third hand-mirrored implementation of any
formula will always drift; the pattern had to make drift structurally
impossible, then generalize to every decision logic in the system.

## The three laws (the ruling)

1. **The lab owns zero math.** Every lab is backed by a simulate endpoint
   that calls the REAL production functions — extracted so production
   delegates to them, making a second copy structurally impossible.
   This applies to thresholds too (the Gauge Lab's flip distances are
   found by PROBING the function, never by hardcoding boundaries).
2. **Pre-seed from live state, prove it.** The lab opens showing today's
   actual inputs, with a parity chip demonstrating it reproduces the
   live result. Seeds are the engine's own emitted exact inputs
   (`score_inputs` / `assess_inputs`) wherever possible — display fields
   are rounded post-decision and can flip a branch at a boundary.
3. **Honest twisted state.** Any deviation from the live inputs is
   visibly flagged; a fresh artifact never silently clobbers an
   experiment; parity claims never survive inputs that weren't actually
   simulated.

## Evidence

Three shipped instances, each with acceptance pins through its endpoint:

| Instance | Endpoint | Shipped |
|---|---|---|
| Score Lab (item 19) | /api/score/simulate | 8d6c060 |
| Gauge Lab (24a) | /api/regime/simulate | 9408656 |
| Position Lab (24b) | /api/position/simulate | a2a6b54 |

The laws earned their edges from adversarial-review findings: the
rounding boundary (law 2's exact-seeds clause), the reseed-clobber and
false-parity-on-garbage-seed cases (law 3's clauses).

## Consequences

Wherever new decision logic ships, the extraction happens at build time
(production delegates to pure functions) — which is also what makes the
logic backtestable (Build 4 reused the same extraction).

## Revisit triggers

A lab whose simulate round-trip latency makes the UI unusable (the
Score Lab ruling pre-authorized a client-mirror + CI-parity-fuzz fallback
for exactly that case — no instance has needed it).

## Retest recipe

```
python3 test_score_lab.py       # incl. the 200-combo drift-proof fuzz
python3 test_gauge_lab.py       # truth table + probed flip distances
python3 test_position_lab.py    # ticket pins + live-row replay
```

## Links

- Jira: PER-508 comment 11713 (item 24, the pattern) / 11709 (item 19, the precedent)
- Commits: 8d6c060, 9408656, a2a6b54
