# D-007 — Theme layer retirement: scanner + quality gate as thesis, R28 dollar enforcement, Option C staged

| | |
|---|---|
| **ID** | D-007 |
| **Date** | 2026-07-12 |
| **Status** | Ruled — **Phases 0-3 COMPLETE** (Phase 3 shipped 2026-07-23) |

## Context

The hand-picked 8-narrative theme layer sat between the trader and the
GICS universe scanner as a qualification gate, ranking themes by
proxy-ETF momentum. A week of live operation produced four exhibits of
the same root cause: **proxy-ETF momentum ≠ constituent tradability.**

## Options considered

Per the strategy-session deliberation (Option C = staged retirement was
ruled; A = keep and patch, B = immediate rip-out rejected for sequencing
risk — the concentration backstop must exist before the theme cap dies).

## Evidence (the week's four exhibits)

1. Coverage blindness: airlines/healthcare/refiners led for weeks,
   invisible to the 8 narratives.
2. Semis ranked #1 on a stale 12W tail while every constituent scored
   REDUCE/EXIT.
3. The theme's momentum leader (MRNA) was the board's most dangerous
   chase all week (see [D-004](D-004-extension-guard.md)).
4. Biotech ranked #1 (composite 2.5) at the Jul 12 review — two days
   after its constituents stopped out the entire deployment.

Plus Build 4's principle: complexity must earn its keep ([D-006](D-006-build4-protocol.md));
measured against the scanner it duplicates, the theme layer doesn't.

## Ruling + rationale

No hand-curated conviction layer between the trader and the scanner.
Condition 5 rewires from "theme qualified" to "ticker's GICS group is in
the current selected universe." Concentration discipline moves to
computed dollars (R28) + per-group caps, not theme counts. SPDR taxonomy
note: the ~20 industry SPDRs re-import the proxy-divergence failure
class; sector ETFs move to the breadth/regime toolkit, never
qualification gates.

**Strict phase order:**

- **Phase 0 (FIRST, standalone): R28 dollar enforcement** — rule engine
  reads positions.json + prices; per-position 5–8% (R15), per-GICS-group
  caps (successor to R5/R16) — *drafted 15% + max 2/group; amended
  2026-07-13 to **20% + max 3/group**: 15% against the 8% max position
  forced undersizing the second name to 1.875-of-full, while 20% fits
  two full-size names with room; the correlated-group tradeoff is
  acknowledged on the record (the July-6 biotech book). The D-008 Q4
  ceiling is unchanged — emergent consequence: 90/20 forces ≥5 groups
  at full deployment* — aggregate exposure
  and cash floor as computed statuses with dollar numbers in
  assessment.json, the framework page, both Slack pushes.
  *Amended same day by [D-008](D-008-gauge-b-architecture.md) Q4 (comment
  11715): the aggregate ceiling is regime-scaled 90/50/25/5, superseding
  R17's static 25% and absorbing R18's 30–40% floor — R28 implements the
  scaled ceiling from day one.*
- **Phase 1:** condition-5 rewire (theme_qualified →
  group_in_current_universe); emit weeks_in_universe and OBSERVE before
  adding any persistence gate (Build 4 lesson: no pre-tuning). Themes
  display-only in parallel. 1B pins: ARWR/MRNA replay identically.
- **Phase 2:** rule-engine triggers key off real holdings' groups; R5
  semantics retire into R28's caps; framework Layer 2 becomes the GICS
  leadership view.
- **Phase 3:** decommission theme_ranker/qualified_themes/leaders.json
  (archived); docs + architecture updated.

**What survives:** R4's anti-whipsaw spirit ([D-002](D-002-r4-sunday-cadence.md))
as an observed-then-decided persistence question at the group level.
**What dies:** 8 narratives as qualification content, proxy-ETF composite
ranking, theme counts as concentration discipline.

## Consequences

R28 becomes the flagship build before any retirement step. Until Phase 3
completes, themes run display-only in parallel — divergence between the
two gates is observable, not harmful.

## Implementation record

- **Phase 0 (R28)**: shipped 2026-07-13, `ba811a0` (caps amended to 20%/3
  same day; real capital $97,500 set in `7b7812a`).
- **Phase 1 (condition-5 rewire + weeks_in_universe)**: shipped 2026-07-18,
  `04e5964`, riding D-011's grade. THE FIRST LIVE DIVERGENCE EXHIBIT fired
  at ship time: the 2026-07-18 rotation dropped Biotechnology (rank 29,
  outranked) while the legacy theme layer still ranked Biotech #1 ACTIVE —
  MRNA's condition 5 honestly fails where the theme gate passed, exactly
  the "observable, not harmful" divergence this ruling anticipated. The
  replay pin asserts it data-driven and self-retires if the group rotates
  back in. weeks_in_universe counting starts at the 2026-07-25 rotation
  (week-keyed — same-week rebuilds carry unchanged).
- **Phase 2 (rule-engine rewire + GICS Layer 2)**: shipped 2026-07-18 —
  active state = real holdings' groups via the ONE shared resolver
  (`resolve_group_map`), R5 → superseded R28 pointer, Layer 2 renders the
  universe's selected groups with breaker chips; legacy themes demoted to
  a visibly-deprecated strip.
- **Phase 3 (decommission)**: shipped 2026-07-23 after five days of
  parallel observation (per this record's own plan). ThemeRanker /
  ConstituentRanker / `qualified_themes.json` / `theme_history.json`
  moved to `archive/` (git history intact); the runner computes no theme
  layer and the artifacts carry no `themes`/`theme_leaders` blocks
  (era-aware consumers render the retirement — `/api/framework/
  leaders.json` is 410 Gone with pointers, the assessment `themes`
  section reports the decommission on post-Phase-3 artifacts and still
  projects rows from older ones). R3/R4 are superseded pointer rows
  (entries → universe rotation + A+ doctrine; exits → close-basis stops
  + group breakers — the R5→R28 treatment). The Sunday weekly-review
  text reduced to its scanner-era form ("regime + universe review").
  The Layer-2 legacy strip and the theme-rotation "Actions This Week"
  panel are gone from the page.
- **The MRNA divergence exhibit's final state**: the exhibit (c5
  honestly failing while the display theme layer ranked Biotech #1)
  RETIRED WITH THE LAYER — the contradiction it existed to demonstrate
  is no longer expressible, which was Phase 3's whole point. The
  replay pin's exhibit assertions self-retire era-aware: they activate
  only on artifacts that carry a themes block.

## Revisit triggers

1. Phase 1 observation shows 1-rotation universe churn making condition
   5 noisier than the theme gate it replaced (the persistence-gate
   question re-opens with data).
2. Any phase's pins failing to replay the current watchers identically.

## Retest recipe

Phase-gated; per phase:

```
python3 test_rule_engine.py          # R28 statuses (Phase 0, once built)
python3 test_position_signals.py     # condition-5 rewire pins (Phase 1)
python3 test_position_lab.py         # watcher replay identical (Phase 1)
```

## Links

- Jira: PER-508 comment 11714 (the strategy-session ruling, 2026-07-12)
- Related: [D-002](D-002-r4-sunday-cadence.md), [D-004](D-004-extension-guard.md), [D-006](D-006-build4-protocol.md)
