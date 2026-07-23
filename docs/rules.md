# Standing Rules R1–R27

> Decision records: [D-003](decisions/D-003-1b-position-engine.md) (1B engine / R11 close-basis) · [D-004](decisions/D-004-extension-guard.md) (extension guard) · [D-007](decisions/D-007-theme-layer-retirement.md) (theme retirement → R28 dollar enforcement) · [D-009](decisions/D-009-exit-timing-1230.md) (exit timing, proposed).

Read-only extraction of the standing-rules layer (Layer 3) as actually
evaluated in code. Source of truth: `framework/rule_engine.py` (evaluation
logic), `framework/config.yaml` `standing_rules:` (rule text) + `themes:`
(thresholds), `framework/theme_ranker.py` (where several thresholds are
actually enforced upstream), and `public/framework.html` (rendering).

**R4 is documented in [docs/regime.md](regime.md)** ("R4 semantics: signals
fast, state slow") and is only summarized here for completeness in the
tables. The other 26 are documented in full below.

## How to read the enforcement class

- **HARD** — code physically blocks or forces the outcome (an action
  cannot happen, or state is mutated regardless of the human).
- **ADVISORY** — code evaluates real state and reports a status
  (`action_needed` / `elevated` / `violation`); a human decides. The
  report changes with inputs but nothing is blocked.
- **DISPLAY-ONLY** — the rule renders as a static reminder; its status
  never varies with real portfolio state (it can only ever be
  `compliant`, or it emits fixed text).

## Architecture note that governs both summary tables

`RuleEngine.evaluate(regime_result, theme_result)` reads **only** the
regime string and the theme-qualification result. Its behavioral
elevation keys off `theme_result["active_themes"]`, which is the
`active` list in `framework/state/qualified_themes.json` — currently
`["Semis", "Biotech"]`.

**The rule engine itself has zero knowledge of `positions.json` — by
design, and still true.** As of PER-508 Phase 0 the dollar computations
live in a separate module: `framework/portfolio_rules.py` (R28,
`assess_portfolio` — a pure function per the Lab laws) reads
positions.json share counts × EOD closes × `account_capital_usd` and
computes R15–R18's numbers as real dollar statuses every framework run
(the `r28` block in framework.json / assessment.json, rendered in
Layer 3). `rule_engine.py` remains dollar-blind; R28 is the computing
authority ([D-007](decisions/D-007-theme-layer-retirement.md),
[D-008](decisions/D-008-gauge-b-architecture.md) Q4).

---

## R1 — Weekly Sunday review

- **Text (framework.html):** "Weekly Sunday review (regime + themes). Do not skip."
- **Code (`_eval_r1`, lines 124–138):** `is_sunday = today.weekday() == 6`.
  Sunday → `action_needed` ("weekly review required"); any other day →
  `compliant` ("Not Sunday. Next review on Sunday.").
- **Enforcement:** ADVISORY. It surfaces a reminder on Sundays; nothing
  blocks or forces the review, and no other code consults R1's status.
- **Reads:** system date only. **Emits:** status + message.
- **Notes:** the only rule that flips on weekday. Independent of the R4/
  theme-qualification "weekly review" cadence in theme_ranker — R1 is a
  UI nudge, the actual Sunday-cadence mutation lives in
  `_weekly_review_due` (see regime.md).

## R2 — Risk-off → no new aggressive positions

- **Text:** "Risk-off regime -> no new aggressive positions."
- **Code (`_eval_r2`):**
  - `regime == "Risk-off"` → `elevated`.
  - otherwise → `compliant`.
- **Enforcement:** ADVISORY (report only — it does not block an entry).
- **Reads:** regime string. **Emits:** status + message.
- **Resolved 2026-07-09:** the former `violation` branch (Risk-off + an
  ENTRY-prefixed signal) was **unreachable in the pipeline** — theme_ranker
  only emits `"ENTRY SIGNAL — …"` when `regime_ok` (regime in Trending/
  Choppy), so a risk-off ENTRY signal can never exist. The branch was
  removed; R2 degrades to `elevated` in Risk-off. The block-the-entry
  intent is enforced upstream in theme_ranker's entry gate.

## R3 — RETIRED → universe rotation + A+ doctrine (D-007 Phase 3, 2026-07-23)

- **Text (historical):** "New theme entry requires #1/#2 rank for 2
  consecutive Sundays AND risk-on regime AND discretionary catalyst
  review."
- **Code (`_eval_r3`):** always `compliant` with a supersession pointer —
  "Superseded → universe rotation + A+ doctrine: entries qualify via the
  weekly top-15 GICS scanner and the D-011 grade." No signal consumption
  remains (the R5→R28 pointer treatment). Theme entry signals no longer
  exist anywhere in the pipeline; the entry gate is condition 5 (group in
  the current universe) + the D-011 A+ grade, both computed.
- **Enforcement:** the pointer row is informational; the real gates are
  COMPUTED (c5 + grade in the position engine).

## R4 — RETIRED → position-engine stops + group breakers (D-007 Phase 3, 2026-07-23)

- **Text (historical):** documented in [docs/regime.md](regime.md) —
  Sunday-cadence theme qualification exits (D-002).
- **Code (`_eval_r4`):** always `compliant` with a supersession pointer —
  "Superseded → close-basis stops (R10/R11, the 1B engine's EXIT_FIRED) +
  per-group breakers." The theme_ranker's `qualified_themes.json`
  mutation died with the layer (module archived; state file in
  `archive/`). Exits are governed by the position engine's stop machine
  and the per-group breaker status — both computed.

## R5 — RETIRED → R28 per-group caps (D-007 Phase 2, 2026-07-18)

- **Text (historical):** "Maximum 2 themes active. Cash is the default."
- **Code (`_eval_r5`):** always `compliant` with a supersession pointer —
  "Superseded → R28 per-group caps: ≤20% / ≤3 positions per GICS group
  (computed, Layer 3)." No count logic and **no violation branch remain**
  (the same pointer treatment R17/R18 received). Concentration is
  enforced in dollars by R28, not theme counts.
- The theme_ranker's display-only active-list cap died with the layer
  (D-007 Phase 3, 2026-07-23 — module archived).

---

## R6–R27 — Behavioral / structural rules

All 22 route through one function, `_eval_behavioral`. Each returns
`elevated` ("{reason} {text}"), idle-`compliant` ("No active positions.
{text}" on a proven-flat book), unknown-`compliant` ("Positions
unavailable — active state unknown. {text}" when positions.json is
unreadable — an outage never renders as a confident flat book), or plain
`compliant`. **Since D-007 Phase 2 (2026-07-18) the active trigger reads
REAL portfolio state** — the holdings\' GICS groups from positions.json
via the shared resolver (unmapped holdings bucket as "TICKER
(ungrouped)", R28\'s convention):

```
elevated_in_defensive = {R10, R13, R15, R16, R17, R18, R26}
    → elevated when regime in ("Risk-off", "Caution")   # both defensive regimes
elevated_when_active  = {R6, R7, R8, R10, R11, R12, R13, R15,
                         R19, R21, R23, R24, R25}
    → elevated when len(active_groups) > 0   # REAL holdings\' groups
```

Order: the defensive branch is checked first (`if`), the active-groups
branch second (`elif`). So a rule in both sets, in Risk-off/Caution with
holdings, reports the "Elevated — regime is X" reason, not the
active-groups reason (affects R10, R13, R15).

**Enforcement class for all of R6–R27: ADVISORY at best, DISPLAY-ONLY for
several** (see per-rule table). Every one of these is a reminder string;
no code enforces the described discipline against actual trades.

### Two cross-cutting defects — both resolved 2026-07-09

1. **`elevated_on_entry` dead set removed.** The set `{R6, R7, R8, R9,
   R15}` was constructed on every call but never referenced (only the
   defensive and active sets are consulted), so its "elevate on a fresh
   entry signal" intent never existed. It was deleted. **R9**, whose only
   membership was this dead set, correctly remains never-elevated
   (DISPLAY-ONLY) — behavior unchanged.

2. **`elevated_in_risk_off` renamed `elevated_in_defensive`.** The set
   also fires in **Caution**, not just Risk-off (R16/R17/R18/R26 elevate
   in both). The name now matches the behavior; no logic change.

### Per-rule detail (R6–R27)

Text is the exact config string (rendered verbatim in framework.html's
"Standing Rules Reference" and inside each `elevated`/`compliant`
message). "Elevates" = the condition under which status becomes
`elevated`; "never" = always `compliant`.

| Rule | Text | Category | Elevates when | Enforcement |
|---|---|---|---|---|
| **R6** | Define invalidation before entry. Written down. Close-based. | invalidation | active themes > 0 | ADVISORY |
| **R7** | Size from risk-to-stop, not gut, not conviction, not max premium. | sizing | active themes > 0 | ADVISORY |
| **R8** | No naked momentum exposure through binary catalysts. | binary_catalyst | active themes > 0 | ADVISORY¹ |
| **R9** | Verify compliance scope before entering any new instrument class. | compliance | **never** (in no set) | DISPLAY-ONLY |
| **R10** | Stop moves UP only. Never lower a stop in adversity. | stop_management | Risk-off/Caution **or** active themes | ADVISORY |
| **R11** | Decisions on the exit-timeframe CLOSE, never on intraday wicks. | close_based | active themes > 0 | ADVISORY² |
| **R12** | Exit timeframe must be slower than entry timeframe. | timeframe | active themes > 0 | ADVISORY |
| **R13** | Never override a fired stop. | stop_override | Risk-off/Caution **or** active themes | ADVISORY |
| **R14** | Bank big winners back to preservation core where tax-efficient. | profit_taking | **never** (in no set) | DISPLAY-ONLY |
| **R15** | Maximum single-position size: 5-8% of relevant account's capital. *Computed by R28 since PER-508 Phase 0 ([D-007](decisions/D-007-theme-layer-retirement.md)): dollar statuses per holding, warning at 7.2%.* | position_size | Risk-off/Caution **or** active themes | **COMPUTED (R28)** |
| **R16** | Maximum single-theme exposure: 15% of total book. *Succeeded by R28's per-GICS-group caps ([D-007](decisions/D-007-theme-layer-retirement.md), Phase 0 caps amended 2026-07-13): group ≤20% of capital AND ≤3 positions, computed in dollars (warning above 18%).* | theme_concentration | Risk-off/Caution only | **COMPUTED (R28)** |
| **R17** | Maximum total theme exposure (across all active themes): 25% of total book. *Superseded by [D-008](decisions/D-008-gauge-b-architecture.md) Q4: the static 25% cap is absorbed into R28's regime-scaled ceiling (90/50/25/5), computed by R28.* | total_theme_exposure | Risk-off/Caution only | **COMPUTED (R28)** |
| **R18** | Minimum cash reserve at all times: 30-40% in money market vehicles. *Amended by [D-008](decisions/D-008-gauge-b-architecture.md) Q4: the floor as written is absorbed into R28's regime-scaled ceiling (implied cash floor 10% at Trending); R28 reports it as the ceiling's complement.* | cash_reserve | Risk-off/Caution only | **COMPUTED (R28)** |
| **R19** | Aggressive sleeve cost basis capped at active theme limits. | cost_basis | active themes > 0 | ADVISORY |
| **R20** | Never fund the aggressive sleeve from preservation core. Never cross-fund between accounts. | cross_funding | **never** (in no set) | DISPLAY-ONLY |
| **R21** | No new entries on same day as stop-out from related name. | same_day_entry | active themes > 0 | ADVISORY |
| **R22** | Hot tape => deploy slowly. | hot_tape | **never** (in no set) | DISPLAY-ONLY |
| **R23** | Conviction expresses through stop placement, not oversizing. | conviction_sizing | active themes > 0 | ADVISORY |
| **R24** | Hitting target = sell signal. No goalpost-moving. | target_hit | active themes > 0 | ADVISORY |
| **R25** | Don't add to position because it's working. Add only on pullbacks per pre-committed plan. | adding_to_position | active themes > 0 | ADVISORY |
| **R26** | No margin leverage on aggressive options trades. | margin | Risk-off/Caution only | DISPLAY-ONLY |
| **R27** | Flexibility means switching themes by rules, not by feel. | theme_switching | **never** (in no set) | DISPLAY-ONLY |

¹ **R8** is also surfaced independently by the PER-510 earnings layer,
which appends "earnings in Nd — R8: binary catalyst window" to a
ticker's reasoning and to position-engine `earnings_note` when earnings
are ≤7 days out. That is a separate display path in `signal_engine.py` /
`position_signals.py`; it does not block or size anything either.

² **R11** (close-based decisions) is genuinely enforced — but in the 1B
position engine, not here: the state machine strips the intraday
live-quote bar and advances only on completed daily closes. The
rule-engine surface is still just a reminder.

   **Intraday stop-breach alerts (PER-510-B)** layer information on top of
   R11 without changing it: `notify_intraday.py` checks each HOLDING's live
   price against its SMA20 stop during market hours and pushes a Slack
   alert on breach (with depth in ×ATR) or when within 0.25×ATR above the
   stop (warn), once per ticker per tier per day. **Cadence reality: it
   rides the throttled hourly update-signals runs (~5×/day), so detection
   lag is up to ~an hour — an hourly check, not a tick watcher.** If that
   is ever insufficient, faster detection is a separate infra decision.
   The alert is information only — nothing is executed, and the close
   still decides. Any change to exit timing itself stays gated on Build 5
   evidence. Full market holidays are skipped (SPY-session check); on the
   ~4 early-close half-days a year, a run between 13:00 and 16:00 ET can
   still alert — stale by hours but truthful vs the stop (accepted
   residual). EXIT_FIRED rows never re-alert: that exit already signaled
   at a prior close and belongs to the post-close report.

³ **HISTORICAL (fixed by R28, PER-508 Phase 0):** through 2026-07-12
these rows carried live numbers that nothing computed — static text
with no dollar comparison, no code reading `positions.json` × prices.
`framework/portfolio_rules.py` (assess_portfolio, a pure function per
the Lab laws) now computes all four against actual dollars every run:
per-position %, per-GICS-group exposure + count, the D-008 Q4
regime-scaled ceiling, and the implied cash floor. Enforcement class
COMPUTED (reporting-hard — it cannot block broker orders and does not
pretend to). A ceiling breach from a regime downshift is action_needed
— derisk via normal exit discipline (stops/R11); never a same-day
forced liquidation.

### Rules that never leave `compliant`

R9, R14, R20, R22, R27 are in no elevation set, so they are effectively
**DISPLAY-ONLY** — they always render `compliant` and exist purely as the
reference list. (Before the 2026-07-09 cleanup R9 sat only in the dead
`elevated_on_entry` set; removing it left R9's never-elevate behavior
unchanged.)

---

## TABLE 1 — Position sizing, deployment & concentration

Every rule that touches size, aggregate deployment, cash floor, or
concentration, with the actual numbers and whether code enforces them.

| Rule | Limit (from text) | Where the number lives | Computed against real $? | Enforcement |
|---|---|---|---|---|
| **R3** | *(retired)* theme entry gate | superseded → universe rotation + A+ doctrine (D-007 Phase 3) | **Yes** — c5 + D-011 grade | **COMPUTED (position engine)** |
| **R4** | *(retired)* theme exit signals | superseded → close-basis stops + group breakers (D-007 Phase 3) | **Yes** — EXIT_FIRED + breakers | **COMPUTED (position engine)** |
| **R5** | *(retired)* theme count | superseded → R28 per-group caps (D-007 Phase 2) | **Yes — R28** (≤20% / ≤3 per group) | **COMPUTED (R28)** |
| **R15** | Single position 5–8% of account | rule text + `portfolio_rules.py` | **Yes — R28** (warning >7.2%, violation >8%) | **COMPUTED (R28)** |
| **R16** | Single theme ≤ 15% of book | `portfolio_rules.py` per-GICS-group caps (config `r28:`) | **Yes — R28** (≤20% AND ≤3 positions/group, amended 2026-07-13) | **COMPUTED (R28)** |
| **R17** | All themes ≤ 25% of book | `portfolio_rules.py` regime-scaled ceiling ([D-008](decisions/D-008-gauge-b-architecture.md) Q4: 90/50/25/5) | **Yes — R28** | **COMPUTED (R28)** — supersedes the static 25% |
| **R18** | Cash reserve 30–40% | `portfolio_rules.py` cash-floor row | **Yes — R28** (informational: the ceiling's complement) | **COMPUTED (R28)** |
| **R7 / R23** | Size from risk-to-stop, not oversizing | rule text only | **No** | ADVISORY reminder |
| **R26** | No margin on aggressive options | rule text only | **No** | DISPLAY-ONLY |

**Is there a HARD aggregate portfolio-deployment ceiling? Still no —
but there is now a COMPUTED one.** Since PER-508 Phase 0, R28 computes
the regime-scaled total-exposure ceiling (Trending 90 / Choppy 50 /
Caution 25 / Risk-off 5, per [D-008](decisions/D-008-gauge-b-architecture.md)
Q4) against actual dollars every run — enforcement class COMPUTED
(reporting-hard): it reports violations and downshift action_needed
statuses with dollar numbers, but cannot block a broker order and does
not pretend to. A downshift breach is an advisory to derisk via normal
exit discipline (stops/R11), never a same-day forced liquidation. R5\'s
two-theme cap — formerly the single HARD constraint — retired into
R28\'s per-group caps on 2026-07-18 (D-007 Phase 2); every numeric
concentration limit is now COMPUTED in dollars.

---

## TABLE 2 — What fires for the current state

State: 3 HELD positions (IWM $6,890 + ARWR $6,061 + BIIB $5,764 ≈
**$18.7K deployed**), Trending regime, ~$81K reserve (~19% of a ~$100K
book), today = **Monday 2026-07-06**. Values below are the actual latest
committed `framework.json` run (20:23 UTC).

**The caveat above is HISTORY as of D-007 Phase 2 (2026-07-18):** the
rule engine now DOES see your positions. The elevations below were
driven by `active_themes` in the era this table records; today the same
book drives them as `active_groups = ["Biotechnology", "IWM
(ungrouped)"]` — real holdings\' GICS groups via the shared resolver —
and a flat book idles all 13 with "No active positions." (the live state
at the Phase-2 ship). The dollar amounts themselves are read by R28
(Layer 3), not this layer.

Current tally: **14 compliant · 13 action_needed · 0 violations.**

| Rule | Status | Why |
|---|---|---|
| **R6, R7, R8, R11, R12, R19, R21, R23, R24, R25** | elevated | `elevated_when_active` and 2 themes are active. Pure reminders — invalidation written down, size-from-stop, catalyst caution, close-based decisions, exit-timeframe, cost-basis cap, no same-day re-entry, conviction-via-stop, target=sell, add-only-on-pullbacks. |
| **R10, R13, R15** | elevated | In both `elevated_when_active` and `elevated_in_defensive`; here the active-themes branch fires (regime is Trending, not Risk-off/Caution). Stop-up-only, never-override-stop, single-position-size reminders. |
| **R1** | compliant | Monday, not Sunday. Flips to `action_needed` this coming Sunday (Jul 12) — which is also the first live weekly-review mutation under the R4 fix. |
| **R2, R3, R4, R5** | compliant | Trending (R2 not triggered); no entry/exit signals this run (R3/R4); R5 now a superseded→R28 pointer (era note: it read "2/2 themes"). |
| **R9, R14, R16, R17, R18, R20, R22, R26, R27** | compliant | Not in any live elevation set for the current Trending regime. R16/R17/R18/R26 would elevate if regime degrades to Caution/Risk-off; R9/R14/R20/R22/R27 never elevate at all. |

**What would change the picture:**
- Regime → Caution/Risk-off adds **R16, R17, R18, R26** to the elevated
  set (+ re-labels R10/R13/R15 to the regime reason) → 17 action_needed.
- A Sunday run flips **R1** to action_needed.
- None of these is a violation; since D-007 Phase 2 NO R1-R27 rule has a
  violation branch at all (R5\'s was retired with it; R2\'s went in the
  2026-07-09 cleanup) — violations now live exclusively in R28\'s
  computed dollar rows.

**Bottom line (as updated for the group era):** nothing here blocks or
forces anything. The 13 reminder items fire on REAL holdings\' groups
now; deployment level, per-name sizes, and the cash posture are checked
by R28\'s computed dollar rows on Layer 3 — this layer carries the
behavioral discipline, R28 carries the numbers.
