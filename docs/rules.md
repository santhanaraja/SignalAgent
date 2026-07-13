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

**The rule engine has zero knowledge of `positions.json`.** The Build 1B
holdings (IWM / ARWR / BIIB) and their dollar values never enter
`rule_engine.py`. No rule computes anything against deployed capital,
share counts, cash balance, or per-name size. Every "sizing" and
"deployment" rule (R15–R18) carries its numbers in **text only**. This
is the single most important fact for interpreting what fires.

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

## R3 — New theme entry requirements

- **Text:** "New theme entry requires #1/#2 rank for 2 consecutive
  Sundays AND risk-on regime AND discretionary catalyst review."
- **Code (`_eval_r3`, lines 163–189):**
  - any `entry_signals` action contains `"ENTRY SIGNAL"` → `action_needed`
    ("Requires discretionary catalyst review before activation").
  - else any contains `"Building"` → `compliant` (lists progress).
  - else → `compliant`.
- **Thresholds:** the "#1/#2 rank for 2 consecutive Sundays AND risk-on"
  logic is **enforced upstream in theme_ranker**, not here:
  `requires_top_n_rank: 2`, `consecutive_sundays_required: 2`,
  `regime_required: [Risk-on / Trending, Risk-on / Choppy]` (config
  `themes.entry_rule`). Counters run on ISO-week-deduped weekly closes
  (see regime.md). R3 only re-reports theme_ranker's verdict.
- **Enforcement:** ADVISORY (the discretionary catalyst review is a human
  gate; code never confirms it happened).
- **Resolved 2026-07-09:** `_eval_r3` formerly took a `regime` parameter
  it never read (the "risk-on regime" clause is checked only in
  theme_ranker). The unused parameter and its call-site argument were
  removed; signature is now `_eval_r3(entry_signals)`.

## R4 — Theme exit conditions

- **Documented in [docs/regime.md](regime.md).** Summary: `_eval_r4`
  (lines 191–225) reports `exit_signals` — `"EXIT SIGNAL"` →
  `action_needed`, `"Warning"` → `compliant`. The rule-engine report is
  ADVISORY, but the underlying theme_ranker **HARD-mutates**
  `qualified_themes.json` on a *confirmed* (2 degraded weekly closes)
  regime-degradation or 3-weekly-close rank exit. Enforcement class
  overall: HARD (state mutation) + ADVISORY (rule-engine surface).

## R5 — Maximum 2 themes active

- **Text:** "Maximum 2 themes active. Cash is the default."
- **Code (`_eval_r5`, lines 227–242):** `max_themes =
  themes.ranking.max_active_themes` (=2). `len(active_themes) >
  max_themes` → `violation`; else `compliant` ("N/2 active themes. Cash
  is the default.").
- **Enforcement:** **HARD.** theme_ranker's `_update_active_state` (line
  ~426) breaks out of the entry loop at `len(updated) >= max_active`,
  so a 3rd theme is physically never persisted to `qualified_themes.json`.
  R5's `violation` branch is a defensive backstop for a hand-edited state
  file — practically unreachable through the pipeline.
- **This is the only hard-enforced numeric limit in the entire
  sizing/deployment family** (see Table 1).

---

## R6–R27 — Behavioral / structural rules

All 22 route through one function, `_eval_behavioral` (lines 244–288).
Each returns either `elevated` (with `message = "{reason} {text}"`) or
`compliant` (with `message = text`). **None reads real portfolio state**
— elevation is driven purely by two conditions:

```
elevated_in_defensive = {R10, R13, R15, R16, R17, R18, R26}
    → elevated when regime in ("Risk-off", "Caution")   # both defensive regimes
elevated_when_active  = {R6, R7, R8, R10, R11, R12, R13, R15,
                         R19, R21, R23, R24, R25}
    → elevated when len(active_themes) > 0
```

Order: the defensive branch is checked first (`if`), the active-themes
branch second (`elif`). So a rule in both sets, in Risk-off/Caution with
active themes, reports the "Elevated — regime is X" reason, not the
active-themes reason (affects R10, R13, R15).

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
| **R15** | Maximum single-position size: 5-8% of relevant account's capital. | position_size | Risk-off/Caution **or** active themes | ADVISORY³ |
| **R16** | Maximum single-theme exposure: 15% of total book. | theme_concentration | Risk-off/Caution only | DISPLAY-ONLY³ |
| **R17** | Maximum total theme exposure (across all active themes): 25% of total book. *Superseded by [D-008](decisions/D-008-gauge-b-architecture.md) Q4: the static 25% cap is absorbed into R28's regime-scaled ceiling (90/50/25/5).* | total_theme_exposure | Risk-off/Caution only | DISPLAY-ONLY³ |
| **R18** | Minimum cash reserve at all times: 30-40% in money market vehicles. *Amended by [D-008](decisions/D-008-gauge-b-architecture.md) Q4: the floor as written is absorbed into R28's regime-scaled ceiling (implied cash floor 10% at Trending).* | cash_reserve | Risk-off/Caution only | DISPLAY-ONLY³ |
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

³ **R15–R18 carry live numbers (5–8%, 15%, 25%, 30–40%) but nothing
computes them.** The concentration limits emit static text in
theme_ranker (`concentration_notes`: `"Max per theme: 15% of book"`,
`"Max total themes: 25% of book"`, lines 257–262) with no dollar
comparison. R15/R18 exist only as reminder text. No code reads
`positions.json` share counts × prices to check any of these.

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
| **R5** | ≤ 2 active themes | `themes.ranking.max_active_themes: 2` | Yes — theme *count* | **HARD** (theme_ranker won't persist a 3rd) |
| **R15** | Single position 5–8% of account | rule text only | **No** | ADVISORY reminder |
| **R16** | Single theme ≤ 15% of book | `concentration_limits.max_per_theme_pct_of_book: 15` | **No** (emitted as static text) | DISPLAY-ONLY |
| **R17** | All themes ≤ 25% of book | `concentration_limits.max_total_themes_pct_of_book: 25` | **No** (emitted as static text) | DISPLAY-ONLY |
| **R18** | Cash reserve 30–40% | rule text only | **No** | DISPLAY-ONLY |
| **R7 / R23** | Size from risk-to-stop, not oversizing | rule text only | **No** | ADVISORY reminder |
| **R26** | No margin on aggressive options | rule text only | **No** | DISPLAY-ONLY |

**Is there a HARD aggregate portfolio-deployment ceiling? No.** There is
no explicit portfolio-level maximum anywhere in code — not even at
Trending; total exposure is bounded only *implicitly* and only *on paper*
by per-name sizing (R15, advisory), the cash floor (R18, display-only),
and the theme-concentration limits (R16/R17, display-only text) — none of
which the code computes against actual dollars — with the single hard
constraint being R5's cap of two active themes.

---

## TABLE 2 — What fires for the current state

State: 3 HELD positions (IWM $6,890 + ARWR $6,061 + BIIB $5,764 ≈
**$18.7K deployed**), Trending regime, ~$81K reserve (~19% of a ~$100K
book), today = **Monday 2026-07-06**. Values below are the actual latest
committed `framework.json` run (20:23 UTC).

**First, the load-bearing caveat:** the rule engine does not see your
positions. What actually drives every elevation below is
`active_themes = ["Semis", "Biotech"]` from `qualified_themes.json`.
IWM's theme ("SmallCap/Broad") is not a framework theme; ARWR/BIIB map to
Biotech, which is already active in the theme layer independent of
whether you hold the names. **These same 13 rules would fire with zero
positions held**, as long as Semis + Biotech stay qualified. Nothing
reads the $18.7K or the $81K.

Current tally: **14 compliant · 13 action_needed · 0 violations.**

| Rule | Status | Why |
|---|---|---|
| **R6, R7, R8, R11, R12, R19, R21, R23, R24, R25** | elevated | `elevated_when_active` and 2 themes are active. Pure reminders — invalidation written down, size-from-stop, catalyst caution, close-based decisions, exit-timeframe, cost-basis cap, no same-day re-entry, conviction-via-stop, target=sell, add-only-on-pullbacks. |
| **R10, R13, R15** | elevated | In both `elevated_when_active` and `elevated_in_defensive`; here the active-themes branch fires (regime is Trending, not Risk-off/Caution). Stop-up-only, never-override-stop, single-position-size reminders. |
| **R1** | compliant | Monday, not Sunday. Flips to `action_needed` this coming Sunday (Jul 12) — which is also the first live weekly-review mutation under the R4 fix. |
| **R2, R3, R4, R5** | compliant | Trending (R2 not triggered); no entry/exit signals this run (R3/R4); 2/2 themes (R5 at the cap but not over). |
| **R9, R14, R16, R17, R18, R20, R22, R26, R27** | compliant | Not in any live elevation set for the current Trending regime. R16/R17/R18/R26 would elevate if regime degrades to Caution/Risk-off; R9/R14/R20/R22/R27 never elevate at all. |

**What would change the picture:**
- Regime → Caution/Risk-off adds **R16, R17, R18, R26** to the elevated
  set (+ re-labels R10/R13/R15 to the regime reason) → 17 action_needed.
- A Sunday run flips **R1** to action_needed.
- None of these is a violation; the only way to reach a `violation` is
  R5 with a hand-edited 3-theme state file (R2 no longer has a violation
  branch as of the 2026-07-09 cleanup).

**Bottom line for tonight's read:** nothing is blocking or forcing
anything. All 13 firing items are advisory reminders triggered by having
any active theme at all; your actual deployment level, per-name sizes,
and 19%-deployed / 81%-cash posture are invisible to this layer and
un-checked by any rule.
