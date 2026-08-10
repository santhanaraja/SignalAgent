# CHAT-RULINGS — operator rulings issued in session, pending or minor

| | |
|---|---|
| **ID** | CHAT-RULINGS |
| **Date** | 2026-08-09 (registry created; entries carry their own dates) |
| **Status** | **Living registry** — never Superseded as a whole; individual entries may be promoted to full D-records or marked superseded |

## Purpose

The D-records capture decisions large enough to earn a file. Rulings
issued in chat — scope calls, interpretive-step approvals, standing
caveats, "not yet, and here is why" — bind the work just as hard but
previously lived only in session transcripts and memory notes. This
registry is their durable home. Content is supplied by the operator;
the scaffold's conventions below are fixed.

## Conventions

- One entry per ruling, newest first, using the entry template below.
- Cross-link like every record: `[D-XXX](D-XXX-....md)` for decisions,
  `[CR-YYYY-MM-DD-n](#cr-yyyy-mm-dd-n)` for rulings in this file.
- An entry that grows a deliberation, evidence, or a build of its own
  is **promoted**: it becomes a D-record and the entry here is reduced
  to one line pointing at it.
- Entries are append-only in spirit: supersession is recorded, never
  silently edited over.

## Entry template

```markdown
### CR-YYYY-MM-DD-n — <short title>

| | |
|---|---|
| **Ruled** | YYYY-MM-DD, in session |
| **Status** | In force / Superseded by [..] / Promoted to [D-XXX](..) |
| **Binds** | <the surface, build, or doctrine the ruling constrains> |

<The ruling, verbatim or faithfully condensed; the reasoning; links.>
```

## Rulings

### CR-2026-08-09-1 — Extension guard: not removed on signal-level evidence

| | |
|---|---|
| **Ruled** | 2026-08-09, in session |
| **Status** | In force (hold) — pending the carry replay |
| **Binds** | [D-004](D-004-extension-guard.md)'s guard / grade row 2; the 5.1 carry replay's arm design |

Build 5.1 ([report](../backtest-ablation.md)) found the extension
guard (≤1.8×ATR above SMA20) is the campaign's volume suspect —
23,893 excluded days earning 1.43% against A+'s 0.83%, both spans,
positive ex-top-5%, 527 tickers. **The guard is nonetheless retained
pending a system-level test.**

**Reason — a mechanism, not process caution.** The stop is the SMA20
and extension measures distance from the SMA20 in ATR, so
`extension ≤1.8×ATR` is the same statement as `risk per share
≤1.8×ATR`. The guard is doing risk control under a selection label.
Removing it at fixed 6.5% sizing roughly doubles risk per trade: the
return advantage is 1.72×, the risk-per-share penalty is about 2×.
Layer A and 5.1 model no stops and structurally cannot see this.

**Supporting precedent — five for five.** Every signal-level finding
in this project has failed to survive contact with real stops: the
V-recovery cost, Layer A's C-beats-A+ into 5B's
no-gate-distinguishable, both ribbon probes, the addendum's benchmark
bands, and [6A](../backtest-target.md) (58.9% of trades improved,
dollars still lost).

**Also:** Build 5.1's own pre-registration forbids shipping until a
suspect survives the carry.

**Required test — carry replay, three arms, own pre-registration:**

- S1 — A+ as-is, guard at 1.8, fixed 6.5% sizing
- S6 — no guard, fixed 6.5% sizing
- S7 — no guard **plus risk-parity sizing with a position cap**

S7 is the arm the evidence points at: it removes the exclusion while
preserving the risk control the guard was silently providing. If S7
wins, the guard was never a selection rule and should be **replaced**
rather than deleted.

**Counter-argument on record:** 5B found no gate measurably better
than any other, which cuts both ways — but 5B never ran "A+ minus
row 2" as an arm, so this specific question is genuinely unmeasured.

**Operational cost of waiting, measured:** on the 2026-08-07 board,
ten names were blocked by this row — MSFT at 4.35×ATR, PFE 3.65,
MSI 2.91, plus GEN, NOW, NTAP, CRWD, LH, J, DGX. It is the most
active single constraint on the candidate list.

### CR-2026-08-09-2 — D-011 + D-019: open, with recommendations on record

| | |
|---|---|
| **Ruled** | 2026-08-09, in session — **OPEN, not ruled** |
| **Status** | Open — evidence base complete, ruling owed |
| **Binds** | Nothing yet; recommendations on record for the joint [D-011](D-011-aplus-doctrine.md) + [D-019](D-019-breaker-coverage.md) revisit |

Both revisit questions remain unruled. The evidence base is complete:
Layer A, the distribution decomposition, 5B, the addendum, 5.1 and
6A. Nothing further needs measuring before a ruling; every additional
study from here is optional. Brief:
[d011-d019-combined-brief](../reference/d011-d019-combined-brief.html)
(and the fuller five-option table in the standalone
[d011-revisit-brief](../reference/d011-revisit-brief.html)).

**Why they must be ruled together.** D-019 decides how often the hard
gate is switched on; D-011 decides what the gate does when it is.
Softening the gate alone enlarges the surface where the undefined
selection rule operates — the lever 5B measured as larger than any
gate effect. Fixing breadth alone returns the system to Trending,
where the grade is advisory anyway, which moots the gate question
rather than answering it.

**Recommendation on record (deliberation, not a ruling):**

- **D-019 → suspend breadth as a throttle, explicitly interim and
  dated**, with the expiry condition written in. Two documented
  misreads in three weeks in opposite directions: 2026-07-29 read a
  market-wide collapse as "broadening"; 2026-08-07 read an advance as
  "narrowing" while IWM sat 0.7% off its own 52-week high. VIX and HY
  continue; chassis, ladder and stops untouched.
- **D-011 → hold the threshold, fix the actionability window.** The
  A+/B threshold is genuinely unresolved and no option except holding
  avoids acting on a difference 5B could not measure. But the one-day
  median A+ spell is a **timing** defect and it is measurable.
  Proposal: an A+ stays actionable for N sessions after it prints
  unless a row fails hard — gated behind a harness replay of S1 with
  N ∈ {1,3,5}, shipping only if entries rise without expectancy
  falling outside the random-K band.
- **Do not touch the stop, the ceiling ladder, or the close-basis
  law.** Every study points the same way: the machinery is the edge.
  Arm drawdown bands never overlapped SPY's under any seed — the one
  seed-independent positive in the whole evidence base.

**Counterweights carried in the brief:** suspending a throttle is a
loosening recommended on three weeks of live observation; the
actionability window is invented, not measured; the gate threshold
stays unresolved either way.

### CR-2026-08-09-3 — Advisory error record

| | |
|---|---|
| **Ruled** | 2026-08-09, in session |
| **Status** | Standing — append-only |
| **Binds** | Calibration: how much to weight advisory judgement relative to measured results |

A record of where the deliberation layer was wrong, kept because it
is itself evidence about how much to weight advisory judgement
relative to measured results.

| Date | Claim | Outcome |
|---|---|---|
| 2026-07-22 | "There is no surface for the position log" | False — it existed and was silently 404-ing |
| 2026-07-22 | "EXIT_FIRED — official" on Tuesday's close | The close run had computed nothing |
| 2026-07-23 | "Keep the corrections branch as built" | It had already been changed by review |
| 2026-08-01 | "auto_adjust returns raw volume — expect split steps" | Disproved empirically across six WFC splits; Yahoo split-adjusts upstream |
| 2026-08-03 | Ranked PFG first, advised skipping NOW | Over four sessions NOW +7.0%, PFG −0.9% — and the reasoning used was structural-quality, which Layer A had already flagged as unsupported |
| 2026-08-07 | Advised caution on GEN citing a falling 200-day | Same unsupported reasoning; GEN's grade was in fact clean, and the objection was the extension row, which [5.1](../backtest-ablation.md) later named as the expensive one |
| 2026-08-08 | Cited HPE's 26-point YTD divergence | Inflated — the two artifacts were baked on different dates; like-for-like the gap was 2.4 points |
| 2026-08-08 | Build 6A decision table | Incomplete — did not cover "one CI clears, the other straddles zero." T1/T2/T5 landed there and were correctly reported as UNCOVERED in the [6A report](../backtest-target.md) |
| 2026-08-09 | Spec instruction "set the plateau at the curve's peak" | Expanded the operator's ruling beyond "cap flat above 100%," lifting the (50,100] band by +4 on no evidence. Surfaced and confirmed rather than inherited — the explicit ruling lives in [D-020a](D-020a-ytd-anchor-cap.md) |

**Pattern worth noting:** the recurring failure mode is reasoning
from plausible mechanism ahead of measurement — precisely the error
the pre-registration apparatus exists to prevent, committed by the
party who designed it. Weight the systemic arguments in these
records; discount the discretionary ones.

### CR-2026-07-25-1 — Reference set: private, GitHub-rendered, .md + .svg source

| | |
|---|---|
| **Ruled** | 2026-07-25, in session |
| **Status** | In force — built partially (pages committed 2026-08-09 to [docs/reference/](../reference/); generator not built) |
| **Binds** | Deliberation-page publication: source format, rendering target, privacy boundary |

Deliberation pages ship to a private repo rather than the public
MarketPulse site. Source of truth is `.md` plus standalone `.svg`;
HTML is a **generated** artifact.

Reason: GitHub's markdown sanitiser strips `<style>`, inline styles
and `class` attributes, and skips inline `<svg>` entirely — so the
dark HTML house style cannot render there. Standalone SVGs referenced
as images do render, provided each carries its own internal
`<style>`, its own background rect, explicit coordinates
(`dominant-baseline` is stripped), and no external fonts or scripts.

`docs/reference/README.md` is the index; GitHub renders it on folder
browse.

**Maintenance law:** a decision change updates its page in the same
session.

**Explicitly not automatable:** diagrams. Each SVG is a design
decision made one page at a time. A generator can inline them; it
cannot invent them.

**Reversible:** porting the pages to the public site later is a file
copy plus an index link, because every page is self-contained by
construction. The gate on that move is privacy, not engineering — the
Render site is world-readable, so doctrine and position-bearing pages
would become public. Likely middle path: publish the generic
explainers, keep decision records and anything position-bearing
repo-only.

### CR-2026-07-25-2 — Second account: profiles, not tabs

| | |
|---|---|
| **Ruled** | 2026-07-25, in session |
| **Status** | In force (direction) — blocked on three open facts |
| **Binds** | Second-account architecture; the dynamic-targets work is designed account-aware because of this ruling |

Adding the MOC tax-free account does **not** get a new tab. Layers 1
and 2 (market read: regime chassis, universe, breakers, macro) stay
**global**. Layers 1.5 and 3 become **account-scoped** behind a
switcher, with an `accounts.yaml` entry per account carrying capital,
vehicle set, rule pack, execution style and restriction set.

Reason: everything above the account is identical across both books;
everything below it differs. A second tab duplicates the identical
half and crams the genuinely different half into a layout built for
stocks-and-grades.

**Blocked on three facts not yet supplied:**

1. Which account carries the 30-day re-entry restrictions?
2. Same capital pot as the existing rotation, or new money?
3. Does tax-free status change the rules (wash-sale, hold
   preferences)?

### CR-2026-07-25-3 — MOC advisory bot: green-lit with five deltas

| | |
|---|---|
| **Ruled** | 2026-07-25, in session |
| **Status** | In force — not built |
| **Binds** | The advisory bot's design contract (five required deltas; advisory-only boundary) |

Standalone repo and Render service. Noon and 3:30 ET weekday crons.
Fetches ~34 MarketPulse API endpoints, sends them plus the position
and `system_prompt.md` to the Claude API, posts a verdict to Slack,
logs to a Google Sheet and a JSONL ledger. **Advises only; never
places trades.**

Approved because it rides MarketPulse's API rather than forking the
data layer, and because it automates an existing manual process
rather than inventing a new one.

**Named honestly in the record:** the bot's decision engine is an LLM
reading a prompt — the opposite of the computed, pinned, replayable
direction the TOS side has been moving toward. Acceptable here
because it is advisory, parallel-run, and mirrors the current manual
loop. The mechanical rules migrate to computed account profiles
later, informed by this bot's divergence ledger.

**Five required deltas:**

1. **Data-completeness gate** — expected-vs-fetched per endpoint; a
   DEGRADED banner naming what is missing; `NO VERDICT` when a
   critical-set endpoint fails. The model must never receive
   silently-partial data and answer confidently.
2. **Ledger provenance** — every row records `system_prompt.md`
   sha256, the exact model string, and the per-endpoint fetch
   manifest.
3. **Position staleness** — if the Sheet's newest row is older than 3
   trading days, or a prior verdict recommended an action with no
   Sheet entry since, the verdict opens with a staleness warning.
4. **Forming-bar caveat** — the 3:30 verdict computes on an
   incomplete session by necessity; every 3:30 verdict says so, and
   close-phrased rules are evaluated with the forming bar labelled as
   such. The noon run is informational only.
5. **Separate Slack channel** — this account's voice never
   interleaves with SignalAgent's.

### CR-2026-07-25-4 — Standing rule: A+ entries come to deliberation

| | |
|---|---|
| **Ruled** | 2026-07-25, in session |
| **Status** | In force — live constraint on real money; expires when [D-011](D-011-aplus-doctrine.md) is re-ruled |
| **Binds** | Any live entry decision on an A+ name |

While D-011's revisit trigger is fired and unruled, **any live entry
decision on an A+ name is adjudicated in deliberation before the
order is placed**, with the Build 5 evidence on the table.

Reason: the doctrine that would authorise the entry is itself under
active reconsideration. Entering on the chip alone would be acting on
a rule the system's own evidence has called into question.

This rule expires when D-011 is re-ruled. It is not a permanent
feature.

**Exercised:** Jul 27 (CSX, NSC), Aug 3 (NTAP, NOW, PFG), Aug 7 (GEN,
and the six-name board). No entry was taken in any instance; in every
case the grade had lapsed before the fill window.

## Promoted entries (pointers only)

Two rulings from this period are already full decision records and
live there, not here: D-020a's rulings (real YTD anchoring, the
plateau cap — including the plateau-12 scope decision and the
newly-visible (50,100] taper logged as a never-measured construct)
are in [D-020a](D-020a-ytd-anchor-cap.md).
