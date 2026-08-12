# D-022 — An unmeasurable market cap is not a small company

| | |
|---|---|
| **ID** | D-022 |
| **Date** | 2026-08-11 |
| **Status** | **Ruled** |
| **Takes effect** | Next rotation. Nothing changes before then. |
| **Shape** | D-019 (an absent input reading as a finding) |

## Context

`universe_builder._qualify` treated a market cap it could not FETCH
exactly like a market cap it had measured and found too small. Both
appended a bare failure to the same list, both fell through
`_gate_status` to the same published status `failed_mcap_gate`, and the
artifact legend read *"market cap below min_market_cap USD (or
unavailable)"* — the parenthesis doing the work of a distinction.

So a company whose size the build failed to read was published as a
company that is too small to trade. That is the D-019 shape exactly: an
absent input arriving at a consumer as a finding.

**Correction to the original wording of this record** (found while
auditing `fd83fa4` for D-023, and corrected here rather than left
standing): this said the name was dropped "with no distinguishable
record". That overstates it. What collapsed was the published `status`,
the legend entry and the dashboard's visible annotation — all three of
which said *small*. The `fails` string did NOT collapse: it read `mcap
unavailable`, already distinct from `mcap $X<$Y`, and the near-miss
strip rendered it in the hover tooltip. So the true defect was narrower
and more insidious than "no record": the artifact carried the truth in
one field while the label beside it, the legend explaining that label,
and the status a consumer would branch on all asserted the opposite.
A reader had to hover the right row to find the contradiction.

That surviving field is not a footnote — it is what makes pre-D-022
artifacts auditable after the fact, and it is how D-023's source
artifact was cleared (see that record's interaction note).

This was noticed while reviewing the ARWR inclusion — a reviewer's
re-run displaced APGE where the original run displaced AMGN, and the
difference was entirely cap-fetch luck — but it is independent of that
change and affects every rotation.

## Evidence

All measured 2026-08-11 against the live pool (550 candidates, 527 with
valid metrics). Harnesses in the session scratchpad; the retest recipe
below re-derives every number from the committed code.

**The fetch fails in production position, and not otherwise.**

| Run | Shape | Unresolved |
|---|---|---|
| Cold, cap fetch only | 534 tickers, 6 workers | **0 / 534** |
| Production position | after the ~550-ticker price download | **48 / 546** |

The cap fetch runs immediately after the price batch and inherits its
rate-limit budget. Every failure was `YFRateLimitError`.

**The failures are not random — they are a contiguous alphabetical
suffix.** `_fetch_market_caps` sorts its input; the 48 failures were
positions **498–545 of 546, with no gaps**. The names lost are not a
sample, they are the end of the alphabet. Two runs of the same rotation
minutes apart therefore disagree about the book for no reason connected
to any company.

**The existing retry was inert.** The single retry pass (2s, 2 workers),
whose docstring says a transient wave "must not become a mass 'mcap
unavailable' disqualification", recovered **0 of 48**. So did pauses of
3s, 8s and 15s. So did the `.info` endpoint (it shares the limiter). A
probe **3.5 minutes later** was still throttled — the limiter's window
outlasts anything a build can wait out.

**34 of the 48 would otherwise have qualified, and none of them was
small.** Every one had a true cap far above the $5B floor:

| | | | |
|---|---|---|---|
| WMT $896.6B | XOM $657.0B | UNH $371.2B | WFC $264.7B |
| VZ $195.4B | UNP $173.6B | WELL $169.4B | VRTX $132.8B |

…down to the smallest casualty, UHS at $10.2B — **double the floor**.
All 34 published as "market cap below min_market_cap USD".

**What it costs the book.** Same metrics, same prices, one shared fetch,
the pure ranker run twice (the counterfactual `rank_and_select` was
extracted for). With the observed failure applied:

| | All caps resolved | Observed failure |
|---|---|---|
| Selected tickers | 77 | 73 |
| Caps degraded | 0 | 46 |
| Blocked (would otherwise qualify) | 0 | 32 |

**Four holdings lost outright: UNH ($371B), VLO ($91B), VSAT ($12B),
ZBRA ($18B)** — across Managed Health Care, Oil & Gas Refining &
Marketing, Communications Equipment, and Electronic Equipment &
Instruments. Nothing was promoted in their place; the groups simply
carried fewer names.

**Neither existing floor sees any of this.**

- `POOL_RETENTION_FLOOR` (D-021) counts *candidates* and is evaluated at
  step 1b, before the price fetch. A wave that strips the cap off 48
  names leaves the candidate count identical. **It cannot move.**
- `MIN_MCAP_COVERAGE = 0.70` catches only a catastrophe. It licenses
  nearly a third of the universe — ~160 names — to be silently
  disqualified while the build reports success. In the measured run
  coverage was **91%** and it stayed quiet through all 34.

**No name in the pool is permanently unmeasurable.** The 12 names absent
from the first cold run were later fetched individually and all 12
resolved (ARM $286B, SHOP $200B, PDD $132B, MELI $93B, …). The healthy
state is genuinely 527/527, so the gate below has no standing deadlock.

## Options considered

| Option | Summary | Why (not) |
|---|---|---|
| A | Fail closed, silently (status quo) | The measured cost is 4 holdings and 34 mislabelled names per wave, with no trace in the status, the legend or the visible dashboard label — only in the raw `fails` string behind a hover tooltip. Rejected. |
| B | Hold the previous rotation's cap | Invents an input. A carried cap is a measurement we did not take, its age is unbounded (a stale artifact can be weeks old after a failed rotation chain), and it would paper over a *systematic alphabetical bias* with plausible-looking numbers. It is the D-019 defect wearing the opposite mask. Rejected. |
| C | Fail closed LOUDLY — distinct status + coverage record | Necessary, and adopted. Not sufficient alone: "loud" in a log nobody reads is silent for a system that runs on cron, and the book still narrows. |
| D | **C, plus refuse to rotate when the missing caps CHANGE the book** | **Ruled.** |

## Ruling + rationale

**An unmeasurable market cap fails closed, publishes as its own outcome,
and stops the rotation when — and only when — it changes what we would
trade.**

Three parts.

**1. The outcomes are separated.** `MCAP_UNMEASURED` is a distinct
`fails` token and `_gate_status` maps it to a distinct published status
`mcap_unavailable`, with its own legend entry. The name is still held
out — *an unverified size must not buy admission* — but it is never
again described as small. The dashboard annotates it "cap unavailable",
not "small cap".

**2. Coverage is recorded, following the `etf_coverage` idiom.**
`mcap_coverage` reports a per-ticker `outcome` (`fast_info` /
`fast_info_retry` / `info_fallback` / `fx_unconvertible` /
`unavailable`) plus a `degraded` count that is **zero in the healthy
state**, and it NAMES the blocked tickers rather than counting them.
`blocked` is the harm number: names whose *only* disqualifier was the
missing measurement.

**3. Materiality decides publication.** The question worth gating on is
not "how many caps are missing" but "did the missing caps change what we
would trade" — and that is answerable exactly. `mcap_materiality` runs
the pure ranker twice over one shared fetch, granting every unusable cap
exactly the floor in the optimistic pass. If the selected book differs,
the rotation raises `UniverseBuildError` and the previous, committed,
known-good universe stays authoritative — the same failure posture
D-021 established for a degraded pool.

Why materiality rather than a coverage threshold: an unmeasured cap on a
name that was never going to be selected costs nothing, and stopping the
week for it would make the rotation fragile for no gain. In the measured
run, 46 caps were missing and 32 names blocked, but only **4** actually
moved the book. A count cannot tell those apart. The counterfactual can.

A false alarm here is the safe side. "The missing caps might have
changed the book" and "they did" are the same state to a system that
cannot tell them apart, and both should stop a rotation rather than
publish a guess.

**Why not carry caps forward (option B), stated plainly:** because the
failure is a *systematic* alphabetical bias, not noise. Carrying stale
caps would let a rotation proceed on a book partly decided by where a
rate limiter started, while looking fully measured. The system already
knows how to fail safely; it should use that instead of inventing data.

## Consequences

- **A rotation can now refuse on cap coverage.** The pipeline degrades
  gracefully: `get_active_industry_groups` catches the error and serves
  the previous universe, so nothing crashes and nothing goes untraded.
  Re-running normally clears it — the limiter resets.
- **A permanently unmeasurable name would block its rotation** until
  handled. The escape hatch is deliberate and on the record:
  `universe.source.manual_exclusions` in `framework/config.yaml`, named
  in the error message. No such name exists in the pool today (all 527
  resolve), but the deadlock path is real and this is its exit.
- **Accepted debt: the rotation can now be refused by an upstream rate
  limiter.** We prefer a refused rotation to a silently wrong one, but
  this trades a silent failure for a visible one that needs a human when
  it recurs. If waves become frequent the answer is to stop tripping the
  limiter (pace or reorder the fetch), not to relax this gate.
- **The two cap paths in the repo now agree in coverage.** Each of
  `universe_builder._fetch_market_caps` and
  `signal_engine.fetch_fundamentals_yfinance` (4e5eec4) tries both
  `fast_info` and `.info`, and each records which endpoint answered.
  They still *prefer* opposite endpoints, which is fine; what mattered
  was that each had a blind spot the other covered.
- **Bounded fallback:** `.info` is skipped above
  `MCAP_INFO_FALLBACK_MAX = 25` unresolved. Above that it is a throttled
  session, not an omitted field, and `.info` shares the limiter — it
  would cost minutes to rescue nothing (measured: 0 of 48).
- **`mcap_coverage` is absent on artifacts baked before this record.**
  Those read as `null`, not as healthy. They are not retro-relabelled.

## Revisit triggers

- **A rotation refuses on materiality more than twice in a row.** That
  is no longer a transient wave; the fetch ordering itself needs work
  (pace the cap fetch, or move it off the price fetch's budget).
- **A name appears in `mcap_coverage.unavailable_tickers` in three
  consecutive rotations.** It is permanently unmeasurable via yfinance;
  rule on excluding it explicitly rather than letting it gate the week.
- **`MIN_MCAP_COVERAGE` fires.** It has never fired; if it does, the
  wave is large enough that the materiality test is downstream of a
  bigger problem.
- **A cap source other than yfinance becomes available.** The whole
  class of failure here is one vendor's rate limiter.
- **Any record reconstructs caps from a committed artifact.** See the
  cross-record note below; the precondition is
  `mcap_coverage.degraded == 0` on that artifact, and pre-D-022
  artifacts must be audited through their `fails` strings instead.

## Cross-record: D-023 reconstructs caps, and the two chips interact

[D-023](D-023-classification-aliases.md) was built in parallel with this
record, by a separate session, and its live cap fetch was blocked by the
same rate limiter measured here. It therefore reconstructs caps from
committed artifacts and, where the artifact only proves a name cleared
the older $500M floor, assigns a **$5B sentinel** — ten of its 78
selected tickers rest on one.

**A sentinel resolves an unknown cap in the opposite direction to this
record's gate.** The sentinel passes by construction; the live gate
fails an unmeasured cap **closed**. Structurally the sentinel arm is
this record's *optimistic* pass — the same object `mcap_materiality`
builds — so on a wave-affected week a reconstruction and a real rotation
would diverge by exactly the set that function reports, in the admitting
direction.

**Checked for that week: D-023 does not inherit this defect.** Its
source artifact `fd83fa4` records zero unmeasured caps (389/389
qualifiers carry one; the only mcap disqualification in the whole
artifact is ARQQ's measured `mcap $399M<$500M`), and all ten
sentinel-backed names are independently confirmed far above the floor.
The conclusion stands; the recipe carries the precondition.

Recorded because a dependency between two records written at the same
time is exactly the kind that neither notices.

## Retest recipe

```
python3 test_pool_builder.py
```

Pins 10–12 (`test_unmeasurable_mcap_is_not_a_small_company`,
`test_unmeasured_caps_that_change_the_book_refuse_to_rotate`,
`test_cap_fetch_retries_and_falls_back_to_the_other_endpoint`) assert
the distinction, the coverage contract, the gate in both directions, and
the fetch ladder. Seven mutations were checked against them, each
turning the suite red: reverting `_gate_status` to fall through;
reverting the reason token to size-verdict shape; a materiality gate
that always says no; a materiality gate that always says yes; a coverage
record that never reports degraded; an unmeasured cap failing OPEN; and
the dashboard calling it "small cap" again.

To re-derive the live measurement (needs network; the numbers move with
the pool and with how throttled Yahoo is):

```
python3 universe_builder.py --dry-run
```

Read `[builder] retrying market cap for N tickers` and the
`*** MARKET CAP UNMEASURABLE` line. The cold-versus-production contrast
requires running `_fetch_market_caps` alone against the same ticker list
and comparing — that harness is not committed, and this record says so
rather than implying the contrast re-runs from the repo.

## Links

- Records: [D-019](D-019-breaker-coverage.md) (the shape),
  [D-021](D-021-pool-builder-v2.md) (`POOL_RETENTION_FLOOR`, the same
  refuse-to-rotate posture)
- Commits: `4e5eec4` (the `.info`/`fast_info` cap fallback this mirrors)
- Code: `universe_builder.py` — `_fetch_market_caps`, `_qualify`,
  `_gate_status`, `mcap_coverage`, `mcap_materiality`,
  `mcap_blocked_tickers`
- Pins: `test_pool_builder.py` §10–12
