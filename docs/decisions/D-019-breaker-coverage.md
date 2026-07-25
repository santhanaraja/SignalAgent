# D-019 — Breaker coverage: record coverage, not outcome

| | |
|---|---|
| **ID** | D-019 |
| **Date** | 2026-07-25 |
| **Status** | Ruled — implemented 2026-07-25 |

## The law

**A group breaker may report `clear` only when every check its
sensitivities call for actually ran.** `clear` is a positive claim about
having looked. A group that could not be fully checked is `degraded`, and
it names which checks did not run and why. A degraded group is never
byte-identical to a clear one — that identity *was* the defect.

A trigger still wins: something that fired is news regardless of what
else could not be measured.

## Context — the defect

Found by the swallowed-fetch remediation's verification fan-out
(2026-07-24) and deliberately deferred to its own build, because it is
the one remaining path that can fabricate a **tradeable** signal.

The chain, every link verified in code:

1. `run_engine` fetched the six `MACRO_TICKERS` (`^GSPC, GLD, UUP, USO,
   UNG, XLE`) one-shot and swallowed every failure with a printed
   `SKIP`.
2. `generate_dynamic_breaker_checks` **omitted** the affected check from
   its `checks` dict entirely — six sites, each an
   `if data is not None and len(...) > N:` with **no `else`**.
3. `check_thesis_breakers` builds its "not triggered" entries by
   iterating `checks.items()`, so an omitted check left **zero trace**
   in `breaker_alerts`.
4. `breaker_status` fell through to its `"clear"` initialiser and was
   written to `signals.json`.

The result was indistinguishable, byte for byte, from a group that was
checked and found healthy. `compute_trade_signal` lost its only macro
kill-switch (`critical → AVOID`, `warning` + net ≤ 0 → `AVOID`), and
after the 2026-07-24 remediation (`6c52aa0`) the search-page gate
**certified the fabricated clear as verified** — `status: "resolved"`
was granted on any non-empty `breaker_status` string, with no coverage
test.

The ambiguity was undetectable from the artifact: a check omitted by an
outage and a check that legitimately does not apply produced identical
records.

## The ruling

### 1. Fetch outcomes are recorded, never swallowed

`run_engine` builds `macro_status = {ticker: {ok, reason, bars}}` and
prints an explicit incomplete-coverage warning. The record travels with
the run and lands in the artifact as `macro_status`.

### 2. The check ↔ sensitivity map is an explicit engine-side table

`BREAKER_CHECK_SPECS` is the source of truth. **The map is not 1:1 and
must never be reconstructed by name-matching:**

| sensitivity | macro input | check id | conditional |
|---|---|---|---|
| `sp500_drawdown` | `^GSPC` | `sp500_drawdown_10pct` | — |
| `group_momentum` | *(none)* | `group_avg_rsi_below_40` | — |
| `group_trend` | *(none)* | `majority_below_ma50` | — |
| `group_ytd` | *(none)* | `avg_ytd_negative` | — |
| `breadth_collapse` | *(none)* | `breadth_collapse` | — |
| `commodity_drop` | `GLD` | `gold_below_threshold` | **only when `commodity_proxy == "GLD"`** |
| `usd_strength` | `UUP` | `usd_strength` | — |
| `oil_collapse` | `USO` | **`oil_below_60`** | — |
| `natgas_collapse` | `UNG` | `natgas_collapse` | — |
| `energy_spike` | `XLE` | `energy_spike` | — |

Four sensitivities need no macro input (they read the group's own member
rows) and therefore can never be degraded by a macro outage.

`commodity_drop` declared with a non-GLD proxy resolves to **no check**.
That is a configuration gap, not an outage — it degrades the group with
its own reason rather than quietly shrinking what "complete" means.

### 3. A trigger still wins — and the caveat rides along

`resolve_breaker_status(alerts, coverage)` is the **one** implementation
of the ladder (`run_engine` and the pins both call it). A fired breaker
outranks incomplete coverage: hiding a fired `critical` behind a
"degraded" label would lose the alarm. When both are true the serve layer
returns the trigger as the verdict and carries `coverage_incomplete` +
`degraded_reasons` alongside it.

### 4. The artifact is a surface too

`compute_trade_signal` returns **`SIGNAL WITHHELD`** for a degraded
breaker. Without this the fix would have been cosmetic: the dashboard
renders `trade_signal` straight out of `signals.json`, so every stock in
a degraded group would still have published an ungated `BUY NOW` — the
same fabricated-clear defect, one layer over. Changing the **value**
rather than adding a flag is deliberate: an older cached page renders an
unfamiliar label harmlessly, whereas it would ignore a new flag and print
the false `BUY NOW`.

### 5. A missing INPUT is not always a missing CHECK

`sp500_ytd` comes from a **separate** fetch (`get_index_data`) and used to
default to `0.0` on failure — so every "beating the S&P" comparison
measured against a fabricated baseline while `breadth_collapse` still
produced a confident number. Expected-vs-run cannot see this, because the
check *did* run. Degraded **inputs** are therefore reported directly
(`degraded_inputs`), and `sp500_baseline_ok` is recorded on the run.

### 6. Coverage is derived, never re-implemented

`breaker_coverage(group_info, checks_run, macro_status)` compares the
generator's **own output** (ground truth for what ran) against the
table (what was called for). The guards that decide computability are
never duplicated, so coverage cannot drift from them.

Per group the artifact gains `breaker_checks_expected`,
`breaker_checks_run`, `breaker_degraded_reasons`.

### 7. Surfaces

- **search.html** — a degraded breaker gates the trade signal exactly as
  an unavailable one does: `SIGNAL WITHHELD`, with a
  "GROUP BREAKER PARTIALLY UNVERIFIED" block naming the unrun checks.
- **framework.html Layer 2** — degraded renders amber with a ⚠ glyph and
  a dashed border, tooltipped with the reasons. Green is reserved for a
  positive `clear`.
- **Close report** — one line when any selected group's breaker is
  degraded, naming the groups and the unrun checks.
- **The serve layer does not trust the label over the record:** an
  artifact stamped `clear` whose `expected` ⊄ `run` still gates.

### 8. Era-awareness

Artifacts written before this ruling carry no coverage fields. They
render exactly as they did in the old world — resolved on `clear`, no
crash, and **no retro-claim** that coverage was verified when it was
never measured.

## Evidence — the pins

`test_breaker_coverage.py`, built to the three-law pin doctrine
([docs/testing.md](../testing.md)):

| Pin | What it demonstrates |
|---|---|
| (a) outage injection | Each macro ticker killed in turn: all 5 sensitive group/ticker combinations read `degraded` with the ticker and reason named; insensitive groups stay `clear` (no over-degradation) |
| (a′) **pre-fix comparison** | The engine at the pre-fix SHA `e81b6d1` (anchored to the COMMIT, never `HEAD` — see the Law-3 note in docs/testing.md) on the *same* injection reports `clear` with 3 checks and **no trace** of `sp500_drawdown_10pct`; the fixed engine reports `degraded`. Law 3: a pin that cannot fail on the old code is not evidence |
| (b) full coverage | A complete day still reads `clear`, `expected == run`, no reasons, alert list unchanged — the fix does not manufacture false degradation |
| (c) table complete | 10 specs, 10 declared sensitivities, zero orphans in either direction; checks are derived from the generator's source, so a new branch without a table entry fails here; the non-1:1 pairings are pinned by name |
| (d) gate + surfaces | Degraded gates; a stamped-clear-but-blind group gates; pre-D-019 artifacts read as the old world; search/Layer-2/close-report renders pinned |
| (e) **writer half** | Drives the real `run_engine` with a stubbed network and one dead macro ticker: asserts `macro_status`, the per-group coverage fields, the degraded verdict, and the withheld published signal are all actually written. Era-awareness makes a writer regression *invisible* — silently stop emitting the fields and every consumer reads the artifact as pre-D-019 and un-gates |
| (f) trigger wins | The ladder's headline rule, in `resolve_breaker_status` AND at the serve layer, with the caveat preserved |
| (g) gate behaviour | Executes the page's own gate predicate over every degraded shape; a fired critical still renders its `AVOID` |
| (h) degraded input | A check that ran on a fabricated baseline degrades; the artifact withholds its published signal; dashboard/leadership/close-report all carry it |

## Consequences

The last path that could fabricate a tradeable signal is closed. "Proven,
never presumed" now holds all the way down: the grade (D-011), the
signal chip (the swallowed-fetch remediation), and now the breaker state
those rest on.

Cost: a macro outage will now visibly degrade groups that previously
sailed through as clear. That is the point — but it means an
intermittent yfinance failure becomes user-visible where it used to be
silent. If that proves noisy, the fix is fetch resilience (retry), not
re-hiding the outage.

## Revisit triggers

1. Degraded appearing routinely from transient fetch failures — add
   retry/backoff to the macro fetch rather than relaxing the law.
2. A new breaker check added without a `BREAKER_CHECK_SPECS` entry (pin
   (c) fails loudly by design).
3. A sensitivity declared with a pairing that has no implemented check
   (currently only `commodity_drop` + non-GLD) — implement it or retire
   the sensitivity.

## Retest recipe

```
python3 test_breaker_coverage.py     # all five pins, incl. the pre-fix comparison
python3 test_serve_guard.py          # the gate, era-aware serve
```

## Links

- Predecessor: the swallowed-fetch remediation (`6c52aa0`, `e81b6d1`) —
  the gate this ruling makes honest.
- [docs/testing.md](../testing.md) — the pin doctrine this record's
  evidence set demonstrates.
