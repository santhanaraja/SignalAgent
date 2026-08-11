# D-021 — Pool builder v2: a third index source, an inclusion list, a $5B floor, and a versioned pool

| | |
|---|---|
| **ID** | D-021 |
| **Date** | 2026-08-10 (ruled by the user's build order; implemented same day) |
| **Status** | **Ruled** |
| **Takes effect** | Next rotation (Saturday). Nothing changes before then. |
| **Pool version** | `v2-2026-08-10`, definition hash `5e0cd19756618550` |

## What the pool is now

```
pool = ETF top-holdings  ∪  S&P 500  ∪  Nasdaq-100  ∪  inclusions  −  exclusions
```

Six changes, taken as one doctrine change rather than six config edits,
because together they redefine what the system is allowed to trade.

---

## 1 — Nasdaq-100 as a third source, from a committed file

`data/pool/nasdaq100.json`, 102 symbols, as of 2026-08-10, retrieved
from `https://api.nasdaq.com/api/quote/list-type/nasdaq100` — the index
publisher itself, not a third-party mirror. The count exceeds 100
because the index admits multiple share classes of one company: `GOOG`
and `GOOGL` are both members.

**There is no live scrape, and the absence is enforced rather than
asserted.** `universe_source._load_nasdaq100()` reads the committed
file; a pin cuts the network and proves the load still succeeds, then
demonstrates that a variant which fetches fails under the same cut.

The list moves only when a human runs `scripts/refresh_nasdaq100.py`,
reads the membership diff it prints, and commits it — at reconstitution
(annual, December) or on an announced ad-hoc change. The reason is not
tidiness: a live scrape hands a third party the ability to change what
the system can trade, silently, between two rotations, through a parser
with no way to tell a reconstitution from an outage.

A missing or malformed list **raises** and fails the rotation. It does
not degrade to two sources. A source that silently contributes nothing
shrinks the tradeable universe without anyone seeing it happen; the
previous universe is committed and known-good, and staying there is the
safe failure.

**On the circulating list being unreliable** — the caution was
warranted. A source claiming TER, NXPI, DASH and DDOG sit outside the
S&P 500 is simply wrong: all four are S&P 500 constituents *and*
Nasdaq-100 members, and all four were already in the pool. Dual
membership is ordinary and is exactly what the dedupe handles.

## 2 — An inclusion list, seeded empty

`data/pool/inclusions.json`: one object per ticker with `ticker`,
`reason` (one line), `added` (ISO date). **Seeded empty.**

> **AMENDED 2026-08-11 — the list is no longer empty, and its scope is
> wider than this section states.** Its first entry is `ARWR`, and the
> rule was widened in the file's own `_doc.why` from *new listings* to
> **"a legitimate large-cap that no index source carries."** New
> listings are the most obvious instance of that rule, not the
> definition of it: ARWR has been public since 2004, is $12.5B, and is
> simply absent from the S&P 500, the Nasdaq-100, and the ~10-deep
> top-holdings slice the ETF feed returns. Scoping the list to new
> listings would have left that larger population unreachable for no
> principled reason, and would have made the mechanism's first entry
> teach the wrong pattern. Pool version **v3-2026-08-11**. The entry
> decision, its impact and its consequences are recorded in the
> commit that added it.

Its purpose is new listings. The index sources lag: a company that
lists today joins no index for months, and until it does the system
cannot see it at all — not to reject it, to *see* it.

**Nothing is waived.** An included name enters the CANDIDATE set and
then faces every gate unchanged: score, history, volume, market cap,
group membership, group rank. Inclusion buys consideration, never
selection. An included name with insufficient history produces **no
ranking row and no error** until it accumulates bars — that is the gate
working, not a failure to fix. A pin covers all three phases: under 63
bars (no metrics at all), 40 bars against the 90-day gate (recorded as
`failed_history_gate`, never a qualifier, group unaffected), and 120
bars (qualifies normally). The gate delays; it does not reject.

**PRECEDENCE: EXCLUSION WINS.** A ticker on both lists is EXCLUDED.
This is structural, not a rule anyone has to remember: exclusions are
subtracted from the *union of every source*, once, at the end. There is
no ordering of the sources by which an inclusion can re-admit an
excluded name. The pin demonstrates the failure directly — the
plausible refactor `((etf|sp|ndx|add) − excl) | inc` re-admits the
excluded ticker, and the pin catches it.

### SPCX: excluded, then reversed the same day — SPCX is NOT excluded

SPCX was not added to the inclusion list. But it is a **Nasdaq-100
member**, so under item 1 it enters the pool through the index source
regardless. It was briefly added to `manual_exclusions`, and **that
exclusion was reversed before it ever reached a rotation.** The
reasoning is kept because the reversal is the more useful record.

**This was the one place the implementation went beyond the literal
instruction** — "do not add SPCX" was about the inclusion list;
excluding it is a different act, and it was the wrong one.

**Two of the three stated reasons did not survive verification:**

| stated reason | verdict |
|---|---|
| ~40 bars against a 90-day history gate | **holds** — 40 bars, first 2026-06-12 |
| a tracking variant, not an operating company | **false** — resolves to an operating company, `quoteType: EQUITY`, exchange NMS |
| $1.75T cap irreconcilable with a $132.87 share price | **loosely stated, and immaterial** — the inconsistency is cap vs *share count* (reported 7.70B shares × price = $1.02T), and the reconcilable figure clears a $5B floor by ~200× either way, so the cap datum argues nothing about the gate |

The surviving reason is **self-expiring**: the history gate stops
holding SPCX out around **2026-09-10**. A reason that lapses on a known
date is a reason to wait, not a reason to bar a name by hand.

**And `HONA` decided it.** HONA is structurally identical — 39 bars,
first listed 2026-06-15, arriving from the same Nasdaq-100 list, and
landing in the *same GICS group* (Aerospace & Defense) — and was never
excluded. One newly listed name barred by hand while its twin passes
through the gates is not a doctrine, it is an inconsistency. Both now go
to the gates, which is what the gates are for.

**Measured effect of the reversal**: pool 548 → **549**. SPCX has 40
bars, so it produces no metrics, contributes nothing to any composite,
and appears only as a `no_valid_data` row in the audit table. Its group
holds 18 members and is not selected. **Zero change to the selected set
or to any qualifier list** — which is exactly the outcome the inclusion
doctrine in this same item predicts for a name that is too new.

## 3 — Market-cap gate: $500M → $5B

`universe.rotation.min_market_cap: 5000000000`. The default in
`universe_builder.DEFAULT_ROTATION` was raised in step, so a missing
config key cannot silently reinstate the looser floor.

**The anomaly, explained: the gate is enforced.** The pool member
sitting below the *old* $500M floor is `ARQQ` at $0.39B, and it carries
`status: failed_mcap_gate`. Nothing slipped through and nothing decayed
past a gate applied only at build time.

The confusion is in what the ranking artifact *is*. It is the full
candidate **audit table**: a row means "this name was considered", not
"this name passed". Every candidate appears with the gate it failed,
which is the transparency working. The two are easy to mistake for one
another precisely because the artifact is complete.

## 4 — The ETF holdings caches are committed

`data/universe_cache/etf_*.json` are now committed;
`sp500.json`, `gics_cache.json` and `subindustry_sectors.json` stay
ignored.

**Committing the files alone would have been worse than leaving them
out.** The caches carry a 7-day TTL and the rotation runs weekly on a
fresh Actions checkout, which finds them expired, refetches, overwrites
— and the workflow previously committed only `data/universe_active.json`
and the two `public/` artifacts, so the refetched caches were discarded.
A committed copy that the build never wrote back would be permanently
stale while still *reading* as evidence of what was sourced. So
`.github/workflows/rotate-universe.yml` now stages `data/universe_cache/`
in both its change-detection and its commit step. The committed caches
are the ones the rotation actually used.

After this change two of the three index sources are committed evidence;
only the S&P 500 list is still fetched at build time with a TTL cache
and a stale fallback. **Closing that remaining half is not done here**
and is offered as a follow-up, not assumed.

### Two regressions this item introduced, found in review and fixed

Committing the caches had consequences that committing them *correctly*
did not remove. Both were found by adversarial review, not by the
implementer, and both are recorded because the mechanism is instructive:
**making a file authoritative changes the behaviour of every path that
reads it.**

**(a) A cache-vs-cadence off-by-one.** `universe_ttl_days` was 7 and the
rotation cron is weekly. The TTL check expires only on age *strictly
greater* than the limit, so the cache and the cadence were the same
length: whenever a run started marginally faster than the previous one
(GitHub queue delay and pip install both vary by minutes), the cache
read fresh and the ETF fetch was skipped for that week. Before the
caches were committed this was **unreachable** — a fresh CI checkout had
no cache to hit — so committing them is precisely what made an existing
off-by-one live. Verified against the real code: a cache one second
short of seven days reads *fresh* at ttl=7. **Fixed: `universe_ttl_days:
6`**, pinned.

**(b) A silent ETF outage — a direct D-019 violation.** On fetch
failure, `_fetch_etf_holdings` falls back to the stale cache. Before,
a fresh CI checkout had no cache, so an outage returned `[]` and
collapsed `by_source.etf_holdings` from ~150 to 0 — loud, and visible
in the published artifact. With the caches committed the stale copy is
*always* present, so the build would sail through on last week's
holdings and be **indistinguishable from a healthy one**. The viability
floor would not catch it: the pool still carries ~500 S&P names.

That is the outage impersonating safety, which D-019 forbids. **Fixed by
recording COVERAGE, not outcome**: `etf_coverage` now carries a
per-ETF outcome (`fresh` / `cache_hit` / `stale_fallback` /
`unavailable` / `empty_source`, with the stale age in days) plus an
`etf_coverage_summary` whose `degraded` count is zero in the healthy
state. The pin demonstrates the point directly — the two paths return
byte-identical holdings, and the coverage record is the *only* thing
that tells them apart.

Related, same cause: an empty holdings response was previously written
to cache as a success, with a fresh timestamp. For GLD and IBIT that is
correct (they hold no equities); for a fund with holdings on record it
is a broken scrape, and committing the caches would have entered "SPY
had no holdings on 2026-08-10" into the repo as evidence. An empty
response over a non-empty cache is now treated as a failure: the stale
holdings are served, the file is not overwritten, and the timestamp is
not re-stamped.

### (c) Recording coverage is not deciding — the pool retention floor

**Ruled 2026-08-10.** Coverage makes a degraded build *visible*; it does
not stop one. Visibility only helps a reader who looks, and a rotation
runs unattended at 01:07 on a Saturday. So the rotation now **refuses**:

> If the assembled candidate pool is **more than 10% smaller** than the
> previous rotation's, raise and do not write.
> (`POOL_RETENTION_FLOOR = 0.90`, checked at step 1b.)

**It triggers on pool shrink, not on a source count**, deliberately.
What matters is how much of the tradeable population survived, and that
is the one number every degradation path has to move — a failed feed, a
truncated response, a silently emptied cache, a bad list edit. A
source-count trigger would miss a source that returns *some* holdings,
which is the more common failure than one that returns none.

Failing is the safe side, and the viability floor already established
why: **a rotation that refuses to write leaves the previous, committed,
known-good universe authoritative.** Building on partial holdings does
not. The asymmetry is the whole argument — one path costs a week of
staleness, the other silently narrows what the system may buy.

The floor catches what nothing else can see. The viability floor is 3
groups / 8 tickers; a pool cut by a **third** still assembles 15 groups
and ~70 tickers, so every existing check stays silent. Pinned, including
that case explicitly, and pinned to run **before** the ~550-ticker price
fetch so a degraded pool fails in seconds. A missing baseline (first
build, wiped cache) never blocks — the floor guards against a pool that
quietly narrowed, not against not knowing.

## 5 — `data/universe_ranking.json`: the premise was wrong

**Reported first, as instructed: nothing in production reads it.** The
only reader anywhere is `scripts/d020a_impact.py`, a completed study.
`ticker_api.py` serves `PUBLIC_DIR`; `scripts/backtest_systems.py` reads
`public/universe_ranking.json` from a pinned commit; the rotation
workflow commits only the `public/` copy. **Hygiene, not a live
defect** — the condition set for calling it a defect was not met.

**It was never tracked.** Not in `HEAD`, no history, not in the index,
ignored at `.gitignore:17`. It is a local build byproduct, frozen at
2026-07-25 because that is when this machine last ran a local rotation;
Actions rotations write only `public/`. The observation was right, the
diagnosis was not — there was no tracked-and-ignored contradiction to
resolve.

So: **properly ignored**, made explicit in `.gitignore` with the reason,
and the stale local file deleted (it regenerates on the next local
build). The `data/` copies of the ranking and candidates artifacts are
documented there as never authoritative.

The real defect the item uncovered was in the reader. A committed study
was reading a gitignored, local-only file — the Build 7 mutable-input
defect in a worse form, since a reviewer on a fresh clone cannot
reproduce it *at all* and one on a stale clone reproduces it against the
wrong population without being told. `scripts/d020a_impact.py` is now
pinned to `5214cfc` (2026-07-24 rotation, 533 names, verified
ticker-for-ticker identical to the local copy its committed results came
from) and raises rather than falling back to the working tree.

## 6 — The pool is versioned

Every pool-derived artifact — `data/universe_active.json`,
`data|public/universe_ranking.json`, `universe_candidates.json` — now
carries `pool_version`, `pool_definition_hash`, `pool_sources`,
`nasdaq100_as_of`, `pool_exclusions`, `pool_inclusions`.

Two identifiers, because each covers the other's blind spot. The
declared `POOL_VERSION` is meaningful but can be forgotten; the
`pool_definition_hash` is a fingerprint of the actual source definition
and moves on any deliberate edit **even when the version string is not
bumped** (pinned).

The hash describes the pool's *definition*, not its *outcome*: the ETF
tickers sourced from, not the holdings they returned this week. Holdings
churn weekly; hashing them would move the fingerprint every rotation and
tell you nothing about whether the pool was redefined.

Gates are deliberately **not** in the hash — they filter the pool, they
do not constitute it, and `rotation_config` already records them
verbatim in the same artifact. Pool version answers *which names could
have been considered*; `rotation_config` answers *what they had to
clear*. Together they fully determine a ranking. (Consequence worth
stating: item 3's gate change does **not** move the pool hash. It moves
`rotation_config`.)

The stamp is **carried by the artifact, never recomputed at read time** —
pinned, because recomputing would brand a historical ranking with
today's pool, which is the exact confusion versioning exists to prevent.
Pre-v2 artifacts report `None` rather than being retro-labelled, the
same treatment `scorer_version` gets under D-020a.

---

## Impact, measured before staging

Three arms over **one shared price+cap fetch**, all calling the
production `rank_and_select`:

| arm | pool | gate | groups | tickers |
|---|---|---|---|---|
| **A** | v1 (535) | $500M | 15 | 74 |
| **B** | v2 (548) | $500M | 15 | 78 |
| **C** | v2 (548) | $5B | 15 | 78 |

One fetch, not three runs: the market was open, and two separate live
builds would have smeared intraday movement across the comparison as if
it were an effect of the change. `rank_and_select` was extracted from
`build_active_universe` for exactly this reason — so the counterfactual
runs the same code the rotation runs.

**Fidelity check first, as required.** Re-applying the selection rule to
the committed 2026-08-07 artifact reproduces its 15 selected groups
**exactly**. The rule that had to be right: `min_candidates` gates a
group's MEMBERSHIP count, and eligibility needs only ONE qualifier. The
misreading is not academic — on committed data it wrongly drops
**Construction & Engineering (4 members, 2 qualifiers)**.

### The three population effects, attributed separately

**NDX adds: 13 net-new candidates** after dedupe against the S&P 500 ∪
ETF union (14 before SPCX is excluded) —
ALAB, ALNY, ARM, CCEP, CRWV, FER, MELI, MSTR, NBIS, PDD, RKLB, SHOP, TRI.

**Inclusions add: 0.** The list is empty by instruction.

**The $5B gate removes: 2 pool names**, `ARQQ` ($0.39B) and `HQ`
($0.67B) — both below $1B, as expected. Only one, `HQ` (score 61,
Systems Software), was a qualifier at all; `ARQQ` already failed the old
$500M floor. **Zero names leave the active universe**, confirmed: arms B
and C are identical in every selected group and every ticker.

**But "zero removed" hides a change in margin, and it reaches a live
holding.** Nothing sits between $1B and $5B today, which is why the
headline is zero — the pool is large-cap by construction (S&P 500 plus
ETF *top-25* holdings). What moved is the distance to the floor:

| | cap | × old $500M floor | × new $5B floor |
|---|---|---|---|
| ORKA (Biotechnology) | $6.18B | 12.4× | **1.24×** |
| **CGON — LIVE HOLDING** | $6.65B | 13.3× | **1.33×** |
| PTGX (Biotechnology) | $9.63B | 19.3× | 1.93× |

Only 2 of the 78 selected tickers sit within $7B, but one of them is
held. A ~25% drawdown in CGON now drops it through the market-cap gate,
where before it had an order of magnitude of headroom. This does not
force a sale — the position ladder is independent of universe
membership — but the name would leave the tradeable universe on a move
that previously would not have touched it. **This is a real change in
behaviour under stress that the "zero removed" number conceals**, and it
is stated here rather than left for someone to discover during a
drawdown.

### Reachable vs merely present

Of the 13 net-new names, **10 clear all four gates** — but almost none
are reachable. **This count is a per-snapshot fact, not a property of the
change**, and it moved within the session that measured it (see the
correction below):

| | count | names |
|---|---|---|
| present in pool | 13 | |
| clear all four gates | 10 | ALAB, ARM, CCEP, CRWV, FER, NBIS, PDD, RKLB, SHOP, TRI |
| **REACHABLE** | **1** | **ALAB** (Semiconductor Materials & Equipment, position 3) |
| qualified, unreachable | 9 | |
| fail a gate | 3 | ALNY (score 26), MELI (42), MSTR (45) |

Two of the nine miss on the **per-group 7-slot cap**, not on group
selection: **ARM** is position 10 in Semiconductors and **CRWV**
position 8 in Systems Software — both in *selected* groups, both
structurally unreachable anyway. The other seven sit in groups that are
not selected.

**CORRECTION — this did not survive the same trading day.** The arms ran
at 14:45 ET on a forming bar. Re-run on Monday's *closed* bars, CRWV
overtakes MSFT by one score point and takes slot 7 in Systems Software.
So on close data the change adds **two** names, ALAB **and CRWV**, and
**removes MSFT** from the tradeable universe. The rotation runs Friday
evening on closed bars, so the close figure is the one that matters.

Two further caveats on this table, both of which weaken it:

- **ALAB's group is a hardcoded classification, not a GICS lookup.** It
  is absent from the GICS cache and resolves through the static legacy
  map at `signal_engine.py:100`, which hand-places it among capital-
  equipment makers in *Semiconductor Materials & Equipment*. Astera Labs
  is a fabless connectivity-semiconductor company — GICS *Semiconductors*
  — where it would sit beside ARM (both score 67) around position 10 of
  14, i.e. unreachable. **The change's clearest single win rests on a
  hardcoded list.**
- **FOUR of the 13 land in phantom groups, and are structurally
  unreachable — so the reachability figure above is OPTIMISTIC.** MELI
  and PDD classify as "Internet Retail", CCEP as "Beverages - Non -
  Alcoholic", TRI as "Specialty Business Services" — Yahoo labels with
  no entry in the 21-line `gics_aliases` map. Measured: those groups
  contain **only those names and nothing else** — `Internet Retail` =
  {MELI, PDD}, `Beverages - Non - Alcoholic` = {CCEP}, `Specialty
  Business Services` = {TRI} — so each is below `min_candidates=3` and
  carries `exclusion_reason: below_min_candidates`. They are quarantined
  from the canonical groups (Broadline Retail, Soft Drinks &
  Non-alcoholic Beverages, Research & Consulting Services) where their
  real peers sit and where they would actually compete.

  This matters for how the table reads. Describing those four as "in
  groups that are not selected" implies they could become reachable if
  their group rose. **They cannot rise.** A one-member group can never
  be eligible at any composite. Their unreachability is a property of an
  incomplete alias map, not of the market — and the alias map is
  **deliberately left alone here** (raised as its own task), because
  repairing classification inside a pool-definition change would mix two
  populations in one diff.

So the honest statement is: **the practical yield is one to two new
tradeable names, on a boundary that moves intraday, and the count is
partly an artifact of classification rather than of the pool.**

### The decisive question: yes, the selected 15 changes

**Construction & Engineering leaves. Pharmaceuticals enters.**

The mechanism is not what it looks like. **No new name was selected into
either group.** `FER` (Ferrovial) joins Construction & Engineering as a
**5th member** — and because a group's composite is a weighted **median
over ALL VALID MEMBERS**, not over its qualifiers, one weak member moves
it:

| | arm A | arm C |
|---|---|---|
| Construction & Engineering composite | **+19.76** | **+13.89** |
| overall rank | 25 | 48 |
| members | 4 | 5 (+FER) |

That drop pushes it out of the top 15 eligible, and Pharmaceuticals —
16th in arm A at +18.27, unchanged in absolute terms — inherits the slot
with `LLY, BMY, PFE, JNJ, MRK`.

**This is the finding that matters** — but its cause is not what it
looks like, and the first version of this record got it wrong.

**CORRECTION: FER's own returns never enter the number.** A group's
composite is a weighted median, and the median of 4 values is the
midpoint of the middle two while the median of 5 is the middle one.
FER's YTD (+2.28) is below the second order statistic, so adding it
collapses the YTD median from `(33.91+57.54)/2 = 45.72` to `33.91` — and
`33.91` is EME's number, not FER's. The −5.91 YTD contribution is
exactly half the EME↔PWR spread times the 0.50 weight. Proof: a
hypothetical 5th member with returns of −999 gives composite 12.65, and
one with +999 gives 26.06 — the entire attainable range from adding
*any* 5th name. Every net-new ticker with YTD below 33.91 produces the
identical result.

So this is a **median-parity discontinuity**, not an effect of the
Nasdaq-100 or of Ferrovial. It fires on every odd↔even membership
transition in a thin group — from an index addition, a delisting, or a
single failed price download. Saying "FER dragged the composite down"
dresses a generic estimator artifact as a property of the change.

The real lesson stands and is broader than the change: **a thin group's
rank is unstable under any membership change at all**, and the selection
rule inherits that instability. A group can be evicted by a name that is
never traded, and by one whose numbers are never even read.

### ⚠ THE SELECTED-SET DIFF IS ONE DRAW, NOT A PREDICTION

**Read this before treating "Construction & Engineering leaves,
Pharmaceuticals enters" as a forecast of what Saturday will select.**

The swap is decided by a composite gap of **0.43** (Semiconductors
+18.70 at slot 15 versus Pharmaceuticals +18.27 at slot 16 in the
baseline arm). The **median one-day move in a group's composite is
1.46** — more than three times the gap that decides the outcome.

The deciding margin sits *well inside* ordinary daily noise. So this
diff is **one draw from a distribution**, not a prediction. A rotation
run on a different day would plausibly produce a different single swap,
or none. Corroborating this from the same measurements: re-running the
same v1 pool on today's data instead of the committed 2026-08-07 data
already moves **2 of 15 groups** — the change's own effect is **1** swap,
i.e. *smaller than three days of doing nothing.*

What is durable here is not which group swapped. It is the mechanism:
**the selection boundary is decided by margins narrower than daily
noise, in groups thin enough that one member flips the estimator.** That
property was already true; this change did not create it and does not
depend on it. Any future statement of the form "this change adds/removes
group X" carries the same caveat unless the margin exceeds ~1.5.

### ⚠ CGON — a live holding now sits 1.33× above the floor

Raising the market-cap gate to $5B removed **zero** names from the
active universe, and that headline hides a real change in **margin**.

`CGON` is a **live holding** at **$6.64B — 1.33× the new floor**. A
drawdown of roughly **25%** (precisely 24.7%) **marks it
thesis-failed**: it would fail the market-cap gate, stop qualifying, and
leave the tradeable universe while still being held. Before this change
the same name sat at 13× the floor and no drawdown short of collapse
could do that.

It is not alone in the band. Measured in the same cap fetch as the
impact arms:

| name | cap | × floor | drawdown that fails the gate |
|---|---|---|---|
| **CGON** *(held)* | $6.64B | 1.33× | **−25%** |
| ORKA | $6.17B | 1.23× | −19% |
| DNTH | $6.09B | 1.22× | −18% |

All three are in **Biotechnology** — the group holding *both* live
biotech positions. A sector drawdown does not move them independently.
For contrast, the other three holdings sit at 5.5× (HPQ), 12.3× (MET)
and 21.7× (NET), where the gate is not a live consideration.

This is the intended consequence of a higher floor, not a defect: a $5B
gate means names near $5B are near the gate. It is recorded because
"zero names affected today" is a statement about today, and the
position most exposed to it is one the book already owns.

### No live position is affected

All five live holdings sit in groups that survive A → C:

| holding | group | in A | in C |
|---|---|---|---|
| HPQ | Technology Hardware, Storage & Peripherals | ✓ | ✓ |
| NET | Systems Software | ✓ | ✓ |
| MET | Life & Health Insurance | ✓ | ✓ |
| BIIB | Biotechnology | ✓ | ✓ |
| CGON | Biotechnology | ✓ | ✓ |

Construction & Engineering's tickers in arm A were `J` and `PWR` —
neither is held. **Biotechnology survives**, so BIIB and CGON at
`weeks_in_universe` 1 are not marked thesis-failed by this change.

### Caveat, stated plainly

Arm A is a **today-data baseline, not the committed universe**. Run
against the same v1 pool definition, today's data already differs from
the committed 2026-08-07 rotation by **2 of 15 groups** (Health Care
REITs and Rail Transportation out; the two Oil & Gas groups in) and 74
tickers vs 67. The change's own effect is **1 group swap**.

**Quantified, this is the most important caveat in the record.** One
trading day of drift, pool held constant, swaps **2** groups. The change
under test swaps **1**. And the margin it turns on is tiny:

| arm | 15th eligible | 16th eligible | gap |
|---|---|---|---|
| A | Semiconductors 18.70 | Pharmaceuticals 18.27 | **0.43** |
| C | Pharmaceuticals 18.27 | Rail Transportation 17.61 | 0.66 |

The one-day distribution of `|Δcomposite|` across 130 shared groups has
median **1.46** and p90 **3.73**. **85% of groups move more than the
0.43 gap that decides this entire conclusion in a single day.** So the
attribution is sound — the arms share one fetch, so the swap *is* caused
by the change — but the effect is **half the size of one day's churn and
sits inside the noise band**. "The selected 15 changes" is true of this
snapshot; it is not a stable property. A replay across N recent closes,
reporting how often the swap survives, is what would settle it, and was
not run.

Three candidates failed to produce metrics in the shared fetch: `FDXF`
and `HONA` (new listings, under 63 bars — correct behaviour) and the
single unclassified candidate. **An earlier version of this record named
`GLW` here; that was wrong** — GLW priced fine and qualifies, and its
group (Electronic Components) has composite 37.56 identically in both
arms, so it cannot bear on the swap.

---

## Pins

`test_pool_builder.py`, 30 assertions across eight invariants, each
demonstrating the failure it guards. The five required, plus three added
in response to the regressions review found:

1. The Nasdaq-100 list is **read, not fetched** — proven with the
   network cut; a fetching variant fails the same pin; a missing file
   raises instead of silently shrinking the pool.
2. **Inclusion ∧ exclusion → excluded**, driven through the real
   assembly entry point with the network cut; the wrong ordering is
   shown to re-admit.
3. An included name with **insufficient history is silent, not fatal** —
   across the <63-bar, 40-bar and 120-bar cases.
4. The **pool version reaches every artifact and is carried, not
   recomputed**; a pre-v2 artifact keeps `None`; the hash moves on a
   list edit with no version bump.
5. **Nothing is both tracked and ignored** — and the ETF caches are
   confirmed *not* ignored, without which the rotation's write-back
   would be a silent no-op. The contradictory state is demonstrated in a
   throwaway git index.
6. **A degraded ETF fetch cannot look like a healthy one** — the two
   paths return identical holdings and are separated only by the
   coverage record; an empty response does not overwrite holdings on
   record or re-stamp their timestamp.
7. **The cache TTL stays strictly below the rotation interval** — a
   cache one second short of a week must expire, and the pin shows the
   same cache being served at the old ttl=7.
8. **A pool that shrank past the floor refuses to rotate** — driven
   through the builder's own `pool_retention_ok`; growth never refuses;
   a missing baseline never blocks; and the pin demonstrates the case
   nothing else sees, a pool cut to 360 that still clears the 8-ticker
   viability floor. It also asserts the check runs *before* the price
   fetch, so a degraded pool fails in seconds rather than after a
   ~550-ticker download.

The refactor was verified independently rather than argued: the
extracted `rank_and_select` was diffed byte-for-byte against the
pre-change block (one comment differs, nothing else) and run against the
old implementation over the real committed artifact, 20 edge suites and
400 randomised tie-heavy seeds — **zero divergences**, with dict identity
aliasing preserved so the caller's later mutations still reach both
`ranking` and `selected`.

## Found in passing, not fixed here

All **pre-existing**, all surfaced by the review fan-out, none repaired
inside a doctrine change:

- **`_norm()` mangles single-character exchange suffixes.** The
  Tokyo-listed ETF holding `6701.T` (Hitachi, from `etf_QTUM.json`)
  becomes `6701-T`, which Yahoo 404s — verified: `6701.T` returns 122
  bars, `6701-T` returns none. It has been an unpriceable pool member
  all along, hidden because the GICS classifier absorbs it into
  `unclassified_tickers`. `.L`, `.F`, `.V` would fail the same way;
  two-character suffixes are safe. **The Nasdaq-100 list contains no
  dotted or foreign symbol, so this change does not worsen it.**
- **One company can occupy two of a group's seven slots.** GOOG and
  GOOGL are both members of *Interactive Media & Services*, and in the
  committed artifact that group's composite (2.59) is set by a single
  issuer — META has no influence on its own group's rank. `min_candidates`
  counts listings, not companies. **Pre-existing via the S&P 500 source
  (GOOG/GOOGL, FOX/FOXA, NWS/NWSA); the Nasdaq-100 adds no new pair.**
- **The published row order disagrees with the selection order.** Audit
  rows sort by `(score, ticker)` while selection sorts by
  `(score, ytd, ticker)`, so within a block of equal scores the table is
  alphabetical. In the committed artifact *Semiconductors* shows MPWR at
  row 7 marked `outranked_within_group` and TXN at row 8 marked
  `selected`. `status` is correct; **reading "the top 7" off the row
  order gives the wrong answer.** A comment now warns at the sort.
- **The market-cap vector is unvalidated.** Caps come from
  `fast_info["marketCap"]` with no second source and no price × shares
  reconciliation — the same field this change documents as unreliable
  for SPCX, with the error direction (upward) being the one that makes a
  cap floor look inert.
- **The GICS alias map is incomplete**, which is what files MELI, PDD,
  CCEP and TRI into phantom groups (above).

## Consequences

- Takes effect at the **next rotation**. Nothing changes before then.
- The first rotation under v2 will also be the first to commit the ETF
  caches, so that diff will be larger than usual.
- `pool_version` is now a thing that must be **bumped deliberately**
  when a source list changes. The hash is the backstop if it is not.
- Small-group composite sensitivity is now a known, measured property
  of the selection rule. It is not addressed here.
- **The rotation can now refuse to run.** A pool more than 10% smaller
  than last week's raises instead of writing. The first time this fires
  it will look like a broken rotation; it is the floor working. Check
  `etf_coverage` in the candidates artifact first, then the source
  lists. If the shrink is a deliberate source change, re-run once the
  new pool is the baseline.
- **Nothing is excluded by hand except PSTG.** The SPCX exclusion was
  raised and reversed within the day; the pool's only hand-barred name
  is the one failing a data-quality gate. New listings go to the gates.
- **Four net-new names are unreachable for a classification reason, not
  a market one**, and the alias map that causes it is deliberately left
  for its own change.
