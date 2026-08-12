# D-023 — A vendor label is a group nobody can reach

| | |
|---|---|
| **ID** | D-023 |
| **Date** | 2026-08-11 |
| **Status** | **Proposed** — staged and held for push approval |
| **Takes effect** | Next rotation (Saturday). Nothing changes before then. |
| **Shape** | D-021 ("Reachable vs merely present"), which found the first four and deliberately did not fix them |
| **Touches** | what the system can trade → doctrine change, measured before staging |

> ### Two findings that limit what this record proves — read these first
>
> **1. "The selected 15 do not change" is TRUE OF TODAY ONLY.** It is a
> property of one Tuesday's data, not a property of the change. Applied to
> the **last** rotation the same merge does not merely "risk" moving the
> selected 15 — it moves it, **unconditionally**: Health Care REITs falls out
> and Paper & Plastic Packaging Products & Materials takes the slot, and no
> value of the incoming ticker's momentum could have prevented it. **§5a**.
>
> **2. A local sweep UNDER-REPORTS this class of defect.** The rotation's
> environment and a developer's laptop classify some tickers *differently*,
> and the laptop is the one that looks clean. Any future classification audit
> must be run against **CI-equivalent labels**, never local ones. Worse, half
> the divergence is invisible to a vocabulary check at all — `APP` and `DD`
> classify into *different canonical groups* in the two environments. **§1b**.

## The mechanism

`gics_classifier` resolves one string per ticker. `universe_builder` then
**groups by that string**. Nothing between them checks that the string is a
GICS sub-industry name.

So a label only Yahoo uses does not produce a mislabelled group. It produces
an **extra** group — holding whichever candidates carry that label, sitting
below `rotation.min_candidates = 3`, and therefore **ineligible at any
composite, permanently**. `exclusion_reason` reads `below_min_candidates`,
which is indistinguishable in the artifact from a genuinely small
sub-industry.

The damage is symmetric, and the second half is the one that was missed:

- the stranded names can never be selected, whatever they do; **and**
- the canonical group where they belong is ranked **without** them — its
  composite is a weighted median over the wrong population, so its rank is
  wrong too, and it can win or lose a top-15 slot on that basis.

D-021 §"Reachable vs merely present" found four of these and left them alone
on purpose: repairing classification inside a pool-definition change would
have mixed two populations in one diff. This is that repair.

---

## 1 — The sweep

Not "check the four names D-021 named". Every group name the current pool
produces, against a canonical vocabulary.

**Method.** Assemble the pool through `UniverseSource` (550 candidates, pool
`v3-2026-08-11`), classify through `GICSClassifier` with the alias view
**off** to recover raw labels, and compare every resulting group name against
the GICS sub-industry names the authoritative S&P feed uses — the same
`GICS Sub-Industry` column the pipeline already consumes to seed the cache.
A name in that column is canonical by construction; a name absent from it is
a **suspect** requiring verification, not a verdict.

**Result: 134 group names as the rotation would build them (133 in a local
run — see the EA case below), 8 of them vendor labels holding 9 candidates.**
Every one is a group of its own; every one is `below_min_candidates`.

| vendor label | candidates | canonical target | verdict |
|---|---|---|---|
| `Internet Retail` | MELI, PDD | Broadline Retail | **aliased** |
| `Beverages - Non - Alcoholic` | CCEP | Soft Drinks & Non-alcoholic Beverages | **aliased** |
| `Medical Devices` | GKOS | Health Care Equipment | **aliased** |
| `REIT - Healthcare Facilities` | CTRE | Health Care REITs | **aliased** |
| `Electronic Gaming & Multimedia` | EA | Interactive Home Entertainment | **aliased** |
| `Specialty Business Services` | TRI | — | **unmappable** |
| `Health Information Services` | BTSG | — | **unmappable** |
| `Capital Markets` | HUT | — | **unmappable** |

Four more than the brief's four. Two of those four are the reverse error and
are **not** fixed here (§3); one was invisible to a local sweep entirely, for
a reason that is a finding in its own right (§1b).

---

## 1b — FINDING: a local sweep under-reports this defect

**This is not an anecdote about EA. It is a method rule, and it invalidates
the obvious way to run this audit.**

`Electronic Gaming & Multimedia` appears in **no** local run of the sweep. It
appears in the **committed 2026-08-07 rotation artifact**, which carries
*both* `Electronic Gaming & Multimedia` (EA, 1 member) and `Interactive Home
Entertainment` (TTWO, 1 member) — one GICS sub-industry split into two
permanently ineligible groups, by a build that had no idea.

### Why the two environments disagree

`GICSClassifier` resolves in the order: static legacy map → **on-disk cache** →
live Yahoo lookup. The cache is *seeded from the S&P 500 CSV*, and
`data/universe_cache/gics_cache.json` is **gitignored** — committed ETF caches
exist, this one does not. So:

- a ticker **in** today's CSV is seeded canonically in both environments and
  never reaches Yahoo;
- a ticker that has **left** the CSV keeps its pre-departure seed on a laptop
  (30-day TTL) while a fresh CI checkout has no cache at all and falls through
  to Yahoo, which answers with its own label.

**A seed is only trustworthy while the ticker is still in the index — and an
index departure is precisely the event that hands a name's classification from
S&P to Yahoo.** The laptop is the environment that looks clean, which is the
worst possible direction for the error to run.

### The mechanism is broader than the instance, and the second half is invisible

The root cause is one line: `GICSClassifier.seed()` **only fills gaps** —
`if ... self._cache_get(tn) is None` — so it never overwrites a live entry
with a fresher CSV answer. A laptop's seed therefore goes stale in **two**
ways, and the sweep only catches one:

| | pool tickers today | caught by the vendor-label sweep? |
|---|---|---|
| **A. ticker left the index** — seed keeps the last CSV answer, CI falls through to Yahoo | **1** — `EA` | **yes** — CI's label is a Yahoo string |
| **B. ticker stayed, its CSV label CHANGED** — seed keeps the old answer, CI seeds the new one | **2** — `APP`, `DD` | **NO** |

> `APP` — laptop `'Application Software'` (seeded 2026-07-25), CI
> `'Advertising'`.
> `DD` — laptop `'Specialty Chemicals'`, CI `'Industrial Conglomerates'`.

**Both of those pairs are canonical GICS names on both sides**, so *no*
vocabulary check can flag them — including the one this record adds. Case B
puts a ticker in a different real group in the two environments and reports
nothing, anywhere. `APP` and `DD` do not self-heal until their entries expire
on 2026-08-24.

So the correct statement of the finding is: **exactly one pool ticker has a
stale seed from an index departure; three have an `sp500_csv` cache entry that
no longer matches what a fresh CI checkout would produce.** The vendor-label
sweep is a detector for case A only.

### The rule, and the recipe that implements it

**Any future classification audit must be run against CI-equivalent labels.**
Concretely, before sweeping, re-resolve every pool ticker that is *not* in the
current `sp500_gics_map` but *does* carry a cache entry with
`source == "sp500_csv"`:

```python
stale = [t for t in pool
         if t not in src.sp500_gics_map
         and (cache.get(t) or {}).get("source") == "sp500_csv"]
```

Run live on 2026-08-11 that list is exactly `['EA']`, and re-resolving it
reproduces the divergence:

> EA — cache says `'Interactive Home Entertainment'` (source `sp500_csv`,
> seeded 2026-07-25, expiring ~2026-08-24); CI resolves
> `'Electronic Gaming & Multimedia'`.

…and for case B, compare the cached label against the *current* CSV value
rather than merely checking presence:

```python
relabelled = [t for t in pool
              if (cache.get(t) or {}).get("source") == "sp500_csv"
              and t in src.sp500_gics_map
              and src.sp500_gics_map[t] != cache[t]["sub_industry"]]
```

**Three tickers today. The count is not the point** — the point is that it is
not zero and nothing reports it. The mechanism fires on every future index
departure and every GICS reclassification, silently, with a lag set by the
30-day cache TTL.

This is also why the pin reads **committed artifacts** rather than sweeping a
live pool: the artifact is what the rotation actually produced, in the
environment that actually produces it.

### Two adjacent exposures, one real and one already closed

- **Real, and untouched by this change:** 44 pool tickers have neither a
  static-map entry nor a current-CSV classification, so a fresh CI checkout
  makes 44 live `.info` calls every rotation (42 of which a laptop serves from
  cache). A failed `.info` returns `None`, lands the ticker in
  `_unclassified`, and it is dropped from `by_gics` contributing nothing —
  and only names on the hand-maintained inclusion list get a degraded-coverage
  warning for that. Yahoo *label drift* on those 42 is equally invisible
  locally.
- **Checked and closed:** the "S&P CSV fetch fails in CI but not locally" path
  cannot cause a quiet mass-reclassification. CI has no `sp500.json` to fall
  back on, so the pool would drop from 550 to 216 (−61%) and
  `POOL_RETENTION_FLOOR = 0.90` raises `UniverseBuildError` first.

---

## 2 — Verified against the taxonomy, not against name similarity

The reverse error — an alias that merges two sub-industries GICS deliberately
separates — is worse than the defect it fixes. A phantom group only strands
its own members; a bad merge silently moves an existing group's composite and
rank, and can put a name in front of the selector that does not belong there.

**So each mapping was tested on data.** S&P 500 tickers have an authoritative
GICS sub-industry in the CSV; asking Yahoo what *it* calls those same tickers
produces a real label → GICS contingency table. 77 names probed, 2026-08-11:

| Yahoo label | probed S&P members → their authoritative GICS | verdict |
|---|---|---|
| `Beverages - Non - Alcoholic` | KO, PEP, KDP, MNST → **4/4** Soft Drinks & Non-alcoholic Beverages | clean 1:1 |
| `Medical Devices` | ABT, BSX, MDT, SYK, EW, DXCM, PODD, ZBH, STE, GEHC → **10/10** Health Care Equipment | clean 1:1 |
| `REIT - Healthcare Facilities` | WELL, VTR, DOC → **3/3** Health Care REITs | clean 1:1 |
| `Internet Retail` | AMZN, EBAY → Broadline Retail; **DASH → Specialized Consumer Services** | modal, one counterexample |
| `Electronic Gaming & Multimedia` | TTWO → Interactive Home Entertainment (EA, RBLX, NTES carry the same label, none in the CSV) | clean on the evidence available |

Two traps the probe closed:

- **`REIT - Healthcare Facilities` → NOT `Health Care Facilities`.** GICS
  *Health Care Facilities* (35102020) is hospital **operators** — HCA, UHS —
  and Yahoo files those under `Medical Care Facilities`, a different label.
  The near-identical name is the trap; the REIT belongs with the REITs.
- **`Medical Devices` is safe; its sibling is not.** Yahoo's `Medical
  Instruments & Supplies` holds ISRG, BAX, RMD (GICS Health Care Equipment)
  *and* WST, COO, ALGN (GICS Health Care Supplies) — one label across two
  sub-industries. It is recorded as unmappable so the obvious next edit does
  not get made.

**The `Internet Retail` residual risk, stated rather than buried.** The alias
is exact for MELI and PDD and for the label's modal population, but Yahoo also
files DoorDash there and GICS does not. The exposure is bounded by the
resolution order: any S&P 500 member is seeded canonically from the CSV and
never reaches the Yahoo path, so only a **non-S&P** delivery-marketplace name
arriving via the Nasdaq-100, an ETF holding, or the inclusion list could be
misfiled. It would land in a 4-member group as a visible row in the audit
table, not silently.

---

## 3 — Three labels a label→label alias cannot repair

This contradicts the brief, which proposed `Specialty Business Services →
Research & Consulting Services`. **The measurement says no.**

- **`Specialty Business Services`.** Probed over the S&P 500, this label holds
  **CTAS and CPRT — both GICS Diversified Support Services — and no Research &
  Consulting Services name at all**; GICS R&CS (EFX, VRSK) sits under Yahoo
  `Consulting Services` instead. TRI is the exception the label does not
  describe. Aliasing to TRI's target would misfile every future
  support-services name into the EFX/VRSK group; aliasing to the label's real
  target would misfile TRI. Neither is correct.
- **`Health Information Services`.** Yahoo files BTSG here, but BTSG is a home
  & community health services and pharmacy provider, not health IT — so the
  label's modal target (Health Care Technology, VEEV) is wrong for the one
  member the pool has.
- **`Capital Markets`.** Over the S&P 500 the label is *clean* (GS, MS, SCHW →
  all Investment Banking & Brokerage) — **and that is exactly why aliasing it
  would be harmful.** It is also where Yahoo files crypto miners: HUT here,
  RIOT already in the cache. An alias would inject a bitcoin miner **into the
  brokers' group**, where it would contribute to that group's composite and
  could be selected as one of its members. It is also a GICS *industry*
  (402030), not a sub-industry.

These are recorded in `universe.gics_unmappable_labels` with their reasons.
That map is not a backlog and not a suppression list: it is what keeps a
**decided** problem from burying an **undecided** one in the coverage record.

### What this change does not fix

TRI, BTSG and HUT stay unreachable. The mechanism they need is a **per-ticker**
override — `universe.gics_ticker_overrides`, applied as the same read-time
view — which is a separate decision, because a per-ticker map is a place to
put opinions and needs its own rules about what may go in it. Measured cost of
not having it (same shared fetch, one placement at a time):

| name | placed into | receiving group before → after | selected 15 |
|---|---|---|---|
| TRI | Research & Consulting Services | 2 → 3 members, rank 123 → 119 | unchanged |
| BTSG | Health Care Services | 5 → 6 members, comp 14.66 → 18.00, rank 45 → 28 | unchanged |
| HUT | *(no correct GICS placement established)* | — | — |

None of the three reaches the top 15 today. The cost of leaving them is real
but small, and it is smaller than the cost of a wrong merge.

---

## 3b — What the sweep also found, and this change does NOT touch

The brief asked whether a *new* alias could merge sub-industries GICS
separates. Running the same contingency method over the **twenty-one aliases
that were already in the map** answers a harder question: several of them
already do.

| existing alias | probed S&P members that contradict it |
|---|---|
| `Software - Infrastructure` → Systems Software | ORCL, SNPS → **Application Software**; GDDY, AKAM → **Internet Services & Infrastructure** (4 of 8 probed) |
| `Diagnostics & Research` → Life Sciences Tools & Services | LH, DGX → **Health Care Services**; RVTY → **Health Care Equipment** (3 of 10) |
| `Drug Manufacturers - Specialty & Generic` → Biotechnology | VTRS, ZTS → **Pharmaceuticals** (2 of 2 probed) |
| `Software - Application` → Application Software | NOW → **Systems Software** (1 of 5) |
| `Communication Equipment` → Communications Equipment | ZBRA → **Electronic Equipment & Instruments** (1 of 3) |
| `Computer Hardware` → Technology Hardware, Storage & Peripherals | ANET → **Communications Equipment** (1 of 1 probed) |
| `Electrical Equipment & Parts` → Electrical Components & Equipment | HUBB → **Industrial Machinery & Supplies & Components** (1 of 1) |

**The exposure is bounded but not zero.** None of the contradicting tickers is
affected: every one is an S&P 500 member, seeded canonically from the CSV
before the Yahoo path is ever reached. The aliases apply only to the **18
non-S&P pool candidates** that carry those labels — and two of them are in a
**currently selected** group:

> `NET` and `NTNX` sit in **Systems Software** (selected, rank 12) by way of
> `Software - Infrastructure`, the least accurate alias in the table. **NET is
> a live holding.** If its GICS sub-industry is Internet Services &
> Infrastructure — where the same alias's S&P counterexamples GDDY and AKAM
> live — then a held position is in the wrong group, which moves that group's
> composite, its rank, and the R28 per-group exposure cap it counts against.

This is **not fixed here, and deliberately so.** Every entry above is an
existing decision affecting groups that are selected today; changing one moves
a live ranking and needs its own measurement, exactly as the five additions
here got theirs. Fixing them inside a diff whose subject is *missing* aliases
would mix a population that is stranded with a population that is misplaced —
the same mistake D-021 avoided by leaving this whole subject alone.

It is recorded because the sweep is what surfaced it and because a reader of
this record should not conclude the alias map is now correct. It is *complete*
for the current pool. That is a different claim.

---

## 4 — What changed

1. **`framework/config.yaml`, `universe.gics_aliases`** — five entries added
   (the table in §1). Applied as a **read-time view** in `GICSClassifier`; the
   on-disk cache keeps raw values, so the change is reversible by deleting the
   lines and takes effect on the next read.
2. **`framework/config.yaml`, `universe.gics_unmappable_labels`** — new map,
   four entries (the three above plus `Medical Instruments & Supplies`), each
   with its measured reason.
3. **`data/pool/gics_vocabulary.json`** — new committed input: the 127 GICS
   sub-industry names the authoritative S&P feed uses. Read, never fetched,
   same doctrine as `nasdaq100.json`. Deliberately **incomplete** — a real
   sub-industry with no S&P 500 constituent is absent and will be flagged if
   it ever appears. That false alarm is the intended error: an omission is
   loud and one line to fix, while padding the list from memory would silently
   admit names nobody verified.
4. **`gics_classifier.py`** — `load_vocabulary()`, `unaliased_labels()`, and
   `GICSClassifier.unaliased_group_names()`. An unreadable vocabulary reports
   `error`, never an empty result: the check's own missing input must not read
   as a clean sweep.
5. **`universe_source.build_universe_candidates`** — records
   `unaliased_labels: {new, known, error}` in the candidates artifact and
   prints a warning when `new` is non-empty. **Never fatal.** A rotation that
   refused on an unreviewed classification would trade a silent mistake for an
   outage; the alias map went stale unnoticed for weeks, and a recorded field
   is what ends that.

---

## 5 — Impact, measured before staging — *a result about today*

Two arms over **one shared price+cap fetch**, both calling the production
`rank_and_select`. Not two live builds: the market moves between them and the
movement lands in the diff looking like an effect of the change.

**Fidelity check first.** Re-applying the selection rule to the committed
2026-08-07 artifact reproduces its 15 selected groups **exactly** — and the
known misreading (`min_candidates` against *qualifiers* rather than
membership) wrongly drops Construction & Engineering, the same name it dropped
in D-021. The rule under test is the rule that ran.

Arms use CI-equivalent classification (EA through Yahoo, per §1).

| | groups | selected | tickers |
|---|---|---|---|
| **A** — current aliases | 134 | 15 | 78 |
| **B** — with the five | 129 | 15 | 78 |

**The selected 15 and all 78 tickers are identical — on today's data.** No
group enters, none leaves, no ticker moves. The boundary is unchanged in both
arms: 15th Pharmaceuticals 18.20, 16th Rail Transportation 17.66, gap 0.54.
**Do not read that as a property of the change; §5a is the same merge run
against last week's rotation, where it decides a slot.**

Five groups change — the five that receive members. Every other group's
composite is identical to the cent (ranks renumber only because five phantom
groups vanish):

| receiving group | members | composite | rank | eligible |
|---|---|---|---|---|
| Broadline Retail | 2 → **4** | 12.24 → **4.03** | 57 → 91 | **False → True** |
| Soft Drinks & Non-alcoholic Beverages | 4 → 5 | 7.30 → **10.85** | 81 → 60 | True |
| Health Care Equipment | 16 → 17 | 4.55 → 4.75 | 94 → 89 | True |
| Health Care REITs | 3 → **4** | 16.24 → **14.06** | 35 → 45 | True |
| Interactive Home Entertainment | 1 → 2 | 4.72 → 3.92 | 93 → 94 | False (still 2 members) |

Read the column that matters: **Broadline Retail crosses from
`below_min_candidates` to eligible.** Before this change AMZN, EBAY, MELI and
PDD were four qualifying large caps in two groups of two, none of them
reachable at any composite. That is the change's actual content — reachability,
not today's selection.

### 5a — At the last rotation this change DOES move the selected 15

Not "would plausibly have". **Recomputed exactly from the committed artifact
(`fd83fa4`), it does — and no plausible market outcome prevents it.**

At that rotation Health Care REITs was the **15th of the 15 selected**, at
composite **17.35**, **1.06** ahead of the 16th eligible group (Paper &
Plastic Packaging Products & Materials, **16.29**), holding DOC, VTR and WELL.
`CTRE` sat alone in the phantom group at composite 8.12. Merging it moves
**all three components**, not just YTD:

| component | 3 members | + CTRE | weight | weighted Δ |
|---|---|---|---|---|
| median YTD | 27.62 | 24.92 | 0.50 | **−1.35** |
| median 3M | 10.63 | 9.125 | 0.30 | **−0.45** |
| median 1M | 1.77 | 1.325 | 0.20 | **−0.09** |
| **composite** | **17.35** | **15.46** | | **−1.89** |

Re-run through the production selection rule, the merged 4-member group lands
at **18th eligible**. **Health Care REITs leaves the tradeable universe —
WELL, VTR and DOC with it — and Paper & Plastic Packaging Products & Materials
takes the slot.** Exactly one group changes.

**The adversarial bound closes it.** CTRE's 3M and 1M returns cannot touch the
YTD median, which is pinned at 24.92 — the merged YTD set is
{15.86, 22.22, 27.62, 37.38} and CTRE is the low member. Driving CTRE's 3M and
1M to **+∞** therefore caps the other two medians at 10.695 and 1.92, for a
best-possible merged composite of **16.05** — still below the **16.29** needed
to hold the slot (16.29 exactly would suffice: at a tie the `(-composite,
name)` sort puts "Health Care REITs" ahead of "Paper & Plastic…"), and below
Pharmaceuticals' 16.27. **No value of the incoming ticker's momentum could
have saved the slot.**

> **Two corrections to the first version of this record**, both found by
> re-deriving the arithmetic rather than re-reading it:
> - The move is **−1.89**, not −1.35. The −1.35 figure was the YTD component
>   alone. Comparing one weighted component against the boundary gap is an
>   **unsound argument form** — it silently assumes the other two components
>   hold still. Here they happened to move the same way, so the conclusion
>   survived; in general a merge that lifts 3M/1M can offset a YTD drop. Do
>   not reuse it as a decision rule.
> - "The full recompute is not available from committed data" was **wrong**.
>   The artifact publishes `r3m` and `r1m` per ticker inside each group's
>   `qualifiers` array; it is only the `tickers` audit rows that carry `ytd`
>   alone. The gap is real but bites only for DISQUALIFIED tickers, and both
>   groups here have none.

One thing this is *not*: a `min_candidates` effect. The merge raises Health
Care REITs from 3 members to 4, so it stays **eligible** — it is **outranked**,
not gated out. `min_candidates` is what imprisoned the phantom, not what
costs Health Care REITs its slot.

D-021's caveat applies unchanged and is the honest frame: the one-day
distribution of |Δcomposite| across groups has median 1.46 and p90 3.73. Two
of the four composite moves here (Broadline Retail −8.21, Soft Drinks +3.55)
exceed the median day's churn and one exceeds p90 — these are not noise-sized
for the receiving group — but none of them reaches the selection boundary
*this week*. Whether a merge decides a slot depends on where the receiving
group happens to sit that Friday. Last Friday it sat on the boundary; this
Tuesday it does not. **Both facts are about the data, not about the change** —
which is why the "no selection change" headline is scoped to today and why
this record leads with that scope rather than burying it.

The same sensitivity is visible elsewhere in that rotation and is worth
carrying forward: the 15/16 gap was **1.06** while a single-ticker
reclassification is worth **~1.9**, and `Data Center REITs` (DLR, EQIX) and
`Self-Storage REITs` (PSA, EXR) both sat two members deep at ranks 31 and 42
— one misplaced ticker from moving in either direction. **At that boundary,
classification is not a cosmetic layer; it is an input to selection with more
leverage than a day of price movement.**

### No live position is affected

None of the nine stranded names is held, and no group holding a live position
receives a member: HPQ and DELL (Technology Hardware, Storage & Peripherals),
NET (Systems Software), MET (Life & Health Insurance), BIIB and CGON
(Biotechnology). All four groups have **identical membership and composite**
in both arms and all four stay selected; their rank *numbers* shift by one or
two purely because five phantom groups above them no longer exist.

### Caveat: the arms ran on laptop labels, re-checked against CI-equivalent ones

§1b applies to this measurement too. The arms were built from the local cache,
which classifies three pool tickers differently from the rotation: `EA`
(corrected by hand in the arms), and `APP` and `DD`, which were not.

Re-run with **all three** at their CI-equivalent labels, the A-vs-B conclusion
is unchanged — selected 15 identical, all 78 tickers identical, in both arms.
It has to be: `APP` and `DD` sit in the same group in *both* arms, so they
cancel out of the difference. What they do distort is the **absolute** ranking
figures quoted for four groups that this change does not touch:

| group | laptop rank / composite | CI-equivalent |
|---|---|---|
| Industrial Conglomerates | 34 / 16.08 | 51 / 12.50 |
| Application Software | 122 / −8.14 | 116 / −5.01 |
| Specialty Chemicals | 65 / 9.52 | 61 / 10.76 |
| Advertising | 128 / −20.83 | 129 / −39.88 |

None is a receiving group, none is selected in either arm, and the relocation
does not move the selected 15. Recorded because "the arms ran on labels the
rotation would not have produced" is exactly the kind of thing that should not
be discovered later.

### Caveat on the cap input

Yahoo's quote endpoint was hard rate-limited on this host during the
measurement (`fast_info` and `.info` both 401/429 across a six-step backoff;
the chart endpoint serving prices was unaffected — 549/550). Market caps were
therefore reconstructed from **committed evidence**, per ticker, with the
provenance recorded: 389 exact values from the 2026-08-07 artifact, 1 exact
value from that artifact's own gate-fail string, 142 "cleared the $5B floor
then, value unknown" (a floor sentinel — `rank_and_select` only compares to
the floor), and 14 covered by D-021's measured statement that the $5B gate
removed exactly two names from the v2 pool, neither of them in this set.

Caps are **shared by both arms**, so this cannot bias the A/B comparison; it
can only make the absolute qualifier counts slightly stale. The composite and
rank figures — the substance of this record — do not depend on caps at all:
`rank_and_select` computes a group's composite from all valid members, never
from its qualifiers.

---

## 6 — Pins

`test_gics_aliases.py`, five invariants, each demonstrating its own failure:

1. **Every group name a real rotation produced is canonical** under the
   current alias map — driven off the committed artifact via `git show`, not a
   hand-built list. Plus the historical instance at its own commit
   (`fd83fa4`): EA and TTWO in two one-member groups, both
   `below_min_candidates`, merged by the new map.
2. **Drop any one of the five aliases and exactly that label comes back** as
   an unreviewed group name — run once per alias against the same input, so
   no alias is pinned by another's presence.
3. **The harm, end to end through the real classifier and the real ranker.**
   Without the alias: two groups of two, both `below_min_candidates`, four
   qualifying names unreachable, nothing selected. With it: one group of four,
   eligible, selected, MELI and PDD competing beside AMZN and EBAY. This is
   the pin that asserts the *consequence* rather than the string.
4. **The two maps stay coherent** — no alias chains (`canon()` is a single
   lookup, demonstrated: with a chain in place it stops on the intermediate
   label), no label both aliased and declared unmappable, every alias target
   canonical, every unmappable entry carrying a reason.
5. **An unreadable vocabulary fails loud** — `load_vocabulary` raises on an
   empty or absent file, and the production caller records `error` rather than
   an empty `new` list.

---

## 7 — Reversibility

Delete the five lines from `universe.gics_aliases`. The alias map is a
read-time view and the on-disk cache stores raw labels, so nothing has to be
rebuilt or re-fetched — the next read groups the old way. The vocabulary file
and the coverage record are inert on their own: they observe, they do not
decide.
