# D-023 — A vendor label is a group nobody can reach

| | |
|---|---|
| **ID** | D-023 |
| **Date** | 2026-08-11 |
| **Status** | **Proposed** — staged and held for push approval |
| **Takes effect** | Next rotation (Saturday). Nothing changes before then. |
| **Shape** | D-021 ("Reachable vs merely present"), which found the first four and deliberately did not fix them |
| **Touches** | what the system can trade → doctrine change, measured before staging |

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
are **not** fixed here (§3); one was invisible to a local sweep entirely (§2).

### The one a live sweep could not see

`Electronic Gaming & Multimedia` does not appear in a local run. It appears in
the **committed 2026-08-07 rotation artifact**, which carries *both*
`Electronic Gaming & Multimedia` (EA, 1 member) and `Interactive Home
Entertainment` (TTWO, 1 member) — one GICS sub-industry split into two
permanently ineligible groups.

The cause is worth stating because it generalises: **EA has left the S&P 500
CSV.** It therefore no longer receives the authoritative seed and falls
through to Yahoo. A local run still reads its pre-departure entry from
`data/universe_cache/gics_cache.json`, which is **gitignored** — so the
rotation's environment (a fresh CI checkout, no GICS cache) and a developer's
laptop classify the same ticker differently, and the laptop is the one that
looks clean.

Measured: of 550 pool candidates, exactly **one** currently has a stale
canonical seed of this kind, and it is EA. That is why the pin reads committed
artifacts rather than a live sweep — the artifact is what the rotation
actually produced.

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

## 5 — Impact, measured before staging

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

**The selected 15 and all 78 tickers are identical.** No group enters, none
leaves, no ticker moves. The boundary is unchanged in both arms: 15th
Pharmaceuticals 18.20, 16th Rail Transportation 17.66, gap 0.54.

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

### "No selection change today" is a statement about today

**At the last committed rotation this change would very likely have moved the
selected 15.** Health Care REITs was the **15th of 15**, at composite 17.35,
with a gap of **1.06** to 16th (Paper & Plastic Packaging Products &
Materials, 16.29) — and with exactly **three members**, the minimum. Merging
CTRE moves its median YTD from 27.62 to 24.92, i.e. **−1.35 on the composite
through the YTD weight alone**, against that 1.06 gap. The full recompute is
not available from committed data (the artifact carries median 3m/1m per
group, not per ticker), but today's live measurement of the same merge is
**−2.18**, consistent in sign and larger. WELL, VTR and DOC would have left
the tradeable universe.

D-021's caveat applies unchanged and is the honest frame: the one-day
distribution of |Δcomposite| across groups has median 1.46 and p90 3.73. Two
of the four composite moves here (Broadline Retail −8.21, Soft Drinks +3.55)
exceed the median day's churn and one exceeds p90 — these are not noise-sized
for the receiving group — but none of them reaches the selection boundary
*this week*. Whether a merge decides a slot depends on where the receiving
group happens to sit that Friday.

### No live position is affected

None of the nine stranded names is held, and no group holding a live position
receives a member: HPQ and DELL (Technology Hardware, Storage & Peripherals),
NET (Systems Software), MET (Life & Health Insurance), BIIB and CGON
(Biotechnology). All four groups have **identical membership and composite**
in both arms and all four stay selected; their rank *numbers* shift by one or
two purely because five phantom groups above them no longer exist.

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
