# archive/ — decommissioned modules

Retired code and state, moved here instead of deleted so history stays
reachable in place (full lineage in git). Nothing in this directory is
imported by the live pipeline, run by CI, or included in the test sweep.

| File | Retired | Ruling | What replaced it |
|---|---|---|---|
| `theme_ranker.py` | 2026-07-23 | D-007 Phase 3 | The weekly universe rotation (top-15 GICS scanner) is the thesis layer |
| `constituent_ranker.py` | 2026-07-23 | D-007 Phase 3 | Layer-2 GICS leadership sub-rows + candidate grades (D-017) |
| `qualified_themes.json` | 2026-07-23 | D-007 Phase 3 | No qualified-theme state exists; groups qualify by rotation |
| `theme_history.json` | 2026-07-23 | D-007 Phase 3 | Weekly theme snapshots ended with the ranker |
| `test_r4_qualification.py` | 2026-07-23 | D-007 Phase 3 | Tests the archived ranker's Sunday cadence (D-002-era); R4 is a superseded pointer now |
