# Study inputs — what every committed backtest reads, and what is pinned

**Swept 2026-08-10**, after Build 7 stopped at its integrity gate
because 5B's panel read a mutable artifact. The operator's ruling:
*"clean on the group map" is not "clean"* — enumerate every mutable
artifact any committed study reads. This is that sweep, and the
pinning it produced.

**Definitions, applied strictly.**

- **PINNED** — the study cannot silently read a changed version:
  either it reads from a fixed git commit, or it **asserts** an
  expected content hash and fails loudly on mismatch.
- **RECORDED-ONLY** — the study writes the input's hash into its
  results but never asserts it. **This is not pinning**: a changed
  input yields a changed result with a changed hash, and nothing
  complains.
- **LIVE** — read from the working tree or regenerated, with nothing
  recorded at all. A gitignored *cache* is LIVE, not safe: a refresh
  script rewrites it.

## What the sweep found — 38 inputs across five studies

**29 LIVE · 4 RECORDED-ONLY · 5 PINNED**, before this session's work.
Two findings mattered beyond the bookkeeping:

**1. A sibling of the group-map defect was already broken.** Layer A's
committed *honesty bridge* was built from a **sliding `git log`
window** — the newest 12 commits of `public/framework.json` for the
grade-parity replay, the newest 20 for the pipeline bridge. The
update-signals cron rewrites that file continuously; ~60 commits
landed after Layer A was reported, and the window rolled off the
artifacts the report was written from. Two consequences, both
verified:

- the report's **"516 recorded candidate grades"** silently replayed
  as **680** across a completely different 12 artifacts — **and the
  pin still passed**, because it asserts only floors, never the
  recorded count. Recorded-but-unasserted rot, exactly the failure
  mode the definitions above name.
- the report's **"129/130 (99.2%)"** pipeline bridge raised
  `bridge too thin: 0`, because the window had rolled past the days
  the price cache covers.

**2. Layer A survived on an aggregate pin, not per-input pins.** Nine
of its inputs are live, yet the study still reproduces
(`input_hash d7df85e8ad6ba244`, rebuilt through the real loop in ~70s).
What saved it is `test_score_v2.py`'s frozen-anchor pin, which rebuilds
the frame and asserts the hash. Real protection — but it has a
`--fast` bypass that skips exactly that check, nothing runs it
automatically, and when it trips it names no input.

## What was pinned in this session

| Mechanism | Covers |
|---|---|
| **Read from a fixed commit** (`git show 1d67d1c:…`, raises if unreachable) | `public/universe_ranking.json` — the group map, in `backtest_systems` so 5B, 6A and 7 all inherit it |
| **Asserted content hash** — `scripts/study_inputs.py`, a single manifest checked at each study's real entry point, raising with **the name of the input that moved** | the price cache (manifest over filename **and** content, so an added/removed ticker moves it — the directory listing *is* the study population), `earnings_dates.json`, `regime_daily.json`, `master_frame.csv.gz`, `master_frame_noguard.csv.gz`, `IRX/SPY/RSP/VIX/OAS.csv` |
| **Pinned commit list** replacing a sliding window | Layer A's honesty bridge (`BRIDGE_COMMITS`, 20 SHAs as of `9b5a450`) |

Wired into: `backtest_doctrine.run` (Layer A), `backtest_systems.load_panel`
(5B + everything downstream), `backtest_ablation.run` (5.1),
`backtest_target.run` (6A), `backtest_carry.load_carry_panel` (7).

**Effect, verified:** the honesty bridge is back to **516 grades** and
**129/130 (99.2%)** — the reported figures — and the previously failing
pin passes. Layer A, 5.1, 6A and 7 all now fail loudly and by name if
any pinned input moves.

## Still not pinned — deliberately, and what would be needed

- **`framework/position_signals.py` (`grade_setup`) and
  `signal_engine.py` (`SCORE_BASE`)** — the studies import *live
  production code*. This is by design: Layer A's Law 1 is that the
  grade verdict comes from the real `grade_setup`, and freezing it
  would defeat the point. It is guarded instead by the parity pins
  (`test_score_ladder_parity`, the frozen-v1 anchors). **A change to
  production doctrine can still move a committed study**, which is the
  intended coupling — but it should be noticed. Candidate follow-up: a
  hash of the *doctrine surface* asserted alongside the data manifest.
- **`requirements.txt` floor pins (pandas/numpy)** — a minor-version
  bump could in principle move a float. Not addressed here; a lockfile
  is the fix, and it is a repo-wide decision rather than a study one.
- **Python runtime version** — same class, same reasoning.
- **`_git_head()` provenance** — RECORDED-ONLY by nature; it labels a
  run, it does not feed one.

## The follow-up this sweep earned (not done here)

The dead-import finding: `backtest_doctrine`'s comment declares the
study "FROZEN to the v1 ladder" and imports `score_ytd_points_v1` to
that end — but **that import is dead**. Only `SCORE_BASE` is consumed;
the ladders are hand-inlined in `score_components_vec`. Editing the
frozen v1 ladder would not move the study at all. The replica is
guarded by `test_score_ladder_parity`, so this is not a correctness
defect today — but **the freeze is a comment, not a mechanism**, and
the comment claims more than the code does. Worth its own session:
either consume the imported ladders or reword the claim.

Also noted: `scripts/build_regime_series.py` writes straight to the
working-tree `data/regime_daily.json` that the studies read — one
command silently rewrites a pinned input. The hash assertion now
*catches* that; it does not prevent it.
