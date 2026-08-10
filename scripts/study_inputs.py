"""Pinned study inputs — one manifest, asserted, naming the mover.

Ruled 2026-08-10 after Build 7 stopped at its integrity gate: a
committed backtest that reads a MUTABLE artifact silently stops
reproducing when the artifact moves, and the committed result becomes
unverifiable through no fault of its code. The universe group map was
the instance; the sweep in docs/backtest-inputs.md found the siblings.

The law, from docs/testing.md's sliding-window entry, applied to a
study's INPUTS: anchor to a fixed commit or a content hash, never to
whatever the working tree currently holds.

Two mechanisms live here:
  · `pinned_universe_ranking()` — artifacts under version control are
    read from a FIXED COMMIT (in backtest_systems, which owns the
    panel).
  · `assert_pinned_inputs()` — gitignored caches cannot be read from a
    commit, so they are pinned by ASSERTED CONTENT HASH. Recording a
    hash in a results file is NOT pinning: nothing checks it. These
    are checked, and a mismatch RAISES and NAMES the input that moved.

Every value below was captured on 2026-08-10 from the working tree
that reproduces the committed studies: Layer A's frame rebuild ->
input_hash d7df85e8ad6ba244, and 5B's S1/score -> $235,200.09 under
the pinned group map.
"""

import glob
import hashlib
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------
# The manifest. A study declares which of these it consumes; anything
# it consumes that is NOT here is unpinned by definition.
# ---------------------------------------------------------------------
EXPECTED = {
    "prices_manifest": "f22f69aac72c5019",       # 530 OHLCV CSVs
    "earnings_dates": "be0082bd26258b4b",
    "regime_daily": "d1343d018ccd7ad6",
    "master_frame": "d036c2276866fcd5",
    "master_frame_noguard": "1ee8b35689d4083b",
    "IRX": "4d8ae808c01f8b78",
    "SPY": "f097894f0aa638f0",
    "RSP": "334fbdc648553bb2",
    "VIX": "6720b33ba8b0aaf5",
    "OAS": "4fe223715fd0f19e",
}
PRICES_EXPECTED_FILES = 530

PATHS = {
    "earnings_dates": "data/doctrine_cache/earnings_dates.json",
    "regime_daily": "data/regime_daily.json",
    "master_frame": "data/doctrine_cache/master_frame.csv.gz",
    "master_frame_noguard": "data/doctrine_cache/master_frame_noguard.csv.gz",
    "IRX": "data/backtest_cache/IRX.csv",
    "SPY": "data/backtest_cache/SPY.csv",
    "RSP": "data/backtest_cache/RSP.csv",
    "VIX": "data/backtest_cache/VIX.csv",
    "OAS": "data/backtest_cache/OAS.csv",
}


def _file_hash(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def prices_manifest_hash():
    """One hash over the whole price cache — filename AND content, so a
    ticker ADDED or REMOVED moves it as surely as a changed close. The
    directory listing is the study population; it must be pinned too."""
    files = sorted(glob.glob(os.path.join(
        REPO, "data", "doctrine_cache", "prices", "*.csv")))
    h = hashlib.sha256()
    for p in files:
        h.update(os.path.basename(p).encode())
        with open(p, "rb") as f:
            h.update(f.read())
    return h.hexdigest()[:16], len(files)


def current(names):
    """{name: hash} for the requested inputs, computed now."""
    out = {}
    for n in names:
        if n == "prices_manifest":
            out[n], _ = prices_manifest_hash()
        else:
            p = os.path.join(REPO, PATHS[n])
            out[n] = _file_hash(p) if os.path.exists(p) else None
    return out


def assert_pinned_inputs(names, label=""):
    """Raise, naming the mover, if any consumed input has changed.

    A study that calls this cannot silently produce a different answer
    from a different input — the failure mode that stopped Build 7 and
    that the sweep found in eight other places.
    """
    problems = []
    if "prices_manifest" in names:
        h, n = prices_manifest_hash()
        if n != PRICES_EXPECTED_FILES:
            problems.append(
                f"prices cache holds {n} files, pinned at "
                f"{PRICES_EXPECTED_FILES} — the study POPULATION moved")
        elif h != EXPECTED["prices_manifest"]:
            problems.append(
                f"prices cache content moved: {h} != "
                f"{EXPECTED['prices_manifest']} (a re-fetch re-adjusts "
                "whole histories for any split/dividend since)")
    for n in names:
        if n == "prices_manifest":
            continue
        path = os.path.join(REPO, PATHS[n])
        if not os.path.exists(path):
            problems.append(f"{n}: MISSING at {PATHS[n]}")
            continue
        got = _file_hash(path)
        if got != EXPECTED[n]:
            problems.append(f"{n} ({PATHS[n]}): {got} != {EXPECTED[n]}")
    if problems:
        raise RuntimeError(
            f"PINNED STUDY INPUTS MOVED{' — ' + label if label else ''}:\n  "
            + "\n  ".join(problems)
            + "\n\nA committed study cannot be trusted against changed "
              "inputs. Either restore the input, or — if the change is "
              "intended — re-run the affected studies, re-baseline "
              "their committed results, and update EXPECTED in "
              "scripts/study_inputs.py in the same commit.")
    return True
