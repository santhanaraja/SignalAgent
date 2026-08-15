#!/usr/bin/env python3
"""Build 5.2 prereg pins — the DURABLE form (ruled 2026-08-15).

Two pins, and deliberately NO FIXED COMMIT SHA anywhere in this file.
Three builds in six days hit the same failure: a SHA minted locally,
cited in artifacts, then rewritten by the push-day rebase — 75a19ef
re-anchored the 5.1/6A pins after the first instance, and 5.2's prereg
citation (cdcfa19 -> b02ccb5, docs/decisions/README.md mapping table)
was the third. A `git show <sha>:` pin is rebase-fragile by
construction; these two constructs are not:

  (a) CONTENT immutability — a sha256 LITERAL of the prereg bytes,
      checked against the working tree. No git object involved: works
      in any clone, survives any rebase (content is preserved; only
      SHAs are rewritten), and a post-hoc amendment to the decision
      table fails loudly — which is the point. A legitimately RULED
      amendment updates the literal in the same commit, so the change
      is visible in one diff.

  (b) ORDER — the prereg strictly precedes the results, proven by
      PATH-resolved ancestry (5.1's test_prereg_precedes_results
      pattern, the one pin that survived the 2026-08-09 rebase
      untouched): resolve whatever commits CURRENTLY hold the two
      files and check ancestry between them. No SHA is named, so no
      rebase can dangle it.

The study's own integrity pins (A+ parity vs the committed frame,
elementwise ladder parity, component recomposition) live in
scripts/backtest_score_ablation.py and run with the study — they need
the gitignored doctrine caches, which this file deliberately does not,
so it can run on a fresh clone.

Run: python3 test_backtest_score_ablation.py
"""

import hashlib
import os
import subprocess

REPO = os.path.dirname(os.path.abspath(__file__))
PREREG = "docs/backtest-score-ablation-prereg.md"
RESULTS = "docs/backtest-score-ablation-results.json"

# sha256 of the prereg as committed (b02ccb5 on main, 2026-08-14) and
# ruled. An edit to the prereg MUST update this literal in the same
# commit — that is the amendment becoming visible, not an inconvenience.
PREREG_SHA256 = ("ca1f3b596d834f7d8a5a42314c6ff44e"
                 "cbb5e88473327b09fab5d511d3245033")


def _git(*args):
    return subprocess.run(["git", "-C", REPO] + list(args),
                          capture_output=True, text=True)


def test_prereg_content_immutable():
    """(a) The working-tree prereg hashes to the ruled literal."""
    with open(os.path.join(REPO, PREREG), "rb") as f:
        h = hashlib.sha256(f.read()).hexdigest()
    assert h == PREREG_SHA256, (
        f"prereg content hash {h[:16]}… != ruled {PREREG_SHA256[:16]}… — "
        "the decision table changed after it was ruled. If the change "
        "was itself ruled, update PREREG_SHA256 in the same commit; "
        "anything else is a post-hoc amendment to a pre-registration.")
    print(f"  (1) prereg content immutable: sha256 {h[:16]}… "
          "matches the ruled literal: OK")


def test_prereg_phrases_present():
    """(a2) The decision table's load-bearing branches exist in the
    text the hash pins — a hash alone would also faithfully pin an
    empty file."""
    txt = " ".join(open(os.path.join(REPO, PREREG)).read().split())
    for phrase in ("PRIMARY SUSPECT", "VINDICATED", "INCONCLUSIVE",
                   "CONFLATES distinguishable states",
                   "jointly AND separately",
                   "fixed now, no additions after results",
                   "the plateau ruling", "Never decreases"):
        assert phrase in txt, f"prereg missing branch text: {phrase!r}"
    print("  (2) prereg branches present: all eight bars' language "
          "found: OK")


def test_prereg_precedes_results():
    """(b) Path-resolved ancestry: the OLDEST commit touching the
    prereg is an ancestor of the OLDEST commit touching the results.
    No fixed SHA — the paths resolve to whatever commits currently
    hold them, so a rebase moves both sides together."""
    pre = _git("log", "--format=%H", "--", PREREG).stdout.split()
    assert pre, f"{PREREG} is not committed"
    res = _git("log", "--format=%H", "--", RESULTS).stdout.split()
    assert res, f"{RESULTS} is not committed"
    first_pre, first_res = pre[-1], res[-1]
    assert first_pre != first_res, (
        "prereg and results first appear in the SAME commit — a "
        "decision table that lands with the answer is not a "
        "pre-registration")
    order = _git("merge-base", "--is-ancestor", first_pre, first_res)
    assert order.returncode == 0, (
        f"prereg's first commit {first_pre[:7]} is not an ancestor of "
        f"the results' first commit {first_res[:7]} — the prereg does "
        "not precede the results")
    print(f"  (3) ordering: prereg ({first_pre[:7]}) strictly precedes "
          f"results ({first_res[:7]}), resolved by path: OK")


def test_committed_results_cite_the_prereg():
    """(b2) The results artifact names the prereg PATH (its SHA
    citation is a breadcrumb resolved by docs/decisions/README.md's
    mapping table — record, not executed; never edited)."""
    import json
    with open(os.path.join(REPO, RESULTS)) as f:
        r = json.load(f)
    assert PREREG in r.get("prereg", ""), (
        f"results 'prereg' field does not name {PREREG}: "
        f"{r.get('prereg')!r}")
    print("  (4) results cite the prereg by path: OK")


if __name__ == "__main__":
    print("\n=== Build 5.2 prereg pins (durable form — no fixed SHA) ===")
    test_prereg_content_immutable()
    test_prereg_phrases_present()
    test_prereg_precedes_results()
    test_committed_results_cite_the_prereg()
    print("\nAll Build 5.2 prereg pins passed.\n")
