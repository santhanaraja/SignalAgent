#!/usr/bin/env python3
"""Build 9 prereg pins — the DURABLE form (1d6d1cf precedent). NO
FIXED COMMIT SHA anywhere: a content-hash literal for immutability,
path-resolved ancestry for order. Runs on a fresh clone; needs none
of the gitignored doctrine caches. At the REPO ROOT because
scripts/run_pins.py enumerates root test_*.py — a pin no runner
executes is an outage impersonating safety.

Build 9 has no amendment: the chain is prereg -> results, two links.
The study script additionally ASSERTS the same content hash at
runtime (a post-hoc prereg edit aborts the run itself).

Run: python3 test_backtest_sizing.py
"""

import hashlib
import os
import subprocess

REPO = os.path.dirname(os.path.abspath(__file__))
PREREG = "docs/backtest-sizing-prereg.md"
RESULTS = "docs/backtest-sizing-results.json"

PREREG_SHA256 = ("af870141594b5a0380ddaf075f69c1fc"
                 "d3c455b5483ffc98d0e38b8c1c1baf8c")


def _git(*args):
    return subprocess.run(["git", "-C", REPO] + list(args),
                          capture_output=True, text=True)


def _content_check(path, ruled_hash, label):
    with open(os.path.join(REPO, path), "rb") as f:
        h = hashlib.sha256(f.read()).hexdigest()
    assert h == ruled_hash, (
        f"{label} content hash {h[:16]}… != ruled {ruled_hash[:16]}… — "
        f"the document changed after it was ruled. A ruled change "
        f"updates the literal in the same commit; anything else is a "
        f"post-hoc amendment.")
    return h


def test_prereg_content_immutable():
    h = _content_check(PREREG, PREREG_SHA256, "prereg")
    print(f"  (1) prereg immutable: sha256 {h[:16]}… matches: OK")


def test_tampered_document_fails():
    """Pin doctrine law 3: a tampered prereg must FAIL, naming both
    hashes."""
    import tempfile
    doctored = open(os.path.join(REPO, PREREG), "rb").read() \
        + b"\namended after results\n"
    h = hashlib.sha256(doctored).hexdigest()
    assert h != PREREG_SHA256
    with tempfile.NamedTemporaryFile(dir=REPO, suffix=".tmp",
                                     delete=False) as tf:
        tf.write(doctored)
        tmp = tf.name
    try:
        try:
            _content_check(os.path.relpath(tmp, REPO), PREREG_SHA256,
                           "prereg")
            raise SystemExit("tampered prereg PASSED the pin — broken")
        except AssertionError as e:
            assert h[:16] in str(e) and PREREG_SHA256[:16] in str(e), \
                "failure does not name both hashes"
    finally:
        os.unlink(tmp)
    print(f"  (2) tampered prereg fails naming both hashes "
          f"({h[:16]}… vs {PREREG_SHA256[:16]}…): OK")


def test_branch_language_present():
    """A hash pins an empty file just as faithfully — the six decision
    branches' load-bearing phrases must exist inside the hashed text."""
    txt = " ".join(open(os.path.join(REPO, PREREG)).read().split())
    for phrase in ("DOMINANCE", "the only free-lunch branch",
                   "WELL-ORDERED FRONTIER", "THE EXCHANGE RATE",
                   "NON-MONOTONE FRONTIER",
                   "ALL INSIDE THE BANDS",
                   "THE LADDER, RULED SEPARATELY",
                   "the ladder is decorative",
                   "FRAGILE regardless of branch",
                   "CASH FLOOR IS ZERO, HARD",
                   "cent-exact, or STOP",
                   "No arm is added after results are seen"):
        assert phrase in txt, f"prereg missing: {phrase!r}"
    print("  (3) branch + feasibility language present in the hashed "
          "text: OK")


def _first_commit(path):
    out = _git("log", "--format=%H", "--", path).stdout.split()
    return out[-1] if out else None


def test_ordering():
    """Path-resolved ancestry: prereg strictly precedes results, with
    same-commit REJECTED. No SHA literals — the paths resolve to
    whatever commits currently hold them."""
    pre = _first_commit(PREREG)
    assert pre, f"{PREREG} is not committed"
    res = _first_commit(RESULTS)
    if not res:
        print(f"  (4) ordering: prereg {pre[:7]} committed "
              "(results not yet committed): OK")
        return
    assert pre != res, (
        "prereg and results first appear in the SAME commit — a "
        "decision table that lands with the answer is not a "
        "pre-registration")
    assert _git("merge-base", "--is-ancestor", pre, res).returncode \
        == 0, f"prereg {pre[:7]} does not precede results {res[:7]}"
    print(f"  (4) ordering by path-resolved ancestry: {pre[:7]} -> "
          f"{res[:7]}: OK")


def test_results_cite_prereg_by_path():
    import json
    p = os.path.join(REPO, RESULTS)
    if not os.path.exists(p):
        print("  (5) results not present yet — citation check pending")
        return
    with open(p) as f:
        r = json.load(f)
    assert PREREG in r.get("prereg", ""), r.get("prereg")
    assert PREREG_SHA256[:16] in r.get("prereg", ""), (
        "results cite the prereg without its content hash")
    print("  (5) results cite the prereg by path AND content hash: OK")


if __name__ == "__main__":
    print("\n=== Build 9 prereg pins (durable form — no fixed SHA) ===")
    test_prereg_content_immutable()
    test_tampered_document_fails()
    test_branch_language_present()
    test_ordering()
    test_results_cite_prereg_by_path()
    print("\nAll Build 9 pins passed.\n")
