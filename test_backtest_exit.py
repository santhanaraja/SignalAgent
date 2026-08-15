#!/usr/bin/env python3
"""Build 8 prereg + amendment pins — the DURABLE form (1d6d1cf
precedent). NO FIXED COMMIT SHA anywhere; content hashes and
path-resolved ancestry only, so every pin runs on a fresh clone and
survives any rebase. Needs none of the gitignored doctrine caches.

Placed at the REPO ROOT, not tests/: scripts/run_pins.py enumerates
root-level test_*.py, and a pin no runner executes is an outage
impersonating safety.

The chain pinned here: the pre-registration (flaw included — it
STANDS, unedited, per the 2026-08-15 ruling), then amendment 1 (the
leverage-clean primary, ruled after the adversarial review and before
the results artifact was read), then the results. Content pins prove
neither document changed after ruling; ordering pins prove the
sequence prereg -> amendment -> results with same-commit rejected.

Run: python3 test_backtest_exit.py
"""

import hashlib
import os
import subprocess

REPO = os.path.dirname(os.path.abspath(__file__))
PREREG = "docs/backtest-exit-prereg.md"
AMEND = "docs/backtest-exit-prereg-amendment.md"
RESULTS = "docs/backtest-exit-results.json"

PREREG_SHA256 = ("9b392b5f7e34c5d27502119cf021d380"
                 "1c49882a9a05c7236adb668713f65a68")
AMEND_SHA256 = ("d7a52c84cd5fef6bbd9adcb7bca51e4c"
                "6682ef0d1eb746bf3dfd76c9d5d58f7d")


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


def test_amendment_content_immutable():
    h = _content_check(AMEND, AMEND_SHA256, "amendment")
    print(f"  (2) amendment immutable: sha256 {h[:16]}… matches: OK")


def test_tampered_document_fails():
    """Pin doctrine law 3: demonstrate the failure. A tampered prereg
    must fail, and the failure must NAME BOTH HASHES."""
    doctored = open(os.path.join(REPO, PREREG), "rb").read() \
        + b"\namended after results\n"
    h = hashlib.sha256(doctored).hexdigest()
    assert h != PREREG_SHA256
    try:
        # the same check, fed the doctored bytes via a temp path
        import tempfile
        with tempfile.NamedTemporaryFile(dir=REPO, suffix=".tmp",
                                         delete=False) as tf:
            tf.write(doctored)
            tmp = tf.name
        try:
            _content_check(os.path.relpath(tmp, REPO), PREREG_SHA256,
                           "prereg")
            raise SystemExit("tampered prereg PASSED the pin — "
                             "the pin is broken")
        except AssertionError as e:
            msg = str(e)
            assert h[:16] in msg and PREREG_SHA256[:16] in msg, (
                "failure does not name both hashes")
        finally:
            os.unlink(tmp)
    except SystemExit:
        raise
    print(f"  (3) tampered prereg fails naming both hashes "
          f"({h[:16]}… vs {PREREG_SHA256[:16]}…): OK")


def test_branch_language_present():
    """A hash pins an empty file just as faithfully — the load-bearing
    language of the decision table AND the amendment must exist inside
    the text the hashes cover."""
    txt = " ".join(open(os.path.join(REPO, PREREG)).read().split())
    for phrase in ("ADOPT", "EXCHANGE RATE", "REJECT that arm",
                   "BY INCUMBENCY", "It is a modifier, not an arm",
                   "FRAGILE regardless of branch", "cent-exact, or STOP",
                   "No additional arm is authorised after results"):
        assert phrase in txt, f"prereg missing: {phrase!r}"
    atxt = " ".join(open(os.path.join(REPO, AMEND)).read().split())
    for phrase in ("LEVERAGE-CLEAN", "UNMEASURABLE on portfolio metrics",
                   "ASYMMETRIC TEST ONLY", "ENFORCED on the adoption",
                   "REDESIGNED, NOT REPAIRED", "unseen gates",
                   "STANDS UNCHANGED, flaw included"):
        assert phrase in atxt, f"amendment missing: {phrase!r}"
    print("  (4) branch language present in both ruled documents: OK")


def _first_commit(path):
    out = _git("log", "--format=%H", "--", path).stdout.split()
    return out[-1] if out else None


def _strictly_precedes(a, b, la, lb):
    assert a != b, (
        f"{la} and {lb} first appear in the SAME commit — a document "
        "that lands with its consequence was not ruled before it")
    r = _git("merge-base", "--is-ancestor", a, b)
    assert r.returncode == 0, (
        f"{la} ({a[:7]}) does not precede {lb} ({b[:7]})")


def test_ordering():
    """Path-resolved ancestry: prereg -> amendment -> results, strictly,
    with same-commit rejected at every link. No SHA literals — the
    paths resolve to whatever commits currently hold them."""
    pre = _first_commit(PREREG)
    amd = _first_commit(AMEND)
    res = _first_commit(RESULTS)
    assert pre, f"{PREREG} is not committed"
    assert amd, f"{AMEND} is not committed"
    _strictly_precedes(pre, amd, "prereg", "amendment")
    if res:
        _strictly_precedes(amd, res, "amendment", "results")
        chain = f"{pre[:7]} -> {amd[:7]} -> {res[:7]}"
    else:
        chain = f"{pre[:7]} -> {amd[:7]} (results not yet committed)"
    print(f"  (5) ordering by path-resolved ancestry: {chain}: OK")


def test_results_cite_both_by_path():
    import json
    p = os.path.join(REPO, RESULTS)
    if not os.path.exists(p):
        print("  (6) results not present yet — citation check pending")
        return
    with open(p) as f:
        r = json.load(f)
    assert PREREG in r.get("prereg", ""), r.get("prereg")
    assert AMEND in r.get("amendment", ""), r.get("amendment")
    print("  (6) results cite prereg AND amendment by path: OK")


if __name__ == "__main__":
    print("\n=== Build 8 prereg+amendment pins (durable form) ===")
    test_prereg_content_immutable()
    test_amendment_content_immutable()
    test_tampered_document_fails()
    test_branch_language_present()
    test_ordering()
    test_results_cite_both_by_path()
    print("\nAll Build 8 pins passed.\n")
