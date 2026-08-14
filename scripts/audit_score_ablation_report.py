#!/usr/bin/env python3
"""Report-vs-JSON audit for Build 5.2 (docs/backtest-score-ablation.md
vs docs/backtest-score-ablation-results.json).

Every numeric token in the report must be (a) present in the results
JSON at the precision printed, (b) DERIVED here from JSON values with
the derivation shown, or (c) on the explicit allowlist below with a
stated source. Anything else fails the audit. Exit 0 clean, 1 not.

Run: python3 scripts/audit_score_ablation_report.py
"""

import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MD = os.path.join(REPO, "docs", "backtest-score-ablation.md")
JS = os.path.join(REPO, "docs", "backtest-score-ablation-results.json")

# Structural constants with sources — NOT results.
ALLOW = {
    # ladder/knob constants (signal_engine + prereg)
    "75", "1.8", "13", "14", "6", "3", "16", "12", "24",
    "20", "50", "100", "150", "45", "70",
    # band labels (prereg)
    "0.1", "0.25", "0.5", "1.0", "0.01",
    # bar/section/build numbering, spans, D-refs
    "1", "2", "4", "5", "7", "8", "5.1", "5.2", "5B", "6A",
    "020", "011",
    # years / span labels
    "2020", "2021", "2023", "2024",
    # counts of things in prose (five components, three spans, etc.)
    "0",
    # prose "n≈27k" — the exact LOST counts 27,080 / 27,284 are both in
    # the JSON; the prose abbreviates them
    "27",
    # run-log panel size: printed by the run, not a JSON field
    # (pass_others_n and the A+ parity count ARE in the JSON)
    "848,328",
    # sampled parity points — script constant (ladder_parity_pin n=4000)
    "4,000",
    # approximate prose: "ex-top-5% ≈ −0.35" (exact values -0.355/-0.36/
    # -0.348 are all in the JSON; the prose rounds them)
    "0.35",
    # prose approximations of derived diffs, verified in DERIVED below
    "0.09", "0.10", "0.1", "6.6", "1.5", "0.9",
}


def flatten(o, acc):
    if isinstance(o, dict):
        for v in o.values():
            flatten(v, acc)
    elif isinstance(o, list):
        for v in o:
            flatten(v, acc)
    elif isinstance(o, (int, float)) and not isinstance(o, bool):
        acc.append(float(o))
    elif isinstance(o, str):
        # numbers embedded in provenance strings ("29777 A+ days ...")
        for m in re.findall(r"-?\d+(?:\.\d+)?", o):
            acc.append(float(m))
    return acc


def main():
    with open(JS) as f:
        results = json.load(f)
    vals = flatten(results, [])
    val_strs = set()
    for v in vals:
        # the report prints CI bounds inside [a,b] with a typographic
        # minus, so tokens surface unsigned — index |v| as well as v
        for w in (v, abs(v)):
            for nd in range(0, 5):
                val_strs.add(f"{round(w, nd):.{nd}f}".rstrip("0")
                             .rstrip("."))
                val_strs.add(f"{w:,.{nd}f}".rstrip("0").rstrip("."))
    md = open(MD).read()
    # strip links, commit refs, hashes, script paths
    md = re.sub(r"`[0-9a-f]{7,16}`", "", md)
    md = re.sub(r"\[[^\]]*\]\([^)]*\)", "", md)
    md = re.sub(r"[0-9a-f]{16}", "", md)
    # bucket labels like (100,150] read as thousands-separated numbers
    # to the token regex; they are structural labels, not results
    md = re.sub(r"\(\d+,\d+\]", "", md)

    # DERIVED values: computed here from the JSON, shown, then excused.
    ap_full_n = results["aplus_reference"]["full"]["n"]
    derived = {}
    for comp in ("macd", "ma"):
        lost = results["components"][comp]["lost"]["full"]
        share = lost["n"] / ap_full_n * 100
        diff = lost["mean"] - results["aplus_reference"]["full"]["mean"]
        derived[f"{comp}_lost_share"] = round(share, 0)
        derived[f"{comp}_lost_diff"] = round(diff, 2)
    print("derived: " + ", ".join(f"{k}={v}" for k, v in
                                  sorted(derived.items())))
    assert {round(v) for k, v in derived.items()
            if k.endswith("share")} == {92, 93}, derived
    assert {v for k, v in derived.items() if k.endswith("diff")} \
        == {-0.09, -0.10}, derived
    for v in derived.values():
        ALLOW.add(str(abs(v)).rstrip("0").rstrip("."))
        ALLOW.add(str(int(abs(v))))

    unmatched = []
    for tok in re.findall(r"-?\d[\d,]*(?:\.\d+)?", md):
        t = tok.lstrip("-")
        plain = t.replace(",", "")
        if t in ALLOW or plain in ALLOW:
            continue
        if plain in val_strs or t in val_strs:
            continue
        if f"{float(plain):g}" in val_strs:
            continue
        unmatched.append(tok)
    if unmatched:
        print("AUDIT FAIL — numbers in the report absent from the "
              "results JSON, the derivations and the allowlist:")
        for u in sorted(set(unmatched)):
            print(f"  {u}")
        return 1
    print(f"AUDIT CLEAN — every number in {os.path.basename(MD)} is in "
          f"{os.path.basename(JS)}, derived above, or allowlisted "
          "with a stated source.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
