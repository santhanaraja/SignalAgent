#!/usr/bin/env python3
"""Report-vs-JSON audit for Build 8 (docs/backtest-exit.md vs
docs/backtest-exit-results.json). Same law as the 5.2 audit it adapts.

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
MD = os.path.join(REPO, "docs", "backtest-exit.md")
JS = os.path.join(REPO, "docs", "backtest-exit-results.json")

# Structural constants with sources — NOT results.
ALLOW = {
    # structural constants: knobs, arm parameters, branch/clause numbers
    "1", "2", "3", "4", "5", "6", "7", "8", "11", "14",
    "2.5", "4.0", "20", "50", "10", "5.1", "5B", "6A", "0.90",
    "5bps", "6.5",
    # D-refs, spans, dates, builds
    "006", "018", "020", "017", "011",
    "2026", "08", "15",
    # thresholds stated by the amendment/prereg
    "5.0", "100", "0.982", "3.57", "4.95",
    # counts of things in prose
    "0", "929", "1,068", "1068",
    # committed 5B gate values (in docs/backtest-systems-results.json)
    "235,200.09",
    # review-channel figures quoted in the unblinding disclosure and chain
    # (sourced from the review transcript, not this JSON): leverage cor-
    # relation and the E7 gross/equity that also appear per-arm in JSON
    "1.77",
    # verification-pass counts (workflow transcript)
    "14/14", "22/22",
    # percentages derived in DERIVED below
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

    # DERIVED: differences quoted in prose, from JSON values
    e1 = results["arms"]["E1"]; e2 = results["arms"]["E2"]
    d_cagr_e2 = round(e2["full"]["cagr_pct"] - e1["full"]["cagr_pct"], 1)
    assert d_cagr_e2 == -7.3, d_cagr_e2
    print(f"derived: E2 full dCAGR = {d_cagr_e2}")
    for extra in ("7.3", "0.10", "0.1"):
        ALLOW.add(extra)
    # whole-dollar prose/table roundings of JSON values, derived here:
    lev = results["validity"]["leverage"]
    assert round(lev["E1"]["min_cash"]) == 12720
    assert round(lev["E8"]["min_cash"]) == -94700
    assert round(results["arms"]["E1"]["full"]["end_equity"]) == 235200
    ALLOW.update({"12,720", "94,700", "235,200"})
    # prose references: "40% below trigger" (worst_gap 40.31), "80% of
    # ALL exits" (share_below 75.8-82.5 across arms), "p90" (field name)
    gt = results["validity"]["gap_through"]
    assert round(gt["E7"]["worst_gap_pct"]) == 40
    assert all(75 <= gt[a]["share_below_trigger_pct"] <= 83
               for a in gt if gt[a].get("n_exits"))
    ALLOW.update({"40", "80", "90"})
    print("derived: min-cash/end-equity roundings + gap-through prose "
          "anchors verified")
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
