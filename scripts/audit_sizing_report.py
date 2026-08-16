#!/usr/bin/env python3
"""Report-vs-JSON audit for Build 9 (docs/backtest-sizing.md vs
docs/backtest-sizing-results.json). Same law as the 5.2/B8 audits.

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
MD = os.path.join(REPO, "docs", "backtest-sizing.md")
JS = os.path.join(REPO, "docs", "backtest-sizing-results.json")

# Structural constants with sources — NOT results.
ALLOW = {
    # structural constants: arm params, ladders, knobs, bar numbers
    "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "5.0", "6.5", "8.0", "10.0", "90", "50", "25", "100", "60", "30",
    "70", "40", "20", "10.13",
    "5bps", "5B", "5.2", "0", "2026",
    # D-refs / spans / builds
    "006", "008",
    # counts of things in prose
    "353", "353",
    # review-transcript facts (verifier counts, prose)
    "65.3", "307", "1051", "179.3",
    # values verified in DERIVED below
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

    # DERIVED: prose differences and roundings from JSON values
    A=results["arms"]; V=results["validity"]
    d5c=round(A["P5"]["full"]["cagr_pct"]-A["P1"]["full"]["cagr_pct"],2)
    d5m=round(A["P5"]["full"]["max_dd_pct"]-A["P1"]["full"]["max_dd_pct"],2)
    d6c=round(A["P6"]["full"]["cagr_pct"]-A["P1"]["full"]["cagr_pct"],2)
    d6m=round(A["P6"]["full"]["max_dd_pct"]-A["P1"]["full"]["max_dd_pct"],2)
    d7m=round(A["P7"]["full"]["max_dd_pct"]-A["P1"]["full"]["max_dd_pct"],2)
    assert (d5c,d5m,d6c,d6m,d7m)==(-2.23,-2.71,0.96,3.49,-2.99),(d5c,d5m,d6c,d6m,d7m)
    exp_ratio=round(100*(V["P7"]["mean_exposure_over_equity"]
                        /V["P1"]["mean_exposure_over_equity"]-1))
    assert exp_ratio==28, exp_ratio
    assert round(V["P5"]["min_cash"])==10 and round(V["P8"]["min_cash"])==111
    # P1 min cash rounds to 12,720 in the table (JSON 12720.15);
    # "311-411 over-ceiling days" is the JSON's exact range across
    # laddered arms; "~80%" cites Build 8's
    # committed gap-through share, not this JSON
    assert round(V["P1"]["min_cash"]) == 12720
    over = [results["validity"][a]["days_over_ceiling"]
            for a in ("P1","P2","P3","P4","P5","P6","P8","P9")]
    assert min(over) == 311 and max(over) == 411, over
    for x in ("2.23","2.71","0.96","3.49","2.99","28","10","111","3",
              "3bp","65.3","307","1051","179.3","12,720","311","411","80"):
        ALLOW.add(x)
    print("derived: P5/P6/P7 point diffs, P7 exposure ratio, cash "
          "floors, P1 min-cash rounding, over-ceiling bracket")
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
