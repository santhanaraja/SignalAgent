#!/usr/bin/env python3
"""Pins for the classification-alias change (2026-08-11, D-023).

The invariant, in one sentence: **no group the pool produces is named with a
vendor's own label** — because universe_builder groups by that string, so an
uncanonicalised label becomes a group of its own that can never reach
`min_candidates`, while the canonical group where its members belong is
computed without them.

Five invariants, each demonstrating the failure it exists to catch:

  1. Every group name a REAL rotation produced is canonical under the current
     alias map — driven off the committed artifact, not a hand-built list.
  2. The failure, demonstrated: drop any one of the five aliases and exactly
     that vendor label comes back as an unreviewed group name.
  3. The HARM, demonstrated end to end through the real classifier and the
     real ranker: without the alias the group exists, sits below
     min_candidates and carries `below_min_candidates` — permanently
     ineligible at any composite. With it, the members compete.
  4. The two maps stay honest: no alias chains (canon() is single-pass), no
     label both mapped and declared unmappable, every target canonical, every
     unmappable entry carrying a reason.
  5. The vocabulary is load-bearing, so an unreadable one fails LOUD — it must
     never report "no vendor labels found" because it could not look.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

import gics_classifier as gc
import universe_builder as ub
import universe_source as us

# The rotation that produced the historical instance this change repairs.
# Anchored to a COMMIT, never to HEAD: once a post-fix rotation is committed,
# HEAD stops carrying the defect and a HEAD-anchored demonstration would
# quietly evaporate at exactly the moment it started costing something.
PRE_FIX_ROTATION = "fd83fa4"      # 2026-08-07 rotation, 15 groups, pool v1

# The vendor labels the 2026-08-11 sweep found in the live pool, with the
# candidates stranded in each. Recorded here so the pin reproduces the
# instance by name rather than re-deriving it from a cache CI does not have
# (data/universe_cache/gics_cache.json is gitignored).
OBSERVED_2026_08_11 = {
    "Internet Retail": ["MELI", "PDD"],
    "Beverages - Non - Alcoholic": ["CCEP"],
    "Medical Devices": ["GKOS"],
    "REIT - Healthcare Facilities": ["CTRE"],
    "Electronic Gaming & Multimedia": ["EA"],      # rotation environment only
    "Specialty Business Services": ["TRI"],        # unmappable by alias
    "Health Information Services": ["BTSG"],       # unmappable by alias
    "Capital Markets": ["HUT"],                    # unmappable by alias
}

# Alias targets that are real GICS sub-industries with no S&P 500 constituent,
# and so are absent from the committed vocabulary. Adding to this set must be
# a deliberate act — that is the point of it being here and not in the config.
TARGETS_OUTSIDE_THE_VOCABULARY = {"Coal & Consumable Fuels"}


def _cfg():
    uni = us.load_universe_config()
    return uni.get("gics_aliases") or {}, uni.get("gics_unmappable_labels") or {}


def _artifact(ref):
    out = subprocess.run(["git", "show", f"{ref}:data/universe_active.json"],
                         capture_output=True, text=True, cwd=REPO)
    if out.returncode != 0:
        raise RuntimeError(
            f"cannot read the committed rotation artifact at {ref} — this pin "
            f"asserts against real production output and must not pass by "
            f"skipping: {out.stderr.strip()[:200]}")
    return json.loads(out.stdout)


# ----------------------------------------------------------------------
# 1. Every group name a real rotation produced is canonical.
# ----------------------------------------------------------------------
def test_no_rotation_group_name_is_a_vendor_label():
    aliases, unmappable = _cfg()
    names = [g["name"] for g in _artifact("HEAD")["ranking"]]
    assert len(names) > 50, f"only {len(names)} group names — assertion would be vacuous"

    # The invariant is about the CURRENT map: whatever labels production has
    # actually emitted, the map in force today must canonicalise all of them.
    canon = {aliases.get(n, n) for n in names}
    flagged = gc.unaliased_labels(canon, aliases=aliases, unmappable=unmappable)
    assert flagged["new"] == [], (
        f"group names in the committed rotation that no alias covers and no "
        f"config entry excuses: {flagged['new']}. Each is a group of its own, "
        f"below min_candidates, permanently ineligible — add an alias, or an "
        f"entry to universe.gics_unmappable_labels with the reason.")
    print(f"  (1a) all {len(canon)} group names in the committed rotation are "
          f"canonical under the current alias map: OK")
    assert set(flagged["known"]) <= set(unmappable), flagged["known"]
    print(f"  (1b) still-unmapped labels are exactly the ones config declares "
          f"unmappable, with reasons: {flagged['known']}: OK")

    # THE HISTORICAL INSTANCE, at its own commit. One GICS sub-industry split
    # into two one-member groups because EA left the S&P 500 CSV and fell
    # through to Yahoo, while TTWO kept the authoritative seed.
    old = [g for g in _artifact(PRE_FIX_ROTATION)["ranking"]
           if g["name"] in ("Electronic Gaming & Multimedia",
                            "Interactive Home Entertainment")]
    assert len(old) == 2, f"the {PRE_FIX_ROTATION} instance is not in this artifact: {old}"
    assert all(g["candidates"] == 1 for g in old), old
    assert all(g["exclusion_reason"] == "below_min_candidates" for g in old), old
    merged = {aliases.get(g["name"], g["name"]) for g in old}
    assert merged == {"Interactive Home Entertainment"}, merged
    print(f"  (1c) the {PRE_FIX_ROTATION} rotation split EA and TTWO into two "
          f"one-member groups, both below_min_candidates; the current map "
          f"merges them into one: OK")


# ----------------------------------------------------------------------
# 2. Drop an alias and the vendor label comes straight back.
# ----------------------------------------------------------------------
def test_dropping_any_alias_reintroduces_the_label():
    aliases, unmappable = _cfg()
    labels = list(OBSERVED_2026_08_11)

    clean = gc.unaliased_labels({aliases.get(l, l) for l in labels},
                                aliases=aliases, unmappable=unmappable)
    assert clean["new"] == [], clean
    assert set(clean["known"]) == {"Specialty Business Services",
                                   "Health Information Services",
                                   "Capital Markets"}, clean
    print(f"  (2a) of the 8 vendor labels observed on 2026-08-11, 5 are "
          f"canonicalised and 3 are declared unmappable: OK")

    # THE FAILURE, DEMONSTRATED — one alias at a time, against the same input.
    for label in [l for l in labels if l in aliases]:
        without = {k: v for k, v in aliases.items() if k != label}
        broken = gc.unaliased_labels({without.get(l, l) for l in labels},
                                     aliases=without, unmappable=unmappable)
        assert broken["new"] == [label], (
            f"removing the {label!r} alias did not resurface it — this pin "
            f"cannot tell a covered label from an uncovered one: {broken}")
        print(f"  (2b) without the alias, {label!r} is back as an unreviewed "
              f"group name holding {OBSERVED_2026_08_11[label]}: OK")


# ----------------------------------------------------------------------
# 3. The harm itself: an unaliased label is a permanently ineligible group.
# ----------------------------------------------------------------------
def _metric(t, score=70, ytd=20.0):
    return {"ticker": t, "score": score, "history_days": 400,
            "avg_volume": 5_000_000, "ytd": ytd, "r3m": 5.0, "r1m": 2.0,
            "score_components": {}, "above_50dma": True}


def test_an_unaliased_label_is_a_group_that_can_never_be_eligible():
    """Driven through the real classifier and the real ranker.

    The string check above says a label is uncovered. This says what that
    COSTS: the members sit in a group of their own, below min_candidates, so
    no composite can ever make them eligible — while the canonical group they
    belong to is ranked without them.
    """
    aliases, _ = _cfg()
    tmp = tempfile.mkdtemp()
    try:
        # A real Yahoo cache, in the shape gics_classifier writes it.
        raw = {"AMZN": "Broadline Retail", "EBAY": "Broadline Retail",
               "MELI": "Internet Retail", "PDD": "Internet Retail"}
        with open(os.path.join(tmp, "gics_cache.json"), "w") as f:
            json.dump({t: {"sub_industry": s, "source": "yfinance",
                           "fetched_at": gc._now_utc().isoformat()}
                       for t, s in raw.items()}, f)

        rot = dict(ub.DEFAULT_ROTATION)
        rot["composite_weights"] = dict(ub.DEFAULT_ROTATION["composite_weights"])
        metrics = {t: _metric(t) for t in raw}
        caps = {t: {"market_cap_usd": 50e9} for t in raw}

        def rank(alias_map):
            clf = gc.GICSClassifier(tmp, aliases=alias_map)
            by_gics = clf.get_universe_by_gics(sorted(raw), allow_remote=False)
            ranking, selected = ub.rank_and_select(by_gics, metrics, caps, rot)
            return by_gics, {g["name"]: g for g in ranking}, selected

        # WITHOUT the alias — the defect.
        by_gics, ranking, selected = rank({k: v for k, v in aliases.items()
                                           if k != "Internet Retail"})
        assert sorted(by_gics) == ["Broadline Retail", "Internet Retail"], by_gics
        phantom = ranking["Internet Retail"]
        assert phantom["candidates"] == 2 and phantom["qualifier_count"] == 2
        assert phantom["eligible"] is False
        assert phantom["exclusion_reason"] == "below_min_candidates"
        assert ranking["Broadline Retail"]["candidates"] == 2
        assert ranking["Broadline Retail"]["eligible"] is False
        assert selected == [], "a 2-member group must not be selectable"
        print("  (3a) unaliased: two groups of two, BOTH below_min_candidates "
              "— four qualifying names, none of them reachable at any "
              "composite, and nothing selected: OK")

        # WITH the alias — one group, above the floor, competing.
        by_gics, ranking, selected = rank(aliases)
        assert sorted(by_gics) == ["Broadline Retail"], by_gics
        g = ranking["Broadline Retail"]
        assert g["candidates"] == 4 and g["qualifier_count"] == 4
        assert g["eligible"] is True and g["exclusion_reason"] is None
        assert [q["ticker"] for q in g["qualifiers"]] == ["AMZN", "EBAY", "MELI", "PDD"]
        assert [s["name"] for s in selected] == ["Broadline Retail"]
        print("  (3b) aliased: one group of four, eligible and selected, with "
              "MELI and PDD competing beside AMZN and EBAY: OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ----------------------------------------------------------------------
# 4. The two maps stay honest.
# ----------------------------------------------------------------------
def test_alias_and_unmappable_maps_are_coherent():
    aliases, unmappable = _cfg()
    vocab = gc.load_vocabulary()

    # (a) canon() is a SINGLE lookup. An alias whose target is itself an alias
    #     key resolves half way and lands on a label nobody meant to keep.
    chains = {k: v for k, v in aliases.items() if v in aliases}
    assert not chains, (
        f"alias chains resolve only one step in GICSClassifier.canon: {chains}")
    print("  (4a) no alias target is itself an alias key — canon() is "
          "single-pass, so a chain would strand names on the middle label: OK")

    tmp = tempfile.mkdtemp()
    try:
        clf = gc.GICSClassifier(tmp, aliases={"A": "B", "B": "Broadline Retail"})
        assert clf.canon("A") == "B", "canon() resolved a chain — update this pin"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  (4b) demonstrated: with a chain in place, canon('A') stops at "
          "'B' rather than reaching the canonical name: OK")

    # (c) a label cannot be both mapped and declared unmappable.
    both = set(aliases) & set(unmappable)
    assert not both, f"labels both aliased and declared unmappable: {both}"

    # (d) every alias target is a canonical name, not another vendor label.
    bad = sorted(t for t in set(aliases.values())
                 if t not in vocab and t not in TARGETS_OUTSIDE_THE_VOCABULARY)
    assert not bad, (
        f"alias targets that are not canonical GICS sub-industries: {bad}. If "
        f"one is a real sub-industry with no S&P 500 member, add it to "
        f"TARGETS_OUTSIDE_THE_VOCABULARY in this pin and say why.")
    print(f"  (4c) all {len(set(aliases.values()))} alias targets are "
          f"canonical; no label is both aliased and unmappable: OK")

    # (e) an unmappable entry is a decision, so it carries the reasoning.
    thin = {k: v for k, v in unmappable.items() if len(str(v).strip()) < 80}
    assert not thin, f"unmappable labels without a real reason: {list(thin)}"
    print(f"  (4d) all {len(unmappable)} unmappable labels carry a stated "
          f"reason: OK")


# ----------------------------------------------------------------------
# 5. An unreadable vocabulary fails loud, never clean.
# ----------------------------------------------------------------------
def test_a_missing_vocabulary_cannot_report_all_clear():
    real = gc.load_vocabulary()
    assert len(real) > 100, f"committed vocabulary has only {len(real)} names"
    print(f"  (5a) the committed vocabulary loads: {len(real)} canonical "
          f"sub-industry names: OK")

    tmp = tempfile.mkdtemp()
    try:
        empty = os.path.join(tmp, "empty.json")
        with open(empty, "w") as f:
            json.dump({"sub_industries": []}, f)
        for path in (empty, os.path.join(tmp, "absent.json")):
            try:
                gc.load_vocabulary(path)
                raise AssertionError(
                    f"load_vocabulary({os.path.basename(path)}) returned "
                    f"instead of raising — an empty vocabulary makes every "
                    f"vendor label look canonical")
            except (ValueError, OSError):
                pass
        print("  (5b) an empty or absent vocabulary RAISES rather than "
              "reporting a clean sweep: OK")

        # And the production caller degrades to `error`, never to a clean
        # empty result — the check not running must not read as nothing found.
        saved = gc.VOCABULARY_PATH
        gc.VOCABULARY_PATH = os.path.join(tmp, "absent.json")
        try:
            out = gc.GICSClassifier(tmp, aliases={}).unaliased_group_names(
                ["Internet Retail", "Broadline Retail"])
        finally:
            gc.VOCABULARY_PATH = saved
        assert out["error"] and out["new"] == [], out
        print("  (5c) with the vocabulary gone the build records an `error` "
              "rather than an empty `new` list: OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    print("\n=== GICS alias pins (D-023, 2026-08-11) ===")
    test_no_rotation_group_name_is_a_vendor_label()
    test_dropping_any_alias_reintroduces_the_label()
    test_an_unaliased_label_is_a_group_that_can_never_be_eligible()
    test_alias_and_unmappable_maps_are_coherent()
    test_a_missing_vocabulary_cannot_report_all_clear()
    print("\nAll GICS alias pins passed.\n")
