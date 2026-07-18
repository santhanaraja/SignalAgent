#!/usr/bin/env python3
"""
Fear & Greed engine tests (D-012 rebuild). Covers:
- per-component raw->0-100 score-curve pins (endpoints from docs/sentiment.md)
- Market Internals breadth: explicit above_50dma field only (the derive
  fallback was removed 2026-07-18 per its expiry note), fixture universe
- daily persistence: once-per-day upsert / new-day append matrix

No network: pure score maps + a fixture file + injected readings.
"""
import os
import json
import tempfile

import fear_greed_engine as fg


def test_score_curves():
    # endpoints as documented in docs/sentiment.md
    assert fg._score_momentum(-8) == 0 and fg._score_momentum(8) == 100 and fg._score_momentum(0) == 50
    # `combined` is the sign-flipped VIX blend: -20 (surging) -> 0, +20 (falling) -> 100
    assert fg._score_put_call(-20) == 0 and fg._score_put_call(20) == 100 and fg._score_put_call(0) == 50
    # % above 50DMA is already a 0-100 breadth reading
    assert fg._score_internals(0) == 0 and fg._score_internals(100) == 100 and fg._score_internals(50) == 50
    assert fg._score_safe_haven(-6) == 0 and fg._score_safe_haven(6) == 100 and fg._score_safe_haven(0) == 50
    assert fg._score_junk(-4) == 0 and fg._score_junk(4) == 100 and fg._score_junk(0) == 50
    # clamp holds beyond the endpoints
    assert fg._score_momentum(-99) == 0 and fg._score_momentum(99) == 100
    assert fg._score_internals(-5) == 0 and fg._score_internals(140) == 100
    print("  score curves: momentum/put-call/internals/safe-haven/junk endpoints + clamp: OK")


def test_above_50dma_field_only():
    # the derive-fallback was removed 2026-07-18 per its expiry note (the
    # rotation confirmed the field populates) — the EXPLICIT field is the
    # only source; anything else is excluded from the breadth count
    assert fg._ticker_above_50dma({"above_50dma": True}) is True
    assert fg._ticker_above_50dma({"above_50dma": False}) is False
    # ma components alone no longer map (the fallback is gone)
    assert fg._ticker_above_50dma({"components": {"ma": 14}}) is None
    assert fg._ticker_above_50dma({"components": {"ma": -14}}) is None
    # non-bool / missing -> None (excluded)
    assert fg._ticker_above_50dma({"above_50dma": 1}) is None
    assert fg._ticker_above_50dma({}) is None
    print("  above-50DMA: explicit field only (fallback removed per expiry), "
          "non-bool excluded: OK")


def test_market_internals_fixture():
    # field-only counting: rows without the explicit field are EXCLUDED
    fixture = {"ranking": [
        {"tickers": [
            {"ticker": "A", "above_50dma": True},              # above (field)
            {"ticker": "B", "components": {"ma": 6}},          # field-less -> EXCLUDED
            {"ticker": "C", "above_50dma": False},             # below (field)
            {"ticker": "D", "above_50dma": False},             # below (field)
        ]},
        {"tickers": [
            {"ticker": "E", "components": {"ma": None}},       # excluded
            {"ticker": "F", "above_50dma": True},              # above
        ]},
    ]}
    p = os.path.join(tempfile.gettempdir(), "fg_uni_fixture.json")
    with open(p, "w") as f:
        json.dump(fixture, f)
    try:
        r = fg.compute_market_internals(path=p)
    finally:
        os.remove(p)
    # above = A,F = 2 of 4 valid (B and E excluded — no field) = 50%
    assert r["score"] == 50, r
    assert r["value"] == "2/4 (50%)", r
    assert r["label"] == "Neutral", r
    # missing file degrades to neutral, never crashes
    assert fg.compute_market_internals(path="/nonexistent/uni.json")["score"] == 50
    print("  market internals: field-only counting, 2/4=50%, missing->neutral: OK")


def test_persistence_once_per_day():
    reading = {"composite_score": 72, "composite_label": "Greed",
               "indicators": [{"name": "Market Internals", "score": 69, "value": "367/530 (69%)"}]}
    p = os.path.join(tempfile.gettempdir(), "fg_hist_matrix.json")
    if os.path.exists(p):
        os.remove(p)
    try:
        # first write of the day -> append
        fg.append_daily_history(path=p, reading=reading, today="2026-07-13")
        h = json.load(open(p)); assert len(h) == 1 and h[0]["composite"] == 72
        # same day again -> upsert (post-close value wins), not a duplicate
        fg.append_daily_history(path=p, reading={**reading, "composite_score": 78}, today="2026-07-13")
        h = json.load(open(p)); assert len(h) == 1 and h[-1]["composite"] == 78
        # new day -> append
        fg.append_daily_history(path=p, reading=reading, today="2026-07-14")
        h = json.load(open(p)); assert len(h) == 2 and h[-1]["date"] == "2026-07-14"
        # entry shape + raw carried through
        e = h[-1]
        assert set(e) == {"date", "composite", "label", "components"}, e
        assert e["components"][0]["raw"] == "367/530 (69%)"
        # non-monotonic history: a same-day entry that is NOT last must still
        # upsert (scan-by-date), never append a duplicate date
        with open(p, "w") as f:
            json.dump([
                {"date": "2026-07-20", "composite": 50, "label": "Neutral", "components": []},
                {"date": "2026-07-19", "composite": 40, "label": "Fear", "components": []},
            ], f)
        fg.append_daily_history(path=p, reading=reading, today="2026-07-20")
        h = json.load(open(p))
        assert sum(1 for x in h if x["date"] == "2026-07-20") == 1, "duplicate date appended"
        assert next(x for x in h if x["date"] == "2026-07-20")["composite"] == 72
        # corrupt/absent file tolerated -> starts fresh, never crashes
        with open(p, "w") as f:
            f.write("{ not json")
        fg.append_daily_history(path=p, reading=reading, today="2026-07-15")
        h = json.load(open(p)); assert len(h) == 1 and h[0]["date"] == "2026-07-15"
    finally:
        if os.path.exists(p):
            os.remove(p)
    print("  persistence: append/upsert/new-day + non-monotonic upsert, entry shape, corrupt-file recovery: OK")


if __name__ == "__main__":
    print("\n=== Fear & Greed engine tests (D-012 rebuild) ===")
    test_score_curves()
    test_above_50dma_field_only()
    test_market_internals_fixture()
    test_persistence_once_per_day()
    print("\nAll Fear & Greed tests passed.")
