#!/usr/bin/env python3
"""
Behavioral Sentiment engine tests (D-013 rebuild). Covers:
- the five behavioural factor maps + the Technical Sentiment aggregator/buckets
- /api/sentiment/simulate parity — the endpoint must return exactly what the
  pure technical_sentiment() returns (Lab law 1: no drift)
- relative-strength math (with and without a GICS group)
- graceful degradation: news fetch failing, universe file missing

No network: pure functions + a Flask test client + a stubbed yfinance.
"""
import sys
import types
import json

import sentiment_engine as se


def test_factor_helpers():
    # 52w range position
    assert se._range_position_score(10, 10, 20) == 0
    assert se._range_position_score(15, 10, 20) == 50
    assert se._range_position_score(20, 10, 20) == 100
    assert se._range_position_score(15, 20, 20) == 50   # degenerate range -> neutral
    # volume trend (accumulation vs distribution)
    assert se._volume_trend_score(100, 0) == 100 and se._volume_trend_score(0, 100) == 0
    assert se._volume_trend_score(50, 50) == 50 and se._volume_trend_score(0, 0) == 50
    # momentum posture (RSI base +/- MACD tilt)
    assert se._momentum_posture_score(60, True, True) == 70    # bullish + rising = +10
    assert se._momentum_posture_score(60, True, False) == 65   # bullish = +5
    assert se._momentum_posture_score(60, False, False) == 50  # bearish + falling = -10
    assert se._momentum_posture_score(98, True, True) == 100   # clamps at 100
    # sma structure: 0/1/2/3 bullish conditions -> 0/33/67/100
    assert se._sma_structure_score(1, 2, 3) == 0                # none
    assert round(se._sma_structure_score(10, 2, 3)) == 67       # price>both, sma20<sma50
    assert se._sma_structure_score(10, 5, 3) == 100             # all three
    # 20d return percentile vs self
    assert se._return_percentile_score(5, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) == 50
    assert se._return_percentile_score(11, [1, 2, 3]) == 100
    assert se._return_percentile_score(1, []) == 50            # no history -> neutral
    print("  factor helpers: range/volume/momentum/sma/percentile endpoints + clamps: OK")


def test_aggregator_and_buckets():
    t = se.technical_sentiment({"range_position": 90, "volume_trend": 80,
                                "momentum_posture": 70, "sma_structure": 100,
                                "return_percentile": 60})
    assert t["score"] == 80 and t["bucket"] == "Bullish"       # mean = 80
    assert se._bucket(60) == "Bullish" and se._bucket(59) == "Trending"
    assert se._bucket(40) == "Bearish" and se._bucket(41) == "Trending"
    # validation: missing / non-numeric / out-of-range factors are rejected
    for bad in ({"range_position": 90}, {**{f: 50 for f in se.FACTORS}, "volume_trend": "x"},
                {**{f: 50 for f in se.FACTORS}, "sma_structure": 140},
                {**{f: 50 for f in se.FACTORS}, "momentum_posture": True}):
        try:
            se.technical_sentiment(bad); assert False, bad
        except (KeyError, ValueError):
            pass
    print("  aggregator: mean+bucket, boundary buckets, input validation: OK")


def test_relative_strength():
    r = se.relative_strength(5.0, 2.0, 3.0, "Semis")
    assert r["vs_spy"] == 3.0 and r["vs_group"] == 2.0 and r["group_vs_market"] == 1.0
    assert r["group_name"] == "Semis"
    r2 = se.relative_strength(5.0, 2.0)                        # no group -> graceful
    assert r2["vs_spy"] == 3.0 and r2["vs_group"] is None and r2["group_vs_market"] is None
    print("  relative strength: vs-market/vs-group/group-vs-market math, group-absent degrade: OK")


def test_simulate_endpoint_parity():
    # The endpoint must return exactly what the pure function returns (no drift).
    import ticker_api
    client = ticker_api.app.test_client()
    factors = {"range_position": 82, "volume_trend": 47, "momentum_posture": 66,
               "sma_structure": 100, "return_percentile": 55}
    resp = client.post("/api/sentiment/simulate", json=factors)
    assert resp.status_code == 200, resp.status_code
    body = resp.get_json()
    pure = se.technical_sentiment(factors)
    assert body["score"] == pure["score"], (body, pure)
    assert body["bucket"] == pure["bucket"]
    assert body["components"] == pure["components"]
    # bad input -> 400, never 500
    assert client.post("/api/sentiment/simulate", json={"range_position": 50}).status_code == 400
    assert client.post("/api/sentiment/simulate", json={**factors, "volume_trend": 140}).status_code == 400
    assert client.post("/api/sentiment/simulate", data="not json",
                       content_type="application/json").status_code == 400
    print("  simulate endpoint: parity with technical_sentiment(), bad-input 400s: OK")


def test_graceful_degradation():
    old = sys.modules.get("yfinance")

    def _use(news_or_raise):
        stub = types.ModuleType("yfinance")

        class _T:
            def __init__(self, *a, **k):
                if news_or_raise == "raise":
                    raise RuntimeError("no network")
            news = None if news_or_raise == "raise" else news_or_raise
        stub.Ticker = _T
        sys.modules["yfinance"] = stub

    try:
        # 1) fetch raising -> [] (strip omitted), never crashes
        _use("raise")
        assert se.fetch_news("AAPL") == []
        # 2) malformed items (str / None / int / dict-without-title) skipped;
        #    the one well-formed item is kept (per-item graceful, review finding)
        _use(["bad string", None, 12345, {"nope": 1},
              {"content": {"title": "Good headline",
                           "provider": {"displayName": "Src"},
                           "canonicalUrl": {"url": "https://x.com/a"}}}])
        out = se.fetch_news("AAPL")
        assert len(out) == 1 and out[0]["title"] == "Good headline", out
        assert out[0]["publisher"] == "Src" and out[0]["link"] == "https://x.com/a"
        # 3) .news not a list -> []
        _use({"unexpected": "shape"})
        assert se.fetch_news("AAPL") == []
    finally:
        if old is not None:
            sys.modules["yfinance"] = old
        else:
            sys.modules.pop("yfinance", None)
    # universe file missing -> (None, None), no crash
    assert se._group_for_symbol("AAPL", path="/nonexistent/universe.json") == (None, None)
    print("  degradation: news fetch failure / malformed items / non-list -> [], missing universe: OK")


if __name__ == "__main__":
    print("\n=== Behavioral Sentiment engine tests (D-013 rebuild) ===")
    test_factor_helpers()
    test_aggregator_and_buckets()
    test_relative_strength()
    test_simulate_endpoint_parity()
    test_graceful_degradation()
    print("\nAll sentiment tests passed.")
