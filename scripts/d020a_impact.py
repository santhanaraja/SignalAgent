#!/usr/bin/env python3
"""D-020a impact report — the pre-staging probe, committed for the record.

A LIVE-DATA snapshot (run 2026-08-09): re-running later reflects later
tape. The committed CSV beside the decision record
(docs/decisions/D-020a-impact-table.csv) is the measurement of record.
Two review corrections are baked in: the boundary names are DERIVED
(old-path >100% straddlers), never hardcoded; and the board regrade is
DATE-CONSISTENT — quality scores recomputed on frames as of the board's
bake date, isolating the scorer change from subsequent tape (a
date-mixed probe manufactured an NTAP B->A+ that vanishes at the bake
date; kept here as the methodology lesson).

Run: python3 scripts/d020a_impact.py
"""
import datetime
import json
import os
import sys

import numpy as np
import pandas as pd
import yfinance as yf

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import signal_engine as se

ranking = json.load(open(os.path.join(REPO, "data/universe_ranking.json")))
names = sorted({t["ticker"] for g in ranking["groups"] for t in g["tickers"]})
print(f"pool: {len(names)} names")

# ---- validate the 6mo slice against a direct fetch (grading frame) ----
d6 = se.fetch_data("HPE", period="6mo")
d1 = se.fetch_data("HPE", period="1y")
cutoff = d6.index[0]
sl = d1[d1.index >= cutoff]
assert len(sl) == len(d6) and sl.index[0] == d6.index[0]
assert float((sl["Close"] - d6["Close"]).abs().max()) < 1e-6
print(f"slice fidelity vs direct 6mo fetch OK — cutoff {cutoff.date()}")

# ---- index-level frame invariance (^GSPC) ----
g6 = se.fetch_data("^GSPC", period="6mo")
g1 = se.fetch_data("^GSPC", period="1y")
assert abs(float(g6["Close"].iloc[-1]) - float(g1["Close"].iloc[-1])) < 1e-6
print("index level frame-invariant (last close equal 6mo vs 1y): OK")
print(f"sp500_ytd: v1(6mo) {se.compute_ytd_return_v1(g6)} -> "
      f"v2(real) {se.compute_ytd_return_v2(g1)}")

# ---- batch 1y frames ----
frames = {}
for i in range(0, len(names), 100):
    chunk = names[i:i + 100]
    df = yf.download(chunk, period="1y", interval="1d", group_by="ticker",
                     auto_adjust=True, threads=True, progress=False)
    for t in chunk:
        try:
            if t in df.columns.get_level_values(0):
                sub = df[t].dropna(subset=["Close"])
                if len(sub) > 20:
                    sub = sub.copy()
                    try:
                        sub.index = sub.index.tz_localize(None)
                    except TypeError:
                        pass
                    frames[t] = sub
        except Exception:
            pass
missing = [t for t in names if t not in frames]
for t in missing[:40]:
    sub = se.fetch_data(t, period="1y")
    if sub is not None and len(sub) > 20:
        frames[t] = sub
missing = [t for t in names if t not in frames]
print(f"coverage {len(frames)}/{len(names)}"
      + (f" missing {missing}" if missing else ""))

rows = []
for t, f1y in frames.items():
    f6 = f1y[f1y.index >= cutoff]
    if len(f6) < 21:
        continue
    ytd_old = se.compute_ytd_return_v1(f6)          # grading today
    ytd_new, basis = se.compute_ytd_return_v2(f1y, with_basis=True)
    s1g, _, d1g = se.score_stock_v1(f6)
    s2g, _, d2g = se.score_stock_v2(f6, ytd_return=ytd_new,
                                    ytd_basis=basis)
    s1u, _, d1u = se.score_stock_v1(f1y)
    s2u, _, d2u = se.score_stock_v2(f1y)
    # stop-condition: non-ytd components identical on each path
    for a, b in ((d1g, d2g), (d1u, d2u)):
        for k in ("rsi", "macd", "ma", "vol"):
            assert a["score_components"][k] == b["score_components"][k], \
                (t, k)
    rows.append(dict(
        t=t, ytd_old=ytd_old, ytd_new=ytd_new, basis=basis,
        p_old=se.score_ytd_points_v1(ytd_old),
        p_new=se.score_ytd_points_v2(ytd_new),
        s1g=s1g, s2g=s2g, s1u=s1u, s2u=s2u,
        ytd_uni_old=d1u["ytd_return"]))
R = pd.DataFrame(rows)
print(f"\nstop condition: non-ytd components identical on BOTH paths for "
      f"all {len(R)} names: PASS")

dg = R.s2g - R.s1g
du = R.s2u - R.s1u
ch = R[dg != 0]
print(f"\n== GRADING-PATH score change: {len(ch)}/{len(R)} names")
print(f"   delta min {dg.min():+.0f} / median(changed) "
      f"{ch.s2g.sub(ch.s1g).median():+.1f} / max {dg.max():+.0f}; "
      f"up {int((dg > 0).sum())}, down {int((dg < 0).sum())}")
chu = R[du != 0]
print(f"== UNIVERSE-PATH score change: {len(chu)}/{len(R)} names; "
      f"min {du.min():+.0f} / median(changed) "
      f"{chu.s2u.sub(chu.s1u).median():+.1f} / max {du.max():+.0f}; "
      f"up {int((du > 0).sum())}, down {int((du < 0).sum())}")

g50_up = R[(R.s1u < 50) & (R.s2u >= 50)]
g50_dn = R[(R.s1u >= 50) & (R.s2u < 50)]
print(f"\n== universe score>=50 gate: {len(g50_up)} cross UP "
      f"({', '.join(g50_up.t)}), {len(g50_dn)} cross DOWN"
      + (f" ({', '.join(g50_dn.t)})" if len(g50_dn) else ""))
g75_up = R[(R.s1g < 75) & (R.s2g >= 75)]
g75_dn = R[(R.s1g >= 75) & (R.s2g < 75)]
print(f"== row-5 score>=75 (grading): {len(g75_up)} cross UP "
      f"({', '.join(g75_up.t)}), {len(g75_dn)} cross DOWN"
      + (f" ({', '.join(g75_dn.t)})" if len(g75_dn) else ""))

over_new = R[R.ytd_new > 100]
over_old = R[R.ytd_old > 100]
print(f"\n== above 100%: real YTD {len(over_new)} names vs 6mo-anchored "
      f"{len(over_old)} — the cap reaches {len(over_new) - len(over_old)} "
      f"further")

# boundary names: OLD paths (universe-v1 vs grading-v1) disagreeing
# across the >100% penalty boundary — DERIVED, not hardcoded (review
# finding: a hardcoded list tabulated the largest-swing set instead)
strad = R[(R.ytd_uni_old > 100) != (R.ytd_old > 100)]
print(f"\n== boundary names, old paths straddling >100% "
      f"({len(strad)}): {sorted(strad.t)}")
print(f"{'name':10} {'ytd_old':>8} {'ytd_new':>8} {'pts_old':>7} "
      f"{'pts_new':>7} {'score_old':>9} {'score_new':>9}")
for r in strad.sort_values("t").itertuples():
    print(f"{r.t:10} {r.ytd_old:8.2f} {r.ytd_new:8.2f} {r.p_old:7d} "
          f"{r.p_new:7d} {r.s1g:9.0f} {r.s2g:9.0f}")

swing = R.assign(a=(R.s2g - R.s1g).abs()).sort_values(
    "a", ascending=False).head(12)
print("\n== largest score swings (grading path — anchor honesty, a "
      "DIFFERENT set from the boundary names):")
print("   " + ", ".join(f"{r.t} {r.s1g:.0f}->{r.s2g:.0f}"
                        for r in swing.itertuples()))

for t in ("SNDK", "ERAS"):
    r = R[R.t == t]
    if not len(r):
        print(f"\n== {t}: no frame")
        continue
    r = r.iloc[0]
    f1y = frames[t]
    pri = f1y[f1y.index.year == f1y.index[-1].year - 1]
    print(f"\n== {t} end-to-end:")
    print(f"   anchor: prior-year last close "
          f"{float(pri['Close'].iloc[-1]):.2f} ({pri.index[-1].date()}) "
          f"-> last close {float(f1y['Close'].iloc[-1]):.2f} "
          f"({f1y.index[-1].date()})")
    print(f"   YTD: 6mo-anchored {r.ytd_old}% -> real {r.ytd_new}% "
          f"(basis {r.basis}); old universe-path 1y ytd {r.ytd_uni_old}%")
    print(f"   ytd points: v1({r.ytd_old})={r.p_old} -> "
          f"v2({r.ytd_new})={r.p_new}")
    print(f"   grading score {r.s1g:.0f} -> {r.s2g:.0f} "
          f"(row5>=75: {r.s1g >= 75} -> {r.s2g >= 75}); universe score "
          f"{r.s1u:.0f} -> {r.s2u:.0f} (gate>=50: {r.s1u >= 50} -> "
          f"{r.s2u >= 50})")

# ---- board regrade, DATE-CONSISTENT (review finding: a probe using
# today's frames mixed 16 days of tape into the counterfactual and
# manufactured an NTAP B->A+ that vanishes at the bake date) ----
sig = json.load(open(os.path.join(REPO, "public/signals.json")))
fw = json.load(open(os.path.join(REPO, "public/framework.json")))
cand = fw.get("candidate_grades") or {}
from framework.position_signals import PositionSignalEngine
eng = PositionSignalEngine({"positions": {}}, fetcher=None)
regime = (fw.get("regime") or {}).get("regime")
universe = {g.get("name") for g in sig.get("groups") or []}
breakers = {g.get("name"): g.get("breaker_status")
            for g in sig.get("groups") or []}
gen = (fw.get("generated_at") or "")[:10]
today = datetime.date.fromisoformat(gen) if gen else None
changes = []
graded = 0
for g in sig.get("groups") or []:
    for row in g.get("stocks") or []:
        t = row.get("ticker")
        rec = cand.get(t)
        gi = row.get("grade_inputs")
        if not rec or rec.get("grade") is None or not isinstance(gi, dict):
            continue
        try:
            h = yf.Ticker(t).history(period="2y", auto_adjust=True)
            h.index = h.index.tz_localize(None)
            h = h.dropna(subset=["Close"])
            h = h[h.index <= gen]
        except Exception:
            continue
        if len(h) < 40:
            continue
        six_b = h[h.index >= pd.Timestamp(gen) - pd.DateOffset(months=6)]
        s1_b, _, _ = se.score_stock_v1(six_b)
        y2_b, b_b = se.compute_ytd_return_v2(h, with_basis=True)
        s2_b, _, _ = se.score_stock_v2(six_b, ytd_return=y2_b,
                                       ytd_basis=b_b)
        graded += 1
        old_q = gi.get("quality_score")
        if s1_b != old_q:
            print(f"   note {t}: v1 bake-replay {s1_b} != recorded "
                  f"{old_q} (adjusted-price drift) — regrade still run")
        row2 = json.loads(json.dumps(row))
        # isolate the scorer change: ONLY quality_score moves, computed
        # on bake-date frames
        row2["grade_inputs"]["quality_score"] = float(s2_b)
        got = eng._grade_one_candidate(g.get("name"), row2, universe,
                                       breakers, regime, today)
        if got.get("grade") != rec.get("grade"):
            changes.append((t, rec.get("grade"), got.get("grade"),
                            old_q, float(s2_b)))
print(f"\n== board regrade under v2, date-consistent as of {gen} "
      f"({graded} recorded candidate grades):")
if not changes:
    print("   NO grade changes")
for t, old, new, oq, nq in changes:
    print(f"   {t}: {old} -> {new} (quality {oq} -> {nq:.0f})")

# ---- group avg_ytd sign flips (group_ytd breaker input) ----
flips = []
for g in sig.get("groups") or []:
    ts = [s_["ticker"] for s_ in g.get("stocks") or []]
    olds = [float(R[R.t == t].ytd_old.iloc[0]) for t in ts
            if len(R[R.t == t])]
    news = [float(R[R.t == t].ytd_new.iloc[0]) for t in ts
            if len(R[R.t == t])]
    if olds and news and (np.mean(olds) < 0) != (np.mean(news) < 0):
        flips.append((g["name"], round(float(np.mean(olds)), 1),
                      round(float(np.mean(news)), 1)))
print(f"\n== group avg_ytd sign flips (group_ytd breaker input): "
      f"{len(flips)}" + (f" {flips}" if flips else " — none"))

out = os.path.join(REPO, "docs", "decisions", "D-020a-impact-table.csv")
R.to_csv(out, index=False)
print(f"\nfull table saved: {out}")
