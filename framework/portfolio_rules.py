#!/usr/bin/env python3
"""
R28 — real-dollar portfolio enforcement (PER-508 Phase 0 flagship).
Rulings D-007 (concentration moves to computed dollars) and D-008 Q4
(the regime-scaled total-exposure ceiling 90/50/25/5) govern.

The core is a PURE function — assess_portfolio() — per Lab law 1
(D-010): the future Sizing Lab and any replay harness call the same
code the pipeline runs. No I/O, no fetches: prices are the last EOD
closes already computed by the position engine (EOD basis, consistent
with every other decision surface).

ENFORCEMENT CLASS: COMPUTED (reporting-hard). R28 computes and reports
dollar-true statuses; it cannot block broker orders and does not
pretend to. In particular, a ceiling breach caused by a REGIME
DOWNSHIFT (e.g. 60% deployed when the gauge drops to Caution/25) is
action_needed — "reduce exposure below the ceiling via normal exit
discipline" — an advisory to derisk through stops and R11, never a
same-day forced liquidation. R28 reports; the exit rules govern
execution.

Distinguishing violation from action_needed without position history:
at the TOP regime (Trending, the 90% ceiling) no higher state exists to
have downshifted from, so a breach there can only come from buying past
the ceiling -> "violation". At any lower regime a breach is presumed to
be (and is anyway best treated as) a downshift -> "action_needed" with
the derisk-via-discipline message. Both roads lead through the same
exits; only the label and urgency differ.

Capital is the config key `account_capital_usd` — user-maintained by
hand until a broker integration exists (documented limitation; the
module degrades to an "unconfigured" status rather than guessing).

Data-integrity conventions (positions.json is hand-maintained, so bad
rows must degrade LOUDLY, never silently skew the aggregates):
- shares must be a positive finite number (long-only book) — anything
  else becomes an "invalid" row, excluded from every aggregate;
- two lots of one ticker aggregate BEFORE valuation (one position, one
  raw-price valuation — never a rounded intermediate);
- unpriced rows still COUNT toward the per-group position cap (the
  position exists whether or not its price resolved) but never toward
  exposure, and any unpriced row is called out on the ceiling line;
- if NO holding can be priced, the block flags degraded=true and the
  ceiling reports "unavailable" — an artifact outage must never render
  as a compliant flat book.
"""

import math

# D-008 Q4 ceilings (Trending 90 / Choppy 50 / Caution 25 / Risk-off 5).
# An unknown/missing regime state resolves to Caution's ceiling — the
# gauge's own outage convention (2+ dark voters -> Caution, never
# Risk-off), applied here for the same reason: an outage is not
# evidence of danger, but it is not permission either.
REGIME_CEILINGS = {
    "Risk-on / Trending": 90.0,
    "Risk-on / Choppy": 50.0,
    "Caution": 25.0,
    "Risk-off": 5.0,
}
FALLBACK_CEILING_STATE = "Caution"

# R15 per-position band (5-8% of capital). Warning at 90% of the limit.
POSITION_BAND = (5.0, 8.0)
POSITION_WARN = 8.0 * 0.9          # 7.2%

# Per-GICS-group limits (successor to R16/R5 per D-007; AMENDED 2026-07-13
# from 15%/2: 15% against an 8% max position forced undersizing the second
# name to 1.875-of-full — 20% fits two full-size positions with room.
# Correlated-group tradeoff acknowledged in the record (the July-6 biotech
# exhibit). Emergent consequence of the unchanged D-008 Q4 ceiling: 90/20
# forces at least 5 groups at full deployment.)
GROUP_MAX_PCT = 20.0
GROUP_MAX_POSITIONS = 3
GROUP_WARN_FRACTION = 0.9          # warning above 90% of the group cap

CEILING_WARN_FRACTION = 0.9        # warning at 90% of the active ceiling


def _usd(v):
    return f"${v:,.0f}"


def assess_portfolio(positions, prices, capital, regime_state, *,
                     group_max_pct=GROUP_MAX_PCT,
                     group_max_positions=GROUP_MAX_POSITIONS):
    """Pure R28 assessment.

    positions: positions.json dict (holdings: [{ticker, shares, ...}]).
    prices: {ticker: last EOD close} — from already-computed artifacts.
    capital: account_capital_usd (None/<=0 -> unconfigured).
    regime_state: the swing regime string from the framework artifact.

    Returns the r28 block: per-position rows (R15), per-group rows
    (R16 successor), the regime-scaled ceiling (R17 successor, D-008
    Q4), the implied cash floor row (R18 successor — informational; it
    IS the ceiling's complement), and a summary. Groups resolve via
    prices' companion group_map when embedded in positions rows or via
    the `group` key; ungrouped vehicles each form their own bucket.
    """
    if not isinstance(capital, (int, float)) or isinstance(capital, bool) \
            or not math.isfinite(capital) or capital <= 0:
        return {
            "status": "unconfigured",
            "message": "account_capital_usd is not set in framework/"
                       "config.yaml — R28 cannot compute dollar limits. "
                       "(User-maintained until a broker integration "
                       "exists.)",
        }
    capital = float(capital)

    ceiling_state = regime_state if regime_state in REGIME_CEILINGS \
        else FALLBACK_CEILING_STATE
    ceiling = REGIME_CEILINGS[ceiling_state]

    holdings = [h for h in (positions.get("holdings") or [])
                if isinstance(h, dict) and h.get("ticker")]

    # --- aggregate lots per ticker BEFORE valuation (one position, one
    # raw-price valuation), validating shares as we go ---
    agg, order, rows = {}, [], []
    for h in holdings:
        t = h["ticker"]
        sh = h.get("shares")
        if isinstance(sh, bool) or not isinstance(sh, (int, float)) \
                or not math.isfinite(sh) or sh <= 0:
            rows.append({"ticker": t, "shares": sh, "price": None,
                         "value_usd": None, "pct_of_capital": None,
                         "group": h.get("group") or f"{t} (ungrouped)",
                         "status": "invalid",
                         "message": f"{t}: invalid shares value {sh!r} — "
                                    f"row excluded from every aggregate; "
                                    f"fix positions.json (long-only book: "
                                    f"shares must be a positive number)"})
            continue
        if t not in agg:
            agg[t] = {"shares": 0.0,
                      "group": h.get("group") or f"{t} (ungrouped)"}
            order.append(t)
        agg[t]["shares"] += float(sh)

    groups = {}
    deployed = 0.0
    unpriced = 0
    for t in order:
        shares = agg[t]["shares"]
        grp = agg[t]["group"]
        # every position registers in its group bucket — the positions-per-group
        # cap is a COUNT and counts unpriced members too (the position
        # exists whether or not its price resolved); only exposure skips
        g = groups.setdefault(grp, {"group": grp, "tickers": [],
                                    "value_usd": 0.0, "count": 0,
                                    "unpriced": []})
        g["tickers"].append(t)
        g["count"] += 1
        price = prices.get(t)
        if not isinstance(price, (int, float)) or isinstance(price, bool) \
                or not math.isfinite(price) or price <= 0:
            unpriced += 1
            g["unpriced"].append(t)
            rows.append({"ticker": t, "shares": shares, "price": None,
                         "value_usd": None, "pct_of_capital": None,
                         "group": grp, "status": "no_price",
                         "message": f"{t}: no EOD price in the artifact — "
                                    f"not counted toward exposure (still "
                                    f"counts toward the group position cap; "
                                    f"fix the data, not the rule)"})
            continue
        value = shares * float(price)      # raw price, whole position
        deployed += value
        g["value_usd"] += value
        rows.append({"ticker": t, "shares": shares,
                     "price": round(float(price), 2),
                     "value_usd": round(value, 2), "group": grp})

    # --- R15 per-position statuses ---
    for r in rows:
        if r.get("status") in ("no_price", "invalid"):
            continue
        pct = r["value_usd"] / capital * 100.0
        r["pct_of_capital"] = round(pct, 2)
        band_note = "" if pct >= POSITION_BAND[0] else \
            f" (below the {POSITION_BAND[0]:.0f}-{POSITION_BAND[1]:.0f}% target band — starter size, not a violation)"
        if pct > POSITION_BAND[1]:
            r["status"] = "violation"
            r["message"] = (f"{r['ticker']} {_usd(r['value_usd'])} = "
                            f"{pct:.1f}% of capital — above the "
                            f"{POSITION_BAND[1]:.0f}% per-position limit "
                            f"({_usd(capital * POSITION_BAND[1] / 100)})")
        elif pct > POSITION_WARN:
            r["status"] = "warning"
            r["message"] = (f"{r['ticker']} {_usd(r['value_usd'])} = "
                            f"{pct:.1f}% of capital — within "
                            f"{POSITION_BAND[1] - pct:.1f}pp of the "
                            f"{POSITION_BAND[1]:.0f}% limit")
        else:
            r["status"] = "compliant"
            r["message"] = (f"{r['ticker']} {_usd(r['value_usd'])} = "
                            f"{pct:.1f}% of capital{band_note}")

    # --- per-group statuses ---
    group_rows = []
    for g in groups.values():
        pct = g["value_usd"] / capital * 100.0
        g["pct_of_capital"] = round(pct, 2)
        g["value_usd"] = round(g["value_usd"], 2)
        problems = []
        if pct > group_max_pct:
            problems.append(f"{pct:.1f}% of capital exceeds the "
                            f"{group_max_pct:.0f}% group cap "
                            f"({_usd(capital * group_max_pct / 100)})")
        if g["count"] > group_max_positions:
            problems.append(f"{g['count']} positions exceed the "
                            f"{group_max_positions}-per-group cap")
        unpriced_note = (f" ({len(g['unpriced'])} unpriced: "
                         f"{', '.join(g['unpriced'])} — exposure "
                         f"understated)") if g["unpriced"] else ""
        if problems:
            g["status"] = "violation"
            g["message"] = f"{g['group']}: " + "; ".join(problems) \
                + unpriced_note
        elif pct > group_max_pct * GROUP_WARN_FRACTION:
            g["status"] = "warning"
            g["message"] = (f"{g['group']}: {_usd(g['value_usd'])} = "
                            f"{pct:.1f}% — within "
                            f"{group_max_pct - pct:.1f}pp of the "
                            f"{group_max_pct:.0f}% group cap"
                            f"{unpriced_note}")
        else:
            g["status"] = "compliant"
            g["message"] = (f"{g['group']}: {_usd(g['value_usd'])} = "
                            f"{pct:.1f}% across {g['count']} position"
                            f"{'s' if g['count'] != 1 else ''}"
                            f"{unpriced_note}")
        group_rows.append(g)
    group_rows.sort(key=lambda g: -g["value_usd"])

    # --- regime-scaled ceiling (D-008 Q4) ---
    deployed_pct = deployed / capital * 100.0
    ceiling_usd = capital * ceiling / 100.0
    priced_count = sum(1 for r in rows if r.get("value_usd") is not None)
    degraded = bool(order) and priced_count == 0
    unpriced_suffix = (f" · {unpriced} unpriced row"
                       f"{'s' if unpriced != 1 else ''} excluded — "
                       f"exposure understated") if unpriced else ""
    if degraded:
        # an artifact outage must never render as a compliant flat book
        c_status = "unavailable"
        c_msg = (f"no holding could be priced ({len(order)} held) — "
                 f"exposure unknown; ceiling not evaluated (artifact "
                 f"outage: fix the data, not the rule)")
    elif deployed_pct > ceiling:
        if regime_state == "Risk-on / Trending":
            c_status = "violation"
            c_msg = (f"deployed {_usd(deployed)} = {deployed_pct:.1f}% "
                     f"exceeds the Trending ceiling {ceiling:.0f}% "
                     f"({_usd(ceiling_usd)}) — bought past the ceiling")
        else:
            c_status = "action_needed"
            c_msg = (f"deployed {_usd(deployed)} = {deployed_pct:.1f}% vs "
                     f"the {ceiling_state} ceiling {ceiling:.0f}% "
                     f"({_usd(ceiling_usd)}) — reduce exposure below "
                     f"{ceiling:.0f}% via normal exit discipline "
                     f"(stops/R11 govern; no same-day forced liquidation)")
    elif deployed_pct > ceiling * CEILING_WARN_FRACTION:
        c_status = "warning"
        c_msg = (f"deployed {_usd(deployed)} = {deployed_pct:.1f}% — within "
                 f"{ceiling - deployed_pct:.1f}pp of the {ceiling_state} "
                 f"ceiling {ceiling:.0f}% ({_usd(ceiling_usd)})")
    else:
        c_status = "compliant"
        c_msg = (f"deployed {_usd(deployed)} = {deployed_pct:.1f}% of the "
                 f"{ceiling_state} ceiling {ceiling:.0f}% "
                 f"({_usd(ceiling_usd)} headroom: "
                 f"{_usd(ceiling_usd - deployed)})")
    if not degraded and unpriced_suffix:
        c_msg += unpriced_suffix

    cash_pct = 100.0 - deployed_pct
    implied_floor = 100.0 - ceiling

    result = {
        "status": "ok",
        "degraded": degraded,
        "enforcement_class": "COMPUTED (reporting-hard — cannot block "
                             "broker orders and does not pretend to)",
        "capital_usd": capital,
        "regime_state": regime_state,
        "ceiling_state": ceiling_state,
        "ceiling_pct": ceiling,
        "deployed_usd": round(deployed, 2),
        "deployed_pct": round(deployed_pct, 2),
        "cash_pct": round(cash_pct, 2),
        "positions": rows,
        "groups": group_rows,
        "ceiling": {"rule": "R17->R28 (D-008 Q4)", "status": c_status,
                    "message": c_msg},
        # informational: the floor IS the ceiling's complement (D-008 Q4
        # absorbed R18); one limit, reported from both sides
        "cash_floor": {
            "rule": "R18->R28 (informational)",
            "status": "info",
            "message": (f"cash {_usd(capital * cash_pct / 100)} = "
                        f"{cash_pct:.1f}% vs the {ceiling_state}-implied "
                        f"floor {implied_floor:.0f}% — the ceiling's "
                        f"complement, not a separate limit"),
        },
    }
    statuses = ([r.get("status") for r in rows]
                + [g["status"] for g in group_rows] + [c_status])
    result["summary"] = {
        "violations": statuses.count("violation"),
        "action_needed": statuses.count("action_needed"),
        "warnings": statuses.count("warning"),
        "compliant": statuses.count("compliant"),
        "no_price": statuses.count("no_price"),
        "invalid": statuses.count("invalid"),
    }
    return result


def build_group_map(universe_active):
    """{ticker: group_name} from the active universe's GICS mapping."""
    out = {}
    for name, g in (universe_active.get("groups") or {}).items():
        for t in (g.get("tickers") or []):
            out[t] = name
    return out
