# D-015 — Survey serious OSS trading frameworks for borrowable techniques

| | |
|---|---|
| **ID** | D-015 |
| **Date** | 2026-07-13 |
| **Status** | Proposed — parked (learning spike gated on the build queue clearing) |

## Context

Public trading/portfolio "skills" and frameworks are everywhere (YouTube-hyped).
This record surveys the landscape and locates where MarketPulse stands. Three
buckets:
1. **Data/broker plumbing** (tradingview-api, Alpaca/IBKR/ccxt wrappers) — useful
   plumbing. TradingView data is parked as [D-014](D-014-tradingview-data-api.md);
   broker-**execution** wrappers become relevant only if/when we cross the "human
   holds the trigger" line into auto-execution — a North Star future decision with
   its own gravity.
2. **Serious backtest/strategy frameworks** (VectorBT, QuantConnect/LEAN,
   Freqtrade, NautilusTrader, Backtrader) — mature, real, thousands of users. **The
   target of this record.**
3. **"AI trading agent" skills** (LLM + data feed + broker API picks trades) — the
   hyped bucket. Almost universally thin: no regime logic, no sizing discipline, no
   backtested edge, no audit trail.

## Options considered

| Option | Summary | Why (not) |
|---|---|---|
| **Harvest techniques** | Survey-and-steal spike; port specific patterns into our system | **Chosen.** Keeps our bespoke ruled architecture; imports only what's genuinely better. |
| Adopt / migrate to a framework | Rebuild on LEAN / VectorBT / Freqtrade | **Rejected** — migration cost fights our bespoke, ruled architecture, and our discipline layer is genuinely better than the public skills. |
| Do nothing | Ignore the landscape | Default until the trigger fires — but a bounded survey is cheap insurance against blind spots. |

## Evidence

**Honest standing vs the landscape:**
- **Ahead (vs bucket 3): not close.** Our discipline architecture — regime-scaled
  sizing (R28), A+ doctrine ([D-011](D-011-aplus-doctrine.md)), close-basis stops
  ([D-003](D-003-1b-position-engine.md)), the decision **registry**, and
  evidence-gating (Build 4 proved our own gauge didn't beat a 200DMA and we rebuilt
  on the evidence) — is exactly what the hyped skills skip. They sell confidence,
  not edge.
- **Not ahead (honest): two spots.** (a) the backtest harness — `backtest_regime.py`
  is honest but **simple** next to LEAN/VectorBT; a multi-asset/options/tick ambition
  would feel its ceiling. (b) the **yfinance data layer** — the weakest link, already
  known ([D-014](D-014-tradingview-data-api.md)). Neither urgent; both known.

## Ruling + rationale

**Parked.** A one-evening **survey-and-steal** spike (when the queue clears): read
how the mature frameworks solve the two things we'll eventually need, and file
borrowable **techniques** as candidate improvements to *our* system — not migrations
to theirs. **Do not adopt/migrate.** Same "evaluate on fit, not hype" discipline as
[D-014](D-014-tradingview-data-api.md).

## Consequences

- No migration; our ruled architecture stays authoritative.
- A future spike yields concrete technique candidates for our harness / 1B /
  fill-modeling — each of which would get its own record before adoption.

## Revisit triggers

The current build queue clears (D-012/D-013 pages, Gauge B, Phase 1, A+ grading,
dynamic targets) **AND** either:
- the backtest-harness ceiling is actually hit (multi-asset / options / tick
  ambitions materialize), OR
- the Build 5 fill-modeling work would benefit from studying LEAN first.

Until then, no action.

## Retest recipe

Survey-and-steal spike (harvest, don't migrate):

```
- VectorBT: walk-forward split handling, vectorized backtest patterns
            (improve our harness fidelity — see D-006).
- LEAN (QuantConnect): slippage + fill modeling (directly relevant to the
            +7.88% EOD-lag finding and the Build 5 exit-variant work).
- Freqtrade: position-management state-machine structure (compare vs our 1B —
            we built the same primitives; look for patterns we're missing).
```

## Links

- Jira: PER-508 comment 11721
- Related: [D-006](D-006-build4-protocol.md) (Build 4 harness/protocol), [D-003](D-003-1b-position-engine.md) (1B state machine), [D-014](D-014-tradingview-data-api.md) (data-layer plumbing)
