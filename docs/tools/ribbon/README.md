# Ribbon probe — methodology backup

These files came from `~/Projects/ribbon-probe/`, a standalone research folder
that was **under no version control of any kind** — no git, no remote, no Time
Machine, no local snapshots. They are backed up here because they are the only
definition of the methodology: the data files they produce are rebuildable, and
these are not.

Every script says of itself "throwaway research; no commits." That was true of
the *results*. It was not true of the *method*, which is what this directory
preserves.

## What is here, and what deliberately is not

| copied | why |
|---|---|
| `regen_data.py`, `probe.py`, `part2.py` | the entire method — see below |
| `ribbon-lab-v2/v4/v5/v6/v7.html` | hand-written UI, no generator exists |

**Not copied: `ribbon_data.json` (7.3 MB) and `ribbon_SPY/QQQ/WFC.csv`
(5.8 MB).** Both are script output and both are rebuildable. See the
reproducibility caveat below before assuming "rebuildable" means "the same."

`v1`, `v3` and `v8` were already gone from the source folder before this backup
was taken. `ribbon-lab-v9.html` — the current version — sits here too; it landed
in the repo earlier and was moved in beside the rest so all six versions live
together.

**v9 in this repo will ALWAYS show synthetic demo data.** It fetches
`ribbon_data.json` relative to itself, that file is deliberately not committed
(see above), and on failure it renders a full lab — equity curve, CAGR,
drawdown, trade stats — on a seeded random walk. Two on-page warnings say so,
but nothing else about the page changes. Its header comment explains this;
treat every number this copy produces as fiction unless you have put a real
`ribbon_data.json` beside it.

## The three things the scripts define

### 1. TradingView SMA-seeded EMA

`ema_tv()` — [probe.py:42](probe.py#L42), duplicated at
[part2.py:39](part2.py#L39). The value at index `n-1` is the **SMA of the
first n closes**; from there `e[i] = a*close[i] + (1-a)*e[i-1]` with
`a = 2/(n+1)`; NaN before index `n-1`.

This is deliberately **not** pandas' `ewm(span=n, adjust=False)`, which seeds at
the first value. probe.py keeps `ema_first_seed()` alongside it and prints both
seedings for SPY specifically so the numbers can be checked against a
TradingView chart. The reason it is implemented by hand rather than delegated:
at `n=2650` the seeding difference persists **for years**, so the two are not
interchangeable at the long end of the ribbon — which is exactly where the
ribbon's trend filter lives.

### 2. Open-to-open forward returns

`fwd_open_to_open()` — [probe.py:62](probe.py#L62).
`fwd_h[t] = open[t+1+h] / open[t+1] - 1`, in percent. Horizons 5, 10, 20, 40.

The signal fires on the close of `T` and the return is measured from the open of
`T+1`. **There is no fill on the signal close** — the bar that generated the
signal is not tradable, and the measurement says so.

### 3. Non-overlapping event-loop backtest

[part2.py:113](part2.py#L113). Signals at the close of `T` set a `pend` flag and
act at the **open of `T+1`**. One position at a time: a buy is ignored unless
flat, a sell unless in position, so positions never overlap and re-entry happens
only once flat. $10,000, cash earns 0, no costs. Entry is the ribbon signal;
exit is `close < EMA21`.

The signal itself, from [probe.py:13-21](probe.py#L13):

```
stacked  = ema105 > ema465 > ema2650
above200 = close > sma200
setup    = stacked AND above200
dip      = within the last k bars, low_j <= anchor_ema_j * (1 + tol)
turn     = close > prev close AND close > ema21
signal   = setup AND dip AND turn
```

Base case: `anchor = ema105`, `tol = 0.005`, `k = 5`. Warmup rows lacking
`ema2650` or `sma200` are dropped before anything is measured.

probe.py sweeps a 3×3 grid over `tol ∈ {0, 0.005, 0.01}` × `k ∈ {3, 5, 10}`, and
states its own reading rule at [probe.py:188](probe.py#L188): *does the result
survive across the grid — not which cell is best. No cell is recommended.*

## THE TRAP — two scripts, one filename, different schemas

**`part2.py` writes `ribbon_data.json` WITHOUT volume.**
[part2.py:65](part2.py#L65) selects only `Open/High/Low/Close`, and
[`write_json`](part2.py#L75) emits `dates/o/h/l/c` — no `v`.

**`regen_data.py` writes the same filename WITH volume** — `dates/o/h/l/c/v`,
[regen_data.py:54](regen_data.py#L54). Its docstring line 3 exists for this
reason: *"Regenerate ribbon_data.json WITH VOLUME."*

`ribbon-lab-v7` and `v9` consume volume and OBV. So running `part2.py` after
`regen_data.py` **silently strips the volume out of the data file** and the
labs' volume panels go quiet against a file that still looks structurally valid.

**`regen_data.py` is the successor. Use it.** `part2.py` is kept for its
cross-sectional table and its event-loop backtest, not for its data writer.

## Reproducibility: regenerable in shape, NOT in content

[regen_data.py:40](regen_data.py#L40) is
`yf.Ticker(sym).history(period="max", auto_adjust=True)` — **no `start=`, no
`end=`, anywhere in the file.** The range is relative to the day you run it, and
[line 34](regen_data.py#L34) stamps `pd.Timestamp.today()` into the output.

Two consequences, and the second is the one that surprises people:

1. A re-run **extends** the series to today. The backed-up snapshot ended
   2026-07-31, stamped `"generated": "2026-08-01"`.
2. A re-run also **rewrites the history**. `auto_adjust=True` back-propagates
   split and dividend factors computed as of the download date, so any
   distribution after the original pull rescales the entire back-history. Values
   are then rounded to 2dp, where sub-percent shifts are visible.

So a regenerated file is a valid file, but it is **not the file**. Any figure
quoted from a lab UI against the 2026-08-01 snapshot will not reproduce exactly.
Two smaller sharp edges: [regen_data.py:42](regen_data.py#L42) silently drops any
symbol returning fewer than 500 bars, and Yahoo may no longer serve the same
depth of history (the snapshot reached 1993-01-29 for SPY and 1972 for WFC).

Nothing else is unrecoverable. There is no argparse, no REPL-typed threshold, no
command-line ticker list — the 23-symbol list, `tol=0.005`, `k=5`, the EMA
lengths 21/105/465/2650 and SMA200 are all literals in the source.

## One empirical finding worth keeping

[regen_data.py:71-99](regen_data.py#L71) is not plumbing — it is a test. It uses
WFC's 2:1 split on 2006-08-14 to check, empirically, whether yfinance adjusts
volume under `auto_adjust=True`: if average daily volume steps up ~2× across the
split while the close does not step, then **OHLC is split-adjusted and VOLUME IS
NOT**. Long-history volume must be read as a stepped series, not a continuous
one. The check prints its own verdict either way rather than assuming the
answer.

## The CSVs

`probe.py` — not `part2.py`, which writes no CSV — emits
`ribbon_{sym}.csv` at [probe.py:208](probe.py#L208) for SPY, QQQ and WFC.
They carry full float precision and the `ema2650` column, but **no volume**, and
only 3 of the 23 symbols. Neither the CSVs nor `ribbon_data.json` reconstructs
the other.

`breadth_probe.py` was not copied: it is a pure availability probe that writes
nothing and produces no artifact.
