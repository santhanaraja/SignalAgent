#!/usr/bin/env python3
"""
Build 5B charts — equity curves (arms + benchmarks) and per-arm
R-multiple distributions. Deterministic reruns of the headline
(score-order) sims; ~1 minute.

Run: python3 scripts/charts_systems.py
"""

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import backtest_systems as bs

IMG = os.path.join(REPO, "docs", "img")
COLORS = {"S1": "#1f77b4", "S2": "#9467bd", "S3": "#2ca02c",
          "S4": "#d62728", "S5": "#ff7f0e",
          "SPY_buy_hold": "#7f7f7f", "EW_pool_buy_hold": "#bcbd22",
          "chassis_on_SPY": "#17becf"}


def main():
    os.makedirs(IMG, exist_ok=True)
    panel = bs.load_panel()
    sims = {arm: bs.run_sim(panel, arm, "score") for arm in bs.ARMS}
    bench = bs.bench_curves(panel)

    # ---- equity curves -----------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 6))
    for arm, sim in sims.items():
        d = [x[0] for x in sim["curve"]]
        v = [x[1] / 1000 for x in sim["curve"]]
        ax.plot(range(len(d)), v, label=arm, color=COLORS[arm], lw=1.4)
    for name, curve in bench.items():
        v = [x[1] / 1000 for x in curve]
        ax.plot(range(len(v)), v, label=name, color=COLORS[name],
                lw=1.0, ls="--", alpha=0.8)
    # tick at each year BOUNDARY (a fixed 252 stride mislabels year ends)
    years = [d[:4] for d in panel["days"]]
    ticks = [0] + [i for i in range(1, len(years))
                   if years[i] != years[i - 1]]
    ax.set_xticks(ticks)
    ax.set_xticklabels([years[i] for i in ticks])
    ax.axvline(panel["days"].index(
        next(d for d in panel["days"] if d > bs.TRAIN_END)),
        color="k", ls=":", lw=1, alpha=0.6)
    ax.text(0.62, 0.03, "train | validate split", transform=ax.transAxes,
            fontsize=8, alpha=0.7)
    ax.set_ylabel("equity ($k, start 100)")
    ax.set_title("Build 5B — equity curves, score-order selection, "
                 "5 bps/side, 6.5% sizing")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "systems_equity.png"), dpi=130)
    plt.close(fig)

    # ---- R distributions ---------------------------------------------
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.4), sharey=True)
    for ax, arm in zip(axes, bs.ARMS):
        rs = np.array([t["r_mult"] for t in sims[arm]["trades"]
                       if t["r_mult"] is not None])
        clipped = np.clip(rs, -5, 10)
        ax.hist(clipped, bins=60, color=COLORS[arm], alpha=0.85)
        ax.axvline(0, color="k", lw=0.8)
        m = bs.trade_metrics(sims[arm]["trades"])
        ax.set_title(f"{arm}  exp(R,$w)={m['expectancy_r']}\n"
                     f"med={m['median_r']}  win={m['win_rate_pct']}%",
                     fontsize=9)
        ax.set_xlabel("R multiple (clipped [-5,10])", fontsize=8)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("trades")
    fig.suptitle("R-multiple distributions — no profit target (R24 gap): "
                 "the right tail is unbounded by design; left tail is the "
                 "doctrine's actual test", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(os.path.join(IMG, "systems_r_dist.png"), dpi=130)
    plt.close(fig)
    print(f"charts -> {IMG}/systems_equity.png, systems_r_dist.png")


if __name__ == "__main__":
    main()
