"""Parameter sensitivity, cost sensitivity (H4), regimes, sector cross-section.
Parameters frozen on the dev sample are used as-is; dev and holdout are reported side by side."""
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from hlpairs import data, plots
from hlpairs.engine import Config, run_all, split, strategy_monthly, top_n
from hlpairs.evaluation import summary


def hl_filter(K, col):
    return lambda p: p.head(K).nsmallest(20, col).index.to_numpy()


def grid_results(px, membership, base, res):
    """Engine reruns (no features) for each formation length x entry threshold; cached."""
    out = {}
    for fm in (6, 12):
        for k in (1.5, 2.0, 2.5):
            path = Path(f"results/cache/grid_f{fm}_k{k}.pkl")
            if (fm, k) == (12, 2.0):
                out[(fm, k)] = res
            elif path.exists():
                out[(fm, k)] = pickle.load(open(path, "rb"))   # local cache written below only
            else:
                out[(fm, k)] = run_all(px, membership, Config(**(base | {"form_months": fm, "k": k})))
                pickle.dump(out[(fm, k)], open(path, "wb"))
    return out


def main():
    cfg = yaml.safe_load(open("experiments/config.yaml"))
    S, C = cfg["sample"], cfg["headline_cost_bps"]
    frozen = yaml.safe_load(open("experiments/frozen.yaml"))
    px = data.load_prices()
    membership = data.load_membership("data/raw/sp500_membership.csv")
    res = pickle.load(open("results/cache/main.pkl", "rb"))   # local cache written by 02_main_run.py only
    plots.style()
    base = cfg["engine"] | {"ou_hold_mults": (), "features": False}

    # ---- parameter sensitivity: formation length x k x N --------------------------------
    rows = []
    for (fm, k), rs in grid_results(px, membership, base, res).items():
        parts = split(rs, S["dev_last_trade_start"], S["holdout_first_trade_start"])
        for N in (5, 20, 50):
            for period in ("dev", "holdout"):
                m = strategy_monthly(parts[period], select=top_n(N), cost_bps=C)
                rows.append({"form_months": fm, "k": k, "N": N, "period": period, **summary(m)})
    sens = pd.DataFrame(rows)
    sens.to_csv("results/sensitivity.csv", index=False)
    plots.save_table(sens.pivot_table(index=["form_months", "k"], columns=["period", "N"], values="sharpe_ann"),
                     "tab_sensitivity", floatfmt="%.2f")
    fig, axes = plt.subplots(2, 2, figsize=(7, 5.6), sharex=True, sharey=True)
    for r_, fm in enumerate((12, 6)):
        for c_, period in enumerate(("dev", "holdout")):
            ax = axes[r_, c_]
            g = sens[(sens.period == period) & (sens.form_months == fm)].pivot(index="k", columns="N", values="sharpe_ann")
            im = ax.imshow(g, cmap="RdBu", vmin=-1, vmax=1)
            ax.grid(False)
            ax.set_xticks(range(3), g.columns)
            ax.set_yticks(range(3), g.index)
            for a in range(3):
                for b in range(3):
                    ax.text(b, a, f"{g.iloc[a, b]:.2f}", ha="center", va="center", fontsize=8)
            ax.set_title(f"{period}, {fm}-month formation", fontsize=9)
    for ax in axes[1]:
        ax.set_xlabel("N pairs")
    for ax in axes[:, 0]:
        ax.set_ylabel("entry threshold k (σ)")
    fig.colorbar(im, ax=axes, shrink=0.7, label="annualised Sharpe (net)")
    fig.suptitle(f"GGR parameter sensitivity, net of {C} bp per leg-side", fontweight="bold", x=0.02, ha="left")
    plots.save(fig, "fig13_sensitivity")

    # ---- cost curves (H4) -----------------------------------------------------------------
    strats = {"GGR top 20": dict(select=top_n(20)),
              "HL-filtered 20": dict(select=hl_filter(frozen["h5"]["K"], frozen["h5"]["estimator"])),
              f"OU (max hold {frozen['ou']['hold_mult']}×HL)": dict(source=f"ou_m{frozen['ou']['hold_mult']}")}
    parts = split(res, S["dev_last_trade_start"], S["holdout_first_trade_start"])
    costs = np.arange(0, 41, 2.5)
    curve = {(n, p): [summary(strategy_monthly(parts[p], cost_bps=c, **kw))["ann_mean"] for c in costs]
             for n, kw in strats.items() for p in ("dev", "holdout")}
    cc = pd.DataFrame(curve, index=costs)
    cc.index.name = "cost_bps_per_leg_side"
    cc.to_csv("results/cost_curves.csv")
    breakeven = {k: float(np.interp(0, -cc[k].to_numpy(), costs)) if cc[k].iloc[0] > 0 else 0.0 for k in cc}
    be = pd.Series(breakeven).unstack()
    be.index.name = "strategy"
    plots.save_table(be, "tab_breakeven_cost", floatfmt="%.1f")
    print("break-even cost (bp per leg-side):\n", be)
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    for (n, p), y in cc.items():
        col = plots.PALETTE[list(strats).index(n)]
        ax.plot(costs, y * 100, ls="-" if p == "dev" else "--", color=col, label=f"{n} — {p}")
    ax.axhline(0, color=plots.INK, lw=0.6)
    ax.set(xlabel="transaction cost per leg per side (bp)", ylabel="annualised mean return (%)",
           title="H4: pairs returns vs transaction costs (solid = dev 2000–12, dashed = holdout 2013–25)")
    ax.legend(fontsize=7, ncol=2)
    plots.save(fig, "fig14_cost_sensitivity")

    # ---- regimes (descriptive; regime labels are ex post) -------------------------------
    m = strategy_monthly(res, select=top_n(20), cost_bps=C)
    mkt = data.load_prices("data/raw/market.parquet")
    vix = mkt.VIX.resample("ME").mean()
    vix.index = vix.index.to_period("M")
    rec = pd.read_csv("data/raw/usrec.csv", index_col=0, parse_dates=True).iloc[:, 0]
    rec.index = rec.index.to_period("M")
    df = pd.DataFrame({"r": m, "vix": vix, "rec": rec}).dropna()
    df["VIX tercile"] = pd.qcut(df.vix, 3, labels=["low", "mid", "high"])
    df["era"] = pd.cut(df.index.year, [1999, 2007, 2009, 2019, 2025], labels=["2000–07", "2008–09", "2010–19", "2020–25"])
    df["NBER recession"] = df.rec.map({0: "expansion", 1: "recession"})
    reg = pd.concat({key: df.groupby(key, observed=True).r.agg(["mean", "std", "count"])
                     for key in ("era", "VIX tercile", "NBER recession")})
    plots.save_table(reg, "tab_regimes", floatfmt="%.4f")
    print(reg)
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.8), sharey=True, width_ratios=[4, 3, 2])
    for ax, key in zip(axes, ["era", "VIX tercile", "NBER recession"]):
        g = reg.loc[key]
        ax.bar([str(x) for x in g.index], g["mean"] * 100, yerr=1.96 * g["std"] / np.sqrt(g["count"]) * 100,
               capsize=3, color=plots.PALETTE[0], width=0.6)
        ax.axhline(0, color=plots.INK, lw=0.6)
        ax.set_title(key, fontsize=9)
        ax.tick_params(axis="x", labelsize=7)
    axes[0].set_ylabel("% per month (net of 10 bp)")
    fig.suptitle("GGR top-20 returns by regime (±95% CI; regime labels are ex post)", fontweight="bold", x=0.02, ha="left")
    plots.save(fig, "fig15_regimes")

    # ---- sector cross-section ---------------------------------------------------------------
    panel = pd.read_parquet("results/pair_panel.parquet")
    sec = pd.read_csv("data/raw/sectors_current.csv", index_col=0)["sector"]
    p20 = panel[panel["rank"] < 20].copy()
    p20["sector"] = np.where(p20.same_sector, p20.i.map(sec), "Cross-sector / unknown")
    cs = p20.groupby("sector").agg(pairs=("ret", "size"), mean_ret=("ret", "mean"),
                                   median_hl=("hl_ols_252", "median"), mean_trips=("n_trips", "mean"))
    cs = cs.sort_values("pairs", ascending=False)
    plots.save_table(cs, "tab_sector_crosssection", floatfmt="%.4f")
    print(cs)
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.scatter(cs.median_hl, cs.mean_ret * 100, s=20 + cs.pairs / cs.pairs.max() * 400, alpha=0.6,
               color=plots.PALETTE[0], edgecolor="white")
    for n, r in cs.iterrows():
        ax.annotate(n, (r.median_hl, r.mean_ret * 100), fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.axhline(0, color=plots.INK, lw=0.6)
    ax.set(xlabel="median formation half-life (days)", ylabel="mean 6-month pair return (%, gross)",
           title="Sector cross-section of GGR top-20 pairs (bubble size = number of pairs)")
    plots.save(fig, "fig16_sector_crosssection")


if __name__ == "__main__":
    main()
