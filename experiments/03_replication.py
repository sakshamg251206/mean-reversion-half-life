"""H1: does the GGR distance rule still earn excess returns on point-in-time S&P 500 data?"""
import pickle
from pathlib import Path

from matplotlib.ticker import FormatStrFormatter, NullFormatter

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from hlpairs import data, plots
from hlpairs.engine import Config, run_all, split, strategy_monthly, top_n
from hlpairs.evaluation import drawdown, summary

STRATS = {"Top 5": dict(select=top_n(5)), "Top 20": dict(select=top_n(20)),
          "Sector top 20": dict(source="sector"), "Random 20": dict(source="random")}


def spy_monthly():
    spy = data.load_prices("data/raw/market.parquet")["SPY"]
    m = spy.resample("ME").last().pct_change().dropna()
    m.index = m.index.to_period("M")
    return m


def main():
    cfg = yaml.safe_load(open("experiments/config.yaml"))
    S = cfg["sample"]
    res = pickle.load(open("results/cache/main.pkl", "rb"))   # local cache written by 02_main_run.py only
    px = data.load_prices()
    membership = data.load_membership("data/raw/sp500_membership.csv")
    sectors = pd.read_csv("data/raw/sectors_current.csv", index_col=0)["sector"].to_dict()
    ec = cfg["engine"] | {"ou_hold_mults": (), "features": False, "wait": 1}
    wait_path = Path("results/cache/wait1.pkl")
    if wait_path.exists():
        res_wait = pickle.load(open(wait_path, "rb"))       # local cache written below only
    else:
        res_wait = run_all(px, membership, Config(**ec), sectors=sectors)
        pickle.dump(res_wait, open(wait_path, "wb"))
    spy_m = spy_monthly()

    rows = []
    for wait, rs in (("0", res), ("1", res_wait)):
        parts = split(rs, S["dev_last_trade_start"], S["holdout_first_trade_start"])
        for period, sub in parts.items():
            for basis in ("committed", "employed"):
                for name, kw in STRATS.items():
                    m = strategy_monthly(sub, basis=basis, **kw)
                    rows.append({"wait": wait, "period": period, "basis": basis, "strategy": name, **summary(m)})
            idx = strategy_monthly(sub, select=top_n(20)).index.intersection(spy_m.index)
            rows.append({"wait": "-", "period": period, "basis": "-", "strategy": "SPY buy-and-hold",
                         **summary(spy_m.loc[idx])})
    tab = pd.DataFrame(rows).drop_duplicates(subset=["wait", "period", "basis", "strategy"])
    tab.to_csv("results/replication_all.csv", index=False)
    cols = ["mean_monthly", "t_nw", "sharpe_ann", "max_dd", "frac_pos", "n"]
    main_tab = tab[tab.basis.isin(["committed", "-"]) & tab.wait.isin(["0", "-"])]
    plots.save_table(main_tab.set_index(["period", "strategy"])[cols], "tab_replication", floatfmt="%.4f")
    plots.save_table(tab[tab.strategy.isin(["Top 5", "Top 20"])].set_index(["period", "strategy", "wait", "basis"])[["mean_monthly", "t_nw", "sharpe_ann"]],
                     "tab_replication_wait_basis", floatfmt="%.4f")

    # GGR Table 2 analogue: trading statistics of top-20 pairs
    p20 = pd.concat([r.pairs.head(20) for r in res])
    stats = pd.Series({"avg pairs opened per period (of 20)": p20.assign(o=p20.n_trips > 0).groupby("trade_start").o.sum().mean(),
                       "avg round trips per pair": p20.n_trips.mean(),
                       "avg trigger gap (2 sigma, normalised price)": 2 * p20.sigma.mean(),
                       "avg stocks in formation universe": p20.n_universe.mean(),
                       "share of same-sector pairs (sector known)": p20.same_sector.mean()}).to_frame("value")
    plots.save_table(stats, "tab_trading_stats", floatfmt="%.3f")
    print(main_tab[["period", "strategy", *cols]].to_string())
    print(stats)

    plots.style()
    fig_example_pair(res, px, S)
    series = {n: strategy_monthly(res, **kw) for n, kw in STRATS.items()}
    series["SPY"] = spy_m.loc[series["Top 20"].index.intersection(spy_m.index)]
    colors = dict(zip(series, [*plots.PALETTE[:4], plots.GREY]))

    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    for n, s in series.items():
        ax.plot(s.index.to_timestamp(), (1 + s).cumprod(), label=n, color=colors[n])
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.axvspan(pd.Timestamp(S["holdout_first_trade_start"]), series["Top 20"].index[-1].to_timestamp(),
               color=plots.GREY, alpha=0.08, label="holdout period")
    ax.set(ylabel="growth of $1 (log scale)", title="GGR pairs portfolios vs baselines (committed capital, gross of costs)")
    ax.legend(ncol=2, fontsize=8)
    plots.save(fig, "fig03_equity_curves")

    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    for n in ("Top 20", "SPY"):
        d = drawdown(series[n])
        ax.fill_between(d.index.to_timestamp(), d * 100, 0, alpha=0.35, color=colors[n], label=n, lw=0)
    ax.set(ylabel="drawdown (%)", title="Drawdowns: GGR top-20 pairs vs the equity market")
    ax.legend()
    plots.save(fig, "fig04_drawdowns")

    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    roll = series["Top 20"].rolling(36).mean() * 100
    ax.plot(roll.index.to_timestamp(), roll, color=colors["Top 20"])
    ax.axhline(0, color=plots.INK, lw=0.6)
    ax.set(ylabel="% per month", title="GGR top-20 pairs: 36-month rolling mean monthly return (gross)")
    plots.save(fig, "fig05_rolling_performance")


def fig_example_pair(res, px, S):
    """Top-1 pair of the first dev formation in which it completes >= 2 round trips."""
    r = next(r for r in res if r.trade_start <= pd.Timestamp(S["dev_last_trade_start"]) and r.pairs.n_trips.iloc[0] >= 2)
    i, j, sig = r.pairs.i.iloc[0], r.pairs.j.iloc[0], r.pairs.sigma.iloc[0]
    f0 = r.trade_start - pd.DateOffset(months=12)
    win = px.loc[f0:r.dates[-1], [i, j]]
    form_end = win.index[win.index < r.trade_start][-1]
    norm = win / win.loc[:form_end].iloc[0]          # normalised to formation start throughout (GGR)
    spread = (norm[i] - norm[j]) / sig
    fig, axes = plt.subplots(3, 1, figsize=(6.5, 7), sharex=True)
    for ax, s, c in ((axes[0], win, None), (axes[1], norm, None)):
        ax.plot(s.index, s[i], label=i, color=plots.PALETTE[0])
        ax.plot(s.index, s[j], label=j, color=plots.PALETTE[1])
    axes[0].set(ylabel="adjusted price ($)", title=f"Pair {i}/{j}: 12-month formation, then 6-month trading")
    axes[0].legend()
    axes[1].set(ylabel="normalised price", title="Cumulative total-return index (1 = formation start)")
    axes[2].plot(spread.index, spread, color=plots.PALETTE[6], lw=1.2)
    for y in (2, -2):
        axes[2].axhline(y, ls="--", color=plots.INK2, lw=0.8)
    axes[2].axhline(0, color=plots.INK2, lw=0.5)
    act = pd.Series(r.active[:, 0], index=r.dates)
    lim = max(4, float(spread.abs().max()) + 0.5)
    axes[2].fill_between(act.index, -lim, lim, where=act.to_numpy(), color=plots.PALETTE[2], alpha=0.15, lw=0,
                         label="position open (mean-reversion episode)")
    axes[2].set(ylabel="spread / formation σ", ylim=(-lim, lim), title="Trading signal: open beyond ±2σ, close when prices cross")
    axes[2].legend(loc="upper left", fontsize=8)
    for ax in axes:
        ax.axvline(r.trade_start, color=plots.INK, lw=0.8)
    axes[0].annotate("trading starts", (r.trade_start, axes[0].get_ylim()[1]), fontsize=7, ha="left", va="top")
    plots.save(fig, "fig02_example_pair")


if __name__ == "__main__":
    main()
