"""Descriptive (post-holdout, no parameter choices): H2 quintile spread by sub-period, and what
the formation half-life correlates with. Explains the dev-vs-holdout gap; changes nothing."""
import matplotlib.pyplot as plt
import pandas as pd

from hlpairs import plots
from hlpairs.evaluation import nw_mean_test, quintile_sort

PERIODS = [("2000–07/06", "2000-01-01", "2007-06-30"), ("2007/07–09 (crisis)", "2007-07-01", "2009-12-31"),
           ("2010–12/06", "2010-01-01", "2012-07-01"), ("2013–19", "2013-01-01", "2019-12-31"),
           ("2020–25", "2020-01-01", "2025-12-31")]


def main():
    p = pd.read_parquet("results/pair_panel.parquet")
    rows = []
    for name, lo, hi in PERIODS:
        q = quintile_sort(p[(p.trade_start >= lo) & (p.trade_start <= hi)], "hl_ols_252")
        m, se, t, pv = nw_mean_test(q[1] - q[5])
        rows.append({"period": name, "formations": len(q), "Q1": q[1].mean(), "Q5": q[5].mean(),
                     "Q1-Q5": m, "se": se, "t_nw": t, "p": pv})
    sub = pd.DataFrame(rows).set_index("period")
    plots.save_table(sub, "tab_h2_subperiods", floatfmt="%.4f")
    dev = p[p.trade_start <= "2012-07-01"]
    corr = dev[["hl_ols_252", "cross", "adf_p", "eg_p", "n_trips", "sigma", "ssd"]].rank().corr()["hl_ols_252"].drop("hl_ols_252")
    plots.save_table(corr.to_frame("spearman_with_hl_ols_252"), "tab_hl_correlates", floatfmt="%.2f")
    print(sub, corr, sep="\n\n")

    plots.style()
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(range(len(sub)), sub["Q1-Q5"] * 100, yerr=1.96 * sub.se * 100, capsize=3, width=0.6,
           color=[plots.PALETTE[0]] * 3 + [plots.PALETTE[1]] * 2)
    ax.set_xticks(range(len(sub)), sub.index, fontsize=7)
    ax.axhline(0, color=plots.INK, lw=0.6)
    ax.set(ylabel="Q1 − Q5 return per 6 months (%)",
           title="The half-life premium decays: fast-minus-slow quintile spread by period\n(blue = dev, orange = holdout; ±95% NW CI)")
    plots.save(fig, "fig20_h2_subperiods")


if __name__ == "__main__":
    main()
