"""Single out-of-sample evaluation (2013–2025) with parameters frozen on the dev sample."""
import hashlib
import pickle
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

from hlpairs import plots
from hlpairs.engine import split, strategy_monthly, top_n
from hlpairs.evaluation import adjust_pvalues, drawdown, fama_macbeth, nw_mean_test, quintile_sort, summary


def hl_filter(K, col):
    return lambda p: p.head(K).nsmallest(20, col).index.to_numpy()


def main():
    dirty = subprocess.run(["git", "status", "--porcelain", "experiments/frozen.yaml", "experiments/config.yaml"],
                           capture_output=True, text=True).stdout.strip()
    if dirty:
        sys.exit("frozen.yaml / config.yaml must be committed and unchanged before the holdout run")
    cfg = yaml.safe_load(open("experiments/config.yaml"))
    frozen = yaml.safe_load(open("experiments/frozen.yaml"))
    assert frozen["config_sha256"] == hashlib.sha256(open("experiments/config.yaml", "rb").read()).hexdigest()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    S, C = cfg["sample"], cfg["headline_cost_bps"]
    res = pickle.load(open("results/cache/main.pkl", "rb"))   # local cache written by 02_main_run.py only
    hold = split(res, S["dev_last_trade_start"], S["holdout_first_trade_start"])["holdout"]
    panel = pd.read_parquet("results/pair_panel.parquet")
    ph = panel[panel.trade_start >= S["holdout_first_trade_start"]].copy()
    ph["ret_net"] = ph.ret - ph.legs * C * 1e-4
    plots.style()

    pvals, out = {}, {}
    # H2: the pre-specified primary estimator (OLS 252d, as in the dev table) and the dev-chosen one
    qtabs = {}
    for col in ("hl_ols_252", frozen["h5"]["estimator"]):
        for ret in ("ret", "ret_net"):
            q = quintile_sort(ph, col, ret=ret)
            qtabs[(col, ret)] = q
            m, se, t, p = nw_mean_test(q[1] - q[5])
            out[f"H2 Q1-Q5 {col} {ret}"] = {"est": m, "t": t, "p": p, **{f"Q{k}": float(q[k].mean()) for k in q}}
    pvals["H2 Q1-Q5 (OLS 252d, gross)"] = out["H2 Q1-Q5 hl_ols_252 ret"]["p"]
    ph["hl_rank"] = ph.groupby("trade_start").hl_ols_252.rank(pct=True)
    ph["ssd_rank"] = ph["rank"] / 100
    fm, _ = fama_macbeth(ph, "ret", ["hl_rank", "ssd_rank"])
    pvals["H2 FM slope (HL rank | SSD rank)"] = fm.loc["hl_rank", "p"]
    out["H2 FM hl_rank"] = fm.loc["hl_rank"].to_dict()
    out["H2 FM ssd_rank"] = fm.loc["ssd_rank"].to_dict()
    rho, p3 = spearmanr(ph.hl_ols_252, ph.realized_hl, nan_policy="omit")
    pvals["H3 persistence (Spearman)"] = p3
    out["H3 rho"] = {"est": rho, "p": p3}

    ggr = strategy_monthly(hold, select=top_n(20), cost_bps=C)
    filt = strategy_monthly(hold, select=hl_filter(frozen["h5"]["K"], frozen["h5"]["estimator"]), cost_bps=C)
    ou = strategy_monthly(hold, source=f"ou_m{frozen['ou']['hold_mult']}", cost_bps=C)
    filt_gross = strategy_monthly(hold, select=hl_filter(frozen["h5"]["K"], frozen["h5"]["estimator"]))
    ggr_gross = strategy_monthly(hold, select=top_n(20))
    d = (filt - ggr).dropna()
    m5, _, t5, p5 = nw_mean_test(d)
    pvals["H5 HL-filtered − GGR (net)"] = p5
    out["H5 diff net"] = {"est": m5, "t": t5, "p": p5}
    s1 = summary(ggr)
    pvals["H1/H4 GGR top-20 net mean > 0"] = s1["p"]
    s5 = summary(filt)
    pvals["H4 HL-filtered net mean > 0"] = s5["p"]
    adj = adjust_pvalues(pd.Series(pvals))
    plots.save_table(adj, "tab_holdout_tests", floatfmt="%.4f")
    perf = pd.DataFrame({"GGR top 20 (gross)": summary(ggr_gross), "GGR top 20 (net)": s1,
                         "HL-filtered 20 (gross)": summary(filt_gross), "HL-filtered 20 (net)": s5,
                         "OU (net)": summary(ou)}).T
    plots.save_table(perf[["mean_monthly", "t_nw", "sharpe_ann", "max_dd", "frac_pos", "n"]],
                     "tab_holdout_performance", floatfmt="%.4f")
    q2 = pd.DataFrame({k: v for k, v in out.items() if k.startswith("H2 Q1")}).T
    plots.save_table(q2, "tab_h2_quintiles_holdout", floatfmt="%.4f")
    record = {"commit": commit, "frozen": frozen,
              "results": {k: {a: float(b) for a, b in v.items()} for k, v in out.items()}}
    yaml.safe_dump(record, open("results/holdout_record.yaml", "w"), sort_keys=False)
    print(adj, perf[["mean_monthly", "t_nw", "sharpe_ann"]], q2, sep="\n\n")

    fig, axes = plt.subplots(2, 1, figsize=(6.5, 4.8), sharex=True, height_ratios=[2, 1])
    for (n, s), col in zip({"GGR top 20": ggr, "HL-filtered 20": filt, "OU": ou}.items(), plots.PALETTE):
        axes[0].plot(s.index.to_timestamp(), (1 + s).cumprod(), label=n, color=col)
        axes[1].plot(s.index.to_timestamp(), drawdown(s) * 100, color=col)
    axes[0].axhline(1, color=plots.INK, lw=0.5)
    axes[0].set(ylabel="growth of $1", title=f"Holdout 2013–2025, net of {C} bp per leg-side, parameters frozen on 2000–2012")
    axes[0].legend()
    axes[1].set(ylabel="drawdown (%)")
    plots.save(fig, "fig17_holdout_equity")

    q = qtabs[("hl_ols_252", "ret")]
    mu = q.mean() * 100
    se = pd.Series({k: nw_mean_test(q[k])[1] * 100 for k in q})
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(mu.index, mu, yerr=1.96 * se, capsize=3, color=plots.PALETTE[1], width=0.6)
    ax.axhline(0, color=plots.INK, lw=0.6)
    ax.set(xlabel="formation half-life quintile (1 = fastest reversion)", ylabel="mean 6-month pair return (%)",
           title="H2 out of sample (holdout 2013–2025)")
    plots.save(fig, "fig18_h2_quintiles_holdout")

    # Combined view: dev vs holdout quintile profile (dev numbers from 05's table)
    dev_tab = pd.read_csv("tables/tab_h2_quintiles_dev.csv").set_index(["estimator", "return"])
    dv = dev_tab.loc[("hl_ols_252", "ret"), [f"Q{k}" for k in range(1, 6)]].to_numpy() * 100
    fig, ax = plt.subplots(figsize=(5.2, 3))
    x = np.arange(1, 6)
    ax.bar(x - 0.18, dv, width=0.36, color=plots.PALETTE[0], label="dev 2000–2012")
    ax.bar(x + 0.18, mu.to_numpy(), width=0.36, color=plots.PALETTE[1], label="holdout 2013–2025")
    ax.axhline(0, color=plots.INK, lw=0.6)
    ax.set(xticks=x, xlabel="formation half-life quintile (1 = fastest)", ylabel="mean 6-month pair return (%)",
           title="Half-life vs future pair return: in and out of sample")
    ax.legend()
    plots.save(fig, "fig19_h2_dev_vs_holdout")


if __name__ == "__main__":
    main()
