"""H2 on the dev sample, plus dev-only selection of the H5 filter and the OU holding rule.
Writes experiments/frozen.yaml; commit it BEFORE running 07_holdout.py."""
import hashlib
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from hlpairs import plots
from hlpairs.engine import split, strategy_monthly, top_n
from hlpairs.evaluation import deflated_sharpe, fama_macbeth, nw_mean_test, quintile_sort, summary

HL = ["hl_ols_63", "hl_ols_126", "hl_ols_252", "hl_kendall", "hl_ou", "cross", "hl_log"]


def hl_filter(K, col):
    """From the K lowest-SSD pairs, trade the 20 with the shortest formation half-life.
    Pair index == SSD rank, so the index values are the column positions."""
    return lambda p: p.head(K).nsmallest(20, col).index.to_numpy()


def main():
    cfg = yaml.safe_load(open("experiments/config.yaml"))
    S, C = cfg["sample"], cfg["headline_cost_bps"]
    res = pickle.load(open("results/cache/main.pkl", "rb"))   # local cache written by 02_main_run.py only
    dev = split(res, S["dev_last_trade_start"], S["holdout_first_trade_start"])["dev"]
    panel = pd.read_parquet("results/pair_panel.parquet")
    pdev = panel[panel.trade_start <= S["dev_last_trade_start"]].copy()
    pdev["ret_net"] = pdev.ret - pdev.legs * C * 1e-4
    plots.style()

    # ---- H2: quintile sorts (top-100 candidates per formation) -------------------------
    rows, qtabs = [], {}
    for c in HL:
        for ret in ("ret", "ret_net"):
            q = quintile_sort(pdev, c, ret=ret)
            qtabs[(c, ret)] = q
            m, se, t, p = nw_mean_test(q[1] - q[5])
            rows.append({"estimator": c, "return": ret, **{f"Q{k}": q[k].mean() for k in range(1, 6)},
                         "Q1-Q5": m, "t_nw": t, "p": p})
    h2 = pd.DataFrame(rows)
    plots.save_table(h2.set_index(["estimator", "return"]), "tab_h2_quintiles_dev", floatfmt="%.4f")
    print(h2.round(4).to_string())

    # ---- H2: Fama-MacBeth --------------------------------------------------------------
    pdev["hl_rank"] = pdev.groupby("trade_start").hl_ols_252.rank(pct=True)
    pdev["log_hl"] = np.log(pdev.hl_ols_252.clip(upper=1000))
    pdev["ssd_rank"] = pdev["rank"] / 100
    fm_rows = {}
    for name, xs in {"(1) HL rank": ["hl_rank"], "(2) HL rank + SSD rank": ["hl_rank", "ssd_rank"],
                     "(3) log HL + SSD rank + EG p": ["log_hl", "ssd_rank", "eg_p"]}.items():
        tab, _ = fama_macbeth(pdev, "ret", xs)
        for v, r in tab.iterrows():
            fm_rows[(name, v)] = r
    fm = pd.DataFrame(fm_rows).T
    plots.save_table(fm, "tab_h2_famamacbeth_dev", floatfmt="%.4f")
    print(fm.round(4).to_string())

    q = qtabs[("hl_ols_252", "ret")]
    mu = q.mean() * 100
    se = pd.Series({k: nw_mean_test(q[k])[1] * 100 for k in q})
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(mu.index, mu, yerr=1.96 * se, capsize=3, color=plots.PALETTE[0], width=0.6)
    ax.axhline(0, color=plots.INK, lw=0.6)
    ax.set(xlabel="formation half-life quintile (1 = fastest reversion)", ylabel="mean 6-month pair return (%)",
           title="H2 (dev 2000–2012): trading return by formation half-life")
    plots.save(fig, "fig11_h2_quintiles_dev")

    # ---- H5 + OU: dev-only grid; every variant logged ----------------------------------
    log = []
    for K in (40, 60, 100):
        for col in HL:
            m = strategy_monthly(dev, select=hl_filter(K, col), cost_bps=C)
            log.append({"family": "H5", "K": K, "estimator": col, **summary(m)})
    for mult in cfg["engine"]["ou_hold_mults"]:
        m = strategy_monthly(dev, source=f"ou_m{mult}", cost_bps=C)
        log.append({"family": "OU", "hold_mult": mult, **summary(m)})
    base = strategy_monthly(dev, select=top_n(20), cost_bps=C)
    log.append({"family": "baseline", "estimator": "GGR top 20", **summary(base)})
    log = pd.DataFrame(log)
    log.to_csv("results/experiment_log.csv", index=False)

    h5 = log[log.family == "H5"].sort_values("sharpe_ann", ascending=False).iloc[0]
    ou = log[log.family == "OU"].sort_values("sharpe_ann", ascending=False).iloc[0]
    sr_m = log[log.family == "H5"].sharpe_ann / np.sqrt(12)
    dsr = deflated_sharpe(h5.sharpe_ann / np.sqrt(12), len(sr_m), sr_m.var(ddof=1), int(h5["n"]), h5["skew"], h5["kurt"])
    frozen = {"h5": {"K": int(h5.K), "estimator": str(h5.estimator), "dev_sharpe_net": float(h5.sharpe_ann),
                     "dev_deflated_sharpe_prob": float(dsr), "n_trials": int(len(sr_m))},
              "ou": {"hold_mult": int(ou.hold_mult), "dev_sharpe_net": float(ou.sharpe_ann)},
              "baseline_dev_sharpe_net": float(log[log.family == "baseline"].sharpe_ann.iloc[0]),
              "config_sha256": hashlib.sha256(open("experiments/config.yaml", "rb").read()).hexdigest()}
    yaml.safe_dump(frozen, open("experiments/frozen.yaml", "w"), sort_keys=False)
    print(yaml.safe_dump(frozen))

    grid = log[log.family == "H5"].pivot(index="estimator", columns="K", values="sharpe_ann").loc[HL]
    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    im = ax.imshow(grid, cmap="Blues")
    ax.grid(False)
    ax.set_xticks(range(grid.shape[1]), grid.columns)
    ax.set_yticks(range(grid.shape[0]), grid.index)
    for a in range(grid.shape[0]):
        for b in range(grid.shape[1]):
            v = grid.iloc[a, b]
            ax.text(b, a, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if v > grid.to_numpy().mean() else plots.INK)
    ax.set(xlabel="SSD candidates K (trade the 20 shortest half-lives)",
           title=f"H5 dev Sharpe, net {C} bp ({len(sr_m)} trials;\nGGR top-20 = {frozen['baseline_dev_sharpe_net']:.2f})")
    fig.colorbar(im, shrink=0.8, label="annualised Sharpe (net)")
    plots.save(fig, "fig12_h5_dev_grid")


if __name__ == "__main__":
    main()
