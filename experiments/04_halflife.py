"""Half-life distributions, estimator agreement, diagnostics and persistence (H3) on the dev sample."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

from hlpairs import data, plots
from hlpairs import halflife as hl

HL = ["hl_ols_63", "hl_ols_126", "hl_ols_252", "hl_kendall", "hl_ou", "cross", "hl_log"]
LABEL = {"hl_ols_63": "OLS, 63d", "hl_ols_126": "OLS, 126d", "hl_ols_252": "OLS, 252d",
         "hl_kendall": "Kendall-corrected", "hl_ou": "OU exact MLE", "cross": "crossing interval",
         "hl_log": "OLS, log-hedged spread"}


def main():
    cfg = yaml.safe_load(open("experiments/config.yaml"))
    panel = pd.read_parquet("results/pair_panel.parquet")
    dev = panel[panel.trade_start <= cfg["sample"]["dev_last_trade_start"]]
    plots.style()

    fin = dev[HL].replace(np.inf, np.nan)
    desc = pd.DataFrame({"median": fin.median(), "p10": fin.quantile(.1), "p90": fin.quantile(.9),
                         "share_inf": np.isinf(dev[HL]).mean(), "share_nan": dev[HL].isna().mean()})
    desc.index = [LABEL[c] for c in desc.index]
    plots.save_table(desc, "tab_halflife_describe", floatfmt="%.2f")
    print(desc)

    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    bins = np.logspace(-0.5, 3, 50)
    for c, col in zip(["hl_ols_252", "hl_kendall", "hl_ou", "hl_ols_63"], plots.PALETTE):
        ax.hist(dev[c].clip(upper=1000), bins=bins, histtype="step", lw=1.6, label=LABEL[c], color=col)
    ax.set(xscale="log", xlabel="formation half-life (trading days; log scale, capped at 1000)", ylabel="pairs",
           title="Formation half-life distributions (dev sample, top-100 SSD pairs)")
    ax.legend()
    plots.save(fig, "fig06_halflife_distributions")

    rk = dev[HL].rank().corr()
    names = [LABEL[c] for c in HL]
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(rk, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(len(HL)), names, rotation=40, ha="right")
    ax.set_yticks(range(len(HL)), names)
    for a in range(len(HL)):
        for b in range(len(HL)):
            v = rk.iloc[a, b]
            ax.text(b, a, f"{v:.2f}", ha="center", va="center", fontsize=7, color="white" if v > 0.6 else plots.INK)
    ax.grid(False)
    fig.colorbar(im, shrink=0.8, label="Spearman rank correlation")
    ax.set_title("Agreement between half-life estimators (dev)")
    plots.save(fig, "fig07_estimator_agreement")

    fig, axes = plt.subplots(1, 3, figsize=(7, 2.6), sharey=True)
    for ax, c, lbl in zip(axes, ["adf_p", "eg_p", "kpss_p"],
                          ["ADF p-value (H0: unit root)", "Engle–Granger p (H0: no coint.)", "KPSS p (H0: stationary)"]):
        ax.hist(dev[c], bins=20, color=plots.PALETTE[0], edgecolor="white", lw=0.5)
        ax.axvline(0.05, color=plots.INK, ls="--", lw=0.8)
        ax.set(xlabel=lbl)
    axes[0].set_ylabel("pairs")
    fig.suptitle("Are GGR spreads stationary / cointegrated? (dev, top-100 pairs; dashed = 5%)", fontweight="bold", x=0.02, ha="left")
    plots.save(fig, "fig08_diagnostics")
    diag = pd.Series({"ADF rejects unit root (5%)": (dev.adf_p < .05).mean(),
                      "Engle-Granger rejects no-cointegration (5%)": (dev.eg_p < .05).mean(),
                      "Johansen trace rejects rank 0 (5%)": dev.joh.mean(),
                      "KPSS rejects stationarity (5%)": (dev.kpss_p <= .05).mean(),
                      "Median variance ratio VR(10)": dev.vr.median()}).to_frame("value")
    top20 = dev[dev["rank"] < 20]
    diag["top-20 only"] = [(top20.adf_p < .05).mean(), (top20.eg_p < .05).mean(), top20.joh.mean(),
                           (top20.kpss_p <= .05).mean(), top20.vr.median()]
    plots.save_table(diag, "tab_diagnostics", floatfmt="%.3f")
    print(diag)

    # H3 persistence: formation vs realised trading-period half-life (Spearman handles inf as largest)
    pers = pd.DataFrame({LABEL[c]: spearmanr(dev[c], dev.realized_hl, nan_policy="omit") for c in HL},
                        index=["spearman_rho", "p_value"]).T
    pers["median_formation_hl"] = [dev[c].replace(np.inf, np.nan).median() for c in HL]
    pers["median_realised_hl"] = dev.realized_hl.replace(np.inf, np.nan).median()
    plots.save_table(pers, "tab_persistence_dev", floatfmt="%.3f")
    print(pers)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    x, y = dev.hl_ols_252.clip(1, 1000), dev.realized_hl.clip(1, 1000)
    hb = ax.hexbin(x, y, xscale="log", yscale="log", gridsize=35, mincnt=1, cmap="Blues", bins="log")
    ax.plot([1, 1000], [1, 1000], color=plots.INK, lw=0.8, ls="--", label="45° line")
    ax.set(xlabel="formation half-life, OLS 252d (days)", ylabel="realised trading-period half-life (days)",
           title=f"H3 persistence (dev): Spearman ρ = {pers.loc[LABEL['hl_ols_252'], 'spearman_rho']:.2f}")
    ax.legend(loc="upper left")
    fig.colorbar(hb, shrink=0.8, label="pairs (log count)")
    plots.save(fig, "fig09_persistence")

    # Rolling half-life of the most frequently selected top-1 dev pair
    px = data.load_prices()
    top1 = dev[dev["rank"] == 0]
    i, j = top1.groupby(["i", "j"]).size().idxmax()
    w = px[[i, j]].dropna()
    s = np.log(w[i]) - np.log(w[j])
    roll = pd.Series({d: hl.halflife_ols(s.loc[:d].iloc[-126:]) for d in s.index[126::5]})
    sel = top1[(top1.i == i) & (top1.j == j)].trade_start
    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    ax.plot(roll.index, roll.clip(upper=500), color=plots.PALETTE[0], lw=1.2, label="126-day rolling half-life")
    ax.scatter(sel, np.full(len(sel), 0.8), marker="|", color=plots.PALETTE[1], s=60, label="selected as top-1 pair")
    ax.set(yscale="log", ylabel="half-life (days, capped 500)",
           title=f"Rolling half-life is unstable even for the most often selected pair ({i}/{j})")
    ax.legend(loc="upper left", fontsize=8)
    plots.save(fig, "fig10_rolling_halflife")


if __name__ == "__main__":
    main()
