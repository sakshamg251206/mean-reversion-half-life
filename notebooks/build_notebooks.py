"""Generate the two walkthrough notebooks (run from the repo root, then execute with nbconvert)."""
import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
SETUP = code("""import os, pathlib
os.chdir(pathlib.Path.cwd().parent if pathlib.Path.cwd().name == "notebooks" else pathlib.Path.cwd())
import numpy as np, pandas as pd, matplotlib.pyplot as plt
pd.set_option("display.width", 160)""")

walk = nbf.v4.new_notebook()
walk.cells = [
    md("""# 01 — One pair, end to end

This notebook follows a single pair through the whole pipeline so every step is visible:
formation-period selection by distance, the spread, four half-life estimators, stationarity /
cointegration tests, and the GGR trading simulation. Everything uses the cached data written by
`experiments/01_data.py` and the library in `src/hlpairs/`."""),
    SETUP,
    code("""from hlpairs import data, halflife as hl, diagnostics as dg
from hlpairs.pairs import normalize, top_pairs
from hlpairs.universe import formation_universe
from hlpairs.backtest import trade_pairs

px = data.load_prices()
membership = data.load_membership("data/raw/sp500_membership.csv")
t0 = pd.Timestamp("2005-01-03")                      # first trading day of the trading period
form_idx = px.index[(px.index >= t0 - pd.DateOffset(months=12)) & (px.index < t0)]
universe = formation_universe(px, membership, form_idx[0], form_idx[-1])
print(len(universe), "stocks eligible (members on formation start, no missing prices)")"""),
    md("""## Step 1 — rank all pairs by distance (SSD)
Prices are normalised to a cumulative total-return index starting at 1, and
$D_{ij} = \\sum_t (P^i_t - P^j_t)^2$ is computed for every pair."""),
    code("""form_px = px.loc[form_idx, universe]
form = normalize(form_px)
pairs = top_pairs(form, 20)
pairs.head(10)"""),
    md("""## Step 2 — the spread and its half-life
For the best pair, the spread $S_t = P^i_t - P^j_t$ is fitted as an AR(1),
$S_t = c + \\phi S_{t-1} + \\varepsilon_t$, giving half-life $h = -\\ln 2 / \\ln \\phi$."""),
    code("""i, j, sigma = pairs.loc[0, ["i", "j", "sigma"]]
s = (form[i] - form[j]).to_numpy()
est = {name: f(s) for name, f in hl.ESTIMATORS.items()}
print(f"pair {i}/{j}; phi_OLS = {hl.ar1_phi(s):.3f}")
pd.Series(est, name="half-life (days)").round(2)"""),
    code("""y, x = np.log(form_px[i]), np.log(form_px[j])
pd.Series({"ADF p (spread)": dg.adf_pvalue(s), "KPSS p (spread)": dg.kpss_pvalue(s),
           "Engle-Granger p (log prices)": dg.eg_pvalue(y, x), "Johansen rejects rank 0": dg.johansen_reject(y, x),
           "variance ratio VR(10)": dg.variance_ratio(s)})"""),
    md("""## Step 3 — trade it for six months
Open when the normalised gap exceeds $2\\sigma$ (formation σ), short the rich leg, long the cheap
leg, close when prices cross or at the end of the period."""),
    code("""trade = px.loc[(px.index >= t0) & (px.index < t0 + pd.DateOffset(months=6)), universe]
pnl, legs, active = trade_pairs(trade, pairs.head(1), base=form_px.iloc[0])
log = pd.DataFrame({"legs traded": legs[:, 0], "in position": active[:, 0], "daily P&L": pnl[:, 0]}, index=trade.index)
print("round trips:", legs.sum() / 4, " total P&L per $1:", round(pnl.sum(), 4))
log[log["legs traded"] > 0]"""),
    code("""spread = ((trade[i] / form_px[i].iloc[0]) - (trade[j] / form_px[j].iloc[0])) / sigma
fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(spread.index, spread); [ax.axhline(v, ls="--", c="k", lw=.7) for v in (2, -2)]
ax.fill_between(spread.index, -4, 4, where=active[:, 0], alpha=.15, color="tab:green")
ax.set(title=f"{i}/{j}: spread in formation sigmas, shaded = open position"); plt.show()"""),
]

tour = nbf.v4.new_notebook()
tour.cells = [
    md("""# 02 — Results tour

Loads every table produced by the experiments and shows the key figures, hypothesis by
hypothesis. Numbers here are the same files the paper `\\input`s."""),
    SETUP,
    code("""from IPython.display import Image, display
show = lambda name: display(Image(f"figures/{name}.png", width=720))
read = lambda name: pd.read_csv(f"tables/{name}.csv")"""),
    md("## Data coverage (survivorship)"), code('read("data_coverage")'), code('show("fig01_coverage")'),
    md("## H1 — GGR replication"), code('read("tab_replication")'), code('show("fig03_equity_curves")'),
    md("## Half-life characterisation and H3 persistence"), code('read("tab_halflife_describe")'),
    code('read("tab_persistence_dev")'), code('show("fig09_persistence")'),
    md("## H2 — does formation half-life predict trading returns?"), code('read("tab_h2_quintiles_dev")'),
    code('read("tab_h2_famamacbeth_dev")'), code('show("fig19_h2_dev_vs_holdout")'),
    md("## H4 — transaction costs"), code('read("tab_breakeven_cost")'), code('show("fig14_cost_sensitivity")'),
    md("## Holdout (H2, H3, H5) with multiple-testing adjustment"), code('read("tab_holdout_tests")'),
    code('read("tab_holdout_performance")'), code('show("fig17_holdout_equity")'),
]

nbf.write(walk, "notebooks/01_pair_walkthrough.ipynb")
nbf.write(tour, "notebooks/02_results_tour.ipynb")
