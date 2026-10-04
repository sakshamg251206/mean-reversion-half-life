# Does Faster Mean Reversion Predict Pairs-Trading Profits?

**A replication and extension of Gatev, Goetzmann & Rouwenhorst (2006) on point-in-time S&P 500 data, 2000–2025, testing whether a pair's mean-reversion half-life predicts its out-of-sample profits.**

📄 **Paper:** [`paper/main.pdf`](paper/main.pdf) · 🧪 **Tests:** 44 unit tests (`make test`) · 🔒 **Holdout:** parameters frozen in [`experiments/frozen.yaml`](experiments/frozen.yaml) before a single out-of-sample run ([`results/holdout_record.yaml`](results/holdout_record.yaml))

> **Zenodo DOI:** will be added once the archive is published. No DOI exists yet.

---

## TL;DR

| | Development 2000–2012 | Holdout 2013–2025 |
|---|---|---|
| GGR top-20 pairs, gross (committed capital) | **+0.24%/month** (t = 2.21) | **0.00%/month** (t = −0.01) |
| Half-life effect: fastest-minus-slowest quintile, 6-month return | **+1.74%** (t = 3.29) | **+0.36%** (t = 1.04, p = 0.30) |
| Half-life-filtered portfolio vs plain GGR, net of 10 bp | Sharpe 0.73 vs 0.37 | Sharpe −0.28 vs −0.47; difference p = 0.32 |
| Break-even cost, GGR top-20 (bp per leg per side) | 24.5 | 0.0 |

In 2000–2012, half-life looks like a strong, cost-robust predictor of pairs returns, with t-statistics above 3 for every reasonable estimator. **It fails a pre-specified holdout.** Estimated half-life barely persists (Spearman correlation 0.03 between the formation-period estimate and the half-life realised in the trading period). It mostly measures how stationary a spread *looked* during formation, and its predictive power faded after 2010. No hypothesis survives a Holm adjustment out of sample. This is a hypothesis test, not a trading strategy.

## Research question

Under the GGR distance rule, does the half-life of a pair's spread, estimated using **only** formation-period data, predict that pair's **out-of-sample** trading-period return? And does the GGR anomaly itself still exist on modern data, net of costs?

## Original paper

Gatev, E., Goetzmann, W. N., & Rouwenhorst, K. G. (2006). *Pairs Trading: Performance of a Relative-Value Arbitrage Rule.* **Review of Financial Studies**, 19(3), 797–827. [doi:10.1093/rfs/hhj020](https://doi.org/10.1093/rfs/hhj020).
Reading notes, the paper's key numbers, and the mean-reversion and half-life mathematics: [`docs/paper_notes.md`](docs/paper_notes.md).

## Motivation

A pair trade is a bet on mean reversion. Practitioners routinely rank or filter pairs by their Ornstein–Uhlenbeck half-life, h = −ln 2 / ln φ. GGR's selection rule (the sum of squared differences between normalised prices) ignores the speed of reversion entirely. If fast reversion persists, it should improve on GGR. That claim had not been tested with a strict development/holdout protocol.

## What is replication, extension and original research

| Label | Content |
|---|---|
| **Replication** | GGR rule exactly: 12-month formation, 6-month trading, minimum sum-of-squared-differences pairs, 2σ entry, exit when prices cross, six overlapping portfolios, committed and employed capital, wait-one-day execution, top-5, top-20, sector and random portfolios |
| **Extension** | Half-life via OLS AR(1), the Kendall bias correction, exact OU maximum likelihood and a model-free crossing interval; lookbacks of 63, 126 and 252 days; two spread definitions; ADF, KPSS, Engle–Granger, Johansen and variance-ratio diagnostics; a cointegration/OU rule with holding time scaled by half-life |
| **Original research** | Quintile sorts and Fama–MacBeth regressions of future returns on half-life (H2); persistence of half-life (H3); costs (H4); a half-life filter (H5) chosen on 2000–2012 and evaluated once on 2013–2025, with Holm/BH adjustments and a deflated Sharpe ratio |

## Methodology (pipeline)

```mermaid
flowchart LR
  A[Point-in-time S&P 500 membership] --> C[Universe at formation start<br/>no missing prices in 12 months]
  B[Yahoo adjusted closes<br/>spell masking + stale screen] --> C
  C --> D[SSD on normalised prices<br/>top-100 candidate pairs]
  D --> E[Formation-only features<br/>half-life x4, ADF/KPSS/EG/Johansen, VR]
  D --> F[6-month trading<br/>2σ open, cross close]
  E --> G[H2/H3 tests<br/>quintiles, Fama-MacBeth]
  F --> G
  G --> H[Dev 2000–12: choose filter<br/>freeze + commit]
  H --> I[Holdout 2013–25: one run]
```

**Look-ahead and leakage checks**
- Membership comes from the formation-start snapshot only.
- Every estimate (σ, β, half-lives, test statistics) uses formation-period data only.
- [`test_no_lookahead`](tests/test_backtest.py) perturbs all prices after date *t* and asserts that every output up to *t* is unchanged.
- [`test_formation_features_do_not_use_trading_data`](tests/test_engine.py) perturbs trading-period prices and asserts that all formation features are unchanged.
- Five portfolios that start in August–December 2012, whose trading periods straddle the split, are excluded from both the development sample and the holdout.

## Dataset

Point-in-time S&P 500 constituents ([fja05680/sp500](https://github.com/fja05680/sp500)) and Yahoo Finance adjusted closes, 1999–2025; SPY, VIX, and NBER recession dates from FRED. **Free data carries survivorship bias.** Yahoo has no prices for 426 of the 1,117 historical members, so coverage rises from 44% (1999) to 97% (2025). Every cleaning rule, the ticker-reuse fix and the bias discussion are in [`data/README.md`](data/README.md).

## Main findings

- **H1 — replication: partly supported.** GGR top-20 earns 0.24%/month gross in 2000–12 and 0.00% in 2013–25, against 0.81% in GGR's 1962–2002 sample. Trading intensity matches GGR (19.4 of 20 pairs open; 1.5 round trips per pair).
- **Spreads aren't really cointegrated.** Only 54% of selected spreads reject a unit root (ADF, 5%) and 37% pass Engle–Granger. The #1 pair in the first formation fails both tests.
- **H3 — persistence: rejected.** Median half-life is 8.8 days in formation but 16.2 days realised; Spearman ρ = 0.03 (dev) and −0.01 (holdout).
- **H2 — prediction: supported in dev, not out of sample.** Fastest-minus-slowest quintile: +1.74% per 6 months, t = 3.29 (dev) → +0.36%, t = 1.04 (holdout). The effect decays gradually after 2010; it is not driven by the crisis.
- **H4 — costs: dev only.** Break-even costs drop from 25–38 bp to 0–4 bp per leg-side.
- **H5 — half-life filter: rejected.** It is best of 21 dev variants (deflated-Sharpe probability 0.99) but does not beat GGR in the holdout (p = 0.32).
- **Regimes.** Profits concentrate in 2008–09, high-VIX months and recessions. Utilities, the most common pairs, earn nothing.

## Visual results

### Data

| | |
|---|---|
| ![coverage](figures/fig01_coverage.png) | **Fig 1 — Survivorship coverage.** Share of each year's index members with Yahoo prices: 44% in 1999, 97% in 2025. |

### Replication (H1)

![example pair](figures/fig02_example_pair.png)
**Fig 2 — One pair through the pipeline:** prices, normalised prices, and the spread in formation σ, with mean-reversion episodes shaded.

| | |
|---|---|
| ![equity](figures/fig03_equity_curves.png) **Fig 3 — Growth of $1.** GGR portfolios, random pairs and SPY. | ![drawdowns](figures/fig04_drawdowns.png) **Fig 4 — Drawdowns.** Top-20 pairs vs SPY. |
| ![rolling](figures/fig05_rolling_performance.png) **Fig 5 — Decay.** 36-month rolling mean return of the top-20 portfolio. | |

### Half-life characterisation (H3)

| | |
|---|---|
| ![hl dist](figures/fig06_halflife_distributions.png) **Fig 6 — Half-life distributions** by estimator. | ![agreement](figures/fig07_estimator_agreement.png) **Fig 7 — Estimator agreement** (rank correlations). |
| ![diagnostics](figures/fig08_diagnostics.png) **Fig 8 — Stationarity/cointegration p-values** of the selected spreads. | ![persistence](figures/fig09_persistence.png) **Fig 9 — Persistence.** Formation vs realised half-life. |
| ![rolling hl](figures/fig10_rolling_halflife.png) **Fig 10 — Rolling half-life** of the most often selected pair. | |

### Prediction (H2, H5)

| | |
|---|---|
| ![h2 dev](figures/fig11_h2_quintiles_dev.png) **Fig 11 — H2 in dev.** Return by half-life quintile. | ![h5 grid](figures/fig12_h5_dev_grid.png) **Fig 12 — The 21 H5 variants tried in dev** (all logged). |
| ![dev vs holdout](figures/fig19_h2_dev_vs_holdout.png) **Fig 19 — In vs out of sample.** | ![subperiods](figures/fig20_h2_subperiods.png) **Fig 20 — The premium decays** after 2010. |

### Robustness and costs (H4)

| | |
|---|---|
| ![sensitivity](figures/fig13_sensitivity.png) **Fig 13 — Parameter sensitivity.** All 18 dev Sharpe ratios positive, all 18 holdout ratios negative. | ![costs](figures/fig14_cost_sensitivity.png) **Fig 14 — Return vs cost** per leg-side. |
| ![regimes](figures/fig15_regimes.png) **Fig 15 — Regimes:** era, VIX, NBER. | ![sectors](figures/fig16_sector_crosssection.png) **Fig 16 — Sector cross-section.** |

### Holdout

| | |
|---|---|
| ![holdout equity](figures/fig17_holdout_equity.png) **Fig 17 — Holdout equity curves and drawdowns**, net of 10 bp. | ![h2 holdout](figures/fig18_h2_quintiles_holdout.png) **Fig 18 — H2 in the holdout.** |

### Key tables

**Holdout tests (2013–2025)** — [`tables/tab_holdout_tests.csv`](tables/tab_holdout_tests.csv)

| Hypothesis | Estimate | p | Holm | BH |
|---|---|---|---|---|
| H2 Q1−Q5 (OLS 252d, gross, 6-month return) | +0.36% | 0.300 | 0.910 | 0.318 |
| H2 Fama–MacBeth slope on half-life rank | −0.0054 | 0.182 | 0.910 | 0.303 |
| H3 Spearman formation vs realised half-life | −0.011 | 0.193 | 0.910 | 0.303 |
| H5 HL-filtered − GGR (net, monthly) | +0.04% | 0.318 | 0.910 | 0.318 |
| GGR top-20 net mean | −0.09% | 0.093 | 0.555 | 0.303 |

All tables (CSV and LaTeX) are in [`tables/`](tables/); the log of every variant tried is in [`results/experiment_log.csv`](results/experiment_log.csv).

## Reproduction

Requires Python 3.13, [uv](https://docs.astral.sh/uv/), and (for the paper) a TeX distribution with `pdflatex` and `bibtex`.

```bash
make env        # uv sync: exact versions from uv.lock (requirements.txt also provided)
make test       # 44 unit tests
make data       # download + clean (~10 min; Yahoo is rate-limited, downloads are cumulative)
make run        # experiments 02–06 (~1 h on a laptop; engine runs in parallel)
make holdout    # 07: refuses to run unless frozen.yaml/config.yaml are committed and unchanged
make notebooks  # builds and executes notebooks/
make paper      # paper/main.pdf
```

`experiments/08_subperiods.py` is a descriptive post-holdout decomposition and makes no parameter choices. Yahoo Finance revises its data and deletes delisted tickers over time, so a later download can differ. [`data/raw/manifest.json`](data/raw/manifest.json) records the SHA-256 of the files used here. The price files are not redistributed (Yahoo's terms). Seeds are fixed (`seed: 0`).

## Repository structure

```
src/hlpairs/        library: halflife, diagnostics, data, universe, pairs, backtest, evaluation, engine, plots
tests/              44 unit tests (OU recovery, Kendall bias, look-ahead, hand-worked trades, FM recovery …)
experiments/        01_data … 08_subperiods, config.yaml, frozen.yaml
notebooks/          01_pair_walkthrough (one pair end to end), 02_results_tour
figures/ tables/    every figure (PNG + PDF) and table (CSV + LaTeX) in the paper
results/            pair panel, experiment log, sensitivity, cost curves, holdout record
paper/              main.tex, references.bib, main.pdf
data/README.md      provenance, cleaning, survivorship analysis
docs/               reading notes, design spec, implementation plan
```

## Limitations

- **Survivorship bias.** Yahoo is missing 426 historical members, which most likely inflates returns in 2000–2012.
- **Narrower universe.** S&P 500 large caps only, not GGR's full CRSP cross-section.
- **Sectors.** GICS sectors are current, not point-in-time.
- **Costs.** Linear costs only: no market impact and no short-borrow fees.
- **Risk adjustment.** No factor-model regressions.
- **Holdout.** A single holdout is one draw.

Details are in the paper, §12.

## Citation

```bibtex
@techreport{garg2026halflife,
  author = {Garg, Saksham},
  title  = {Does Faster Mean Reversion Predict Pairs-Trading Profits? Half-Life Evidence from a Replication of Gatev, Goetzmann and Rouwenhorst (2006)},
  year   = {2026},
  type   = {Working paper},
  note   = {Code and data documentation: this repository}
}
```

See also [`CITATION.cff`](CITATION.cff). Zenodo DOI: *pending — not yet published.*

## License

MIT — see [`LICENSE`](LICENSE). Not investment advice.
