# Design Spec — Mean-Reversion Half-Life and Pairs-Trading Performance

Date: 2026-10-05 · Status: approved design, pending spec review

## 1. Research question

Does the **mean-reversion half-life** of a pair's spread, estimated using only
formation-period data, predict that pair's **out-of-sample trading-period return**
under the Gatev, Goetzmann & Rouwenhorst (2006, RFS 19(3)) distance rule?

Secondary: does the GGR anomaly survive in 2000–2025 point-in-time S&P 500 data,
net of costs, and is half-life a persistent pair characteristic?

## 2. Labelling of work

| Label | Content |
|---|---|
| **Replication** | GGR distance method: 12m formation, 6m trading, SSD ranking, 2σ open, cross to close, top-5/top-20/sector pairs, committed vs fully-invested returns, 1-day-wait variant. |
| **Extension** | Half-life estimation on GGR pairs (several estimators, lookbacks, spread definitions); stationarity/cointegration diagnostics; cointegration/OU selection with half-life-scaled rules (Approach B). |
| **Original research** | Cross-sectional predictive test of half-life (quintile sorts, Fama–MacBeth), half-life persistence, SSD+half-life filter evaluated on an untouched 2013–2025 holdout, cost break-evens, regime analysis. |

## 3. Hypotheses

- **H1 (replication):** GGR excess returns are positive; magnitude is lower than GGR's 1962–2002 estimates.
- **H2 (core):** shorter formation half-life ⇒ higher trading-period return (negative slope).
- **H3:** formation half-life correlates positively with realized trading-period half-life.
- **H4:** any H2 effect survives realistic transaction costs.
- **H5:** SSD + half-life filter beats SSD alone on the 2013–2025 holdout.

Each is stated with a test statistic and a pre-declared significance level (5%,
two-sided, with Holm and BH adjustments reported across the hypothesis family).

## 4. Data

- **Membership:** `fja05680/sp500` historical components (point-in-time ticker lists by date).
- **Prices:** Yahoo Finance (`yfinance`) daily adjusted close (total-return proxy), 1999-01-01 → 2025-12-31.
  Raw downloads cached to `data/raw/` (git-ignored) with a manifest (download date, tickers, row counts, SHA-256).
- **Market/regime series:** SPY, ^VIX (Yahoo); NBER recession dates (FRED `USREC`).
- **Sectors:** current GICS from the membership/Wikipedia list. *Caveat: not point-in-time (mild look-ahead); documented.*
- **Coverage report:** fraction of historical constituents with no or partial Yahoo data, by year. Residual survivorship bias is a stated limitation.

Cleaning rules: drop non-positive prices; no forward fill across >5 days; a stock enters a formation universe only if it is an index member on the formation start date and has no missing prices in the formation window (GGR rule). No requirement to exist in the trading period — delisted/missing mid-trade positions close at the last available price.

## 5. Methodology

### 5.1 GGR replication
Normalized price \(P_t = \prod_{s\le t}(1+r_s)\), rebased to 1 at formation start.
Distance \(D_{ij} = \sum_t (P^i_t - P^j_t)^2\). Select the N lowest-D pairs (N ∈ {5, 20}; plus top-20 within sector).
Trading: open when \(|P^i_t-P^j_t| > 2\hat\sigma_{ij}\) (σ̂ from formation), long the loser / short the winner, \$1 each leg; close at the next crossing or at period end. Re-entry allowed.
Portfolios start every month; each month's return is the average over the 6 overlapping portfolios (GGR). Returns reported on committed capital and on employed capital. Wait-one-day variant executes signals at t+1 close.

### 5.2 Half-life estimation (formation data only)
Spread definitions: (a) GGR \(S_t = P^i_t - P^j_t\); (b) \(S_t = \log p^i_t - \beta \log p^j_t\), β by OLS in formation.
Estimators:
1. AR(1) OLS: \(S_t = c + \phi S_{t-1} + \varepsilon_t\), \(HL = -\ln 2/\ln\phi\) (undefined/∞ if φ ≥ 1).
2. Exact-discretization OU MLE (κ, θ, σ); \(HL=\ln 2/\kappa\).
3. Kendall bias-corrected φ: \(\tilde\phi = \hat\phi + (1+3\hat\phi)/T\).
4. Empirical: mean time between zero-crossings of demeaned spread (model-free).
Lookbacks: 63, 126, 252 trading days ending at formation end.

### 5.3 Diagnostics
ADF, KPSS on spreads; Engle–Granger (and Johansen check) on log prices; variance ratio. Reported as distributions across pairs and as features in the cross-section.

### 5.4 Predictive tests
- Quintile sorts of selected pairs (top-20 per formation and broader top-100 for power) by formation HL; mean trading-period return per quintile and Q1–Q5 spread with Newey–West SE (lag 6, overlap).
- Fama–MacBeth: \(R_{ij,\text{trade}} = a + b\,\log HL_{ij} + c\,\text{SSD rank} + d\,\text{controls}\); time-series mean of slopes, NW SE.
- Persistence: rank correlation of formation HL vs realized trading HL.

### 5.5 Extension B — cointegration/OU strategy
Select by Engle–Granger p-value among SSD top-100; trade z-score of OU spread (entry |z|>2, exit z crosses 0, max holding = k·HL). Compared with GGR on the same timeline.

### 5.6 Validation protocol
- Development sample: formations starting 2000–2012. All filters/thresholds chosen here.
- Holdout: 2013–2025, run once after parameters are frozen (config committed with a hash before the holdout run).
- All variants tried are logged in `results/experiment_log.csv`; report Holm/BH-adjusted p-values and a deflated Sharpe ratio.
- Baselines: random pairs from the same universe, plain GGR, SPY buy-and-hold.

### 5.7 Costs
Per-side cost c ∈ {0, 5, 10, 20, 30} bp applied per leg per trade (open + close); break-even c*; wait-one-day as GGR's bid-ask proxy. Short-borrow cost noted as a limitation (no free data).

### 5.8 Robustness / regimes
Sub-periods (2000–07, 2008–09, 2010–19, 2020–25), VIX terciles, NBER recession months, entry threshold ∈ {1.5, 2, 2.5}σ, formation length ∈ {6, 12} months, N ∈ {5, 20, 50}.

## 6. Look-ahead / leakage checks (explicit)
- Universe uses membership as of formation start only.
- Every estimate (σ̂, β, HL, test stats) uses formation window only.
- Unit test: perturbing prices after date t leaves all signals ≤ t unchanged.
- Trades at close of signal day (or t+1 in wait variant); no same-bar decision on data not yet observed beyond that close.
- Holdout untouched until final run.

## 7. Repository layout
```
src/hlpairs/   data.py universe.py pairs.py halflife.py diagnostics.py backtest.py costs.py evaluation.py plots.py
experiments/   01_data.py 02_replication.py 03_halflife.py 04_predictive.py 05_ou_extension.py 06_robustness.py 07_holdout.py  + configs/*.yaml
notebooks/     walkthrough notebooks (read results, re-plot)
tests/         OU recovery, look-ahead, backtest accounting, SSD selection
figures/ tables/ results/ paper/ (LaTeX → PDF) data/README.md
README.md LICENSE (MIT) CITATION.cff pyproject.toml requirements.lock
```
Python 3.13 via `uv`, pinned lockfile, seeds recorded.

## 8. Figures (minimum set)
Price & normalized price examples; spread and z-score with trade episodes shaded; HL distributions by estimator/lookback; rolling HL; ADF/EG p-value distributions; HL quintile vs return; formation vs realized HL scatter; equity curves & drawdowns (GGR, HL-filtered, baselines); cost sensitivity; parameter heatmaps; regime bars; sector cross-section; coverage/survivorship chart.

## 9. Publication
GitHub and Zenodo are outward-facing and require the user's credentials: performed only after explicit confirmation. DOI is written to README/CITATION.cff only after Zenodo returns it.

## 10. Known limitations (to carry into paper)
Yahoo lacks many delisted tickers (residual survivorship bias); adjusted close approximates total return; GICS not point-in-time; no borrow costs/short constraints; S&P 500 large-cap universe differs from GGR's CRSP universe.
