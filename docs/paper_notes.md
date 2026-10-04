# Reading notes — Gatev, Goetzmann & Rouwenhorst (2006)

*Pairs Trading: Performance of a Relative-Value Arbitrage Rule.* Review of Financial
Studies 19(3), 797–827. doi:10.1093/rfs/hhj020. Read in full from the published
version (31 pp.); the March 1999 NBER working paper w7032 (1962–1997 sample) was
also consulted for the introduction. Page numbers below refer to the RFS version.

## 1. Research question

Does a simple, practitioner-style relative-value rule — "find two stocks that moved
together, short the winner and buy the loser when they diverge, unwind on
convergence" — earn excess returns, and are those returns explained by known
mean-reversion (short-term reversal), risk factors, transaction costs, or
short-selling frictions? The authors frame it as a test of a near-"Law of One
Price" for close economic substitutes (pp. 800–801).

## 2. Data

- CRSP daily files, all stocks, July 1962 – December 2002 (474 monthly observations
  of the strategy, July 1963 – Dec 2002).
- Liquidity screen: in each formation period drop any stock with **one or more
  days with no trade** (p. 803).
- Prices are cumulative **total-return** indices (dividends reinvested).
- Delisting: if a stock is delisted during trading, the position is closed using
  the delisting return or the last available price (p. 804, fn. 4).

## 3. Pair formation (p. 803)

1. Formation period = 12 months; trading period = the next 6 months. Both
   "chosen arbitrarily and have remained our horizons since the beginning of the study".
2. Normalise each stock's cumulative total-return index to 1 at the start of the
   formation period: \(P^i_t = \prod_{s \le t}(1 + r^i_s)\).
3. Distance: \(D_{ij} = \sum_{t \in \text{formation}} (P^i_t - P^j_t)^2\).
4. Rank pairs by D. Portfolios: **top 5, top 20, pairs 101–120, all pairs**.
   Each stock is matched to its minimum-distance partner.
5. Sector variant: both stocks within one of four S&P groups (Utilities,
   Transportation, Financials, Industrials) assigned by SIC code.

## 4. Trading rule (pp. 803–805)

- Trading starts the day after formation ends.
- **Open** when the normalised price difference exceeds **2 historical standard
  deviations** of that difference, estimated in the formation period.
- Long $1 of the lower-priced (loser), short $1 of the higher-priced (winner).
- **Close** at the next crossing of the normalised prices; otherwise at the
  last trading day of the trading period. Pairs may re-open after closing.
- Figure 1 (Kennecott/Uniroyal) shows trading-period normalised prices starting
  near 1.1, i.e. prices **stay normalised to the formation start** — they are not
  re-based at the start of trading. (This replication follows that.)
- "Wait one day" variant: open/close at the close of the day after the signal, to
  strip out bid–ask bounce. After Table 1, all results use wait-one-day.

## 5. Return computation (p. 805)

- Positions marked to market daily. Daily portfolio return
  \(r_{P,t} = \sum_i w_{i,t} r_{i,t} / \sum_i w_{i,t}\), with buy-and-hold weights
  \(w_{i,t} = (1 + r_{i,1}) \cdots (1 + r_{i,t-1})\); daily returns compounded to monthly.
- Cash not in an open pair earns zero (conservative, fn. 5).
- **Return on committed capital**: payoffs divided by the number of pairs
  selected (e.g. 20). **Fully invested / employed capital**: payoffs divided by the
  number of pairs that actually open during the trading period.
- A new portfolio starts every month; each calendar month's return is the average
  over the six overlapping portfolios (Jegadeesh & Titman 1993 overlap correction).
- t-statistics: Newey–West with six lags.

## 6. Key findings (numbers as read)

| Item | Value | Source |
|---|---|---|
| Top-5, fully invested, no wait | 1.31%/month (t = 8.84) | Table 1A |
| Top-20, fully invested, no wait | 1.44%/month (t = 11.56) | Table 1A |
| Top-5 / Top-20, committed capital, no wait | 0.78% / 0.81%/month | Table 1A |
| Top-20, fully invested, wait one day | 0.90%/month (t = 9.29) | Table 1B |
| Top-20, committed, wait one day | 0.52%/month | Table 1B |
| Avg. round trips per top-20 pair per 6 months | 1.96 | Table 2 |
| Avg. time a top-20 pair is open | 3.76 months | Table 2 |
| Share of utilities in top-20 pairs | 71% | Table 2B |
| Implied cost from waiting one day | 162 bp per round trip per pair (≈81 bp effective spread) | p. 810 |
| Pre-1989 vs post-1988 top-20 (wait one day) | 1.18% vs 0.38%/month | Table 8 |
| True out-of-sample 1999–2002, top-20 fully invested | 10.4% p.a., NW t = 4.82 | p. 800 |
| Random-pairs bootstrap (matched on prior-month return decile) | slightly negative returns | Table 6 |
| Factor model (FF3 + momentum + reversal) alpha, top-20 | 0.76%/month (t = 7.08) | Table 4 |

Interpretation offered: profits are compensation for enforcing the Law of One
Price, linked to a latent common factor whose importance fell after 1988; not
explained by short-term reversal, bankruptcy risk, or short-sale constraints.

## 7. What can be replicated here (free data)

| Element | Replicable? | How / gap |
|---|---|---|
| 12/6-month timeline, monthly overlapping portfolios | Yes | exact |
| SSD on normalised total-return indices | Yes | Yahoo adjusted close ≈ total-return index |
| Liquidity screen (no missing days) | Approx. | "no missing Yahoo price" in formation; no volume-based zero-trade check |
| Universe = all CRSP stocks | **No** | point-in-time S&P 500 only (large caps, ~500 names vs ~2,000+) |
| Delisted stocks | Partial | many delisted S&P members missing in Yahoo → residual survivorship bias |
| Top 5 / top 20 / 101–120 / sector | Yes (sector approx.) | GICS sector (current, not point-in-time) instead of SIC-based 4 groups |
| Wait-one-day, committed vs employed | Yes | exact rules |
| Risk-factor regressions | Not in scope | possible extension with Ken French data |
| Bootstrap random pairs | Simplified | uniformly random pairs, not matched on prior-month return decile |
| Sample period | Different | 2000–2025 instead of 1962–2002: an out-of-sample test of GGR itself |

## 8. Mean reversion and half-life — the mathematics

**Mean reversion.** A spread \(S_t\) is mean-reverting if deviations from a
long-run level \(\theta\) tend to shrink: \(E[S_{t+h} - \theta \mid S_t]\) decays
toward 0 as h grows. The canonical continuous-time model is the
**Ornstein–Uhlenbeck (OU)** process

\[ dS_t = \kappa(\theta - S_t)\,dt + \sigma\, dW_t, \qquad \kappa > 0 . \]

κ is the speed of reversion, θ the mean, σ the volatility, W a Brownian motion.

**Exact discretisation.** Solving the SDE over one step (Δt = 1 trading day):

\[ S_{t+1} = \theta(1 - e^{-\kappa}) + e^{-\kappa} S_t + \varepsilon_{t+1},\quad
\varepsilon \sim N\!\left(0, \tfrac{\sigma^2}{2\kappa}(1 - e^{-2\kappa})\right). \]

This is an AR(1), \(S_{t+1} = c + \phi S_t + \varepsilon_{t+1}\), with
\(\phi = e^{-\kappa}\) and \(c = \theta(1 - \phi)\).

**Half-life.** The expected deviation after h days is
\(E[S_{t+h} - \theta \mid S_t] = \phi^h (S_t - \theta)\). Setting \(\phi^h = 1/2\):

\[ h_{1/2} = \frac{\ln 2}{\kappa} = -\frac{\ln 2}{\ln \phi}. \]

φ → 1 means h → ∞ (random walk, no reversion); φ ≤ 0 means the AR(1) oscillates
and the OU interpretation breaks down (half-life undefined).

**Assumptions** behind reading the half-life as "speed": a linear, constant-
parameter, Gaussian, stationary process. Real spreads have regime changes, jumps
(earnings, M&A), and time-varying parameters — so a formation-period half-life
is an *estimate* of an unstable quantity. Whether it predicts anything out of
sample is exactly what this project tests.

**Estimation.**
1. *OLS*: regress \(S_t\) on \(S_{t-1}\) with intercept; \(\hat h = -\ln 2/\ln\hat\phi\).
   Equivalent to regressing \(\Delta S_t\) on \(S_{t-1}\) (slope \(\phi - 1\)).
2. *Exact MLE*: maximise the Gaussian likelihood of the AR(1) including the
   stationary distribution of the first observation,
   \(S_0 \sim N(\theta, \sigma_\varepsilon^2/(1 - \phi^2))\). Conditional MLE (dropping
   that term) is identical to OLS; the exact version differs in short samples.
3. *Kendall bias correction*: in finite samples OLS φ̂ is biased downward,
   \(E[\hat\phi] - \phi \approx -(1 + 3\phi)/T\) (Kendall 1954). Downward-biased φ
   ⇒ half-lives that look **too short**, most severely for persistent spreads and
   short windows. Corrected \(\tilde\phi = \hat\phi + (1 + 3\hat\phi)/T\).
4. *Model-free*: mean time between zero crossings of the demeaned spread.

**Why the GGR spread need not be stationary.** GGR select on small SSD in
*levels*, not on a stationarity test. Two random walks can stay close for a year
by chance (spurious closeness), and the spread \(P^i - P^j\) of normalised
prices is not a cointegrating combination unless β = 1 happens to be the right
hedge ratio. GGR acknowledge this ("spuriously correlated prices, which are not
de facto co-integrated", p. 802). The extension therefore measures how many
selected spreads pass ADF/KPSS/Engle–Granger tests, and whether the estimated
half-life adds information beyond the SSD rank.
