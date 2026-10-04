# Mean-Reversion Half-Life — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replicate the GGR (2006) distance pairs rule on point-in-time S&P 500 data (1999–2025) and test whether formation-period mean-reversion half-life predicts out-of-sample trading-period returns; ship as a reproducible repo, paper PDF, and (gated) GitHub + Zenodo release.

**Architecture:** A small library `src/hlpairs/` (pure functions, unit-tested against known answers) plus numbered experiment scripts that read cached data, run the rolling formation/trading engine once, and write every figure/table. Analysis that chooses parameters reads only the development sample; the holdout script runs once against a committed frozen config.

**Tech Stack:** Python 3.13 (uv-managed venv, `uv.lock`), numpy, pandas, scipy, statsmodels, matplotlib, yfinance, pyarrow, pyyaml, lxml, jinja2, pytest, jupyter/nbconvert, pdflatex + bibtex.

**Spec:** `docs/superpowers/specs/2026-10-05-mean-reversion-half-life-design.md`

## Global Constraints

- Python `>=3.13,<3.14`; every dependency pinned through `uv.lock`; `requirements.txt` exported from the lock.
- Sample: prices 1999-01-01 → 2025-12-31. Formation 12 calendar months, trading 6 calendar months, a new portfolio on the first trading day of every month.
- Development sample = portfolios with trade start ≤ 2012-07-01; **embargo** = trade starts 2012-08 … 2012-12 (excluded from both); **holdout** = trade starts ≥ 2013-01-01.
- Every estimate used to select or trade a pair uses formation-window data only.
- Universe on a formation date = S&P 500 members on the formation start date with no missing price in the formation window. No requirement to survive the trading period.
- Costs: per-side cost in bp charged per leg per trade (round trip on a pair = 4 × c). Grid {0, 5, 10, 20, 30}; headline net figure uses 10 bp.
- Significance: 5% two-sided; Newey–West lag 6 for overlapping monthly series; Holm and BH reported across the hypothesis family.
- Random seeds: base seed 0, recorded in every results file.
- Never write a DOI, Zenodo link, or GitHub URL into any file until it actually exists.

## Review Focus

1. **A leg stops trading mid-position (delisting, ticker reuse masking):** position closes at the last available price, no further P&L, no crash — test in Task 6 (`test_delisting_closes_at_last_price`).
2. **A pair that diverges and never reconverges:** position is force-closed on the last trading day and closing costs are charged — test in Task 6 (`test_forced_close_at_end`).
3. **Non-mean-reverting or oscillating spread (φ ≥ 1 or φ ≤ 0):** half-life returns `inf` / `nan`, never raises; quintile sorts and Fama–MacBeth treat `inf` as slowest, drop `nan` — tests in Task 2 and Task 7.
4. **Stock that lists mid-formation window or is not a member on the formation start date:** excluded from that universe — test in Task 4.
5. **Formation month with < 2 eligible stocks (data gaps):** engine skips the month and returns `None` — test in Task 8.

---

## File Structure

```
pyproject.toml, uv.lock, requirements.txt, .gitignore, LICENSE, CITATION.cff, README.md
docs/paper_notes.md                      Task 0: reading notes on GGR + mean-reversion math
src/hlpairs/__init__.py
src/hlpairs/halflife.py                  half-life estimators (AR1/OLS, Kendall, OU exact MLE, crossing interval)
src/hlpairs/diagnostics.py               ADF, KPSS, Engle–Granger, Johansen, variance ratio
src/hlpairs/data.py                      membership parsing, spell masking, Yahoo download, cleaning, caching
src/hlpairs/universe.py                  formation universe
src/hlpairs/pairs.py                     normalisation, SSD ranking, pair frames
src/hlpairs/backtest.py                  single-pair simulator, GGR/OU pair trading, portfolio & monthly returns
src/hlpairs/evaluation.py                NW t-stat, summary stats, quintile sorts, Fama–MacBeth, multiple testing, DSR
src/hlpairs/engine.py                    rolling formation/trading engine, FormationResult, strategy_monthly
src/hlpairs/plots.py                     shared matplotlib style + save helper
tests/test_*.py                          one file per module
experiments/config.yaml                  fixed (non-tuned) settings
experiments/frozen.yaml                  dev-selected parameters (written by 05, committed before 07)
experiments/01_data.py … 07_holdout.py   produce data/, results/, figures/, tables/
notebooks/01_pair_walkthrough.ipynb, notebooks/02_results_tour.ipynb
data/README.md                           data provenance, cleaning, known biases
paper/main.tex, paper/references.bib, paper/main.pdf
```

---

### Task 0: Read the paper and write the theory notes

**Files:** Create `docs/paper_notes.md`

- [ ] **Step 1:** Download the NBER working-paper version of GGR (w7032) into the scratchpad: `curl -sL https://www.nber.org/system/files/working_papers/w7032/w7032.pdf -o $SCRATCH/ggr.pdf`. Read it fully (Read tool with `pages`). If the download fails, record that and work from the published RFS abstract plus secondary descriptions, saying so explicitly in the notes.
- [ ] **Step 2:** Write `docs/paper_notes.md` with: research question; data (CRSP 1962–2002, liquidity screens); pair formation (normalised cumulative total-return index, SSD, top 5/20/101–120, industry pairs); trading rule (2σ open, cross close, one-day wait); return computation (committed vs employed capital, six overlapping portfolios); key findings (≈11% annualised excess for top-20, robustness to wait-one-day, bootstrap vs random pairs, decline over time); what we can and cannot replicate with free data. Quote only numbers actually read in the paper; mark anything not verified.
- [ ] **Step 3:** In the same file, derive mean reversion and half-life: OU SDE `dS = κ(θ − S)dt + σ dW`; exact discretisation `S_{t+1} = θ(1 − e^{−κ}) + e^{−κ} S_t + ε`, φ = e^{−κ}; half-life `h = ln 2 / κ = −ln 2 / ln φ`; assumptions (linear, Gaussian, constant parameters); small-sample bias `E[φ̂] − φ ≈ −(1 + 3φ)/T` (Kendall 1954) and why it makes half-lives look too short; why the GGR spread is not guaranteed stationary.
- [ ] **Step 4:** Commit: `git add docs/paper_notes.md && git commit -m "docs: GGR reading notes and half-life theory"`

---

### Task 1: Environment and package skeleton

**Files:** Create `pyproject.toml`, `src/hlpairs/__init__.py`, `tests/test_smoke.py`; modify `.gitignore`.

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "hlpairs"
version = "1.0.0"
description = "Mean-reversion half-life and pairs-trading performance: a GGR (2006) replication and extension"
requires-python = ">=3.13,<3.14"
dependencies = [
  "numpy", "pandas", "scipy", "statsmodels", "matplotlib", "yfinance",
  "pyarrow", "pyyaml", "lxml", "jinja2", "requests",
]

[dependency-groups]
dev = ["pytest", "jupyter", "nbconvert", "nbformat", "ipykernel"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2:** `src/hlpairs/__init__.py` containing `"""Mean-reversion half-life research library."""`. `tests/test_smoke.py`:

```python
def test_import():
    import hlpairs  # noqa: F401
```

- [ ] **Step 3:** Run `uv sync --python 3.13 && uv run pytest -q` → `1 passed`. Then `uv export --format requirements-txt --no-hashes > requirements.txt`.
- [ ] **Step 4:** Append to `.gitignore`: `results/cache/`, `paper/*.aux`, `paper/*.log`, `paper/*.bbl`, `paper/*.blg`, `paper/*.out`.
- [ ] **Step 5:** Commit `pyproject.toml uv.lock requirements.txt src tests .gitignore` — `"build: uv environment and package skeleton"`.

---

### Task 2: Half-life estimators

**Files:** Create `src/hlpairs/halflife.py`, `tests/test_halflife.py`

**Interfaces — Produces:** `phi_to_halflife(phi) -> float`, `ar1_phi(s) -> float`, `kendall_phi(s) -> float`, `halflife_ols(s)`, `halflife_kendall(s)`, `halflife_ou_mle(s)`, `crossing_interval(s)` (all take a 1-D array-like, ignore NaN, return float days; `inf` = no reversion, `nan` = φ ≤ 0).

- [ ] **Step 1: Failing tests**

```python
import numpy as np
import pytest
from hlpairs import halflife as hl


def simulate_ar1(phi, T, rng, mu=0.0):
    s = np.empty(T)
    s[0] = mu
    e = rng.standard_normal(T)
    for t in range(1, T):
        s[t] = mu + phi * (s[t - 1] - mu) + e[t]
    return s


def test_phi_to_halflife_edges():
    assert hl.phi_to_halflife(0.5) == pytest.approx(1.0)
    assert hl.phi_to_halflife(1.0) == np.inf
    assert hl.phi_to_halflife(1.02) == np.inf
    assert np.isnan(hl.phi_to_halflife(-0.2))
    assert np.isnan(hl.phi_to_halflife(np.nan))


@pytest.mark.parametrize("est", [hl.halflife_ols, hl.halflife_kendall, hl.halflife_ou_mle])
def test_recovers_true_halflife_long_sample(est):
    true_hl = 10.0
    s = simulate_ar1(0.5 ** (1 / true_hl), 20_000, np.random.default_rng(1), mu=3.0)
    assert est(s) == pytest.approx(true_hl, rel=0.1)


def test_kendall_reduces_small_sample_bias():
    phi = 0.5 ** (1 / 20)
    rng = np.random.default_rng(2)
    ols, ken = [], []
    for _ in range(400):
        s = simulate_ar1(phi, 126, rng)
        ols.append(hl.ar1_phi(s))
        ken.append(hl.kendall_phi(s))
    assert np.mean(ols) < phi                      # OLS biased down
    assert abs(np.mean(ken) - phi) < abs(np.mean(ols) - phi)


def test_random_walk_has_long_halflife():
    s = np.cumsum(np.random.default_rng(3).standard_normal(5000))
    assert hl.halflife_ols(s) > 100


def test_crossing_interval_faster_for_faster_reversion():
    rng = np.random.default_rng(4)
    fast = hl.crossing_interval(simulate_ar1(0.5, 5000, rng))
    slow = hl.crossing_interval(simulate_ar1(0.98, 5000, rng))
    assert fast < slow


def test_nan_ignored_and_constant_series():
    s = simulate_ar1(0.8, 500, np.random.default_rng(5))
    s[10] = np.nan
    assert np.isfinite(hl.halflife_ols(s))
    assert hl.crossing_interval(np.ones(50)) == np.inf
```

- [ ] **Step 2:** `uv run pytest tests/test_halflife.py -q` → FAIL (module missing).
- [ ] **Step 3: Implement `src/hlpairs/halflife.py`**

```python
"""Half-life estimators for a mean-reverting spread.

An AR(1) spread S_t = c + phi*S_{t-1} + e_t is the exact daily discretisation of an
Ornstein-Uhlenbeck process dS = kappa*(theta - S)dt + sigma*dW, with phi = exp(-kappa).
A deviation from theta decays like phi**h, so it halves after
h = ln(0.5)/ln(phi) = ln(2)/kappa trading days.
"""
import numpy as np
from scipy.optimize import minimize

LN2 = np.log(2.0)


def _clean(s) -> np.ndarray:
    s = np.asarray(s, dtype=float)
    return s[~np.isnan(s)]


def phi_to_halflife(phi: float) -> float:
    """Days for a deviation to halve. inf if phi >= 1 (no reversion), nan if phi <= 0 (oscillation)."""
    if not np.isfinite(phi) or phi <= 0:
        return np.nan
    if phi >= 1:
        return np.inf
    return float(-LN2 / np.log(phi))


def ar1_phi(s) -> float:
    """OLS slope of S_t on S_{t-1} (with intercept)."""
    s = _clean(s)
    x, y = s[:-1], s[1:]
    xc = x - x.mean()
    denom = xc @ xc
    return float(xc @ (y - y.mean()) / denom) if denom > 0 else np.nan


def kendall_phi(s) -> float:
    """OLS phi plus Kendall's (1954) first-order bias correction (1 + 3*phi)/T."""
    s = _clean(s)
    phi = ar1_phi(s)
    return phi + (1 + 3 * phi) / (len(s) - 1)


def halflife_ols(s) -> float:
    return phi_to_halflife(ar1_phi(s))


def halflife_kendall(s) -> float:
    return phi_to_halflife(kendall_phi(s))


def halflife_ou_mle(s) -> float:
    """Exact Gaussian likelihood of the discretised OU process, including the stationary
    distribution of the first observation (this is what distinguishes it from OLS)."""
    s = _clean(s)
    phi0 = ar1_phi(s)
    if not 0 < phi0 < 1:
        return phi_to_halflife(phi0)
    mu0 = s.mean()
    resid = s[1:] - mu0 - phi0 * (s[:-1] - mu0)

    def nll(p):
        phi = 1 / (1 + np.exp(-p[0]))          # logistic keeps 0 < phi < 1
        mu, se2 = p[1], np.exp(2 * p[2])
        e = s[1:] - mu - phi * (s[:-1] - mu)
        v0 = se2 / (1 - phi**2)                  # stationary variance of S_0
        return 0.5 * (np.log(v0) + (s[0] - mu) ** 2 / v0 + len(e) * np.log(se2) + e @ e / se2)

    x0 = [np.log(phi0 / (1 - phi0)), mu0, np.log(resid.std() + 1e-12)]
    res = minimize(nll, x0, method="L-BFGS-B")
    return phi_to_halflife(1 / (1 + np.exp(-res.x[0])))


def crossing_interval(s) -> float:
    """Model-free speed proxy: mean number of days between sign changes of the demeaned spread."""
    s = _clean(s)
    d = np.sign(s - s.mean())
    d = d[d != 0]
    n = np.count_nonzero(d[1:] != d[:-1])
    return len(s) / n if n else np.inf


ESTIMATORS = {
    "ols": halflife_ols,
    "kendall": halflife_kendall,
    "ou_mle": halflife_ou_mle,
    "crossing": crossing_interval,
}
```

- [ ] **Step 4:** `uv run pytest tests/test_halflife.py -q` → all pass.
- [ ] **Step 5:** Commit — `"feat: half-life estimators with OU-recovery tests"`.

---

### Task 3: Stationarity and cointegration diagnostics

**Files:** Create `src/hlpairs/diagnostics.py`, `tests/test_diagnostics.py`

**Interfaces — Produces:** `adf_pvalue(s)`, `kpss_pvalue(s)`, `eg_pvalue(y, x)`, `johansen_reject(y, x) -> bool`, `variance_ratio(s, q=10)`.

- [ ] **Step 1: Failing tests**

```python
import numpy as np
from hlpairs import diagnostics as dg


def ar1(phi, T, seed):
    rng = np.random.default_rng(seed)
    s = np.zeros(T)
    for t in range(1, T):
        s[t] = phi * s[t - 1] + rng.standard_normal()
    return s


def test_adf_and_kpss_separate_stationary_from_random_walk():
    stat, rw = ar1(0.5, 1000, 0), np.cumsum(np.random.default_rng(1).standard_normal(1000))
    assert dg.adf_pvalue(stat) < 0.01
    assert dg.adf_pvalue(rw) > 0.05
    assert dg.kpss_pvalue(stat) >= 0.05
    assert dg.kpss_pvalue(rw) <= 0.05


def test_engle_granger_and_johansen():
    rng = np.random.default_rng(2)
    x = np.cumsum(rng.standard_normal(1000))
    y = 0.8 * x + ar1(0.5, 1000, 3)
    z = np.cumsum(rng.standard_normal(1000))
    assert dg.eg_pvalue(y, x) < 0.01
    assert dg.eg_pvalue(z, x) > 0.05
    assert dg.johansen_reject(y, x)
    assert not dg.johansen_reject(z, x)


def test_variance_ratio():
    assert dg.variance_ratio(ar1(0.8, 5000, 4)) < 0.7
    assert abs(dg.variance_ratio(np.cumsum(np.random.default_rng(5).standard_normal(5000))) - 1) < 0.15
```

- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3: Implement**

```python
"""Stationarity / cointegration diagnostics used to characterise pair spreads."""
import warnings

import numpy as np
from statsmodels.tsa.stattools import adfuller, coint, kpss
from statsmodels.tsa.vector_ar.vecm import coint_johansen


def adf_pvalue(s) -> float:
    """H0: unit root. Small p => evidence the spread is stationary."""
    return float(adfuller(np.asarray(s, float), autolag="AIC", maxlag=10)[1])


def kpss_pvalue(s) -> float:
    """H0: level-stationary. Small p => evidence against stationarity. Reported p is clipped to [0.01, 0.10]."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(kpss(np.asarray(s, float), regression="c", nlags="auto")[1])


def eg_pvalue(y, x) -> float:
    """Engle-Granger two-step test, H0: no cointegration between y and x."""
    return float(coint(np.asarray(y, float), np.asarray(x, float), maxlag=10, autolag="aic")[1])


def johansen_reject(y, x) -> bool:
    """Johansen trace test of rank 0 at 5%: True => cointegrated."""
    r = coint_johansen(np.column_stack([y, x]), det_order=0, k_ar_diff=1)
    return bool(r.lr1[0] > r.cvt[0, 1])


def variance_ratio(s, q: int = 10) -> float:
    """Lo-MacKinlay VR(q) of the spread's increments: <1 mean reversion, ~1 random walk."""
    s = np.asarray(s, float)
    d1 = np.diff(s)
    dq = s[q:] - s[:-q]
    return float(dq.var(ddof=1) / (q * d1.var(ddof=1)))
```

- [ ] **Step 4:** Run → pass.
- [ ] **Step 5:** Commit — `"feat: stationarity and cointegration diagnostics"`.

---

### Task 4: Data layer and formation universe

**Files:** Create `src/hlpairs/data.py`, `src/hlpairs/universe.py`, `tests/test_data.py`

**Interfaces — Produces:**
- `to_yahoo(t: str) -> str`
- `load_membership(path) -> pd.Series` (index: snapshot date, values: `frozenset` of Yahoo tickers)
- `members_on(membership, date) -> frozenset`
- `membership_spells(membership) -> dict[str, list[tuple[Timestamp, Timestamp]]]`
- `mask_outside_spells(px, spells, tail_months=18) -> DataFrame`
- `clean_prices(px, max_gap=5) -> DataFrame`
- `download_prices(tickers, start, end, chunk=200) -> DataFrame`
- `formation_universe(px, membership, start, end) -> list[str]` (in `universe.py`)

- [ ] **Step 1: Failing tests**

```python
import numpy as np
import pandas as pd
from hlpairs import data
from hlpairs.universe import formation_universe


def membership():
    return pd.Series(
        [frozenset({"A", "B"}), frozenset({"A", "C"}), frozenset({"A", "B"})],
        index=pd.to_datetime(["2000-01-01", "2001-01-01", "2003-01-01"]),
    )


def test_to_yahoo():
    assert data.to_yahoo("BRK.B") == "BRK-B"


def test_members_on_uses_last_snapshot_on_or_before():
    m = membership()
    assert data.members_on(m, pd.Timestamp("2000-06-30")) == {"A", "B"}
    assert data.members_on(m, pd.Timestamp("2001-01-01")) == {"A", "C"}
    assert data.members_on(m, pd.Timestamp("1999-01-01")) == frozenset()


def test_spells_and_masking_remove_reused_ticker_history():
    m = membership()
    spells = data.membership_spells(m, end=pd.Timestamp("2004-01-01"))
    assert spells["B"] == [(pd.Timestamp("2000-01-01"), pd.Timestamp("2001-01-01")),
                           (pd.Timestamp("2003-01-01"), pd.Timestamp("2004-01-01"))]
    idx = pd.bdate_range("1999-01-01", "2004-12-31")
    px = pd.DataFrame(1.0, index=idx, columns=["A", "B", "Z"])
    out = data.mask_outside_spells(px, spells, tail_months=18)
    assert np.isnan(out.loc["1999-06-01", "B"])                        # before first spell
    assert out.loc["2002-06-03", "B"] == 1.0                           # within 18m tail of spell 1
    assert np.isnan(out.loc["2002-08-01", "B"])                        # gap between spells
    assert out["Z"].isna().all()                                       # never a member


def test_clean_prices_fills_short_gaps_only_inside_history():
    idx = pd.bdate_range("2000-01-03", periods=12)
    s = pd.Series([1, 2, np.nan, 3, -1, 4, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan], index=idx, dtype=float)
    out = data.clean_prices(s.to_frame("X"), max_gap=5)["X"]
    assert out.iloc[2] == 2           # interior gap filled
    assert out.iloc[4] == 3           # non-positive -> nan -> filled from previous
    assert out.iloc[6:].isna().all()  # trailing (delisting) gap never filled


def test_formation_universe_requires_membership_and_full_history():
    idx = pd.bdate_range("2000-01-03", periods=30)
    px = pd.DataFrame(1.0, index=idx, columns=["A", "B", "C"])
    px.loc[idx[:5], "B"] = np.nan                   # B lists mid-window
    u = formation_universe(px, membership(), idx[0], idx[-1])
    assert u == ["A"]                               # C not a member on 2000-01-03, B incomplete
```

- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3: Implement `src/hlpairs/data.py`**

```python
"""Point-in-time S&P 500 membership, Yahoo prices, cleaning and caching."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

MEMBERSHIP_URL = ("https://raw.githubusercontent.com/fja05680/sp500/master/"
                  "S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv")


def to_yahoo(t: str) -> str:
    return t.strip().replace(".", "-")


def load_membership(path) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["date"])
    sets = [frozenset(to_yahoo(t) for t in s.split(",") if t.strip()) for s in df["tickers"]]
    return pd.Series(sets, index=df["date"]).sort_index()


def members_on(membership: pd.Series, date) -> frozenset:
    snap = membership.loc[:date]
    return snap.iloc[-1] if len(snap) else frozenset()


def membership_spells(membership: pd.Series, end=None) -> dict:
    """Contiguous membership intervals per ticker: [(first snapshot in, first snapshot out)].
    An open spell ends at `end` (default: last snapshot date)."""
    end = membership.index[-1] if end is None else end
    spells, open_since, prev = {}, {}, frozenset()
    for date, cur in membership.items():
        for t in cur - prev:
            open_since[t] = date
        for t in prev - cur:
            spells.setdefault(t, []).append((open_since.pop(t), date))
        prev = cur
    for t, start in open_since.items():
        spells.setdefault(t, []).append((start, end))
    return spells


def mask_outside_spells(px: pd.DataFrame, spells: dict, tail_months: int = 18) -> pd.DataFrame:
    """Keep a ticker's prices only from a spell's start to spell end + tail_months.
    A portfolio formed while the stock is a member needs at most formation + trading
    (18 months) of data after the formation start, so this never removes a usable price,
    but it does remove history Yahoo attaches to a reused ticker (a different company)."""
    keep = pd.DataFrame(False, index=px.index, columns=px.columns)
    for t in px.columns:
        for s, e in spells.get(t, []):
            keep.loc[s:e + pd.DateOffset(months=tail_months), t] = True
    return px.where(keep)


def clean_prices(px: pd.DataFrame, max_gap: int = 5) -> pd.DataFrame:
    """Non-positive prices -> nan; forward-fill interior gaps of at most `max_gap` days.
    Gaps after a stock's last price (delisting) are never filled."""
    px = px.where(px > 0)
    filled = px.ffill(limit=max_gap)
    return filled.where(px.bfill().notna())


def download_prices(tickers, start, end, chunk: int = 200) -> pd.DataFrame:
    """Daily dividend/split-adjusted closes from Yahoo Finance (a total-return proxy)."""
    import yfinance as yf

    frames = []
    for k in range(0, len(tickers), chunk):
        df = yf.download(list(tickers[k:k + chunk]), start=start, end=end,
                         auto_adjust=True, progress=False, threads=True)["Close"]
        frames.append(df)
    px = pd.concat(frames, axis=1)
    px = px.loc[:, ~px.columns.duplicated()].dropna(axis=1, how="all")
    px.index = pd.to_datetime(px.index).tz_localize(None)
    return px.sort_index()


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_manifest(paths, out="data/raw/manifest.json") -> None:
    meta = {str(p): {"sha256": sha256(p), "bytes": Path(p).stat().st_size,
                     "downloaded": pd.Timestamp.now(tz="UTC").isoformat()} for p in paths}
    Path(out).write_text(json.dumps(meta, indent=2))


def load_prices(path="data/raw/prices_clean.parquet") -> pd.DataFrame:
    return pd.read_parquet(path)
```

`src/hlpairs/universe.py`:

```python
"""Formation-period universe: point-in-time members with a complete price history."""
import pandas as pd

from .data import members_on


def formation_universe(px: pd.DataFrame, membership: pd.Series, start, end) -> list[str]:
    members = members_on(membership, start)
    cols = [c for c in px.columns if c in members]
    win = px.loc[start:end, cols]
    return win.columns[win.notna().all()].tolist()
```

- [ ] **Step 4:** Run → pass.
- [ ] **Step 5:** Commit — `"feat: point-in-time membership, spell masking, cleaning, universe"`.

---

### Task 5: Pair formation (GGR distance)

**Files:** Create `src/hlpairs/pairs.py`, `tests/test_pairs.py`

**Interfaces — Produces:** `normalize(px) -> DataFrame` (each column / its first value), `pair_frame(norm, i_idx, j_idx) -> DataFrame[i, j, ssd, sigma]`, `top_pairs(norm, n, sectors=None) -> DataFrame[i, j, ssd, sigma]` sorted by `ssd` ascending, index 0..n−1.

- [ ] **Step 1: Failing tests**

```python
import itertools

import numpy as np
import pandas as pd
from hlpairs.pairs import normalize, top_pairs


def frame(seed=0, n=8, T=120):
    rng = np.random.default_rng(seed)
    px = pd.DataFrame(np.exp(np.cumsum(rng.normal(0, 0.01, (T, n)), axis=0)),
                      columns=[f"S{k}" for k in range(n)])
    px["S7"] = px["S3"] * np.exp(rng.normal(0, 0.001, T))   # near-twin of S3
    return px


def test_normalize_starts_at_one():
    assert np.allclose(normalize(frame()).iloc[0], 1.0)


def test_top_pairs_matches_brute_force():
    norm = normalize(frame())
    brute = sorted(((((norm[a] - norm[b]) ** 2).sum(), a, b)
                    for a, b in itertools.combinations(norm.columns, 2)))[:5]
    got = top_pairs(norm, 5)
    assert list(zip(got.i, got.j)) == [(a, b) for _, a, b in brute]
    assert np.allclose(got.ssd, [d for d, _, _ in brute])
    assert (got.i.iloc[0], got.j.iloc[0]) == ("S3", "S7")
    assert np.isclose(got.sigma.iloc[0], (norm["S3"] - norm["S7"]).std(ddof=1))


def test_sector_restriction():
    norm = normalize(frame())
    sectors = ["X"] * 4 + ["Y"] * 4          # S3 (X) and S7 (Y) differ
    got = top_pairs(norm, 3, sectors=sectors)
    assert ("S3", "S7") not in set(zip(got.i, got.j))
    lookup = dict(zip(norm.columns, sectors))
    assert all(lookup[a] == lookup[b] for a, b in zip(got.i, got.j))
```

- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3: Implement**

```python
"""GGR pair formation: rank all pairs by the sum of squared differences (SSD)
between normalised cumulative-return price paths."""
import numpy as np
import pandas as pd


def normalize(px: pd.DataFrame) -> pd.DataFrame:
    """Cumulative total-return index starting at 1 (adjusted close / first adjusted close)."""
    return px / px.iloc[0]


def pair_frame(norm: pd.DataFrame, i_idx, j_idx) -> pd.DataFrame:
    X = norm.to_numpy()
    diff = X[:, i_idx] - X[:, j_idx]
    cols = norm.columns.to_numpy()
    return pd.DataFrame({"i": cols[i_idx], "j": cols[j_idx],
                         "ssd": (diff ** 2).sum(0), "sigma": diff.std(0, ddof=1)})


def top_pairs(norm: pd.DataFrame, n: int, sectors=None) -> pd.DataFrame:
    """The n lowest-SSD pairs. SSD_ij = |x_i|^2 + |x_j|^2 - 2 x_i.x_j, computed for all
    pairs at once from the Gram matrix."""
    X = norm.to_numpy()
    sq = (X ** 2).sum(0)
    D = sq[:, None] + sq[None, :] - 2 * X.T @ X
    iu, ju = np.triu_indices(X.shape[1], 1)
    d = D[iu, ju]
    if sectors is not None:
        sec = np.asarray(sectors, dtype=object)
        d = np.where(sec[iu] == sec[ju], d, np.inf)
    order = np.argsort(d, kind="stable")[:n]
    order = order[np.isfinite(d[order])]
    return pair_frame(norm, iu[order], ju[order])
```

- [ ] **Step 4:** Run → pass.
- [ ] **Step 5:** Commit — `"feat: SSD pair formation"`.

---

### Task 6: Trading simulator and portfolio accounting

**Files:** Create `src/hlpairs/backtest.py`, `tests/test_backtest.py`

**Interfaces — Consumes:** `normalize` (Task 5). **Produces:**
- `simulate_pair(x, ri, rj, k, w=1.0, wait=0, max_hold=None) -> (pnl, legs, active)` 1-D arrays length T.
- `trade_pairs(trade_px, pairs, k=2.0, wait=0) -> (pnl, legs, active)` 2-D `(T, n_pairs)` (GGR signal).
- `trade_ou(trade_px, cand, k=2.0, wait=0, hold_mult=None) -> (pnl, legs, active)`; `cand` has columns `i, j, beta, ls_mean, ls_sd, hl_log`.
- `portfolio_returns(pnl, legs, active, cost_bps=0.0, cols=None) -> (committed, employed)` 1-D arrays.
- `monthly_from_portfolios(daily: list[pd.Series]) -> pd.Series` (PeriodIndex monthly).

- [ ] **Step 1: Failing tests**

```python
import numpy as np
import pandas as pd
import pytest
from hlpairs.backtest import monthly_from_portfolios, portfolio_returns, simulate_pair

nan = np.nan


def test_hand_worked_trade():
    x = np.array([0, 2.5, 1.0, -0.1, 0])
    ri = np.array([nan, 0, 0.1, 0, 0])
    rj = np.array([nan, 0, 0, 0, 0])
    pnl, legs, active = simulate_pair(x, ri, rj, k=2)
    # open at close 1: short i, long j. Day 2 i rises 10% -> lose 0.10. Close at day 3 (crossed).
    assert np.allclose(pnl, [0, 0, -0.1, 0, 0])
    assert legs.tolist() == [0, 2, 0, 2, 0]
    assert active.tolist() == [False, False, True, True, False]


def test_wait_one_day_shifts_execution():
    x = np.array([0, 2.5, 1.0, -0.1, 0, 0])
    r0 = np.zeros(6); r0[0] = nan
    _, legs, _ = simulate_pair(x, r0, r0, k=2, wait=1)
    assert legs.tolist() == [0, 0, 2, 0, 2, 0]


def test_delisting_closes_at_last_price():
    x = np.array([0, 2.5, 1.0, nan, nan])
    ri = np.array([nan, 0, 0.1, nan, nan])
    rj = np.array([nan, 0, 0, 0, 0])
    pnl, legs, _ = simulate_pair(x, ri, rj, k=2)
    assert np.allclose(pnl, [0, 0, -0.1, 0, 0])
    assert legs.tolist() == [0, 2, 2, 0, 0]


def test_forced_close_at_end():
    x = np.array([0, 3, 3, 3])
    r = np.array([nan, 0, 0, 0])
    _, legs, _ = simulate_pair(x, r, r, k=2)
    assert legs.tolist() == [0, 2, 0, 2]


def test_max_hold():
    x = np.array([0, 3, 3, 3, 3, 3])
    r = np.array([nan, 0, 0, 0, 0, 0])
    _, legs, _ = simulate_pair(x, r, r, k=2, max_hold=2)
    assert legs.tolist() == [0, 2, 0, 2, 2, 2]      # closes after 2 days held, re-opens, forced close at end


def test_reinvested_leg_values():
    # long j (x>0 => short i), j gains 10% two days running: P&L 0.10 then 0.11
    x = np.array([0, 3, 3, 3])
    ri = np.array([nan, 0, 0, 0]); rj = np.array([nan, 0, 0.1, 0.1])
    pnl, _, _ = simulate_pair(x, ri, rj, k=2)
    assert np.allclose(pnl, [0, 0, 0.1, 0.11])


@pytest.mark.parametrize("wait", [0, 1])
def test_no_lookahead(wait):
    rng = np.random.default_rng(0)
    x = rng.normal(0, 2, 200); ri = rng.normal(0, 0.01, 200); rj = rng.normal(0, 0.01, 200)
    ri[0] = rj[0] = nan
    base = simulate_pair(x, ri, rj, k=2, wait=wait)
    for t in (50, 120):
        x2, ri2, rj2 = x.copy(), ri.copy(), rj.copy()
        x2[t + 1:] = rng.normal(0, 5, 199 - t); ri2[t + 1:] *= -3; rj2[t + 1:] *= 2
        alt = simulate_pair(x2, ri2, rj2, k=2, wait=wait)
        for a, b in zip(base, alt):
            assert np.array_equal(a[: t + 1], b[: t + 1])


def test_portfolio_returns_costs_and_bases():
    pnl = np.array([[0.0, 0.0], [0.01, 0.0], [0.02, 0.0]])
    legs = np.array([[2, 0], [0, 0], [2, 0]])
    active = np.array([[False, False], [True, False], [True, False]])
    committed, employed = portfolio_returns(pnl, legs, active, cost_bps=10)
    assert np.allclose(committed, [-0.002 / 2, 0.01 / 2, (0.02 - 0.002) / 2])
    assert np.allclose(employed, [-0.002, 0.01, 0.018])


def test_monthly_compounds_then_averages_portfolios():
    idx = pd.to_datetime(["2001-01-02", "2001-01-03", "2001-02-01"])
    a = pd.Series([0.1, 0.1, 0.0], index=idx)
    b = pd.Series([0.0, 0.0, 0.2], index=idx)
    m = monthly_from_portfolios([a, b])
    assert m.loc[pd.Period("2001-01")] == pytest.approx((0.21 + 0.0) / 2)
    assert m.loc[pd.Period("2001-02")] == pytest.approx(0.1)
```

- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3: Implement**

```python
"""Pair-trading simulation and GGR portfolio accounting.

Timing convention: x[t] is known at the close of day t. An order decided at close t
executes at close t + wait, and the position earns returns from the next day on.
P&L is measured per $1 committed to the pair and legs are buy-and-hold inside a trade
(leg values compound), as in GGR's marked-to-market return calculation.
"""
import numpy as np
import pandas as pd

from .pairs import normalize


def _decide(pos, x, k, held, max_hold):
    if pos == 0:
        return -int(np.sign(x)) if abs(x) > k else None
    if pos * x >= 0 or (max_hold is not None and held >= max_hold):
        return 0
    return None


def simulate_pair(x, ri, rj, k, w=1.0, wait=0, max_hold=None):
    """pos = +1: long $1 of i, short $w of j; pos = -1: the reverse.
    Opens when |x| > k (shorting the rich leg), closes when x crosses zero, after
    max_hold days, when a leg stops trading (at its last price), or on the last day."""
    T = len(x)
    pnl, legs, active = np.zeros(T), np.zeros(T), np.zeros(T, dtype=bool)
    pos, a, b, held, pending = 0, 0.0, 0.0, 0, None
    for t in range(T):
        if pos != 0:
            if np.isnan(ri[t]) or np.isnan(rj[t]):     # leg stopped trading: closed at t-1 close
                legs[t - 1] += 2
                pos = 0
                break
            pnl[t] = pos * (a * ri[t] - b * rj[t])
            a *= 1 + ri[t]
            b *= 1 + rj[t]
            held += 1
            active[t] = True
        if np.isnan(x[t]):
            continue
        new = None
        if pending is not None:
            if pending[0] == t:
                new, pending = pending[1], None
        else:
            new = _decide(pos, x[t], k, held, max_hold)
            if new is not None and wait:
                pending, new = (t + wait, new), None
        if new is not None:
            pos, a, b, held = new, 1.0, w, 0
            legs[t] += 2
    if pos != 0:
        legs[T - 1] += 2                                # forced close at end of trading period
    return pnl, legs, active


def _stack(results, T):
    if not results:
        return np.zeros((T, 0)), np.zeros((T, 0)), np.zeros((T, 0), dtype=bool)
    return tuple(np.column_stack(z) for z in zip(*results))


def trade_pairs(trade_px: pd.DataFrame, pairs: pd.DataFrame, k: float = 2.0, wait: int = 0):
    """GGR rule: signal = (P_i - P_j) / sigma_formation with prices re-normalised to 1 on
    the first trading day."""
    P = normalize(trade_px).to_numpy()
    R = trade_px.pct_change(fill_method=None).to_numpy()
    col = {c: n for n, c in enumerate(trade_px.columns)}
    out = [simulate_pair((P[:, col[i]] - P[:, col[j]]) / s, R[:, col[i]], R[:, col[j]], k, wait=wait)
           for i, j, s in zip(pairs.i, pairs.j, pairs.sigma)]
    return _stack(out, len(trade_px))


def trade_ou(trade_px: pd.DataFrame, cand: pd.DataFrame, k: float = 2.0, wait: int = 0, hold_mult=None):
    """Cointegration/OU rule: signal = z-score of log p_i - beta log p_j using formation
    mean and sd; hedge $1 of i against $beta of j; optional max holding = hold_mult * half-life."""
    L = np.log(trade_px).to_numpy()
    R = trade_px.pct_change(fill_method=None).to_numpy()
    col = {c: n for n, c in enumerate(trade_px.columns)}
    out = []
    for r in cand.itertuples():
        z = (L[:, col[r.i]] - r.beta * L[:, col[r.j]] - r.ls_mean) / r.ls_sd
        mh = None if hold_mult is None or not np.isfinite(r.hl_log) else max(1, int(round(hold_mult * r.hl_log)))
        out.append(simulate_pair(z, R[:, col[r.i]], R[:, col[r.j]], k, w=r.beta, wait=wait, max_hold=mh))
    return _stack(out, len(trade_px))


def portfolio_returns(pnl, legs, active, cost_bps: float = 0.0, cols=None):
    """Daily portfolio returns. Committed capital: divide by all pairs selected.
    Employed capital: divide by pairs with an open position (or trading) that day.
    Costs: cost_bps per leg traded, per $1 notional."""
    if cols is not None:
        pnl, legs, active = pnl[:, cols], legs[:, cols], active[:, cols]
    net = (pnl - legs * cost_bps * 1e-4).sum(1)
    n = pnl.shape[1]
    committed = net / n if n else np.zeros(len(net))
    busy = (active | (legs > 0)).sum(1)
    employed = np.divide(net, busy, out=np.zeros(len(net)), where=busy > 0)
    return committed, employed


def monthly_from_portfolios(daily: list) -> pd.Series:
    """Compound each portfolio's daily returns within calendar months, then average the
    (up to six) overlapping portfolios active in each month, as in GGR."""
    m = pd.concat([(1 + s).groupby(s.index.to_period("M")).prod() - 1 for s in daily], axis=1)
    return m.mean(axis=1).sort_index()
```

- [ ] **Step 4:** Run → pass. (If `test_no_lookahead` fails, the simulator is reading future data — fix the simulator, never the test.)
- [ ] **Step 5:** Commit — `"feat: pair simulator, GGR/OU trading, portfolio accounting"`.

---

### Task 7: Evaluation statistics

**Files:** Create `src/hlpairs/evaluation.py`, `tests/test_evaluation.py`

**Interfaces — Produces:** `nw_mean_test(x, lags=6) -> (mean, se, t, p)`, `summary(monthly) -> dict`, `drawdown(monthly) -> Series`, `quintile_sort(panel, col, ret="ret", by="trade_start", q=5) -> DataFrame`, `fama_macbeth(panel, y, xs, by="trade_start", lags=6) -> (table, slopes)`, `adjust_pvalues(p) -> DataFrame[p, holm, bh]`, `deflated_sharpe(sr, n_trials, sr_var, T, skew=0, kurt=3) -> float`.

- [ ] **Step 1: Failing tests**

```python
import numpy as np
import pandas as pd
import pytest
from hlpairs import evaluation as ev


def test_nw_mean_test():
    x = 0.01 + np.random.default_rng(0).normal(0, 0.01, 600)
    m, se, t, p = ev.nw_mean_test(x)
    assert m == pytest.approx(x.mean())
    assert t > 10 and p < 1e-6


def test_drawdown_and_summary():
    r = pd.Series([0.1, -0.5, 0.2])
    assert ev.drawdown(r).min() == pytest.approx(-0.5)
    s = ev.summary(r)
    assert s["max_dd"] == pytest.approx(-0.5) and s["n"] == 3


def make_panel(slope, seed=0, F=60, N=100):
    rng = np.random.default_rng(seed)
    rows = []
    for f in range(F):
        h = rng.lognormal(3, 1, N)
        h[0] = np.inf
        rows.append(pd.DataFrame({"trade_start": f, "hl": h,
                                  "ret": slope * np.log(np.minimum(h, 500)) + rng.normal(0, 0.05, N)}))
    return pd.concat(rows, ignore_index=True)


def test_quintile_sort_monotone_and_inf_is_slowest():
    tab = ev.quintile_sort(make_panel(-0.02), "hl")
    assert list(tab.columns) == [1, 2, 3, 4, 5]
    means = tab.mean()
    assert means.is_monotonic_decreasing


def test_fama_macbeth_recovers_slope():
    p = make_panel(-0.02)
    p["log_hl"] = np.log(np.minimum(p.hl, 500))
    table, slopes = ev.fama_macbeth(p, "ret", ["log_hl"])
    assert table.loc["log_hl", "coef"] == pytest.approx(-0.02, abs=0.003)
    assert table.loc["log_hl", "t"] < -5
    assert len(slopes) == 60


def test_adjust_pvalues():
    out = ev.adjust_pvalues(pd.Series([0.01, 0.04, 0.03], index=list("abc")))
    assert (out["holm"] >= out["p"]).all() and (out["bh"] >= out["p"]).all()


def test_deflated_sharpe_falls_with_trials():
    one = ev.deflated_sharpe(0.3, 1, 0.01, 120)
    many = ev.deflated_sharpe(0.3, 50, 0.01, 120)
    assert 0 < many < one < 1
```

- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3: Implement**

```python
"""Performance statistics and cross-sectional tests."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import kurtosis, norm, skew
from statsmodels.stats.multitest import multipletests


def nw_mean_test(x, lags: int = 6):
    """Mean with Newey-West (HAC) standard error; lags cover the 6-month portfolio overlap."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    res = sm.OLS(x, np.ones(len(x))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(res.params[0]), float(res.bse[0]), float(res.tvalues[0]), float(res.pvalues[0])


def drawdown(monthly: pd.Series) -> pd.Series:
    wealth = (1 + monthly).cumprod()
    return wealth / wealth.cummax().clip(lower=1.0) - 1


def summary(monthly: pd.Series, lags: int = 6) -> dict:
    m, se, t, p = nw_mean_test(monthly, lags)
    sd = monthly.std(ddof=1)
    return {"mean_monthly": m, "ann_mean": 12 * m, "t_nw": t, "p": p, "sd_monthly": sd,
            "sharpe_ann": np.sqrt(12) * m / sd if sd > 0 else np.nan,
            "max_dd": float(drawdown(monthly).min()), "skew": float(skew(monthly)),
            "kurt": float(kurtosis(monthly, fisher=False)), "frac_pos": float((monthly > 0).mean()),
            "n": int(len(monthly))}


def quintile_sort(panel: pd.DataFrame, col: str, ret: str = "ret", by: str = "trade_start", q: int = 5):
    """Within each formation, bucket pairs by `col` (1 = smallest); mean `ret` per bucket.
    inf ranks as the largest value (slowest reversion); nan is dropped."""
    rows = {}
    for key, g in panel.dropna(subset=[col]).groupby(by):
        if len(g) < q:
            continue
        b = pd.qcut(g[col].rank(method="first"), q, labels=range(1, q + 1))
        rows[key] = g.groupby(b, observed=True)[ret].mean()
    return pd.DataFrame(rows).T


def fama_macbeth(panel: pd.DataFrame, y: str, xs: list, by: str = "trade_start", lags: int = 6):
    """One OLS per formation; report the time-series mean slope with NW standard errors."""
    rows = {}
    for key, g in panel.groupby(by):
        g = g[[y, *xs]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(g) <= len(xs) + 1:
            continue
        X = np.column_stack([np.ones(len(g)), g[xs].to_numpy()])
        rows[key] = np.linalg.lstsq(X, g[y].to_numpy(), rcond=None)[0]
    slopes = pd.DataFrame(rows, index=["const", *xs]).T
    table = pd.DataFrame([nw_mean_test(slopes[c], lags) for c in slopes],
                         index=slopes.columns, columns=["coef", "se", "t", "p"])
    return table, slopes


def adjust_pvalues(p: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"p": p, "holm": multipletests(p, method="holm")[1],
                         "bh": multipletests(p, method="fdr_bh")[1]}, index=p.index)


def deflated_sharpe(sr: float, n_trials: int, sr_var: float, T: int, skew: float = 0.0, kurt: float = 3.0) -> float:
    """Bailey & Lopez de Prado (2014): probability the true Sharpe exceeds the expected
    maximum Sharpe of n_trials unskilled strategies. Sharpe in per-period (monthly) units."""
    g = 0.5772156649
    sr0 = 0.0 if n_trials <= 1 else np.sqrt(sr_var) * (
        (1 - g) * norm.ppf(1 - 1 / n_trials) + g * norm.ppf(1 - 1 / (n_trials * np.e)))
    return float(norm.cdf((sr - sr0) * np.sqrt(T - 1) / np.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr ** 2)))
```

- [ ] **Step 4:** Run → pass.
- [ ] **Step 5:** Commit — `"feat: evaluation statistics (NW, FM, sorts, multiple testing, DSR)"`.

---

### Task 8: Rolling engine

**Files:** Create `src/hlpairs/engine.py`, `tests/test_engine.py`

**Interfaces — Consumes:** Tasks 2–7. **Produces:**
- `Config` dataclass (fields below).
- `FormationResult(trade_start, dates, pairs, pnl, legs, active, extra)`; `extra: dict[str, tuple(pnl, legs, active)]` for `"random"`, `"sector"`, `"ou_m{m}"`.
- `trade_start_dates(index, form_months, trade_months) -> list[Timestamp]`
- `run_formation(px, membership, t0, cfg, sectors=None) -> FormationResult | None`
- `run_all(px, membership, cfg, sectors=None, workers=None) -> list[FormationResult]`
- `strategy_monthly(results, select=None, source=None, cost_bps=0.0, basis="committed") -> pd.Series`
- `pair_panel(results) -> DataFrame`; `split(results, cfg_sample) -> dict[str, list]` with keys `dev`, `holdout`, `full`.

Pair-panel columns: `trade_start, rank, i, j, ssd, sigma, ret, legs, n_trips, realized_hl, hl_ols_63, hl_ols_126, hl_ols_252, hl_kendall, hl_ou, cross, hl_log, beta, ls_mean, ls_sd, adf_p, kpss_p, eg_p, joh, vr, same_sector`.

- [ ] **Step 1: Failing tests**

```python
import numpy as np
import pandas as pd
from hlpairs.engine import Config, run_formation, strategy_monthly, trade_start_dates


def synthetic(seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2000-01-03", "2002-12-31")
    T = len(idx)
    px = pd.DataFrame(np.exp(np.cumsum(rng.normal(0, 0.015, (T, 6)), axis=0)),
                      index=idx, columns=list("ABCDEF"))
    s = np.zeros(T)
    for t in range(1, T):
        s[t] = 0.9 * s[t - 1] + rng.normal(0, 0.01)
    px["B"] = px["A"] * np.exp(s)                      # B mean-reverts around A
    membership = pd.Series([frozenset(px.columns)], index=[idx[0]])
    return px, membership


CFG = Config(n_candidates=10, n_random=3, ou_n=3, ou_hold_mults=(2,))


def test_trade_start_dates_respect_windows():
    px, _ = synthetic()
    d = trade_start_dates(px.index, 12, 6)
    assert d[0] == pd.Timestamp("2001-01-01")
    assert d[-1] <= pd.Timestamp("2002-07-01")


def test_run_formation_finds_the_cointegrated_pair():
    px, m = synthetic()
    r = run_formation(px, m, pd.Timestamp("2001-07-02"), CFG)
    assert {r.pairs.i.iloc[0], r.pairs.j.iloc[0]} == {"A", "B"}
    assert r.pnl.shape == (len(r.dates), 10)
    assert r.pairs.hl_ols_252.iloc[0] < 20
    assert set(r.extra) == {"random", "ou_m2"}


def test_formation_features_do_not_use_trading_data():
    px, m = synthetic()
    t0 = pd.Timestamp("2001-07-02")
    a = run_formation(px, m, t0, CFG)
    px2 = px.copy()
    px2.loc[t0:] *= np.exp(np.random.default_rng(9).normal(0, 0.2, px2.loc[t0:].shape))
    b = run_formation(px2, m, t0, CFG)
    outcome = ["ret", "legs", "n_trips", "realized_hl"]
    pd.testing.assert_frame_equal(a.pairs.drop(columns=outcome), b.pairs.drop(columns=outcome))


def test_too_few_stocks_returns_none():
    px, m = synthetic()
    px.iloc[:, 1:] = np.nan
    assert run_formation(px, m, pd.Timestamp("2001-07-02"), CFG) is None


def test_strategy_monthly_runs():
    px, m = synthetic()
    res = [run_formation(px, m, t, CFG) for t in trade_start_dates(px.index, 12, 6)[:3]]
    s = strategy_monthly(res, select=lambda p: np.arange(5))
    assert s.index[0] == pd.Period("2001-01") and s.notna().all()
```

- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3: Implement**

```python
"""Rolling GGR timeline: each month, form pairs on the past 12 months and trade the next 6."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import diagnostics as dg
from . import halflife as hl
from .backtest import monthly_from_portfolios, portfolio_returns, trade_ou, trade_pairs
from .pairs import normalize, pair_frame, top_pairs
from .universe import formation_universe


@dataclass
class Config:
    form_months: int = 12
    trade_months: int = 6
    n_candidates: int = 100        # SSD-ranked pairs kept per formation (cross-section for H2)
    k: float = 2.0                 # entry threshold in formation sigmas
    wait: int = 0                  # 1 = GGR wait-one-day execution
    features: bool = True          # half-life / diagnostics (slow); off for robustness reruns
    lookbacks: tuple = (63, 126, 252)
    n_random: int = 20
    n_sector: int = 20
    ou_n: int = 20
    ou_hold_mults: tuple = (1, 2, 3)
    seed: int = 0


@dataclass
class FormationResult:
    trade_start: pd.Timestamp
    dates: pd.DatetimeIndex
    pairs: pd.DataFrame
    pnl: np.ndarray
    legs: np.ndarray
    active: np.ndarray
    extra: dict = field(default_factory=dict)


def trade_start_dates(index: pd.DatetimeIndex, form_months: int, trade_months: int) -> list:
    """First trading day of each month with a full formation and trading window in the data."""
    firsts = pd.Series(index, index=index).groupby(index.to_period("M")).first()
    lo = index[0].to_period("M").start_time + pd.DateOffset(months=form_months)
    hi = index[-1] + pd.Timedelta(days=1) - pd.DateOffset(months=trade_months)
    return [d for d in firsts if lo <= d <= hi]


def pair_features(form_norm: pd.DataFrame, form_px: pd.DataFrame, pairs: pd.DataFrame, lookbacks) -> pd.DataFrame:
    """Everything here uses the formation window only."""
    logp = np.log(form_px)
    rows = []
    for i, j in zip(pairs.i, pairs.j):
        s = (form_norm[i] - form_norm[j]).to_numpy()
        y, x = logp[i].to_numpy(), logp[j].to_numpy()
        beta = np.polyfit(x, y, 1)[0]
        ls = y - beta * x
        r = {f"hl_ols_{L}": hl.halflife_ols(s[-L:]) for L in lookbacks}
        r |= {"hl_kendall": hl.halflife_kendall(s), "hl_ou": hl.halflife_ou_mle(s),
              "cross": hl.crossing_interval(s), "hl_log": hl.halflife_ols(ls), "beta": beta,
              "ls_mean": ls.mean(), "ls_sd": ls.std(ddof=1), "adf_p": dg.adf_pvalue(s),
              "kpss_p": dg.kpss_pvalue(s), "eg_p": dg.eg_pvalue(y, x),
              "joh": dg.johansen_reject(y, x), "vr": dg.variance_ratio(s)}
        rows.append(r)
    return pd.DataFrame(rows, index=pairs.index)


def run_formation(px, membership, t0, cfg: Config, sectors: dict | None = None):
    f0 = t0 - pd.DateOffset(months=cfg.form_months)
    t1 = t0 + pd.DateOffset(months=cfg.trade_months)
    form_idx = px.index[(px.index >= f0) & (px.index < t0)]
    universe = formation_universe(px, membership, form_idx[0], form_idx[-1])
    if len(universe) < 2:
        return None
    form_px = px.loc[form_idx, universe]
    form = normalize(form_px)
    trade = px.loc[(px.index >= t0) & (px.index < t1), universe]

    pairs = top_pairs(form, cfg.n_candidates)
    pnl, legs, active = trade_pairs(trade, pairs, cfg.k, cfg.wait)
    pairs["trade_start"], pairs["rank"] = t0, np.arange(len(pairs))
    if sectors is not None:
        pairs["same_sector"] = [sectors.get(i) is not None and sectors.get(i) == sectors.get(j)
                                for i, j in zip(pairs.i, pairs.j)]
    extra = {}

    rng = np.random.default_rng([cfg.seed, int(t0.strftime("%Y%m"))])
    n = len(universe)
    flat = rng.choice(n * (n - 1) // 2, size=min(cfg.n_random, n * (n - 1) // 2), replace=False)
    iu, ju = np.triu_indices(n, 1)
    extra["random"] = trade_pairs(trade, pair_frame(form, iu[flat], ju[flat]), cfg.k, cfg.wait)

    if sectors is not None:
        sec = [sectors.get(t, f"_none_{t}") for t in universe]   # unknown sector never pairs
        extra["sector"] = trade_pairs(trade, top_pairs(form, cfg.n_sector, sectors=sec), cfg.k, cfg.wait)

    if cfg.features:
        pairs = pairs.join(pair_features(form, form_px, pairs, cfg.lookbacks))
        cand = pairs[pairs.beta > 0].nsmallest(cfg.ou_n, "eg_p")
        for m in cfg.ou_hold_mults:
            extra[f"ou_m{m}"] = trade_ou(trade, cand, cfg.k, cfg.wait, hold_mult=m)

    pairs["ret"] = pnl.sum(0)
    pairs["legs"] = legs.sum(0)
    pairs["n_trips"] = legs.sum(0) / 4
    tn = normalize(trade)
    pairs["realized_hl"] = [hl.halflife_ols((tn[i] - tn[j]).to_numpy()) for i, j in zip(pairs.i, pairs.j)]
    return FormationResult(t0, trade.index, pairs, pnl, legs, active, extra)


_G = {}


def _init(px, membership, cfg, sectors):
    _G.update(px=px, membership=membership, cfg=cfg, sectors=sectors)


def _run_one(t0):
    return run_formation(_G["px"], _G["membership"], t0, _G["cfg"], _G["sectors"])


def run_all(px, membership, cfg: Config, sectors=None, workers=None) -> list:
    starts = trade_start_dates(px.index, cfg.form_months, cfg.trade_months)
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(px, membership, cfg, sectors)) as ex:
        return [r for r in ex.map(_run_one, starts) if r is not None]


def strategy_monthly(results, select=None, source=None, cost_bps=0.0, basis="committed") -> pd.Series:
    """Monthly GGR-style return series. `select(pairs) -> column positions` picks from the
    SSD candidates; `source` names a precomputed variant in `extra` (all its pairs)."""
    daily = []
    for r in results:
        if source is not None:
            pnl, legs, active = r.extra[source]
            cols = None
        else:
            pnl, legs, active = r.pnl, r.legs, r.active
            cols = select(r.pairs)
        committed, employed = portfolio_returns(pnl, legs, active, cost_bps, cols)
        daily.append(pd.Series(committed if basis == "committed" else employed, index=r.dates))
    return monthly_from_portfolios(daily)


def top_n(n):
    return lambda p: np.arange(min(n, len(p)))


def pair_panel(results) -> pd.DataFrame:
    return pd.concat([r.pairs for r in results], ignore_index=True)


def split(results, dev_last: str, holdout_first: str) -> dict:
    """Dev / holdout by portfolio trade start; starts in between are an embargo gap."""
    dev = [r for r in results if r.trade_start <= pd.Timestamp(dev_last)]
    hold = [r for r in results if r.trade_start >= pd.Timestamp(holdout_first)]
    return {"dev": dev, "holdout": hold, "full": list(results)}
```

- [ ] **Step 4:** Run `uv run pytest -q` → all tests pass.
- [ ] **Step 5:** Commit — `"feat: rolling formation/trading engine"`.

---

### Task 9: Data acquisition and coverage report (experiment 01)

**Files:** Create `experiments/config.yaml`, `experiments/01_data.py`, `src/hlpairs/plots.py`, `data/README.md`

- [ ] **Step 1: `experiments/config.yaml`**

```yaml
data: {start: "1999-01-01", end: "2025-12-31"}
sample: {dev_last_trade_start: "2012-07-01", holdout_first_trade_start: "2013-01-01"}
engine: {form_months: 12, trade_months: 6, n_candidates: 100, k: 2.0, wait: 0, n_random: 20, n_sector: 20, ou_n: 20, ou_hold_mults: [1, 2, 3], seed: 0}
costs_bps: [0, 5, 10, 20, 30]
headline_cost_bps: 10
```

- [ ] **Step 2: `src/hlpairs/plots.py`** (invoke the `dataviz` skill first and adopt its palette values in `PALETTE`):

```python
"""Shared figure style so every chart in the paper reads as one system."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PALETTE = ["#2E5EAA", "#E07A1F", "#3A9E6F", "#B8406A", "#7A5CC2", "#8A8A8A"]
FIG_DIR = Path("figures")


def style():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 300, "font.size": 9, "axes.titlesize": 10,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.alpha": 0.25, "axes.prop_cycle": matplotlib.cycler(color=PALETTE),
        "legend.frameon": False, "figure.constrained_layout.use": True,
    })


def save(fig, name: str):
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def save_table(df, name: str, floatfmt: str = "%.3f", **kw):
    Path("tables").mkdir(exist_ok=True)
    df.to_csv(f"tables/{name}.csv")
    df.to_latex(f"tables/{name}.tex", float_format=floatfmt, **kw)
```

- [ ] **Step 3: `experiments/01_data.py`**

```python
"""Download membership, prices, market series and sectors; clean; write coverage report."""
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import yaml

from hlpairs import data, plots

cfg = yaml.safe_load(open("experiments/config.yaml"))
RAW = Path("data/raw"); RAW.mkdir(parents=True, exist_ok=True)
start, end = cfg["data"]["start"], cfg["data"]["end"]

mem_path = RAW / "sp500_membership.csv"
mem_path.write_bytes(requests.get(data.MEMBERSHIP_URL, timeout=60).content)
membership = data.load_membership(mem_path)
membership = membership[membership.index <= end]
ever = sorted(set().union(*membership[membership.index >= start]))
print(f"{len(ever)} tickers were S&P 500 members at some point {start}..{end}")

raw = data.download_prices(ever + ["SPY", "^VIX"], start, pd.Timestamp(end) + pd.Timedelta(days=1))
raw.to_parquet(RAW / "prices_raw.parquet")
market = raw[["SPY", "^VIX"]].rename(columns={"^VIX": "VIX"})
market.to_parquet(RAW / "market.parquet")

stocks = raw.drop(columns=["SPY", "^VIX"])
spells = data.membership_spells(membership, end=pd.Timestamp(end))
clean = data.clean_prices(data.mask_outside_spells(stocks, spells))
clean = clean.dropna(axis=1, how="all")
clean.to_parquet(RAW / "prices_clean.parquet")

usrec = pd.read_csv("https://fred.stlouisfed.org/graph/fredgraph.csv?id=USREC", parse_dates=[0], index_col=0)
usrec.to_csv(RAW / "usrec.csv")

html = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", timeout=60,
                    headers={"User-Agent": "hlpairs-research/1.0"}).text
wiki = pd.read_html(StringIO(html))[0]
sectors = pd.Series(wiki["GICS Sector"].to_numpy(), index=wiki["Symbol"].map(data.to_yahoo), name="sector")
sectors.to_csv(RAW / "sectors_current.csv")

data.write_manifest(sorted(RAW.glob("*.parquet")) + [mem_path, RAW / "usrec.csv", RAW / "sectors_current.csv"])

# ---- coverage / survivorship report -------------------------------------------------
years = range(int(start[:4]), int(end[:4]) + 1)
rows = []
for y in years:
    d0, d1 = pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y}-12-31")
    mem = data.members_on(membership, d0)
    cols = [t for t in mem if t in clean.columns]
    win = clean.loc[d0:d1, cols]
    rows.append({"year": y, "members": len(mem), "any_data": int(win.notna().any().sum()),
                 "full_year": int(win.notna().all().sum()),
                 "sector_known": int(sum(t in sectors.index for t in mem))})
cov = pd.DataFrame(rows).set_index("year")
cov["pct_full_year"] = cov.full_year / cov.members
plots.save_table(cov, "data_coverage", floatfmt="%.2f")
rets = clean.pct_change(fill_method=None)
print("daily |r| > 50% observations:", int((rets.abs() > 0.5).sum().sum()))

plots.style()
fig, ax = plt.subplots(figsize=(6.5, 3))
ax.plot(cov.index, cov.members, label="index members (1 Jan)", color=plots.PALETTE[5])
ax.plot(cov.index, cov.full_year, label="with complete Yahoo prices that year")
ax.plot(cov.index, cov.sector_known, label="with a (current) GICS sector", linestyle="--")
ax.set(ylabel="stocks", title="Point-in-time coverage: survivorship gap shrinks toward the present")
ax.legend(loc="lower right")
plots.save(fig, "fig01_coverage")
```

- [ ] **Step 4:** Run `uv run python experiments/01_data.py`. Expected: prints ticker count (~1,100) and writes `data/raw/*.parquet`, `tables/data_coverage.*`, `figures/fig01_coverage.*`. Inspect `tables/data_coverage.csv`: record the actual `pct_full_year` numbers; if any year < 0.6, note it in `data/README.md`. Look at the figure.
- [ ] **Step 5: `data/README.md`**: sources with URLs and access date; the spell-masking rule and why (ticker reuse example: `AAL` in 1996 ≠ American Airlines Group); cleaning rules; coverage table numbers from Step 4; known biases (missing delisted names → residual survivorship bias toward survivors; adjusted close ≈ total return; GICS not point-in-time; Yahoo data errors). State that `data/raw/` is git-ignored and regenerated by `01_data.py`; the manifest records SHA-256s of the files used for the paper.
- [ ] **Step 6:** Commit `experiments/config.yaml experiments/01_data.py src/hlpairs/plots.py data/README.md data/raw/manifest.json tables figures` — `"data: acquisition, cleaning and coverage report"`. (Un-ignore the manifest: add `!data/raw/manifest.json` to `.gitignore`.)

---

### Task 10: Main engine run (experiment 02)

**Files:** Create `experiments/02_main_run.py`

- [ ] **Step 1: Write**

```python
"""Run the rolling engine once on the full sample; cache results and the pair panel."""
import pickle
import time
from pathlib import Path

import pandas as pd
import yaml

from hlpairs import data
from hlpairs.engine import Config, pair_panel, run_all

cfg = yaml.safe_load(open("experiments/config.yaml"))
px = data.load_prices()
membership = data.load_membership("data/raw/sp500_membership.csv")
sectors = pd.read_csv("data/raw/sectors_current.csv", index_col=0)["sector"].to_dict()
ec = cfg["engine"] | {"ou_hold_mults": tuple(cfg["engine"]["ou_hold_mults"])}
t = time.time()
results = run_all(px, membership, Config(**ec), sectors=sectors)
print(f"{len(results)} formations in {time.time() - t:.0f}s")
Path("results/cache").mkdir(parents=True, exist_ok=True)
pickle.dump(results, open("results/cache/main.pkl", "wb"))
panel = pair_panel(results)
panel.to_parquet("results/pair_panel.parquet")
print(panel.describe().T.to_string())
```

- [ ] **Step 2:** Run `uv run python experiments/02_main_run.py` (background; expect ~300 formations). Sanity checks before moving on: number of formations ≈ 300; median universe ≥ 350 stocks (print `len` of universe by adding a column if needed); `ret` has no NaN; `hl_ols_252` median between 5 and 100 days; top-ranked pairs are visibly same-industry names (print 10 random top-1 pairs). Any failure → `superpowers:systematic-debugging`, not parameter tweaking.
- [ ] **Step 3:** Commit `experiments/02_main_run.py results/pair_panel.parquet` — `"exp: main engine run and pair panel"`.

---

### Task 11: GGR replication (experiment 03) — H1

**Files:** Create `experiments/03_replication.py`

- [ ] **Step 1: Write** — loads `results/cache/main.pkl` and `data/raw/market.parquet`; produces:
  - `tables/tab_replication`: rows = Top 5, Top 20, Sector top 20, Random 20, SPY excess-free benchmark; columns = mean monthly, NW t, Sharpe, max DD, % positive months, avg round trips per pair; for {committed, employed} × {wait 0 (main), wait 1 (rerun with `Config(features=False, wait=1)`)} × {full, dev, holdout}.
  - `fig02_example_pair`: for the top-1 pair of one dev formation (first formation with a pair that trades ≥ 2 round trips): panel A raw prices; B normalized prices across formation+trading with the trade-start line; C spread with ±2σ bands, open/close markers and shaded open episodes.
  - `fig03_equity_curves`: cumulative wealth (log scale) of Top 5, Top 20, Sector, Random, SPY; holdout region shaded; embargo marked.
  - `fig04_drawdowns`: drawdown series for Top 20 vs SPY.
  - `fig05_rolling_performance`: 36-month rolling mean monthly return of Top 20 (H1 decay).

```python
"""H1: does the GGR distance rule still earn excess returns on point-in-time S&P 500 data?"""
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from hlpairs import data, plots
from hlpairs.engine import Config, run_all, split, strategy_monthly, top_n
from hlpairs.evaluation import drawdown, summary

cfg = yaml.safe_load(open("experiments/config.yaml"))
S = cfg["sample"]
res = pickle.load(open("results/cache/main.pkl", "rb"))
px = data.load_prices()
membership = data.load_membership("data/raw/sp500_membership.csv")
ec = cfg["engine"] | {"ou_hold_mults": tuple(cfg["engine"]["ou_hold_mults"]), "features": False, "wait": 1}
res_wait = run_all(px, membership, Config(**ec))
pickle.dump(res_wait, open("results/cache/wait1.pkl", "wb"))
spy = data.load_prices("data/raw/market.parquet")["SPY"]
spy_m = spy.resample("ME").last().pct_change().dropna()
spy_m.index = spy_m.index.to_period("M")

STRATS = {"Top 5": dict(select=top_n(5)), "Top 20": dict(select=top_n(20)),
          "Sector top 20": dict(source="sector"), "Random 20": dict(source="random")}

rows = []
for wait, rs in (("0", res), ("1", res_wait)):
    parts = split(rs, S["dev_last_trade_start"], S["holdout_first_trade_start"])
    for period, sub in parts.items():
        for basis in ("committed", "employed"):
            for name, kw in STRATS.items():
                m = strategy_monthly(sub, basis=basis, **kw)
                rows.append({"wait": wait, "period": period, "basis": basis, "strategy": name, **summary(m)})
        rng = spy_m.loc[strategy_monthly(sub, select=top_n(20)).index.intersection(spy_m.index)]
        rows.append({"wait": "-", "period": period, "basis": "-", "strategy": "SPY buy-and-hold", **summary(rng)})
tab = pd.DataFrame(rows).drop_duplicates(subset=["wait", "period", "basis", "strategy"])
tab.to_csv("results/replication_all.csv", index=False)
main = tab[(tab.basis.isin(["committed", "-"])) & (tab.wait.isin(["0", "-"]))]
plots.save_table(main.set_index(["period", "strategy"])[["mean_monthly", "t_nw", "sharpe_ann", "max_dd", "frac_pos", "n"]],
                 "tab_replication", floatfmt="%.4f")
plots.save_table(tab[tab.strategy == "Top 20"].set_index(["period", "wait", "basis"])[["mean_monthly", "t_nw", "sharpe_ann"]],
                 "tab_replication_wait_basis", floatfmt="%.4f")
trips = pd.concat([r.pairs.head(20).n_trips for r in res]).mean()
print("avg round trips per top-20 pair per 6m:", round(trips, 2))

plots.style()
# --- fig02 example pair ---
r = next(r for r in res if r.trade_start <= pd.Timestamp(S["dev_last_trade_start"]) and r.pairs.n_trips.iloc[0] >= 2)
i, j, sig = r.pairs.i.iloc[0], r.pairs.j.iloc[0], r.pairs.sigma.iloc[0]
f0 = r.trade_start - pd.DateOffset(months=12)
win = px.loc[f0:r.dates[-1], [i, j]]
fig, axes = plt.subplots(3, 1, figsize=(6.5, 7), sharex=True)
axes[0].plot(win.index, win[i], label=i); axes[0].plot(win.index, win[j], label=j)
axes[0].set(ylabel="adjusted price ($)", title=f"Pair {i}/{j}: formation then trading"); axes[0].legend()
form_n = win.loc[:r.trade_start - pd.Timedelta(days=1)] / win.iloc[0]
trade_n = win.loc[r.dates] / win.loc[r.dates[0]]
for ax_, part in ((axes[1], form_n), (axes[1], trade_n)):
    ax_.plot(part.index, part[i], color=plots.PALETTE[0]); ax_.plot(part.index, part[j], color=plots.PALETTE[1])
axes[1].set(ylabel="normalised price", title="Normalised (re-based to 1 at trade start)")
spread = (trade_n[i] - trade_n[j]) / sig
axes[2].plot(spread.index, spread, color=plots.PALETTE[4])
axes[2].axhline(2, ls="--", c="k", lw=0.8); axes[2].axhline(-2, ls="--", c="k", lw=0.8); axes[2].axhline(0, c="k", lw=0.5)
act = pd.Series(r.active[:, 0], index=r.dates)
axes[2].fill_between(act.index, -4, 4, where=act, color=plots.PALETTE[2], alpha=0.15, label="position open")
axes[2].set(ylabel="spread / formation σ", ylim=(-4, 4), title="Trading signal and mean-reversion episodes"); axes[2].legend()
for ax_ in axes:
    ax_.axvline(r.trade_start, c="k", lw=0.8)
plots.save(fig, "fig02_example_pair")

# --- fig03 equity, fig04 drawdowns, fig05 rolling ---
series = {n: strategy_monthly(res, **kw) for n, kw in STRATS.items()}
series["SPY"] = spy_m.loc[series["Top 20"].index.intersection(spy_m.index)]
fig, ax = plt.subplots(figsize=(6.5, 3.5))
for n, s in series.items():
    ax.plot(s.index.to_timestamp(), (1 + s).cumprod(), label=n)
ax.set_yscale("log"); ax.axvspan(pd.Timestamp(S["holdout_first_trade_start"]), series["Top 20"].index[-1].to_timestamp(), color="grey", alpha=0.1, label="holdout")
ax.set(ylabel="growth of $1 (log)", title="GGR pairs portfolios vs baselines (committed capital, gross)"); ax.legend(ncol=2)
plots.save(fig, "fig03_equity_curves")

fig, ax = plt.subplots(figsize=(6.5, 2.8))
for n in ("Top 20", "SPY"):
    d = drawdown(series[n]); ax.fill_between(d.index.to_timestamp(), d, 0, alpha=0.4, label=n)
ax.set(ylabel="drawdown", title="Drawdowns: pairs portfolio vs equity market"); ax.legend()
plots.save(fig, "fig04_drawdowns")

fig, ax = plt.subplots(figsize=(6.5, 2.8))
roll = series["Top 20"].rolling(36).mean() * 100
ax.plot(roll.index.to_timestamp(), roll); ax.axhline(0, c="k", lw=0.6)
ax.set(ylabel="% per month", title="Top-20 pairs: 36-month rolling mean monthly return")
plots.save(fig, "fig05_rolling_performance")
```

- [ ] **Step 2:** Run; read `tables/tab_replication.csv`; look at all four figures. Record H1 verdict (sign, magnitude vs GGR's top-20 ≈ 0.9%/month reported in the paper notes, t-stat).
- [ ] **Step 3:** Commit `experiments/03_replication.py tables figures results/replication_all.csv` — `"exp: GGR replication (H1)"`.

---

### Task 12: Half-life characterisation (experiment 04) — H3 + diagnostics

**Files:** Create `experiments/04_halflife.py`

- [ ] **Step 1: Write** — uses **dev formations only** for anything descriptive that feeds later choices; persistence (H3) reported on dev and, after freeze, re-reported in 07.

```python
"""Half-life distributions, estimator agreement, diagnostics and persistence (H3) on the dev sample."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

from hlpairs import halflife as hl, plots
from hlpairs import data
from hlpairs.pairs import normalize

cfg = yaml.safe_load(open("experiments/config.yaml"))
panel = pd.read_parquet("results/pair_panel.parquet")
dev = panel[panel.trade_start <= cfg["sample"]["dev_last_trade_start"]]
HL = ["hl_ols_63", "hl_ols_126", "hl_ols_252", "hl_kendall", "hl_ou", "cross", "hl_log"]
plots.style()

desc = pd.DataFrame({c: {"median": dev[c].replace(np.inf, np.nan).median(),
                         "p10": dev[c].replace(np.inf, np.nan).quantile(.1),
                         "p90": dev[c].replace(np.inf, np.nan).quantile(.9),
                         "share_inf": np.isinf(dev[c]).mean(), "share_nan": dev[c].isna().mean()} for c in HL}).T
plots.save_table(desc, "tab_halflife_describe", floatfmt="%.2f")

fig, ax = plt.subplots(figsize=(6.5, 3.2))
bins = np.logspace(0, 3, 50)
for c in ["hl_ols_252", "hl_kendall", "hl_ou", "hl_ols_63"]:
    ax.hist(dev[c].clip(upper=1000), bins=bins, histtype="step", lw=1.4, label=c)
ax.set(xscale="log", xlabel="half-life (trading days, capped at 1000)", ylabel="pairs",
       title="Formation half-life distributions (dev sample, top-100 SSD pairs)"); ax.legend()
plots.save(fig, "fig06_halflife_distributions")

rk = dev[HL].rank().corr(method="pearson")
fig, ax = plt.subplots(figsize=(4.8, 4))
im = ax.imshow(rk, vmin=-1, vmax=1, cmap="RdBu_r")
ax.set_xticks(range(len(HL)), HL, rotation=45, ha="right"); ax.set_yticks(range(len(HL)), HL)
for a in range(len(HL)):
    for b in range(len(HL)):
        ax.text(b, a, f"{rk.iloc[a, b]:.2f}", ha="center", va="center", fontsize=7)
fig.colorbar(im, shrink=0.8); ax.set_title("Rank correlation between half-life estimators"); ax.grid(False)
plots.save(fig, "fig07_estimator_agreement")

fig, axes = plt.subplots(1, 3, figsize=(7, 2.6))
for ax_, c, lbl in zip(axes, ["adf_p", "eg_p", "kpss_p"], ["ADF p (H0 unit root)", "Engle–Granger p (H0 no coint.)", "KPSS p (H0 stationary)"]):
    ax_.hist(dev[c], bins=20); ax_.axvline(0.05, c="k", ls="--", lw=0.8); ax_.set(xlabel=lbl)
axes[0].set_ylabel("pairs"); fig.suptitle("Stationarity / cointegration of GGR spreads (dev)")
plots.save(fig, "fig08_diagnostics")
diag = pd.Series({"share ADF reject 5%": (dev.adf_p < .05).mean(), "share EG reject 5%": (dev.eg_p < .05).mean(),
                  "share KPSS reject 5%": (dev.kpss_p <= .05).mean(), "share Johansen reject 5%": dev.joh.mean(),
                  "median VR(10)": dev.vr.median()}).to_frame("value")
plots.save_table(diag, "tab_diagnostics", floatfmt="%.3f")

# H3 persistence: formation vs realised trading-period half-life
pers = {c: spearmanr(dev[c], dev.realized_hl, nan_policy="omit")[0] for c in HL}
plots.save_table(pd.Series(pers).to_frame("spearman_rho_vs_realised"), "tab_persistence_dev", floatfmt="%.3f")
fig, ax = plt.subplots(figsize=(4.5, 4))
x, y = dev.hl_ols_252.clip(upper=1000), dev.realized_hl.clip(upper=1000)
ax.hexbin(x, y, xscale="log", yscale="log", gridsize=35, mincnt=1, cmap="Blues")
ax.plot([1, 1000], [1, 1000], c="k", lw=0.8, ls="--")
ax.set(xlabel="formation half-life (days)", ylabel="realised trading half-life (days)",
       title=f"H3 persistence, Spearman ρ = {pers['hl_ols_252']:.2f}")
plots.save(fig, "fig09_persistence")

# Rolling half-life of the example pair from 03 (first dev top-1 pair with ≥2 trips)
px = data.load_prices()
first = dev[(dev["rank"] == 0) & (dev.n_trips >= 2)].iloc[0]
w = px.loc[:, [first.i, first.j]].dropna()
w = w.loc[first.trade_start - pd.DateOffset(years=3): first.trade_start + pd.DateOffset(years=3)]
s = (np.log(w[first.i]) - np.log(w[first.j]))
roll = pd.Series({d: hl.halflife_ols(s.loc[:d].iloc[-126:]) for d in s.index[126::5]})
fig, ax = plt.subplots(figsize=(6.5, 2.8))
ax.plot(roll.index, roll.clip(upper=500)); ax.axvline(first.trade_start, c="k", lw=0.8)
ax.set(yscale="log", ylabel="half-life (days, 126d window)", title=f"Rolling half-life is unstable: {first.i}/{first.j}")
plots.save(fig, "fig10_rolling_halflife")
```

- [ ] **Step 2:** Run; inspect every figure and table. Record H3 verdict.
- [ ] **Step 3:** Commit — `"exp: half-life characterisation and persistence (H3)"`.

---

### Task 13: Predictive tests and dev-only tuning (experiment 05) — H2, H5 selection, OU selection

**Files:** Create `experiments/05_predictive.py`, `experiments/frozen.yaml` (generated), `results/experiment_log.csv`

- [ ] **Step 1: Write**

```python
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

cfg = yaml.safe_load(open("experiments/config.yaml"))
S, C = cfg["sample"], cfg["headline_cost_bps"]
res = pickle.load(open("results/cache/main.pkl", "rb"))
dev = split(res, S["dev_last_trade_start"], S["holdout_first_trade_start"])["dev"]
panel = pd.read_parquet("results/pair_panel.parquet")
pdev = panel[panel.trade_start <= S["dev_last_trade_start"]].copy()
pdev["ret_net"] = pdev.ret - pdev.legs * C * 1e-4
HL = ["hl_ols_63", "hl_ols_126", "hl_ols_252", "hl_kendall", "hl_ou", "cross", "hl_log"]
plots.style()

# ---- H2: quintile sorts (top-100 candidates) and Fama-MacBeth --------------------------
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

pdev["hl_rank"] = pdev.groupby("trade_start").hl_ols_252.rank(pct=True)
pdev["log_hl"] = np.log(pdev.hl_ols_252.clip(upper=1000))
pdev["ssd_rank"] = pdev["rank"] / 100
fm_rows = {}
for name, xs in {"HL rank": ["hl_rank"], "HL rank + SSD rank": ["hl_rank", "ssd_rank"],
                 "log HL + SSD rank + EG p": ["log_hl", "ssd_rank", "eg_p"]}.items():
    tab, _ = fama_macbeth(pdev, "ret", xs)
    for v, r in tab.iterrows():
        fm_rows[(name, v)] = r
plots.save_table(pd.DataFrame(fm_rows).T, "tab_h2_famamacbeth_dev", floatfmt="%.4f")

q = qtabs[("hl_ols_252", "ret")]
fig, ax = plt.subplots(figsize=(5, 3))
mu, se = q.mean() * 100, q.std(ddof=1) / np.sqrt(len(q)) * 100 * np.sqrt(6)   # overlap-inflated SE (approx.)
ax.bar(mu.index, mu, yerr=1.96 * se, capsize=3, color=plots.PALETTE[0])
ax.set(xlabel="formation half-life quintile (1 = fastest)", ylabel="mean 6-month pair return (%)",
       title="H2 (dev): trading-period return by formation half-life")
plots.save(fig, "fig11_h2_quintiles_dev")

# ---- H5 + OU: dev-only grid, every variant logged ---------------------------------------
def hl_filter(K, col):
    return lambda p: p.head(K).nsmallest(20, col).index.to_numpy()   # pairs index = SSD rank 0..n-1

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
dsr = deflated_sharpe(h5.sharpe_ann / np.sqrt(12), len(sr_m), sr_m.var(ddof=1), int(h5.n), h5.skew, h5.kurt)
frozen = {"h5": {"K": int(h5.K), "estimator": h5.estimator, "dev_sharpe": float(h5.sharpe_ann), "dev_dsr": dsr,
                 "n_trials": int(len(sr_m))},
          "ou": {"hold_mult": int(ou.hold_mult), "dev_sharpe": float(ou.sharpe_ann)},
          "config_sha256": hashlib.sha256(open("experiments/config.yaml", "rb").read()).hexdigest()}
yaml.safe_dump(frozen, open("experiments/frozen.yaml", "w"), sort_keys=False)
print(yaml.safe_dump(frozen))

grid = log[log.family == "H5"].pivot(index="estimator", columns="K", values="sharpe_ann")
fig, ax = plt.subplots(figsize=(4.5, 3.6))
im = ax.imshow(grid, cmap="viridis"); ax.grid(False)
ax.set_xticks(range(grid.shape[1]), grid.columns); ax.set_yticks(range(grid.shape[0]), grid.index)
for a in range(grid.shape[0]):
    for b in range(grid.shape[1]):
        ax.text(b, a, f"{grid.iloc[a, b]:.2f}", ha="center", va="center", color="w", fontsize=8)
ax.set(xlabel="SSD candidates K", title=f"H5 dev Sharpe at {C} bp (all {len(sr_m)} trials)"); fig.colorbar(im, shrink=0.8)
plots.save(fig, "fig12_h5_dev_grid")
```

- [ ] **Step 2:** Run; read outputs; record H2 dev verdict.
- [ ] **Step 3:** Commit `experiments/05_predictive.py experiments/frozen.yaml results/experiment_log.csv tables figures` — `"exp: H2 dev tests; freeze H5/OU parameters before holdout"`. **This commit must precede any run of 07.**

---

### Task 14: Robustness, costs and regimes (experiment 06) — H4

**Files:** Create `experiments/06_robustness.py`

- [ ] **Step 1: Write** — reruns the engine with `features=False` for each grid point; all statistics on full sample with dev/holdout columns; also cost curves and regimes.

```python
"""Parameter sensitivity, cost sensitivity (H4), regimes, sectors."""
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from hlpairs import data, plots
from hlpairs.engine import Config, run_all, split, strategy_monthly, top_n
from hlpairs.evaluation import summary

cfg = yaml.safe_load(open("experiments/config.yaml"))
S = cfg["sample"]
frozen = yaml.safe_load(open("experiments/frozen.yaml"))
px = data.load_prices()
membership = data.load_membership("data/raw/sp500_membership.csv")
res = pickle.load(open("results/cache/main.pkl", "rb"))
plots.style()
base = cfg["engine"] | {"ou_hold_mults": (), "features": False}

def hl_filter(K, col):
    return lambda p: p.head(K).nsmallest(20, col).index.to_numpy()   # pairs index = SSD rank 0..n-1

# ---- parameter sensitivity: k x formation length x N --------------------------------
rows = []
for fm in (6, 12):
    for k in (1.5, 2.0, 2.5):
        rs = res if (fm, k) == (12, 2.0) else run_all(px, membership, Config(**(base | {"form_months": fm, "k": k})))
        parts = split(rs, S["dev_last_trade_start"], S["holdout_first_trade_start"])
        for N in (5, 20, 50):
            for period in ("dev", "holdout"):
                m = strategy_monthly(parts[period], select=top_n(N), cost_bps=cfg["headline_cost_bps"])
                rows.append({"form_months": fm, "k": k, "N": N, "period": period, **summary(m)})
sens = pd.DataFrame(rows)
sens.to_csv("results/sensitivity.csv", index=False)
plots.save_table(sens.pivot_table(index=["form_months", "k"], columns=["period", "N"], values="mean_monthly"),
                 "tab_sensitivity", floatfmt="%.4f")
fig, axes = plt.subplots(1, 2, figsize=(7, 3), sharey=True)
for ax_, period in zip(axes, ("dev", "holdout")):
    g = sens[(sens.period == period) & (sens.form_months == 12)].pivot(index="k", columns="N", values="sharpe_ann")
    im = ax_.imshow(g, cmap="RdBu_r", vmin=-1, vmax=1); ax_.grid(False)
    ax_.set_xticks(range(3), g.columns); ax_.set_yticks(range(3), g.index); ax_.set(xlabel="N pairs", title=period)
    for a in range(3):
        for b in range(3):
            ax_.text(b, a, f"{g.iloc[a, b]:.2f}", ha="center", va="center", fontsize=8)
axes[0].set_ylabel("entry threshold k (σ)"); fig.colorbar(im, ax=axes, shrink=0.8)
fig.suptitle(f"Net Sharpe ({cfg['headline_cost_bps']} bp) across parameters, 12-month formation")
plots.save(fig, "fig13_sensitivity")

# ---- cost curves (H4) -----------------------------------------------------------------
strats = {"GGR top 20": dict(select=top_n(20)),
          "HL-filtered 20 (frozen)": dict(select=hl_filter(frozen["h5"]["K"], frozen["h5"]["estimator"])),
          f"OU (m={frozen['ou']['hold_mult']})": dict(source=f"ou_m{frozen['ou']['hold_mult']}")}
parts = split(res, S["dev_last_trade_start"], S["holdout_first_trade_start"])
costs = np.arange(0, 41, 2.5)
curve = {(n, p): [summary(strategy_monthly(parts[p], cost_bps=c, **kw))["ann_mean"] for c in costs]
         for n, kw in strats.items() for p in ("dev", "holdout")}
cc = pd.DataFrame(curve, index=costs); cc.index.name = "cost_bps_per_side"
cc.to_csv("results/cost_curves.csv")
breakeven = {k: float(np.interp(0, -cc[k].to_numpy(), costs)) if cc[k].iloc[0] > 0 else 0.0 for k in cc}
plots.save_table(pd.Series(breakeven).unstack(), "tab_breakeven_cost", floatfmt="%.1f")
fig, ax = plt.subplots(figsize=(6.5, 3.2))
for (n, p), y in cc.items():
    ax.plot(costs, y * 100, ls="-" if p == "holdout" else ":", label=f"{n} ({p})")
ax.axhline(0, c="k", lw=0.6)
ax.set(xlabel="cost per leg per side (bp)", ylabel="annualised mean return (%)", title="H4: returns vs transaction costs")
ax.legend(fontsize=7, ncol=2); plots.save(fig, "fig14_cost_sensitivity")

# ---- regimes ----------------------------------------------------------------------------
m = strategy_monthly(res, select=top_n(20), cost_bps=cfg["headline_cost_bps"])
mkt = data.load_prices("data/raw/market.parquet")
vix = mkt.VIX.resample("ME").mean(); vix.index = vix.index.to_period("M")
rec = pd.read_csv("data/raw/usrec.csv", index_col=0, parse_dates=True).iloc[:, 0]; rec.index = rec.index.to_period("M")
df = pd.DataFrame({"r": m, "vix": vix, "rec": rec}).dropna()
df["VIX tercile"] = pd.qcut(df.vix, 3, labels=["low", "mid", "high"])
df["era"] = pd.cut(df.index.year, [1999, 2007, 2009, 2019, 2025], labels=["2000–07", "2008–09", "2010–19", "2020–25"])
reg = pd.concat({"VIX tercile": df.groupby("VIX tercile", observed=True).r.agg(["mean", "std", "count"]),
                 "NBER recession": df.groupby("rec").r.agg(["mean", "std", "count"]),
                 "era": df.groupby("era", observed=True).r.agg(["mean", "std", "count"])})
plots.save_table(reg, "tab_regimes", floatfmt="%.4f")
fig, axes = plt.subplots(1, 3, figsize=(7, 2.6), sharey=True)
for ax_, key in zip(axes, ["era", "VIX tercile", "NBER recession"]):
    g = reg.loc[key]
    ax_.bar([str(x) for x in g.index], g["mean"] * 100, yerr=1.96 * g["std"] / np.sqrt(g["count"]) * 100, capsize=3)
    ax_.axhline(0, c="k", lw=0.6); ax_.set(title=key)
axes[0].set_ylabel("% per month (net)"); fig.suptitle("Top-20 pairs returns by regime (descriptive; ex-post regime labels)")
plots.save(fig, "fig15_regimes")

# ---- sector cross-section ---------------------------------------------------------------
panel = pd.read_parquet("results/pair_panel.parquet")
sec = pd.read_csv("data/raw/sectors_current.csv", index_col=0)["sector"]
p20 = panel[panel["rank"] < 20].copy()
p20["sector"] = np.where(p20.same_sector, p20.i.map(sec), "cross-sector / unknown")
cs = p20.groupby("sector").agg(pairs=("ret", "size"), mean_ret=("ret", "mean"), median_hl=("hl_ols_252", "median"))
plots.save_table(cs.sort_values("pairs", ascending=False), "tab_sector_crosssection", floatfmt="%.4f")
fig, ax = plt.subplots(figsize=(5.5, 3.6))
ax.scatter(cs.median_hl, cs.mean_ret * 100, s=cs.pairs / cs.pairs.max() * 300, alpha=0.6)
for n, r in cs.iterrows():
    ax.annotate(n, (r.median_hl, r.mean_ret * 100), fontsize=6)
ax.set(xlabel="median formation half-life (days)", ylabel="mean 6-month pair return (%)", title="Sector cross-section (top-20 pairs, full sample)")
plots.save(fig, "fig16_sector_crosssection")
```

- [ ] **Step 2:** Run (background; 5 extra engine runs). Inspect outputs.
- [ ] **Step 3:** Commit — `"exp: robustness, costs (H4), regimes, sectors"`.

---

### Task 15: Holdout (experiment 07) — H2, H3, H5 out of sample

**Files:** Create `experiments/07_holdout.py`

- [ ] **Step 1:** Verify the frozen file is committed and unchanged: the script aborts unless `git status --porcelain experiments/frozen.yaml experiments/config.yaml` is empty and `frozen.config_sha256` matches `config.yaml`.

```python
"""Single out-of-sample evaluation with parameters frozen on the dev sample."""
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

if subprocess.run(["git", "status", "--porcelain", "experiments/frozen.yaml", "experiments/config.yaml"],
                  capture_output=True, text=True).stdout.strip():
    sys.exit("frozen.yaml / config.yaml must be committed and unchanged before the holdout run")
cfg = yaml.safe_load(open("experiments/config.yaml"))
frozen = yaml.safe_load(open("experiments/frozen.yaml"))
assert frozen["config_sha256"] == hashlib.sha256(open("experiments/config.yaml", "rb").read()).hexdigest()
commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
S, C = cfg["sample"], cfg["headline_cost_bps"]
res = pickle.load(open("results/cache/main.pkl", "rb"))
hold = split(res, S["dev_last_trade_start"], S["holdout_first_trade_start"])["holdout"]
panel = pd.read_parquet("results/pair_panel.parquet")
ph = panel[panel.trade_start >= S["holdout_first_trade_start"]].copy()
plots.style()

def hl_filter(K, col):
    return lambda p: p.head(K).nsmallest(20, col).index.to_numpy()   # pairs index = SSD rank 0..n-1

pvals, out = {}, {}
q = quintile_sort(ph, "hl_ols_252")
m, se, t, p = nw_mean_test(q[1] - q[5]); pvals["H2 Q1-Q5 (hl_ols_252)"] = p
out["H2 Q1-Q5"] = {"est": m, "t": t, "p": p}
ph["hl_rank"] = ph.groupby("trade_start").hl_ols_252.rank(pct=True)
fm, _ = fama_macbeth(ph, "ret", ["hl_rank"])
pvals["H2 FM slope (HL rank)"] = fm.loc["hl_rank", "p"]; out["H2 FM slope"] = fm.loc["hl_rank"].to_dict()
rho, p3 = spearmanr(ph.hl_ols_252, ph.realized_hl, nan_policy="omit")
pvals["H3 persistence"] = p3; out["H3 rho"] = {"est": rho, "p": p3}
ggr = strategy_monthly(hold, select=top_n(20), cost_bps=C)
filt = strategy_monthly(hold, select=hl_filter(frozen["h5"]["K"], frozen["h5"]["estimator"]), cost_bps=C)
ou = strategy_monthly(hold, source=f"ou_m{frozen['ou']['hold_mult']}", cost_bps=C)
d = (filt - ggr).dropna()
m5, _, t5, p5 = nw_mean_test(d); pvals["H5 filtered − GGR"] = p5; out["H5 diff"] = {"est": m5, "t": t5, "p": p5}
s1 = summary(ggr); pvals["H1/H4 GGR net mean > 0"] = s1["p"]
adj = adjust_pvalues(pd.Series(pvals))
plots.save_table(adj, "tab_holdout_tests", floatfmt="%.4f")
perf = pd.DataFrame({"GGR top 20": s1, "HL-filtered 20": summary(filt), "OU": summary(ou)}).T
plots.save_table(perf[["ann_mean", "t_nw", "sharpe_ann", "max_dd", "frac_pos", "n"]], "tab_holdout_performance", floatfmt="%.4f")
yaml.safe_dump({"commit": commit, "frozen": frozen, "results": {k: {a: float(b) for a, b in v.items()} for k, v in out.items()}},
               open("results/holdout_record.yaml", "w"), sort_keys=False)

fig, axes = plt.subplots(2, 1, figsize=(6.5, 4.8), sharex=True, height_ratios=[2, 1])
for n, s in {"GGR top 20": ggr, "HL-filtered 20": filt, "OU": ou}.items():
    axes[0].plot(s.index.to_timestamp(), (1 + s).cumprod(), label=n)
    axes[1].plot(s.index.to_timestamp(), drawdown(s))
axes[0].set(ylabel="growth of $1", title=f"Holdout 2013–2025, net of {C} bp per leg-side, parameters frozen on 2000–2012")
axes[0].legend(); axes[1].set(ylabel="drawdown")
plots.save(fig, "fig17_holdout_equity")

fig, ax = plt.subplots(figsize=(5, 3))
mu = q.mean() * 100
ax.bar(mu.index, mu, color=plots.PALETTE[1]); ax.axhline(0, c="k", lw=0.6)
ax.set(xlabel="formation half-life quintile (1 = fastest)", ylabel="mean 6-month pair return (%)",
       title="H2 out of sample (2013–2025)")
plots.save(fig, "fig18_h2_quintiles_holdout")
```

- [ ] **Step 2:** Run once. Do **not** change any parameter after seeing the output; any bug fix that changes numbers is documented in the paper as a post-holdout correction, with before/after values.
- [ ] **Step 3:** Commit — `"exp: one-shot holdout evaluation"`.

---

### Task 16: Notebooks

**Files:** Create `notebooks/build_notebooks.py`, `notebooks/01_pair_walkthrough.ipynb`, `notebooks/02_results_tour.ipynb`

- [ ] **Step 1:** `notebooks/build_notebooks.py` uses `nbformat.v4.new_notebook/new_markdown_cell/new_code_cell` to write two notebooks:
  - `01_pair_walkthrough`: markdown explaining each step; code cells that load cached prices, pick the example pair from `fig02`, compute SSD rank, spread, all four half-life estimators, ADF/EG, and simulate it with `simulate_pair`, printing the trade log (open/close dates, P&L).
  - `02_results_tour`: loads every `tables/*.csv` and displays them with one markdown cell per hypothesis stating the verdict text that will appear in the paper; displays key figures with `IPython.display.Image`.
- [ ] **Step 2:** `uv run python notebooks/build_notebooks.py && uv run jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb` → both execute without errors.
- [ ] **Step 3:** Commit — `"docs: walkthrough and results notebooks"`.

---

### Task 17: Research paper

**Files:** Create `paper/main.tex`, `paper/references.bib`, output `paper/main.pdf`

- [ ] **Step 1: Verify every reference** via Crossref (`curl -s "https://api.crossref.org/works?query.bibliographic=<title>&rows=1"`) and write `references.bib` only with entries whose title/authors/year/journal/DOI match. Candidate list: GGR (2006, RFS); Do & Faff (2010, FAJ); Do & Faff (2012, JFR); Engle & Granger (1987); Vidyamurthy (2004, book); Elliott, van der Hoek & Malcolm (2005, QF); Avellaneda & Lee (2010, QF); Krauss (2017, JES); Rad, Low & Faff (2016, QF); Kendall (1954, Biometrika); Dickey & Fuller (1979, JASA); Kwiatkowski et al. (1992, JoE); Johansen (1991, Econometrica); Lo & MacKinlay (1988, RFS); Newey & West (1987); Fama & MacBeth (1973, JPE); Bailey & López de Prado (2014, JPM); Harvey, Liu & Zhu (2016, RFS); Holm (1979); Benjamini & Hochberg (1995); Uhlenbeck & Ornstein (1930); Jegadeesh (1990, JF); Lehmann (1990, QJE); Chan (2013, book). Drop any that cannot be verified.
- [ ] **Step 2: Write `paper/main.tex`** (article class, `booktabs`, `graphicx`, `amsmath`, `natbib`, `hyperref`), sections exactly: Abstract, Introduction, Literature, Hypotheses, Data, Methodology, Paper Replication, Results, Robustness, Out-of-Sample Results, Transaction Costs, Economic Interpretation, Limitations, Conclusion, Future Research, References. Rules:
  - Every number in prose is copied from a file in `tables/` or `results/`; tables are `\input{../tables/<name>.tex}`; figures `\includegraphics{../figures/<name>.pdf}`.
  - Methodology contains the OU/AR(1) derivation, half-life formula, Kendall correction, SSD, trading rule, return definitions, NW and FM formulas, DSR.
  - Each hypothesis section states the verdict (supported / not supported / mixed) with the test statistic, and labels the content **Replication / Extension / Original**.
  - Limitations reproduce spec §10 plus anything found during execution (coverage numbers from `tab_data_coverage`).
  - Statement that the backtest is a hypothesis test, not evidence of a deployable strategy.
- [ ] **Step 3:** Build: `cd paper && pdflatex -interaction=nonstopmode main && bibtex main && pdflatex -interaction=nonstopmode main && pdflatex -interaction=nonstopmode main`. Expected: `main.pdf` exists, `grep -c "Warning.*undefined" main.log` = 0. Open the PDF pages with Read and check every figure/table renders.
- [ ] **Step 4:** Commit `paper/main.tex paper/references.bib paper/main.pdf` — `"paper: full research paper"`.

---

### Task 18: README, LICENSE, CITATION, reproduction

**Files:** Create `README.md`, `LICENSE` (MIT, holder "Saksham Garg", year 2026), `CITATION.cff`, `Makefile`

- [ ] **Step 1: `Makefile`** with targets `env` (`uv sync`), `test` (`uv run pytest -q`), `data`, `run` (scripts 02–06 in order), `holdout` (07), `notebooks`, `paper`, `all`.
- [ ] **Step 2: `README.md`** sections: title + one-paragraph abstract; Research question; Original paper (full citation + link); Motivation; Methodology (pipeline diagram as a Markdown list/mermaid); Dataset; Main findings (bullet per hypothesis, numbers from tables); **Visual results** (every figure fig01–fig18 embedded with a one-sentence caption, grouped: data, replication, half-life, predictive, robustness, holdout); key tables rendered as Markdown from the CSVs; Reproduction (`make all`, expected runtime, Python 3.13, uv); Repository structure; Paper link (`paper/main.pdf`); Citation (BibTeX); **Zenodo DOI: section present with the text "DOI will be added on publication" until Task 20 produces a real DOI**; Limitations / disclaimer.
- [ ] **Step 3: `CITATION.cff`** (cff-version 1.2.0, type software, title, authors `Garg, Saksham`, version 1.0.0, date-released, license MIT, `preferred-citation` of type article for the paper). No `doi` field yet.
- [ ] **Step 4: Fresh reproduction check:** `git clone . $SCRATCH/repro && cd $SCRATCH/repro && uv sync && uv run pytest -q` → pass. (Full data rerun optional; record whether done.)
- [ ] **Step 5:** Final review: invoke `superpowers:requesting-code-review` on the whole repo; fix findings; `uv run pytest -q`.
- [ ] **Step 6:** Commit — `"docs: README, license, citation, Makefile"`.

---

### Task 19: GitHub publication (GATED — ask the user first)

- [ ] **Step 1:** Ask the user: repo name (default `mean-reversion-half-life`), public/private, and confirm publishing. Check `gh auth status`; if not authenticated, tell the user to run `gh auth login` (cannot be done here).
- [ ] **Step 2:** After explicit yes: `gh repo create <name> --public --source . --push --description "..."`. Report the URL.

---

### Task 20: Zenodo publication (GATED — ask the user first)

- [ ] **Step 1:** Ask the user to confirm publishing and to provide a Zenodo personal access token via an environment variable `ZENODO_TOKEN` (scopes `deposit:write`, `deposit:actions`) set in their own shell — never pasted into chat or files. Confirm creator name/affiliation/ORCID for metadata.
- [ ] **Step 2:** Create a draft deposition with `prereserve_doi: true` via `POST https://zenodo.org/api/deposit/depositions`; read the reserved DOI from `metadata.prereserve_doi.doi`.
- [ ] **Step 3:** Write the reserved DOI into `README.md` (badge + citation) and `CITATION.cff` (`doi:`), rebuild paper if it cites the DOI, commit, tag `v1.0.0`, push (if GitHub done).
- [ ] **Step 4:** `git archive --format=zip -o $SCRATCH/hlpairs-v1.0.0.zip v1.0.0`; upload it and `paper/main.pdf` to the deposition bucket; set metadata (upload_type `publication`/`publicationtype` `workingpaper` or `software` per user choice, title, creators, description, keywords, license `mit`, related identifier = GitHub URL, references GGR DOI).
- [ ] **Step 5:** Publish only after the user re-confirms (`POST .../actions/publish`). Verify the DOI resolves (`curl -sI https://doi.org/<doi>` → 302 to zenodo). Only then report it as published.
