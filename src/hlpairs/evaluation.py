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
