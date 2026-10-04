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
