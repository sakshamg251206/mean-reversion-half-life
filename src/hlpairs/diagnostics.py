"""Stationarity / cointegration diagnostics used to characterise pair spreads."""
import warnings

import numpy as np
from statsmodels.tsa.stattools import adfuller, coint, kpss
from statsmodels.tsa.vector_ar.vecm import coint_johansen


def adf_pvalue(s) -> float:
    """H0: unit root. Small p => evidence the spread is stationary."""
    return float(adfuller(np.asarray(s, float), autolag="AIC", maxlag=10, result_object=False)[1])


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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")      # statsmodels internally casts tiny complex eigenvalues
        r = coint_johansen(np.column_stack([y, x]), det_order=0, k_ar_diff=1)
    return bool(r.lr1[0] > r.cvt[0, 1])


def variance_ratio(s, q: int = 10) -> float:
    """Lo-MacKinlay VR(q) of the spread's increments: <1 mean reversion, ~1 random walk."""
    s = np.asarray(s, float)
    d1 = np.diff(s)
    dq = s[q:] - s[:-q]
    return float(dq.var(ddof=1) / (q * d1.var(ddof=1)))
