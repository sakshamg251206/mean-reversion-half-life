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
