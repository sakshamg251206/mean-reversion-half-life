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
