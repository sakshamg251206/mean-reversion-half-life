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
