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
