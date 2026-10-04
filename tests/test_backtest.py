import numpy as np
import pandas as pd
import pytest
from hlpairs.backtest import monthly_from_portfolios, portfolio_returns, simulate_pair, trade_pairs

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


def test_trade_pairs_keeps_formation_normalisation():
    # i and j both start trading at 1.1 relative to formation start, but i was 1.3 vs j 1.0
    trade = pd.DataFrame({"I": [1.3, 1.3, 1.0], "J": [1.0, 1.0, 1.0]})
    base = pd.Series({"I": 1.0, "J": 1.0})
    pairs = pd.DataFrame({"i": ["I"], "j": ["J"], "sigma": [0.1]})
    _, legs, _ = trade_pairs(trade, pairs, k=2, base=base)
    assert legs[:, 0].tolist() == [2, 0, 2]          # gap 0.3 = 3 sigma opens on day 0; crosses on day 2
    _, legs_rebased, _ = trade_pairs(trade, pairs, k=2)
    assert legs_rebased[:, 0].tolist() == [0, 0, 4]  # re-based: no gap on day 0; opens on last day, force-closed


def test_portfolio_returns_costs_and_bases():
    pnl = np.array([[0.0, 0.0], [0.01, 0.0], [0.02, 0.0]])
    legs = np.array([[2, 0], [0, 0], [2, 0]])
    active = np.array([[False, False], [True, False], [True, False]])
    committed, employed = portfolio_returns(pnl, legs, active, cost_bps=10)
    assert np.allclose(committed, [-0.002 / 2, 0.01 / 2, (0.02 - 0.002) / 2])
    assert np.allclose(employed, [-0.002, 0.01, 0.018])   # only one of two pairs ever opens


def test_employed_divides_by_pairs_opened_during_period():
    pnl = np.array([[0.0, 0.0], [0.01, 0.0], [0.0, 0.0]])
    legs = np.array([[2, 0], [0, 0], [2, 4]])            # pair 1 opens (and closes) on the last day
    active = np.zeros((3, 2), dtype=bool)
    _, employed = portfolio_returns(pnl, legs, active)
    assert employed[1] == pytest.approx(0.005)            # GGR: divide by the 2 pairs that opened


def test_monthly_compounds_then_averages_portfolios():
    idx = pd.to_datetime(["2001-01-02", "2001-01-03", "2001-02-01"])
    a = pd.Series([0.1, 0.1, 0.0], index=idx)
    b = pd.Series([0.0, 0.0, 0.2], index=idx)
    m = monthly_from_portfolios([a, b])
    assert m.loc[pd.Period("2001-01")] == pytest.approx((0.21 + 0.0) / 2)
    assert m.loc[pd.Period("2001-02")] == pytest.approx(0.1)
