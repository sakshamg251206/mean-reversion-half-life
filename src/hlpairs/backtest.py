"""Pair-trading simulation and GGR portfolio accounting.

Timing convention: x[t] is known at the close of day t. An order decided at close t
executes at close t + wait, and the position earns returns from the next day on.
P&L is measured per $1 committed to the pair and legs are buy-and-hold inside a trade
(leg values compound), as in GGR's marked-to-market return calculation.
"""
import numpy as np
import pandas as pd


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


def trade_pairs(trade_px: pd.DataFrame, pairs: pd.DataFrame, k: float = 2.0, wait: int = 0, base=None):
    """GGR rule: signal = (P_i - P_j) / sigma_formation. `base` = prices on the first
    formation day, so trading-period prices stay normalised to the formation start as in
    GGR's Figure 1; without it prices are re-based to the first trading day."""
    base = trade_px.iloc[0] if base is None else base[trade_px.columns]
    P = (trade_px / base).to_numpy()
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
    Employed ("fully invested") capital: divide by the pairs holding a position (or
    trading) that day — GGR eq. (2), where the portfolio return is weighted over open
    positions. Costs: cost_bps per leg traded, per $1 notional."""
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
