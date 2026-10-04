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
    base = form_px.iloc[0]                       # trading prices stay normalised to formation start
    trade = px.loc[(px.index >= t0) & (px.index < t1), universe]

    pairs = top_pairs(form, cfg.n_candidates)
    pnl, legs, active = trade_pairs(trade, pairs, cfg.k, cfg.wait, base=base)
    pairs["trade_start"], pairs["rank"] = t0, np.arange(len(pairs))
    pairs["n_universe"] = len(universe)
    if sectors is not None:
        pairs["same_sector"] = [sectors.get(i) is not None and sectors.get(i) == sectors.get(j)
                                for i, j in zip(pairs.i, pairs.j)]
    extra = {}

    rng = np.random.default_rng([cfg.seed, int(t0.strftime("%Y%m"))])
    n = len(universe)
    n_all = n * (n - 1) // 2
    flat = rng.choice(n_all, size=min(cfg.n_random, n_all), replace=False)
    iu, ju = np.triu_indices(n, 1)
    extra["random"] = trade_pairs(trade, pair_frame(form, iu[flat], ju[flat]), cfg.k, cfg.wait, base=base)

    if sectors is not None:
        sec = [sectors.get(t, f"_none_{t}") for t in universe]   # unknown sector never pairs
        extra["sector"] = trade_pairs(trade, top_pairs(form, cfg.n_sector, sectors=sec), cfg.k, cfg.wait, base=base)

    if cfg.features:
        pairs = pairs.join(pair_features(form, form_px, pairs, cfg.lookbacks))
        cand = pairs[pairs.beta > 0].nsmallest(cfg.ou_n, "eg_p")
        for m in cfg.ou_hold_mults:
            extra[f"ou_m{m}"] = trade_ou(trade, cand, cfg.k, cfg.wait, hold_mult=m)

    pairs["ret"] = pnl.sum(0)
    pairs["legs"] = legs.sum(0)
    pairs["n_trips"] = legs.sum(0) / 4
    tn = trade / base
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
