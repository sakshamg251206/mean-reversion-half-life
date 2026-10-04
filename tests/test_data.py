import numpy as np
import pandas as pd
from hlpairs import data
from hlpairs.universe import formation_universe


def membership():
    return pd.Series(
        [frozenset({"A", "B"}), frozenset({"A", "C"}), frozenset({"A", "B"})],
        index=pd.to_datetime(["2000-01-01", "2001-01-01", "2003-01-01"]),
    )


def test_to_yahoo():
    assert data.to_yahoo("BRK.B") == "BRK-B"


def test_members_on_uses_last_snapshot_on_or_before():
    m = membership()
    assert data.members_on(m, pd.Timestamp("2000-06-30")) == {"A", "B"}
    assert data.members_on(m, pd.Timestamp("2001-01-01")) == {"A", "C"}
    assert data.members_on(m, pd.Timestamp("1999-01-01")) == frozenset()


def test_spells_and_masking_remove_reused_ticker_history():
    m = membership()
    spells = data.membership_spells(m, end=pd.Timestamp("2004-01-01"))
    assert spells["B"] == [(pd.Timestamp("2000-01-01"), pd.Timestamp("2001-01-01")),
                           (pd.Timestamp("2003-01-01"), pd.Timestamp("2004-01-01"))]
    idx = pd.bdate_range("1999-01-01", "2004-12-31")
    px = pd.DataFrame(1.0, index=idx, columns=["A", "B", "Z"])
    out = data.mask_outside_spells(px, spells, tail_months=18)
    assert np.isnan(out.loc["1999-06-01", "B"])                        # before first spell
    assert out.loc["2002-06-03", "B"] == 1.0                           # within 18m tail of spell 1
    assert np.isnan(out.loc["2002-08-01", "B"])                        # gap between spells
    assert out["Z"].isna().all()                                       # never a member


def test_clean_prices_fills_short_gaps_only_inside_history():
    idx = pd.bdate_range("2000-01-03", periods=12)
    s = pd.Series([1, 2, np.nan, 3, -1, 4, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan], index=idx, dtype=float)
    out = data.clean_prices(s.to_frame("X"), max_gap=5)["X"]
    assert out.iloc[2] == 2           # interior gap filled
    assert out.iloc[4] == 3           # non-positive -> nan -> filled from previous
    assert out.iloc[6:].isna().all()  # trailing (delisting) gap never filled


def test_formation_universe_requires_membership_and_full_history():
    idx = pd.bdate_range("2000-01-03", periods=30)
    px = pd.DataFrame(1.0, index=idx, columns=["A", "B", "C"])
    px.loc[idx[:5], "B"] = np.nan                   # B lists mid-window
    u = formation_universe(px, membership(), idx[0], idx[-1])
    assert u == ["A"]                               # C not a member on 2000-01-03, B incomplete


def test_drop_stale_removes_securities_with_many_zero_returns():
    idx = pd.bdate_range("2000-01-03", periods=100)
    rng = np.random.default_rng(0)
    live = 50 * np.exp(np.cumsum(rng.normal(0, 0.01, 100)))
    stale = live.copy()
    stale[::2] = np.roll(stale, 1)[::2]                 # every other day unchanged: 50% zero returns
    px = pd.DataFrame({"LIVE": live, "STALE": stale}, index=idx)
    out, dropped = data.drop_stale(px, max_zero_frac=0.10)
    assert list(out.columns) == ["LIVE"] and dropped == ["STALE"]
