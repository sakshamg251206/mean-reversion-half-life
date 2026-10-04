"""Point-in-time S&P 500 membership, Yahoo prices, cleaning and caching."""
import hashlib
import json
from pathlib import Path

import pandas as pd

MEMBERSHIP_URL = ("https://raw.githubusercontent.com/fja05680/sp500/master/"
                  "S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv")


def to_yahoo(t: str) -> str:
    return t.strip().replace(".", "-")


def load_membership(path) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["date"])
    sets = [frozenset(to_yahoo(t) for t in s.split(",") if t.strip()) for s in df["tickers"]]
    return pd.Series(sets, index=df["date"]).sort_index()


def members_on(membership: pd.Series, date) -> frozenset:
    snap = membership.loc[:date]
    return snap.iloc[-1] if len(snap) else frozenset()


def membership_spells(membership: pd.Series, end=None) -> dict:
    """Contiguous membership intervals per ticker: [(first snapshot in, first snapshot out)].
    An open spell ends at `end` (default: last snapshot date)."""
    end = membership.index[-1] if end is None else end
    spells, open_since, prev = {}, {}, frozenset()
    for date, cur in membership.items():
        for t in cur - prev:
            open_since[t] = date
        for t in prev - cur:
            spells.setdefault(t, []).append((open_since.pop(t), date))
        prev = cur
    for t, start in open_since.items():
        spells.setdefault(t, []).append((start, end))
    return spells


def mask_outside_spells(px: pd.DataFrame, spells: dict, tail_months: int = 18) -> pd.DataFrame:
    """Keep a ticker's prices only from a spell's start to spell end + tail_months.
    A portfolio formed while the stock is a member needs at most formation + trading
    (18 months) of data after the formation start, so this never removes a usable price,
    but it does remove history Yahoo attaches to a reused ticker (a different company)."""
    keep = pd.DataFrame(False, index=px.index, columns=px.columns)
    for t in px.columns:
        for s, e in spells.get(t, []):
            keep.loc[s:e + pd.DateOffset(months=tail_months), t] = True
    return px.where(keep)


def clean_prices(px: pd.DataFrame, max_gap: int = 5) -> pd.DataFrame:
    """Non-positive prices -> nan; forward-fill interior gaps of at most `max_gap` days.
    Gaps after a stock's last price (delisting) are never filled."""
    px = px.where(px > 0)
    filled = px.ffill(limit=max_gap)
    return filled.where(px.bfill().notna())


def download_prices(tickers, start, end, chunk: int = 200) -> pd.DataFrame:
    """Daily dividend/split-adjusted closes from Yahoo Finance (a total-return proxy)."""
    import yfinance as yf

    frames = []
    for k in range(0, len(tickers), chunk):
        df = yf.download(list(tickers[k:k + chunk]), start=start, end=end,
                         auto_adjust=True, progress=False, threads=True)["Close"]
        frames.append(df)
    px = pd.concat(frames, axis=1)
    px = px.loc[:, ~px.columns.duplicated()].dropna(axis=1, how="all")
    px.index = pd.to_datetime(px.index).tz_localize(None)
    return px.sort_index()


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_manifest(paths, out="data/raw/manifest.json") -> None:
    meta = {str(p): {"sha256": sha256(p), "bytes": Path(p).stat().st_size,
                     "downloaded": pd.Timestamp.now(tz="UTC").isoformat()} for p in paths}
    Path(out).write_text(json.dumps(meta, indent=2))


def load_prices(path="data/raw/prices_clean.parquet") -> pd.DataFrame:
    return pd.read_parquet(path)
