"""Download membership, prices, market series and sectors; clean; write coverage report."""
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import requests
import yaml

from hlpairs import data, plots

cfg = yaml.safe_load(open("experiments/config.yaml"))
RAW = Path("data/raw"); RAW.mkdir(parents=True, exist_ok=True)
start, end = cfg["data"]["start"], cfg["data"]["end"]

mem_path = RAW / "sp500_membership.csv"
mem_path.write_bytes(requests.get(data.MEMBERSHIP_URL, timeout=60).content)
membership = data.load_membership(mem_path)
membership = membership[membership.index <= end]
ever = sorted(set().union(*membership[membership.index >= start]) | set(data.members_on(membership, start)))
print(f"{len(ever)} tickers were S&P 500 members at some point {start}..{end}")

import time

want = ever + ["SPY", "^VIX"]
stop = pd.Timestamp(end) + pd.Timedelta(days=1)
cache = RAW / "prices_raw.parquet"
raw = pd.read_parquet(cache) if cache.exists() else pd.DataFrame()
# Yahoo drops tickers transiently (rate limits), so downloads are cumulative: keep what we
# have and retry only the missing tickers, in progressively smaller batches.
for attempt, chunk in enumerate((200, 50, 10)):
    missing = [t for t in want if t not in raw.columns]
    print(f"attempt {attempt + 1}: {len(missing)} tickers missing")
    if not missing:
        break
    got = data.download_prices(missing, start, stop, chunk=chunk)
    raw = pd.concat([raw, got], axis=1).sort_index()
    time.sleep(20)
raw.to_parquet(RAW / "prices_raw.parquet")
market = raw[["SPY", "^VIX"]].rename(columns={"^VIX": "VIX"})
market.to_parquet(RAW / "market.parquet")

stocks = raw.drop(columns=["SPY", "^VIX"])
spells = data.membership_spells(membership, end=pd.Timestamp(end))
clean = data.clean_prices(data.mask_outside_spells(stocks, spells))
clean = clean.dropna(axis=1, how="all")
clean, stale = data.drop_stale(clean)
print(f"dropped {len(stale)} stale/reused-ticker securities (>10% zero-return days): {stale}")
clean.to_parquet(RAW / "prices_clean.parquet")
print(f"Yahoo returned {stocks.shape[1]} of {len(ever)} tickers; {clean.shape[1]} keep data after spell masking")

usrec = pd.read_csv("https://fred.stlouisfed.org/graph/fredgraph.csv?id=USREC", parse_dates=[0], index_col=0)
usrec.to_csv(RAW / "usrec.csv")

html = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", timeout=60,
                    headers={"User-Agent": "hlpairs-research/1.0"}).text
wiki = pd.read_html(StringIO(html))[0]
sectors = pd.Series(wiki["GICS Sector"].to_numpy(), index=wiki["Symbol"].map(data.to_yahoo), name="sector")
sectors.to_csv(RAW / "sectors_current.csv")

data.write_manifest(sorted(RAW.glob("*.parquet")) + [mem_path, RAW / "usrec.csv", RAW / "sectors_current.csv"])

# ---- coverage / survivorship report -------------------------------------------------
years = range(int(start[:4]), int(end[:4]) + 1)
rows = []
for y in years:
    d0, d1 = pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y}-12-31")
    mem = data.members_on(membership, d0)
    cols = [t for t in mem if t in clean.columns]
    win = clean.loc[d0:d1, cols]
    rows.append({"year": y, "members": len(mem), "any_data": int(win.notna().any().sum()),
                 "full_year": int(win.notna().all().sum()),
                 "sector_known": int(sum(t in sectors.index for t in mem))})
cov = pd.DataFrame(rows).set_index("year")
cov["pct_full_year"] = cov.full_year / cov.members
plots.save_table(cov, "data_coverage", floatfmt="%.2f")
print(cov.to_string())
rets = clean.pct_change(fill_method=None)
print("daily |r| > 50% observations:", int((rets.abs() > 0.5).sum().sum()))

plots.style()
fig, ax = plt.subplots(figsize=(6.5, 3))
ax.plot(cov.index, cov.members, label="index members (1 Jan)", color=plots.GREY)
ax.plot(cov.index, cov.full_year, label="with complete Yahoo prices that year", color=plots.PALETTE[0])
ax.plot(cov.index, cov.sector_known, label="with a (current) GICS sector", color=plots.PALETTE[1], linestyle="--")
ax.set(ylabel="stocks", title="Point-in-time coverage: the survivorship gap shrinks toward the present")
ax.set_ylim(0, 540); ax.legend(loc="lower right")
plots.save(fig, "fig01_coverage")
