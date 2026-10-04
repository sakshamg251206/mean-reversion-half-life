"""Run the rolling engine once on the full sample; cache results and the pair panel."""
import pickle
import time
from pathlib import Path

import pandas as pd
import yaml

from hlpairs import data
from hlpairs.engine import Config, pair_panel, run_all

if __name__ == "__main__":                       # guard needed: run_all uses worker processes
    cfg = yaml.safe_load(open("experiments/config.yaml"))
    px = data.load_prices()
    membership = data.load_membership("data/raw/sp500_membership.csv")
    sectors = pd.read_csv("data/raw/sectors_current.csv", index_col=0)["sector"].to_dict()
    ec = cfg["engine"] | {"ou_hold_mults": tuple(cfg["engine"]["ou_hold_mults"])}
    t = time.time()
    results = run_all(px, membership, Config(**ec), sectors=sectors)
    print(f"{len(results)} formations in {time.time() - t:.0f}s")
    Path("results/cache").mkdir(parents=True, exist_ok=True)
    pickle.dump(results, open("results/cache/main.pkl", "wb"))
    panel = pair_panel(results)
    panel.to_parquet("results/pair_panel.parquet")
    print(panel.describe().T.to_string())
