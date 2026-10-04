import itertools

import numpy as np
import pandas as pd
from hlpairs.pairs import normalize, top_pairs


def frame(seed=0, n=8, T=120):
    rng = np.random.default_rng(seed)
    px = pd.DataFrame(np.exp(np.cumsum(rng.normal(0, 0.01, (T, n)), axis=0)),
                      columns=[f"S{k}" for k in range(n)])
    px["S7"] = px["S3"] * np.exp(rng.normal(0, 0.001, T))   # near-twin of S3
    return px


def test_normalize_starts_at_one():
    assert np.allclose(normalize(frame()).iloc[0], 1.0)


def test_top_pairs_matches_brute_force():
    norm = normalize(frame())
    brute = sorted(((((norm[a] - norm[b]) ** 2).sum(), a, b)
                    for a, b in itertools.combinations(norm.columns, 2)))[:5]
    got = top_pairs(norm, 5)
    assert list(zip(got.i, got.j)) == [(a, b) for _, a, b in brute]
    assert np.allclose(got.ssd, [d for d, _, _ in brute])
    assert (got.i.iloc[0], got.j.iloc[0]) == ("S3", "S7")
    assert np.isclose(got.sigma.iloc[0], (norm["S3"] - norm["S7"]).std(ddof=1))


def test_sector_restriction():
    norm = normalize(frame())
    sectors = ["X"] * 4 + ["Y"] * 4          # S3 (X) and S7 (Y) differ
    got = top_pairs(norm, 3, sectors=sectors)
    assert ("S3", "S7") not in set(zip(got.i, got.j))
    lookup = dict(zip(norm.columns, sectors))
    assert all(lookup[a] == lookup[b] for a, b in zip(got.i, got.j))
