"""GGR pair formation: rank all pairs by the sum of squared differences (SSD)
between normalised cumulative-return price paths."""
import numpy as np
import pandas as pd


def normalize(px: pd.DataFrame) -> pd.DataFrame:
    """Cumulative total-return index starting at 1 (adjusted close / first adjusted close)."""
    return px / px.iloc[0]


def pair_frame(norm: pd.DataFrame, i_idx, j_idx) -> pd.DataFrame:
    X = norm.to_numpy()
    diff = X[:, i_idx] - X[:, j_idx]
    cols = norm.columns.to_numpy()
    return pd.DataFrame({"i": cols[i_idx], "j": cols[j_idx],
                         "ssd": (diff ** 2).sum(0), "sigma": diff.std(0, ddof=1)})


def top_pairs(norm: pd.DataFrame, n: int, sectors=None) -> pd.DataFrame:
    """The n lowest-SSD pairs. SSD_ij = |x_i|^2 + |x_j|^2 - 2 x_i.x_j, computed for all
    pairs at once from the Gram matrix."""
    X = norm.to_numpy()
    sq = (X ** 2).sum(0)
    D = sq[:, None] + sq[None, :] - 2 * X.T @ X
    iu, ju = np.triu_indices(X.shape[1], 1)
    d = D[iu, ju]
    if sectors is not None:
        sec = np.asarray(sectors, dtype=object)
        d = np.where(sec[iu] == sec[ju], d, np.inf)
    order = np.argsort(d, kind="stable")[:n]
    order = order[np.isfinite(d[order])]
    return pair_frame(norm, iu[order], ju[order])
