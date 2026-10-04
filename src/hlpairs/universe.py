"""Formation-period universe: point-in-time members with a complete price history."""
import pandas as pd

from .data import members_on


def formation_universe(px: pd.DataFrame, membership: pd.Series, start, end) -> list[str]:
    members = members_on(membership, start)
    cols = [c for c in px.columns if c in members]
    win = px.loc[start:end, cols]
    return win.columns[win.notna().all()].tolist()
