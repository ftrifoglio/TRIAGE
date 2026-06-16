"""Bipartite (ku, ki)-core decomposition on a (user, item) edge list."""

import pandas as pd


def kcore(d: pd.DataFrame, ku: int, ki: int, verbose: bool) -> pd.DataFrame:
    """
    Decompose bipartite graph into the (ku, ki)-core.

    Returns the unique fixed-point subgraph in which every user has degree >= ku and
    every item has degree >= ki.

    Parameters
    ----------
    d : pd.DataFrame
        Edge list with `user` and `item` columns.
    ku : int
        Minimum user degree retained in the core.
    ki : int
        Minimum item degree retained in the core.
    verbose : bool
        If True, print per-iteration interaction / user / item counts.

    Returns
    -------
    pd.DataFrame
        The (ku, ki)-core edge list with a reset index.

    """
    iteration = 0
    while True:
        u_counts = d["user"].value_counts()
        i_counts = d["item"].value_counts()
        valid_u = u_counts[u_counts >= ku].index
        valid_i = i_counts[i_counts >= ki].index
        new = d[d["user"].isin(valid_u) & d["item"].isin(valid_i)]
        if verbose:
            print(
                f"  iter {iteration:>2}: {len(new):>10,} interactions  "
                f"({new['user'].nunique():>6,} users, "
                f"{new['item'].nunique():>6,} items)"
            )
        if len(new) == len(d):
            return new.reset_index(drop=True)
        d = new
        iteration += 1
