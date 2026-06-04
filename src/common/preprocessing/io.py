"""Read / write adjacency-list dataset files."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd


def save_adjacency(
    df: pd.DataFrame,
    out_dir: Path,
    test_ratio: float,
    seed: int | None = None,
    split_strategy: str = "temporal",
    time_col: str = "timestamp",
) -> None:
    """
    Remap user/item to 0-based ints, split per-user, write the four files.

    Users with only one interaction go entirely to train. Users with at least two
    interactions get at least one item in test (`ceil(test_ratio * n)`).

    Parameters
    ----------
    df : pd.DataFrame
        Interaction frame with `user` and `item` columns (original IDs). Under
        `split_strategy="temporal"` it must also carry `time_col`.
    out_dir : Path
        Directory the `user_list.txt`, `item_list.txt`, `train.txt`, and `test.txt`
        files are written to; created if missing.
    test_ratio : float
        Fraction of each user's interactions placed in the test split.
    seed : int | None, default None
        Seed for the random generator used to shuffle each user's items under
        `split_strategy="random"`; unused under `"temporal"`. If None and
        `split_strategy="random"`, defaults to 42.
    split_strategy : str, default "temporal"
        `"temporal"` (leave-last-out by `time_col`) or `"random"` (uniform per-user
        holdout). Under `"temporal"`, `train.txt` is serialised in chronological order
        (oldest first) so a downstream most-recent validation tail can be carved.
    time_col : str, default "timestamp"
        Name of the timestamp column used to order interactions under
        `split_strategy="temporal"`.

    """
    out_dir.mkdir(parents=True, exist_ok=True)

    users = sorted(df["user"].unique())
    items = sorted(df["item"].unique())
    u_remap = {u: i for i, u in enumerate(users)}
    i_remap = {it: i for i, it in enumerate(items)}

    _write_id_list(out_dir / "user_list.txt", u_remap)
    _write_id_list(out_dir / "item_list.txt", i_remap)

    work = pd.DataFrame(
        {
            "user": df["user"].map(u_remap),
            "item": df["item"].map(i_remap),
        }
    )
    if split_strategy == "temporal":
        if time_col not in df.columns:
            raise ValueError(
                f"split_strategy='temporal' requires a {time_col!r} column in df"
            )
        work["t"] = df[time_col].to_numpy()
    elif split_strategy != "random":
        raise ValueError(f"unknown split_strategy={split_strategy!r}")

    if seed is None:
        seed = 42

    rng = np.random.default_rng(seed)
    train_lines: list[str] = []
    test_lines: list[str] = []
    for _, group in work.groupby("user", sort=True):
        user = int(group["user"].iloc[0])
        if split_strategy == "temporal":
            # Order by time (ties broken by item index for determinism) before splitting
            ordered = group.sort_values(["t", "item"], kind="stable")["item"].to_numpy()
            n = len(ordered)
            # Remaining interactions after floor(test_ratio * n) go to train
            n_test = max(1, int(np.ceil(n * test_ratio))) if n >= 2 else 0
            train_items, test_items = ordered[: n - n_test], ordered[n - n_test :]
        else:
            ordered = group["item"].to_numpy().copy()
            rng.shuffle(ordered)
            n = len(ordered)
            n_test = max(1, int(np.ceil(n * test_ratio))) if n >= 2 else 0
            test_items, train_items = ordered[:n_test], ordered[n_test:]
        # Under the temporal strategy, train.txt is in chronological order
        # (oldest first) so the a validation set can be carved from the tail
        train_out = train_items if split_strategy == "temporal" else sorted(train_items)
        if len(train_items) > 0:
            train_lines.append(f"{user} " + " ".join(map(str, train_out)))
        if len(test_items) > 0:
            test_lines.append(f"{user} " + " ".join(map(str, sorted(test_items))))

    (out_dir / "train.txt").write_text("\n".join(train_lines) + "\n")
    (out_dir / "test.txt").write_text("\n".join(test_lines) + "\n")


def _write_id_list(path: Path, remap: dict) -> None:
    with open(path, "w") as f:
        f.write("org_id remap_id\n")
        for org, idx in remap.items():
            f.write(f"{org} {idx}\n")


def _iter_edges(paths: list[Path]) -> Iterator[tuple[int, int]]:
    for path in paths:
        with open(path) as f:
            for line in f:
                toks = line.split()
                if not toks:
                    continue
                u = int(toks[0])
                for t in toks[1:]:
                    yield u, int(t)


def load_adjacency(paths: Path | list[Path]) -> pd.DataFrame:
    """
    Load one or more adjacency-list files as a DataFrame[user, item] of ints.

    Parameters
    ----------
    paths : Path or list[Path]
        A single adjacency-list file path or a list of them; a single `Path` is wrapped
        into a one-element list.

    Returns
    -------
    pd.DataFrame
        Frame with integer `user` and `item` columns, one row per edge.

    """
    if isinstance(paths, Path):
        paths = [paths]
    return pd.DataFrame(_iter_edges(paths), columns=["user", "item"])


def load_splits(splits_dir: Path) -> tuple[set, set, set[tuple]]:
    """
    Load reference user / item / edge sets keyed by original IDs.

    Needs `user_list.txt` and `item_list.txt` to map remapped idx back to original
    IDs.

    Parameters
    ----------
    splits_dir : Path
        Directory containing the reference `user_list.txt`, `item_list.txt`,
        `train.txt`, and `test.txt` files.

    Returns
    -------
    tuple[set, set, set[tuple]]
        Original-ID user set, original-ID item set, and the set of `(user, item)`
        original-ID edge pairs.

    """
    u_df = pd.read_csv(splits_dir / "user_list.txt", sep=" ")
    i_df = pd.read_csv(splits_dir / "item_list.txt", sep=" ")
    u_org = u_df.set_index("remap_id")["org_id"].to_dict()
    i_org = i_df.set_index("remap_id")["org_id"].to_dict()

    files = [
        p for p in (splits_dir / "train.txt", splits_dir / "test.txt") if p.exists()
    ]
    edges = {(u_org[u], i_org[i]) for u, i in _iter_edges(files)}
    return set(u_df["org_id"]), set(i_df["org_id"]), edges


def load_test_edges(splits_dir: Path) -> set[tuple]:
    """
    Load the `test.txt` edge set keyed by original IDs.

    Needs `user_list.txt` and `item_list.txt` to map remapped idx back to original
    IDs.

    Parameters
    ----------
    splits_dir : Path
        Directory containing `user_list.txt`, `item_list.txt`, and `test.txt`.

    Returns
    -------
    set[tuple]
        Set of `(user, item)` original-ID test edges.

    """
    u_df = pd.read_csv(splits_dir / "user_list.txt", sep=" ")
    i_df = pd.read_csv(splits_dir / "item_list.txt", sep=" ")
    u_org = u_df.set_index("remap_id")["org_id"].to_dict()
    i_org = i_df.set_index("remap_id")["org_id"].to_dict()
    return {(u_org[u], i_org[i]) for u, i in _iter_edges([splits_dir / "test.txt"])}
