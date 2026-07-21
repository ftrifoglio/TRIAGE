"""Dataset loading, building, splitting and processing."""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

from common.preprocessing.io import load_adjacency


@dataclass
class InteractionDataset:
    """
    Train and test edges with optional carved-out validation edges.

    Attributes
    ----------
    name : str
        Dataset name, taken from the directory name it was loaded from.
    n_users : int
        Number of users. Users occupy index range `[0, n_users)`.
    n_items : int
        Number of items. Items occupy index range `[n_users, n_users + n_items)`.
    train_edges : np.ndarray of shape (E, 2), int64
        `(user, item)` training interactions, with any carved-out validation edges
        excluded.
    test_edges : np.ndarray of shape (E_test, 2), int64
        `(user, item)` test interactions.
    user_pos_items : dict[int, np.ndarray]
        Maps each user to its training items. These are the positives the model trains
        on, and the mask used when evaluating on the validation split.
    val_user_items : dict[int, np.ndarray]
        Maps each user to its validation items. Empty when `val_ratio == 0`.
    test_user_items : dict[int, np.ndarray]
        Maps each user to its test items.
    train_val_user_items : dict[int, np.ndarray]
        Maps each user to its full `train.txt` items. This is the mask used when
        evaluating on the test split.
    item_degrees : np.ndarray of shape (n_items,), int64
        Item popularity over the full `train.txt`, so the head/tail partition is
        invariant to `val_ratio`.

    """

    name: str
    n_users: int
    n_items: int
    train_edges: np.ndarray
    test_edges: np.ndarray
    user_pos_items: dict[int, np.ndarray]
    val_user_items: dict[int, np.ndarray]
    test_user_items: dict[int, np.ndarray]
    train_val_user_items: dict[int, np.ndarray]
    item_degrees: np.ndarray

    @classmethod
    def from_dir(
        cls,
        dataset_dir: Path,
        *,
        val_ratio: float = 0.0,
        val_seed: int | None = None,
        val_carve: str = "temporal",
    ) -> "InteractionDataset":
        """
        Load `train.txt` and `test.txt` and optionally carve out a validation split.

        Parameters
        ----------
        dataset_dir : Path
            Directory containing `train.txt` and `test.txt`.
        val_ratio : float, default 0.0
            Fraction of each user's training items to carve into the validation split.
            `0.0` (default) disables carving. Must be in range [0.0, 1.0). Users with
            fewer than two training items keep all items in train.
        val_seed : int | None, default None
            Seed for the random carve. Used only when `val_carve="random"`. Ignored for
            the deterministic temporal carve. If None and `val_carve="random"`, defaults
            to 42.
        val_carve : str, default "temporal"
            `"temporal"` carves the most-recent (chronological) tail of the time-ordered
            `train.txt`, mirroring the leave-last-out test horizon; `"random"` carves a
            uniform random subset seeded by `val_seed`.

        Returns
        -------
        InteractionDataset
            The loaded dataset with edges and derived statistics populated.

        """
        dataset_dir = Path(dataset_dir)
        if (
            not isinstance(val_ratio, (int, float))
            or val_ratio < 0.0
            or val_ratio >= 1.0
        ):
            raise ValueError(
                f"val_ratio must be in range [0.0, 1.0), got {val_ratio!r}"
            )
        if val_carve not in ("temporal", "random"):
            raise ValueError(
                f"val_carve must be 'temporal' or 'random', got {val_carve!r}"
            )
        if val_seed is None:
            val_seed = 42

        train_df = load_adjacency(dataset_dir / "train.txt")
        test_df = load_adjacency(dataset_dir / "test.txt")

        n_users = int(max(train_df["user"].max(), test_df["user"].max())) + 1
        n_items = int(max(train_df["item"].max(), test_df["item"].max())) + 1

        full_train_edges = train_df.to_numpy(dtype=np.int64)
        test_edges = test_df.to_numpy(dtype=np.int64)

        train_val_items: dict[int, list[int]] = defaultdict(list)
        for u, i in full_train_edges:
            train_val_items[int(u)].append(int(i))

        rng = np.random.default_rng(val_seed)
        keep_items: dict[int, np.ndarray] = {}  # items that survive the val-carving
        val_items: dict[int, np.ndarray] = {}
        for u in sorted(train_val_items):
            items = np.asarray(train_val_items[u], dtype=np.int64)
            if val_ratio > 0.0 and len(items) >= 2:
                n_val = max(1, int(round(len(items) * val_ratio)))
                n_val = min(n_val, len(items) - 1)  # keep >= 1 train item
                if val_carve == "random":
                    perm = rng.permutation(len(items))
                    val_idx = np.sort(perm[:n_val])
                    keep_idx = np.sort(perm[n_val:])
                    val_items[u] = items[val_idx]
                    keep_items[u] = items[keep_idx]
                else:
                    val_items[u] = items[-n_val:]
                    keep_items[u] = items[:-n_val]
            else:
                keep_items[u] = items

        if val_ratio > 0.0:
            train_edges = np.array(
                [(u, i) for u in sorted(keep_items) for i in keep_items[u]],
                dtype=np.int64,
            ).reshape(-1, 2)
        else:
            train_edges = full_train_edges

        test_user_items: dict[int, list[int]] = defaultdict(list)
        for u, i in test_edges:
            test_user_items[int(u)].append(int(i))

        item_degrees = np.bincount(full_train_edges[:, 1], minlength=n_items)

        return cls(
            name=dataset_dir.name,
            n_users=n_users,
            n_items=n_items,
            train_edges=train_edges,
            test_edges=test_edges,
            user_pos_items=keep_items,
            val_user_items=val_items,
            test_user_items={
                u: np.asarray(v, dtype=np.int64) for u, v in test_user_items.items()
            },
            train_val_user_items={
                u: np.asarray(v, dtype=np.int64) for u, v in train_val_items.items()
            },
            item_degrees=item_degrees,
        )

    @property
    def n_train(self) -> int:
        """Number of training edges."""
        return len(self.train_edges)


def build_norm_adj(
    dataset: InteractionDataset,
    extra_edges: np.ndarray | None = None,
) -> torch.Tensor:
    """
    Build the degree-normalised bipartite adjacency as a sparse CSR tensor.

    Assembles `A = [[0, R], [R^T, 0]]` over the joint user+item index space from the
    training edges and applies the symmetric LightGCN normalisation `D^{-1/2} A
    D^{-1/2}`, where `D` is the degree matrix of `A`.

    Parameters
    ----------
    dataset : InteractionDataset
        Source of the training edges and the index sizes `n_users`/`n_items`.
    extra_edges : np.ndarray of shape (n_aug, 2), int64 or None, default None
        Additional `(user, item)` edges inserted as unit entries alongside the training
        edges before degrees are computed and the normalisation applied.

    Returns
    -------
    torch.Tensor of shape (n_users + n_items, n_users + n_items), float32
        Sparse CSR tensor on CPU.

    """
    n_users, n_items = dataset.n_users, dataset.n_items
    n = n_users + n_items
    edges = dataset.train_edges
    if extra_edges is not None and len(extra_edges):
        edges = np.concatenate([edges, extra_edges.astype(edges.dtype)])
    rows = edges[:, 0]
    cols = edges[:, 1] + n_users

    # Build A = R | R^T (size n x n) via COO.
    all_rows = np.concatenate([rows, cols])
    all_cols = np.concatenate([cols, rows])
    data = np.ones_like(all_rows, dtype=np.float32)
    a = sp.coo_matrix((data, (all_rows, all_cols)), shape=(n, n))

    # A is symmetric so one degree vector serves both sides
    deg = np.asarray(a.sum(axis=1)).flatten()
    # `deg == 0` occurs for nodes with no training edge (items seen only in test or
    # carved entirely into validation). The clamp avoids the divide-by-zero warning and
    # zeroing states the convention
    # (https://github.com/Coder-Yu/SELFRec/blob/5b02294/data/graph.py#L15) that isolated
    # nodes get weight 0. Neither changes a_norm.
    d_inv_sqrt = np.power(np.maximum(deg, 1.0), -0.5)
    d_inv_sqrt[deg == 0] = 0.0
    d = sp.diags(d_inv_sqrt)
    a_norm = (d @ a @ d).tocoo()

    indices = torch.from_numpy(np.vstack([a_norm.row, a_norm.col]).astype(np.int64))
    values = torch.from_numpy(a_norm.data.astype(np.float32))
    adj = torch.sparse_coo_tensor(indices, values, (n, n)).coalesce()

    # LightGCNEncoder multiplies by this adjacency n_layers times per batch (forward and
    # backward) via `adj_mm` (`torch.sparse.mm`) whose CUDA CSR path uses cuSPARSE
    # directly, so we return a CSR.
    return adj.to_sparse_csr()
