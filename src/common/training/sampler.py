"""Sampling primitives."""

from collections.abc import Iterator
from typing import Protocol

import numpy as np
import torch
import torch.nn as nn

from .data import InteractionDataset


class BPRSampler:
    """Uniform BPR triple sampler. One negative per positive, drawn until unseen."""

    def __init__(
        self,
        dataset: InteractionDataset,
        batch_size: int,
        seed: int,
        sampling: str = "per_edge",
    ) -> None:
        """
        Initialise the sampler and cache each user's positive-item set.

        Parameters
        ----------
        dataset : InteractionDataset
            Source of training edges and per-user positive items.
        batch_size : int
            Number of triples per mini-batch. Counts edges under `"per_edge"` and users
            under `"per_user"`.
        seed : int
            Seed for the NumPy random generator driving shuffling and negatives.
        sampling : str, default "per_edge"
            Triple sampling convention: `"per_edge"` (default) permutes the training
            edges, weighting users by degree; `"per_user"` draws users uniformly and one
            uniform positive each, over `n_train // batch_size + 1` batches.

        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.sampling = sampling
        self.rng = np.random.default_rng(seed)
        self._user_pos_set = {
            u: set(items.tolist()) for u, items in dataset.user_pos_items.items()
        }
        # Users with at least one training positive so a uniform draw always has a
        # positive to return
        self._exist_users = np.fromiter(
            (u for u, s in self._user_pos_set.items() if s), dtype=np.int64
        )
        self._user_pos_arr = {
            u: np.asarray(items, dtype=np.int64)
            for u, items in dataset.user_pos_items.items()
        }

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Iterate over shuffled mini-batches of BPR triples for one epoch.

        Permutes the training edges and yields batches of `(user, pos_item, neg_item)`,
        drawing one unseen negative per positive.

        Yields
        ------
        users : torch.Tensor of shape (B,), int64
            Batch user idx.
        pos : torch.Tensor of shape (B,), int64
            Positive item idx, aligned to `users`.
        neg : torch.Tensor of shape (B,), int64
            Unseen negative item idx, aligned to `users`.

        """
        if self.sampling == "per_user":
            yield from self._iter_per_user()
            return

        edges = self.dataset.train_edges
        order = self.rng.permutation(len(edges))
        n_items = self.dataset.n_items
        bs = self.batch_size

        for start in range(0, len(order), bs):
            batch_idx = order[start : start + bs]
            users = edges[batch_idx, 0]
            pos = edges[batch_idx, 1]
            neg = self._sample_negatives(users, n_items)
            yield (
                torch.from_numpy(users.astype(np.int64)),
                torch.from_numpy(pos.astype(np.int64)),
                torch.from_numpy(neg.astype(np.int64)),
            )

    def _iter_per_user(
        self,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Iterate batches under the canonical per-user convention.

        Runs `n_train // batch_size + 1` batches; each batch draws `batch_size` users
        uniformly (without replacement within a batch when possible) and one uniform
        positive per user, so the epoch keeps ~`n_train` steps while weighting users
        uniformly rather than by degree.

        Yields
        ------
        users : torch.Tensor of shape (B,), int64
            Batch user idx.
        pos : torch.Tensor of shape (B,), int64
            Positive item idx, aligned to `users`.
        neg : torch.Tensor of shape (B,), int64
            Unseen negative item idx, aligned to `users`.

        """
        n_items = self.dataset.n_items
        bs = self.batch_size
        exist = self._exist_users
        replace = bs > len(exist)

        for _ in range(self.dataset.n_train // bs + 1):
            users = self.rng.choice(exist, size=bs, replace=replace)
            pos = self._sample_positives(users)
            neg = self._sample_negatives(users, n_items)
            yield (
                torch.from_numpy(users.astype(np.int64)),
                torch.from_numpy(pos.astype(np.int64)),
                torch.from_numpy(neg.astype(np.int64)),
            )

    def _sample_positives(self, users: np.ndarray) -> np.ndarray:
        """
        Draw one uniformly-random training positive item per user.

        Parameters
        ----------
        users : np.ndarray of shape (B,), int64
            User idx to draw a positive for.

        Returns
        -------
        np.ndarray of shape (B,), int64
            One positive item idx per user, aligned to `users`.

        """
        out = np.empty(len(users), dtype=np.int64)
        for k, u in enumerate(users):
            items = self._user_pos_arr[int(u)]
            out[k] = items[self.rng.integers(0, len(items))]
        return out

    def _sample_negatives(self, users: np.ndarray, n_items: int) -> np.ndarray:
        """
        Draw one uniformly-random unseen negative item per user.

        For each user, resamples until the drawn item is not in that user's training
        positive set.

        Parameters
        ----------
        users : np.ndarray of shape (B,), int64
            User idx to draw a negative for.
        n_items : int
            Number of items, the upper bound of the uniform draw.

        Returns
        -------
        np.ndarray of shape (B,), int64
            One negative item idx per user, aligned to `users`.

        """
        out = self.rng.integers(0, n_items, size=len(users))
        for k, u in enumerate(users):
            seen = self._user_pos_set.get(int(u), set())
            while int(out[k]) in seen:
                out[k] = self.rng.integers(0, n_items)
        return out

    def __len__(self) -> int:
        """Return number of mini-batches per epoch."""
        if self.sampling == "per_user":
            return self.dataset.n_train // self.batch_size + 1
        return (self.dataset.n_train + self.batch_size - 1) // self.batch_size


class NeighbourhoodSampler(Protocol):
    """
    Protocol for adjacency-shaping samplers.

    Implementations decide what sparse propagation matrix the encoder reads from in a
    given epoch.
    """

    def sample(self, epoch: int, encoder_state: dict[str, nn.Module]) -> torch.Tensor:
        """
        Return the sparse adjacency to propagate over for `epoch`.

        Parameters
        ----------
        epoch : int
            Current training epoch, letting implementations vary the adjacency.
        encoder_state : dict[str, nn.Module]
            Dictionary keyed by "encoder" containing the encoder module an
            implementation may inspect to shape the adjacency.

        Returns
        -------
        torch.Tensor of shape (n_users + n_items, n_users + n_items), float32
            The sparse CSR propagation adjacency for this epoch.

        """
        ...


class IdentitySampler:
    """Always return the precomputed full normalised adjacency."""

    def __init__(self, norm_adj: torch.Tensor) -> None:
        """
        Store the precomputed normalised adjacency to return each epoch.

        Parameters
        ----------
        norm_adj : torch.Tensor of shape (n_users + n_items, n_users + n_items), float32
            The full sparse normalised propagation adjacency, returned unchanged.

        """
        self.norm_adj = norm_adj

    def sample(self, epoch: int, encoder_state: dict[str, nn.Module]) -> torch.Tensor:
        """
        Return the precomputed full normalised adjacency, ignoring inputs.

        Parameters
        ----------
        epoch : int
            Unused.
        encoder_state : dict[str, nn.Module]
            Unused.

        Returns
        -------
        torch.Tensor of shape (n_users + n_items, n_users + n_items), float32
            The stored full sparse normalised adjacency.

        """
        return self.norm_adj
