"""Sparse-adjacency products for the propagation hot path."""

import weakref
from typing import Any

import torch

# Symmetry verdict per adjacency tensor, keyed by `id()` and computed once on first use.
# A finalizer drops the entry when the tensor is freed, so an adjacency that is rebuilt
# (e.g. a per-epoch augmented graph) gets a fresh check rather than a stale verdict from
# a reused id. (A `WeakKeyDictionary` cannot hold sparse CSR tensors: its lookup
# compares keys, which CSR does not support.)
_SYMMETRIC: dict[int, bool] = {}


class _SymmetricSpMM(torch.autograd.Function):
    """`adj @ x` with a backward that reuses `adj` (exactly symmetric `adj` only)."""

    @staticmethod
    def forward(ctx: Any, adj: torch.Tensor, x: torch.Tensor) -> torch.Tensor:  # noqa: ANN401
        """
        Return `adj @ x`, keeping `adj` for the backward.

        Parameters
        ----------
        ctx : Any
            Autograd context (`FunctionCtx`; typed `Any` as in torch, since it carries
            ad-hoc attributes).
        adj : torch.Tensor of shape (n, n), float32
            Exactly symmetric sparse adjacency; receives no gradient.
        x : torch.Tensor of shape (n, d), float32
            Dense right-hand side.

        Returns
        -------
        torch.Tensor of shape (n, d), float32
            The product `adj @ x`.

        """
        ctx.adj = adj
        return torch.sparse.mm(adj, x)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor) -> tuple[None, torch.Tensor]:  # noqa: ANN401
        """
        Return `adj @ grad` (= `adj^T @ grad` by symmetry) without transposing.

        Parameters
        ----------
        ctx : Any
            Autograd context holding `adj`.
        *grad_outputs : torch.Tensor of shape (n, d), float32
            Single entry: gradient of the loss with respect to the product.

        Returns
        -------
        grad_adj : None
            No gradient for `adj`.
        grad_x : torch.Tensor of shape (n, d), float32
            The gradient with respect to `x`.

        """
        (grad,) = grad_outputs
        return None, torch.sparse.mm(ctx.adj, grad)


def _is_symmetric(adj: torch.Tensor) -> bool:
    """
    Whether `adj` equals its transpose exactly (indices and values), cached per tensor.

    Parameters
    ----------
    adj : torch.Tensor of shape (n, n), float32
        Sparse square adjacency.

    Returns
    -------
    bool
        True only if every stored entry `(i, j, v)` has a bitwise-equal `(j, i, v)`.

    """
    verdict = _SYMMETRIC.get(id(adj))
    if verdict is None:
        with torch.no_grad():
            a = adj.to_sparse_coo().coalesce()
            t = a.t().coalesce()
            verdict = (
                adj.shape[0] == adj.shape[1]
                and torch.equal(a.indices(), t.indices())
                and torch.equal(a.values(), t.values())
            )
        _SYMMETRIC[id(adj)] = verdict
        weakref.finalize(adj, _SYMMETRIC.pop, id(adj), None)
    return verdict


def adj_mm(adj: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Sparse propagation product `adj @ x`, with a transpose-free backward when exact.

    Parameters
    ----------
    adj : torch.Tensor of shape (n, n), float32
        Sparse propagation adjacency.
    x : torch.Tensor of shape (n, d), float32
        Dense embedding table.

    Returns
    -------
    torch.Tensor of shape (n, d), float32
        `adj @ x`.

    """
    if (
        torch.is_grad_enabled()
        and x.requires_grad
        and not adj.requires_grad
        and _is_symmetric(adj)
    ):
        return _SymmetricSpMM.apply(adj, x)
    return torch.sparse.mm(adj, x)


# Module created by claude-opus-4-8 as part of its response to prompt in
# `src/common/training/encoder.py:59-99`.
