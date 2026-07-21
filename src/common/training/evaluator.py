"""
Full-ranking evaluator.

Scores every ground-truth user of a split against every item, masks the requested
interactions, and ranks the rest. Accuracy metrics are reported overall and for the
head/tail item bands. A band metric keeps only that band's hits and positives, and
skips users without positives in the band.
"""

from collections.abc import Callable

import numpy as np
import torch

from .config import EvalCfg
from .data import InteractionDataset
from .metrics import (
    effective_coverage,
    item_coverage,
    ndcg_at_k,
    pru_at_k,
    pru_plus,
    recall_at_k,
    self_information,
)

ScoreAdjust = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
"""
Post-hoc score adjustment `(scores, item_pop) -> scores`.

`scores` is `(n_chunk_users, n_items)` with masked entries at `-inf`; `item_pop` is the
`(n_items,)` item training degrees. The returned matrix must keep masked entries at
`-inf`.
"""


class Evaluator:
    """Full-ranking evaluator."""

    def __init__(
        self,
        dataset: InteractionDataset,
        cfg: EvalCfg,
    ) -> None:
        """
        Build the evaluator and the head/tail item partition.

        Parameters
        ----------
        dataset : InteractionDataset
            Dataset providing test users, train/test interactions, and item degrees.
        cfg : EvalCfg
            Evaluator configurations.

        """
        self.dataset = dataset
        self.cfg = cfg

        deg = dataset.item_degrees
        if len(deg) == 0:
            self.head_mask = np.zeros(0, dtype=bool)
        else:
            cutoff = np.quantile(deg, cfg.head_quantile)
            self.head_mask = deg >= cutoff
        self.tail_mask = ~self.head_mask

    @torch.no_grad()
    def evaluate(
        self,
        user_emb: torch.Tensor,
        item_emb: torch.Tensor,
        gt_items: dict[int, np.ndarray],
        mask_items: dict[int, np.ndarray],
        return_top_items: bool = False,
        score_adjust: ScoreAdjust | None = None,
    ) -> dict[str, float]:
        """
        Score every ground-truth user against every item (full-ranking).

        Parameters
        ----------
        user_emb : torch.Tensor of shape (n_users, d), float32
            User embeddings.
        item_emb : torch.Tensor of shape (n_items, d), float32
            Item embeddings.
        gt_items : dict[int, np.ndarray]
            Maps each user to its held-out ground-truth items for this split.
        mask_items : dict[int, np.ndarray]
            Maps each user to the items to set to `-inf` before top-K. During validation
            user's train items are masked out; for test it is train plus val.
        return_top_items : bool, default False
            When true, additionally keep every scored user's top-`max(cfg.k_list)` item
            idx, best first, from the same ranking pass, and stash them on `top_items`
            as an array of shape `(n_scored_users, max_k)`. Lets any top-K metric be
            recomputed later, per user, without the model.
        score_adjust : ScoreAdjust or None, default None
            Applied to each chunk's masked scores before ranking, e.g. popularity
            compensation. `None` ranks the raw scores.

        Returns
        -------
        dict[str, float]
            For each K in `k_list`: `recall@K`, `ndcg@K` (overall plus `_head`/`_tail`
            band variants), `novelty@K`, `item_coverage@K`, `eff_coverage@K` and
            `pru@K`, and `pru_plus`, when self.cfg.compute_pru is `True`.

        """
        device = user_emb.device
        ks = sorted(self.cfg.k_list)
        max_k = ks[-1]
        n_items = self.dataset.n_items
        eval_users = np.array(sorted(gt_items.keys()), dtype=np.int64)

        # Item popularity; numpy for PRU metrics; tensor for novelty and score_adjust
        pop_np = self.dataset.item_degrees.astype(np.float64)
        pop_t = torch.from_numpy(pop_np).to(device)

        bands = {
            "": None,
            "_head": torch.from_numpy(self.head_mask).to(device),
            "_tail": torch.from_numpy(self.tail_mask).to(device),
        }

        # Running sums across user chunks
        recall_sum = {band: {k: 0.0 for k in ks} for band in bands}
        ndcg_sum = {band: {k: 0.0 for k in ks} for band in bands}
        band_users = {band: 0 for band in bands}  # users with >= 1 positive in band
        rec_counts = {k: torch.zeros(n_items, device=device) for k in ks}
        novelty_sum = {k: torch.zeros((), device=device) for k in ks}
        novelty_cnt = {k: torch.zeros((), device=device) for k in ks}
        pru_plus_sum, pru_plus_cnt = 0.0, 0
        pru_k_sum = {k: 0.0 for k in ks}
        pru_k_cnt = {k: 0 for k in ks}
        top_chunks: list[np.ndarray] = []

        chunk = self.cfg.user_chunk
        for start in range(0, len(eval_users), chunk):
            users = eval_users[start : start + chunk]
            scores = user_emb[torch.from_numpy(users).to(device)] @ item_emb.t()
            scores.masked_fill_(_indicator(users, mask_items, n_items, device), -np.inf)
            if score_adjust is not None:
                scores = score_adjust(scores, pop_t)
            gt = _indicator(users, gt_items, n_items, device)

            # (n_chunk_users, max_k)
            top_idx = torch.topk(scores, k=max_k, dim=-1).indices
            top_hits = gt.gather(1, top_idx)  # (n_chunk_users, max_k) bool
            if return_top_items:
                top_chunks.append(top_idx.cpu().numpy().astype(np.int32))

            # Accuracy, per band
            for band, mask in bands.items():
                if mask is None:
                    band_gt, band_hits = gt, top_hits
                else:
                    band_gt, band_hits = gt & mask, top_hits & mask[top_idx]
                n_pos = band_gt.sum(dim=-1)  # (n_chunk_users,)
                has_pos = n_pos > 0
                band_users[band] += int(has_pos.sum().item())
                for k in ks:
                    recall = recall_at_k(band_hits[:, :k], n_pos)
                    ndcg = ndcg_at_k(band_hits[:, :k], n_pos)
                    recall_sum[band][k] += float(recall[has_pos].sum().item())
                    ndcg_sum[band][k] += float(ndcg[has_pos].sum().item())

            # Coverage and novelty
            for k in ks:
                top_k = top_idx[:, :k]
                flat = top_k.reshape(-1)
                rec_counts[k].index_add_(
                    0, flat, torch.ones_like(flat, dtype=rec_counts[k].dtype)
                )
                deg = pop_t[top_k]
                # Zero-degree items have no defined self-information and are left out
                known = deg > 0
                si = self_information(deg, self.dataset.n_users)
                novelty_sum[k] += si[known].sum()
                novelty_cnt[k] += known.sum()

            # PRU
            if self.cfg.compute_pru:
                d, c = _sum_defined(pru_plus(scores, gt, pop_np, chunk))
                pru_plus_sum += d
                pru_plus_cnt += c
                for k in ks:
                    pru_k_sum[k], c = _sum_defined(
                        pru_at_k(top_idx, pop_np, k), start=pru_k_sum[k]
                    )
                    pru_k_cnt[k] += c

        out: dict[str, float] = {}
        for band in bands:
            denom = max(band_users[band], 1)
            for k in ks:
                out[f"recall@{k}{band}"] = recall_sum[band][k] / denom
                out[f"ndcg@{k}{band}"] = ndcg_sum[band][k] / denom
        catalogue = max(n_items, 1)
        for k in ks:
            out[f"item_coverage@{k}"] = item_coverage(rec_counts[k], catalogue)
            out[f"eff_coverage@{k}"] = effective_coverage(rec_counts[k], catalogue)
            out[f"novelty@{k}"] = (
                float((novelty_sum[k] / novelty_cnt[k]).item())
                if float(novelty_cnt[k].item()) > 0
                else float("nan")
            )
        if self.cfg.compute_pru:
            out["pru_plus"] = _negated_mean(pru_plus_sum, pru_plus_cnt)
            for k in ks:
                out[f"pru@{k}"] = _negated_mean(pru_k_sum[k], pru_k_cnt[k])

        if return_top_items:
            self.top_users = eval_users
            self.top_items = (
                np.concatenate(top_chunks)
                if top_chunks
                else np.zeros((0, max_k), dtype=np.int32)
            )

        return out


def _indicator(
    users: np.ndarray,
    items_by_user: dict[int, np.ndarray],
    n_items: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Build the user-item indicator matrix of one user chunk.

    Parameters
    ----------
    users : np.ndarray of shape (n_chunk_users,), int64
        User idx of the chunk.
    items_by_user : dict[int, np.ndarray]
        Maps a user to its items; users missing from it get an all-false row.
    n_items : int
        Catalogue size.
    device : torch.device
        Device of the returned matrix.

    Returns
    -------
    torch.Tensor of shape (n_chunk_users, n_items), bool
        User-item indicator of the chunk.

    """
    out = torch.zeros(len(users), n_items, dtype=torch.bool, device=device)
    for row, u in enumerate(users):
        items = items_by_user.get(int(u))
        if items is not None and len(items) > 0:
            out[row, torch.from_numpy(items).to(device)] = True
    return out


def _sum_defined(values: np.ndarray, start: float = 0.0) -> tuple[float, int]:
    """
    Add the non-`NaN` entries of `values` onto `start`, left to right, and count them.

    Parameters
    ----------
    values : np.ndarray of shape (n_chunk_users,), float64
        Per-user values, `NaN` where undefined.
    start : float, default 0.0
        Value the sum starts from.

    Returns
    -------
    tuple[float, int]
        The sum, and the number of entries added.

    """
    total, count = start, 0
    for v in values:
        if not np.isnan(v):
            total += v
            count += 1
    return total, count


def _negated_mean(total: float, count: int) -> float:
    """Return `-(total / count)`, or `NaN` when `count` is zero."""
    return -(total / count) if count else float("nan")
