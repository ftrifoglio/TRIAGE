"""
Popularity Compensation (PC).

Ziwei Zhu, Yun He, Xing Zhao, Yin Zhang, Jianling Wang, and James Caverlee. 2021.
Popularity-Opportunity Bias in Collaborative Filtering. In Proceedings of the 14th ACM
International Conference on Web Search and Data Mining (WSDM '21). Association for
Computing Machinery, New York, NY, USA, 85-93. https://doi.org/10.1145/3437963.3441820
"""

import torch


def popularity_compensation(
    scores: torch.Tensor, pop: torch.Tensor, pc_a: float, pc_b: float
) -> torch.Tensor:
    """
    Post-hoc Popularity Compensation of a user-item score matrix.

    Parameters
    ----------
    scores : torch.Tensor of shape (n_chunk_users, n_items), float32
        Score rows, train/val positions already set to `-inf`.
    pop : torch.Tensor of shape (n_items,), float64
        Item training degrees; clamped to `>= 1` for the reciprocal.
    pc_a : float
        Compensation strength; `0` disables it.
    pc_b : float
        Mixing coefficient between the score-dependent (`pc_b`) and constant (`1 -
        pc_b`) parts of the correction. Inert when `pc_a` is `0`.

    Returns
    -------
    torch.Tensor of shape (n_chunk_users, n_items), float64
        The compensated score matrix, with masked entries unchanged.

    """
    # https://github.com/fuxiAIlab/r-AdjNorm/blob/6b6edd3/src/utility/batch_test.py#L220-L236
    # without the per-user /= (M - user_item_cnt) inside norm() as it divides both n and
    # m by the same scalar, so it cancels in n / m.
    # we mask c for any pc_b, whereas the reference only zeroes -inf entries in norm()
    # at pc_b = 0, -inf * 0 = nan on masked entries, so m, and with it the whole
    # compensated row, becomes nan
    finite = torch.isfinite(scores)
    z = torch.zeros_like(scores)
    s = torch.where(finite, scores, z)
    inv_pop = (1.0 / pop.clamp(min=1.0)).unsqueeze(0)
    c = torch.where(finite, inv_pop * (s * pc_b + (1.0 - pc_b)), z)
    n = s.norm(dim=1)
    m = c.norm(dim=1).clamp(min=1e-12)
    adj = pc_a * c * (n / m).unsqueeze(1)
    return torch.where(finite, scores + adj, scores)
