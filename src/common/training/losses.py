"""Loss functions."""

import torch
import torch.nn.functional as F


def bpr_loss(u: torch.Tensor, i_pos: torch.Tensor, i_neg: torch.Tensor) -> torch.Tensor:
    """
    Bayesian Personalised Ranking loss.

    Parameters
    ----------
    u : torch.Tensor of shape (B, d), float32
        User embeddings.
    i_pos : torch.Tensor of shape (B, d), float32
        Positive (interacted) item embeddings.
    i_neg : torch.Tensor of shape (B, d), float32
        Negative (sampled) item embeddings.

    Returns
    -------
    torch.Tensor of shape (), float32
        BPR loss averaged over the batch.

    """
    pos_score = (u * i_pos).sum(dim=-1)
    neg_score = (u * i_neg).sum(dim=-1)
    return -F.logsigmoid(pos_score - neg_score).mean()


def l2_reg_squared(*tensors: torch.Tensor) -> torch.Tensor:
    """
    Squared-L2 sum over given tensors, halved and divided by batch size.

    Parameters
    ----------
    *tensors : torch.Tensor of shape (n_k, d), float32
        Embedding tensors to regularise; the first one's row count is the batch size.

    Returns
    -------
    torch.Tensor of shape (), float32
        Regularisation term, or `tensor(0.0)` if no tensors are given.

    """
    # https://github.com/kuandeng/LightGCN/blob/b067ec0/LightGCN.py#L344-L346
    # https://github.com/wujcan/SSM-TensorFlow/blob/661767c/util/tool.py#L259-L260
    if not tensors:
        return torch.tensor(0.0)
    n = tensors[0].shape[0]
    return torch.stack([(t * t).sum() for t in tensors]).sum() / (2.0 * max(n, 1))


def l2_reg(*tensors: torch.Tensor) -> torch.Tensor:
    """
    Frobenius-norm sum over given tensors, divided by batch size.

    Parameters
    ----------
    *tensors : torch.Tensor of shape (n_k, d), float32
        Embedding tensors to regularise; the first one's row count is the batch size.

    Returns
    -------
    torch.Tensor of shape (), float32
        Regularisation term, or `tensor(0.0)` if no tensors are given.

    """
    # https://github.com/Coder-Yu/SELFRec/blob/5b02294/util/loss_torch.py#L18-L22
    if not tensors:
        return torch.tensor(0.0)
    n = tensors[0].shape[0]
    return torch.stack([torch.norm(t, p=2) for t in tensors]).sum() / max(n, 1)


def ssm_margin_loss(
    u: torch.Tensor,
    i_pos: torch.Tensor,
    i_neg: torch.Tensor,
    temperature: float,
    margin: float,
) -> torch.Tensor:
    """
    Shared-negative sampled-softmax loss with an additive margin.

    Cosine similarities over L2-normalised embeddings form a `(B, 1 + N)` logit matrix
    per batch: column 0 is each user's matched positive, the remaining `N` columns are
    one set of sampled negatives shared across the batch's positive pairs.

    The positive logit is handicapped by a constant additive cosine margin `s_ui - m`
    (the CosFace/AM-Softmax form).

    Parameters
    ----------
    u : torch.Tensor of shape (B, d), float32
        User embeddings, L2-normalised.
    i_pos : torch.Tensor of shape (B, d), float32
        Positive item embeddings, L2-normalised. Row `b` is the matched positive of user
        `b` (logit column 0).
    i_neg : torch.Tensor of shape (N, d), float32
        Shared negative item embeddings, L2-normalised. Every row scores the same `N`
        negatives.
    temperature : float
        Softmax temperature scaling the cosine logits.
    margin : float
        Additive cosine margin `m` (similarity units) subtracted from the positive logit
        before temperature scaling. `0.0` recovers the plain sampled softmax.

    Returns
    -------
    torch.Tensor of shape (), float32
        Sampled-softmax loss averaged over the batch.

    """
    u = F.normalize(u, dim=-1)
    i_pos = F.normalize(i_pos, dim=-1)
    i_neg = F.normalize(i_neg, dim=-1)
    pos = (u * i_pos).sum(dim=-1) - margin
    neg = u @ i_neg.t()
    logits = torch.cat([pos.unsqueeze(1), neg], dim=1) / temperature
    targets = torch.zeros(u.size(0), dtype=torch.long, device=u.device)
    return F.cross_entropy(logits, targets)


def info_nce(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Single-directional InfoNCE on two L2-normalised views of the same nodes.

    Equivalent to a single cross-entropy over the similarity rows. Each row of `z1` is
    the anchor and all rows of `z2` form its denominator.

    Parameters
    ----------
    z1 : torch.Tensor of shape (n, d), float32
        Anchor view of the node embeddings, L2-normalised internally.
    z2 : torch.Tensor of shape (n, d), float32
        Second view of the node embeddings, L2-normalised internally. Its rows form the
        denominator (positive on the diagonal, all others negatives).
    temperature : float
        Softmax temperature scaling the similarity logits.

    Returns
    -------
    torch.Tensor of shape (), float32
        Single-directional InfoNCE loss.

    """
    # https://github.com/Coder-Yu/SELFRec/blob/5b02294/util/loss_torch.py#L35-L50
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    logits = z1 @ z2.t() / temperature
    targets = torch.arange(z1.size(0), device=z1.device)
    return F.cross_entropy(logits, targets)


def mmd_stat_multikernel(
    x: torch.Tensor,
    y: torch.Tensor,
    kernel_mul: float,
    kernel_num: int,
) -> torch.Tensor:
    """
    Multi-kernel estimate of squared MMD.

    Implements the estimator referenced by AURL before Eq. 11. The kernel is the sum
    of `kernel_num` Gaussian kernels `exp(-||a - b||^2 / b_h)` with bandwidths
    `b_h = kernel_mul ** (h - kernel_num // 2) * b_0`. The base bandwidth `b_0` is the
    mean off-diagonal squared distance of the pooled sample, detached, so no gradient
    flows through the bandwidths. The estimate is the biased (V-statistic) squared MMD.

    Parameters
    ----------
    x : torch.Tensor of shape (m, d), float32
        First sample set.
    y : torch.Tensor of shape (n, d), float32
        Second sample set.
    kernel_mul : float
        Geometric ratio between consecutive kernel bandwidths.
    kernel_num : int
        Number `H` of RBF kernels in the geometric mixture.

    Returns
    -------
    torch.Tensor of shape (), float32
        Scalar squared-MMD estimate; `tensor(0.0)` when either sample has fewer than two
        points.

    References
    ----------
    Miaomiao Cai, Min Hou, Lei Chen, Le Wu, Haoyue Bai, Yong Li, and Meng Wang. 2024.
    Mitigating Recommendation Biases via Group-Alignment and Global-Uniformity in
    Representation Learning. ACM Trans. Intell. Syst. Technol. 15, 5, Article 101
    (October 2024), 27 pages. https://doi.org/10.1145/3664931

    Yongchun Zhu, Fuzhen Zhuang, and Deqing Wang. 2019. Aligning domain-specific
    distribution and classifier for cross-domain classification from multiple sources.
    In Proceedings of the Thirty-Third AAAI Conference on Artificial Intelligence and
    Thirty-First Innovative Applications of Artificial Intelligence Conference and Ninth
    AAAI Symposium on Educational Advances in Artificial Intelligence
    (AAAI'19/IAAI'19/EAAI'19), Vol. 33. AAAI Press, Article 735, 5989-5996.
    https://doi.org/10.1609/aaai.v33i01.33015989

    """
    # https://github.com/easezyc/deep-transfer-learning/blob/2969ed6/MUDA/MFSAN/MFSAN_2src/mmd.py
    # `mean(K_xx) + mean(K_yy) - mean(K_xy) - mean(K_yx)` instead of
    # `mean(XX + YY - XY - YX)` so unequal sample sizes are allowed.
    # Pairwise squared distances via ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a^T b, which
    # avoids the (N, N, d) broadcast
    # https://github.com/scikit-learn/scikit-learn/blob/afe5bcd/sklearn/metrics/pairwise.py#L409-L411
    m = x.shape[0]
    n = y.shape[0]
    if m < 2 or n < 2:
        return torch.zeros((), device=x.device)

    total = torch.cat([x, y], dim=0)
    n_total = total.shape[0]
    sq = (total * total).sum(dim=1)
    G = total @ total.t()
    l2 = (sq.unsqueeze(0) + sq.unsqueeze(1) - 2.0 * G).clamp_min(0.0)

    bandwidth = l2.detach().sum() / (n_total**2 - n_total)
    bandwidth = bandwidth / (kernel_mul ** (kernel_num // 2))
    kernels = torch.stack(
        [torch.exp(-l2 / (bandwidth * kernel_mul**h)) for h in range(kernel_num)]
    ).sum(dim=0)

    k_xx = kernels[:m, :m]
    k_yy = kernels[m:, m:]
    k_xy = kernels[:m, m:]
    k_yx = kernels[m:, :m]
    return k_xx.mean() + k_yy.mean() - k_xy.mean() - k_yx.mean()


def _group_mmd(
    emb: torch.Tensor,
    idx: torch.Tensor,
    pop_mask: torch.Tensor,
    kernel_mul: float,
    kernel_num: int,
    mmd_batch: int,
) -> torch.Tensor | None:
    """
    Popular-vs-unpopular multi-kernel MMD over the entities in `idx`.

    Parameters
    ----------
    emb : torch.Tensor of shape (n_entities, d), float32
        Full embedding table for this side.
    idx : torch.Tensor of shape (B,), int64
        Entity idx present in the batch (batch users or batch positive items).
    pop_mask : torch.Tensor of shape (n_entities,), bool
        Mask over entities, True where the entity is popular.
    kernel_mul : float
        Geometric ratio between consecutive RBF kernel bandwidths.
    kernel_num : int
        Number of RBF kernels in the mixture.
    mmd_batch : int
        Maximum number of entities kept per group, taken in batch order.

    Returns
    -------
    torch.Tensor of shape (), float32 or None
        Multi-kernel MMD between the popular (stop-gradient) and unpopular embeddings,
        or None when either group has fewer than two entities in the batch.

    """
    is_pop = pop_mask[idx]
    pop_idx = idx[is_pop][:mmd_batch]
    unpop_idx = idx[~is_pop][:mmd_batch]
    if pop_idx.numel() < 2 or unpop_idx.numel() < 2:
        return None
    # Stop-gradient on the popular group so only the unpopular embeddings move
    return mmd_stat_multikernel(
        emb[pop_idx].detach(), emb[unpop_idx], kernel_mul, kernel_num
    )


def _group_uniformity(
    emb: torch.Tensor, idx: torch.Tensor, t: float, uniform_batch: int
) -> torch.Tensor | None:
    """
    Wang & Isola uniformity over the entities in `idx`.

    Log of the mean pairwise Gaussian potential `exp(-t * ||.||^2)` between the
    unit-sphere-normalised embeddings (AURL Eq. 14-15).

    Parameters
    ----------
    emb : torch.Tensor of shape (n_entities, d), float32
        Full embedding table for this side.
    idx : torch.Tensor of shape (B,), int64
        Entity idx present in the batch (batch users or batch positive items).
    t : float
        Gaussian-potential temperature.
    uniform_batch : int
        Maximum number of entities kept, taken in batch order.

    Returns
    -------
    torch.Tensor of shape (), float32 or None
        Uniformity loss, or None when fewer than two entities are present.

    References
    ----------
    Tongzhou Wang and Phillip Isola. 2020. Understanding Contrastive Representation
    Learning through Alignment and Uniformity on the Hypersphere. In Proceedings of the
    37th International Conference on Machine Learning (Proceedings of Machine Learning
    Research), PMLR, 9929-9939. https://proceedings.mlr.press/v119/wang20k.html

    Miaomiao Cai, Min Hou, Lei Chen, Le Wu, Haoyue Bai, Yong Li, and Meng Wang. 2024.
    Mitigating Recommendation Biases via Group-Alignment and Global-Uniformity in
    Representation Learning. ACM Trans. Intell. Syst. Technol. 15, 5, Article 101
    (October 2024), 27 pages. https://doi.org/10.1145/3664931

    """
    idx = idx[:uniform_batch]
    if idx.numel() < 2:
        return None
    e = F.normalize(emb[idx], dim=-1)
    return torch.pdist(e, p=2).pow(2).mul(-t).exp().mean().log()


def _mean_of_sides(
    terms: tuple[torch.Tensor | None, torch.Tensor | None], device: torch.device
) -> torch.Tensor:
    """
    Average the user-side and item-side terms, dropping the degenerate ones.

    Parameters
    ----------
    terms : tuple[torch.Tensor | None, torch.Tensor | None]
        User-side and item-side terms, each a float32 tensor of shape `()`, or None for
        a degenerate side.
    device : torch.device
        Device of the zero returned when both sides are degenerate.

    Returns
    -------
    torch.Tensor of shape (), float32
        Mean of the non-degenerate terms, or zero when both sides are degenerate.

    """
    kept = [x for x in terms if x is not None]
    if not kept:
        return torch.zeros((), device=device)
    return torch.stack(kept).mean()


def aurl_align_loss(
    user_emb: torch.Tensor,
    item_emb: torch.Tensor,
    users: torch.Tensor,
    pos_items: torch.Tensor,
    user_pop_mask: torch.Tensor,
    item_pop_mask: torch.Tensor,
    kernel_mul: float,
    kernel_num: int,
    mmd_batch: int,
) -> torch.Tensor:
    """
    Unweighted AURL group-alignment loss (AURL Eq. 11-13).

    Mean of the user-side and item-side popular-vs-unpopular multi-kernel MMD over the
    batch entities.

    Parameters
    ----------
    user_emb : torch.Tensor of shape (n_users, d), float32
        Full user embedding table.
    item_emb : torch.Tensor of shape (n_items, d), float32
        Full item embedding table.
    users : torch.Tensor of shape (B,), int64
        Batch user idx.
    pos_items : torch.Tensor of shape (B,), int64
        Batch positive item idx.
    user_pop_mask : torch.Tensor of shape (n_users,), bool
        Mask over users, True where the user is popular.
    item_pop_mask : torch.Tensor of shape (n_items,), bool
        Mask over items, True where the item is popular.
    kernel_mul : float
        Geometric ratio between consecutive RBF kernel bandwidths.
    kernel_num : int
        Number of RBF kernels in the multi-kernel MMD estimator.
    mmd_batch : int
        Maximum number of entities kept per group per side.

    Returns
    -------
    torch.Tensor of shape (), float32
        Alignment loss. A side with fewer than two entities in either group is dropped,
        and the loss is zero when both sides are dropped.

    References
    ----------
    Miaomiao Cai, Min Hou, Lei Chen, Le Wu, Haoyue Bai, Yong Li, and Meng Wang. 2024.
    Mitigating Recommendation Biases via Group-Alignment and Global-Uniformity in
    Representation Learning. ACM Trans. Intell. Syst. Technol. 15, 5, Article 101
    (October 2024), 27 pages. https://doi.org/10.1145/3664931

    """
    user_mmd = _group_mmd(
        user_emb, users, user_pop_mask, kernel_mul, kernel_num, mmd_batch
    )
    item_mmd = _group_mmd(
        item_emb, pos_items, item_pop_mask, kernel_mul, kernel_num, mmd_batch
    )
    return _mean_of_sides((user_mmd, item_mmd), item_emb.device)


def aurl_uniform_loss(
    user_emb: torch.Tensor,
    item_emb: torch.Tensor,
    users: torch.Tensor,
    pos_items: torch.Tensor,
    t: float,
    uniform_batch: int,
) -> torch.Tensor:
    """
    Unweighted AURL global-uniformity loss (AURL Eq. 14-15).

    Mean of the user-side and item-side Wang & Isola uniformity over the batch entities.

    Parameters
    ----------
    user_emb : torch.Tensor of shape (n_users, d), float32
        Full user embedding table.
    item_emb : torch.Tensor of shape (n_items, d), float32
        Full item embedding table.
    users : torch.Tensor of shape (B,), int64
        Batch user idx.
    pos_items : torch.Tensor of shape (B,), int64
        Batch positive item idx.
    t : float
        Gaussian-potential temperature.
    uniform_batch : int
        Maximum number of entities kept per side.

    Returns
    -------
    torch.Tensor of shape (), float32
        Uniformity loss. A side with fewer than two entities is dropped, and the loss is
        zero when both sides are dropped.

    """
    user_uni = _group_uniformity(user_emb, users, t, uniform_batch)
    item_uni = _group_uniformity(item_emb, pos_items, t, uniform_batch)
    return _mean_of_sides((user_uni, item_uni), item_emb.device)
