"""SimGCL bundle: encoder + identity sampler + aux InfoNCE loss."""

import torch

from common.training.config import Config
from common.training.data import InteractionDataset, build_norm_adj
from common.training.encoder import SimGCLEncoder
from common.training.losses import info_nce
from common.training.registry import ModelBundle, register_model
from common.training.sampler import IdentitySampler
from common.training.trainer import AuxLoss, BatchContext


def make_cl_loss(encoder: SimGCLEncoder, weight: float, tau: float) -> AuxLoss:
    """
    Build the SimGCL InfoNCE loss over two noise-perturbed views.

    Parameters
    ----------
    encoder : SimGCLEncoder
        Encoder whose noisy propagation produces the two contrastive views.
    weight : float
        Weight applied to the sum of the user- and item-side InfoNCE terms.
    tau : float
        Temperature for the InfoNCE loss.

    Returns
    -------
    AuxLoss
        Loss callable returning the weighted InfoNCE loss over the batch users and
        positive items.

    """
    # https://github.com/Coder-Yu/SELFRec/blob/5b02294/model/graph/SimGCL.py#L43-L50
    # Indices are deduplicated before contrasting.
    # A node repeated across interactions would otherwise appear as its own
    # near-identical, which is a degree-correlated false negative.

    def fn(ctx: BatchContext) -> torch.Tensor:
        e0 = encoder.initial_table()
        view1 = encoder.propagate(ctx.adj, e0, perturbed=True)
        view2 = encoder.propagate(ctx.adj, e0, perturbed=True)
        users = torch.unique(ctx.users)
        pos_items = torch.unique(ctx.pos_items)
        u1 = view1[: encoder.n_users][users]
        u2 = view2[: encoder.n_users][users]
        i1 = view1[encoder.n_users :][pos_items]
        i2 = view2[encoder.n_users :][pos_items]
        return weight * (info_nce(u1, u2, tau) + info_nce(i1, i2, tau))

    return fn


@register_model("simgcl")
def build(cfg: Config, dataset: InteractionDataset, device: str) -> ModelBundle:
    """
    Build the SimGCL bundle.

    Parameters
    ----------
    cfg : Config
        Full run configuration.
    dataset : InteractionDataset
        Train graph and index sizes used to size the encoder and build the normalised
        adjacency.
    device : str
        Torch device onto which the normalised adjacency is moved.

    Returns
    -------
    ModelBundle
        Bundle of the SimGCL encoder, an `IdentitySampler` over the normalised
        adjacency, and the InfoNCE auxiliary loss.

    """
    tau = float(cfg.model.get("tau", 0.2))
    cl_weight = float(cfg.model.get("cl_weight", 0.5))

    encoder = SimGCLEncoder(
        n_users=dataset.n_users,
        n_items=dataset.n_items,
        emb_dim=cfg.encoder.emb_dim,
        n_layers=cfg.encoder.n_layers,
        eps=cfg.encoder.eps,
        readout_ego=cfg.encoder.readout_ego,
    )
    norm_adj = build_norm_adj(dataset).to(device)
    return ModelBundle(
        encoder=encoder,
        neighbourhood_sampler=IdentitySampler(norm_adj),
        aux_losses=[make_cl_loss(encoder, cl_weight, tau)],
    )
